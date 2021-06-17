#!/usr/bin/env bash

# Copyright 2021 ASAPP (author: Kyu J. Han)
# MIT License

# This recipe is based on a paper titled "Multistream CNN for Robust Acoustic Modeling", 
# https://arxiv.org/abs/2005.10470.

# %WER 15.7 | 2628 21594 | 86.5 9.5 4.0 2.2 15.7 50.8 | exp/chain/multistream_cnn_1a_sp/decode_eval2000_fsh_sw1_fg/score_7_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 12.6 | 4459 42989 | 89.1 7.6 3.4 1.6 12.6 48.8 | exp/chain/multistream_cnn_1a_sp/decode_eval2000_fsh_sw1_fg/score_8_0.0/eval2000_hires.ctm.filt.sys
# %WER 9.2 | 1831 21395 | 91.8 5.6 2.6 1.1 9.2 44.7 | exp/chain/multistream_cnn_1a_sp/decode_eval2000_fsh_sw1_fg/score_10_0.5/eval2000_hires.ctm.swbd.filt.sys

# Copyright 2017 University of Chinese Academy of Sciences (UCAS) Gaofeng Cheng
# Apache 2.0

set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/multistream_cnn_1a # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=

# training options
leftmost_questions_truncate=-1
num_epochs=6
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
num_jobs_initial=8
num_jobs_final=8
minibatch_size=32
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}$suffix
build_tree_train_set=train_nodup
train_set=train_nodup_sp
build_tree_ali_dir=exp/tri5a_ali
treedir=exp/chain/tri6_tree
lang=data/lang_chain

# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $build_tree_ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri5a exp/tri5a_lats_nodup$suffix
  rm exp/tri5a_lats_nodup$suffix/fsts.*.gz # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --leftmost-questions-truncate $leftmost_questions_truncate \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 11000 data/$build_tree_train_set $lang $build_tree_ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_first_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.002"
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.0"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # MFCC to filterbank
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat

  linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025
  batchnorm-component name=idct-batchnorm input=idct

  spec-augment-layer name=idct-spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20
  combine-feature-maps-layer name=combine_inputs input=Append(idct-spec-augment, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40
  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  
  relu-batchnorm-dropout-layer name=tdnn6a $affine_opts input=cnn5 dim=512
  tdnnf-layer name=tdnnf7a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf8a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf9a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf10a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf11a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf12a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf13a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf14a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf15a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf16a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf17a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf18a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf19a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf20a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf21a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf22a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6
  tdnnf-layer name=tdnnf23a $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=6

  relu-batchnorm-dropout-layer name=tdnn6b $affine_opts input=cnn5 dim=512
  tdnnf-layer name=tdnnf7b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf8b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf9b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf10b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf11b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf12b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf13b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf14b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf15b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf16b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf17b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf18b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf19b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf20b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf21b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf22b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9
  tdnnf-layer name=tdnnf23b $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=9

  relu-batchnorm-dropout-layer name=tdnn6c $affine_opts input=cnn5 dim=512
  tdnnf-layer name=tdnnf7c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf8c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf9c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf10c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf11c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf12c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf13c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf14c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf15c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf16c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf17c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf18c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf19c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf20c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf21c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf22c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12
  tdnnf-layer name=tdnnf23c $tdnnf_opts dim=512 bottleneck-dim=80 time-stride=12

  relu-batchnorm-dropout-layer name=tdnn17 $affine_opts input=Append(tdnnf23a,tdnnf23b,tdnnf23c) dim=768
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri5a_lats_nodup$suffix \
    --dir $dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_fsh_sw1_tg $dir $dir/graph_fsh_sw1_tg
fi

decode_suff=fsh_sw1_tg
graph_dir=$dir/graph_fsh_sw1_tg
if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in rt03 eval2000; do
    (
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 50 --cmd "$decode_cmd" $iter_opts \
      --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
      $graph_dir data/${decode_set}_hires \
      $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_${decode_suff} || exit 1;
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_fsh_sw1_{tg,fg} data/${decode_set}_hires \
      $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_fsh_sw1_{tg,fg} || exit 1;
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

test_online_decoding=true
lang=data/lang_fsh_sw1_tg
if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in rt03 eval2000; do
    (
    # note: we just give it "$decode_set" as it only uses the wav.scp, the
    # feature type does not matter.

    steps/online/nnet3/decode.sh --nj 50 --cmd "$decode_cmd" $iter_opts \
      --acwt 1.0 --post-decode-acwt 10.0 \
      $graph_dir data/${decode_set}_hires \
      ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_fsh_sw1_{tg,fg} data/${decode_set}_hires \
      ${dir}_online/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_fsh_sw1_{tg,fg} || exit 1;
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in online decoding"
    exit 1
  fi
fi

exit 0;
