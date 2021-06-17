#!/usr/bin/env bash

# Copyright 2021 ASAPP (author: Kyu J. Han)
# MIT License

# This recipe is based on a paper titled "Multistream CNN for Robust Acoustic Modeling",
# https://arxiv.org/abs/2005.10470.

set -e

# steps/info/chain_dir_info.pl exp/chain_cleaned/multistream_cnn_1a
# exp/chain_cleaned/multistream_cnn_1a: num-iters=931 nj=8..8 num-params=20.4M dim=40+100->5992 combine=-0.039->-0.038 (over 10) xent:train/valid[619,930,final]=(-0.868,-0.735,-0.729/-0.917,-0.808,-0.799) logprob:train/valid[619,930,final]=(-0.054,-0.040,-0.039/-0.064,-0.053,-0.053)

# local/chain/compare_wer.sh exp/chain_cleaned/multistream_cnn_1a
# System                     multistream_cnn_1a
# WER on dev(fglarge)              3.20
# WER on dev(tglarge)              3.32
# WER on dev(tgmed)                4.01
# WER on dev(tgsmall)              4.43
# WER on dev_other(fglarge)        7.68
# WER on dev_other(tglarge)        8.03
# WER on dev_other(tgmed)          9.74
# WER on dev_other(tgsmall)       10.50
# WER on test(fglarge)             3.54
# WER on test(tglarge)             3.70
# WER on test(tgmed)               4.37
# WER on test(tgsmall)             4.81
# WER on test_other(fglarge)       7.87
# WER on test_other(tglarge)       8.17
# WER on test_other(tgmed)         9.83
# WER on test_other(tgsmall)      10.57
# Final train prob              -0.0392
# Final valid prob              -0.0530
# Final train prob (xent)       -0.7292
# Final valid prob (xent)       -0.7994
# Num-parameters               20440944

# configs for 'chain'
stage=0
decode_nj=50
train_set=train_960_cleaned
gmm=tri6b_cleaned
nnet3_affix=_cleaned

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# TDNN options
frames_per_eg=150,110,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=true  # if true, it will run the last decoding stage.
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
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm 6 --num-processes 3 \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/multistream_cnn${affix:+_$affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

# if we are using the speed-perturbed data we need to generate
# alignments for it.

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

# Please take this as a reference on how to specify all the options of
# local/chain/run_chain_common.sh
local/chain/run_chain_common.sh --stage $stage \
                                --gmm-dir $gmm_dir \
                                --ali-dir $ali_dir \
                                --lores-train-data-dir ${lores_train_data_dir} \
                                --lang $lang \
                                --lat-dir $lat_dir \
                                --num-leaves 7000 \
                                --tree-dir $tree_dir || exit 1;

if [ $stage -le 14 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.0"
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

if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{09,10,11,12}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 2500000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 8 \
    --trainer.optimization.num-jobs-final 8 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.00001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

graph_dir=$dir/graph_tgsmall
if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang_test_tgsmall $dir $graph_dir
  # remove <UNK> from the graph, and convert back to const-FST.
  fstrmsymbols --apply-to-output=true --remove-arcs=true "echo 3|" $graph_dir/HCLG.fst - | \
    fstconvert --fst_type=const > $graph_dir/temp.fst
  mv $graph_dir/temp.fst $graph_dir/HCLG.fst
fi

iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in test_clean test_other dev_clean dev_other; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall || exit 1
      steps/lmrescore.sh --cmd "$decode_cmd" --self-loop-scale 1.0 data/lang_test_{tgsmall,tgmed} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,tgmed} || exit 1
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,tglarge} || exit 1
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,fglarge} || exit 1
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if $test_online_decoding && [ $stage -le 18 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/nnet3${nnet3_affix}/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for data in test_clean test_other dev_clean dev_other; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $graph_dir data/${data} ${dir}_online/decode_${data}_tgsmall || exit 1

    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


exit 0;
