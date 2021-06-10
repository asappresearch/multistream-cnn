Multistream Convolutional Neural Network (CNN)
==============================================

A multistream CNN is a novel neural network architecture for robust acoustic modeling in speech recognition tasks. It processes input speech with diverse resolutions by applying different dilation rates to convolutional neural networks across multiple streams to achieve the robustness. The dilation rate of 3 are selected from the multiples of a sub-sampling rate of 3 frames. Each stream stacks TDNN-F layers (a variant of 1D CNN), and output embedding vectors from the streams are concatenated then projected to the final layer, as illustrated below:
[My image](khan-asapp.github.com/multistream-cnn/asapp/multistream-cnn.png)



