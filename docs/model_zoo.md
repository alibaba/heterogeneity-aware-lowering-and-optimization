<!--
### An Object Dection Example

[This example](docs/yolo-examples.md) includes a complete end-to-end workflow of
deploying YOLOv3 object detection model.
It demonstrates the use of HALO to compile a subgraph, to change the data layout,
to override input dimensions.

DEAD LINK:
| [Resnet V1-18](https://arxiv.org/abs/1512.03385)                                                                  | [ONNX](https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v1-7.onnx)                                                        | models/vision/classification/resnet       |
| [ResNet V2-50](https://arxiv.org/abs/1603.05027)                                                                  | [ONNX](https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx)                                                        | models/vision/classification/resnet       |
| [ResNet V2-101](https://arxiv.org/abs/1603.05027)                                                                 | [ONNX](https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v2-7.onnx)                                                       | models/vision/classification/resnet       |
| [CaffeNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | [BVLC/Caffe](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)                                                                         | models/vision/classification/caffenet     |

### Using Inside Python

HALO generated code can also be used inside Python.

[This example](docs/using-inside-python.md) shows how to do image classification
with [CaffeNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
inside Python using HALO.
-->

### List of Demos

[models directory](../models/) contains scripts for the following models, which download the pretrained models, compile and deploy them using HALO on X86-CPU or NVGPU.
Please refer to [Instruction.md](../models/Instruction.md) for more details about how to run the examples.


#### Image Classification

| Model Class                                                                                                       | Model Source                                                                                                                                                   | HALO Examples                             |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| [AlexNet](https://arxiv.org/abs/1404.5997)                                                                        | [PyTorch](https://pytorch.org/hub/pytorch_vision_alexnet/)                                                                                                     | models/vision/classification/alexnet      |
| [DenseNet-121](https://arxiv.org/abs/1608.06993)                                                                  | [PyTorch](https://pytorch.org/hub/pytorch_vision_densenet/)                                                                                                    | models/vision/classification/densenet     |
| [GoogleNet](https://arxiv.org/abs/1409.4842)                                                                      | [PyTorch](https://pytorch.org/hub/pytorch_vision_googlenet/)                                                                                                   | models/vision/classification/googlenet    |
| [Inception_V1](https://arxiv.org/abs/1409.4842)                                                                   | [ONNX](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v1)                                                  | models/vision/classification/inception    |
| [Inception_V3](https://arxiv.org/abs/1512.00567)                                                                  | [PyTorch](https://pytorch.org/hub/pytorch_vision_inception_v3/)                                                                                                | models/vision/classification/inception    |
| [MNIST](http://yann.lecun.com/exdb/publis)                                                                        | [TensorFlow Tutorial](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/tutorials/mnist/beginners/index.md) | models/vision/classification/mnist_simple |
| [MobileNet_V2](https://arxiv.org/abs/1801.04381)                                                                  | [PyTorch](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)                                                                                                | models/vision/classification/mobilenet    |
| [ShuffleNet](https://arxiv.org/abs/1707.01083)                                                                    | [ONNX](https://github.com/onnx/models/blob/master/vision/classification/shufflenet/model/shufflenet-9.onnx)                                                    | models/vision/classification/shufflenet   |
| [ShuffleNet_V2](https://arxiv.org/abs/1707.01083)                                                                 | [ONNX](https://github.com/onnx/models/blob/master/vision/classification/shufflenet/model/shufflenet-v2-10.onnx)                                                |
| [SqueezeNet_10](https://arxiv.org/abs/1602.07360)                                                                 | [PyTorch](https://pytorch.org/hub/pytorch_vision_squeezenet/)                                                                                                  | models/vision/classification/squeezenet   |
| [SqueezeNet_11](https://arxiv.org/abs/1602.07360)                                                                 | [PyTorch](https://pytorch.org/hub/pytorch_vision_squeezenet/)                                                                                                  | models/vision/classification/squeezenet   |
| [VGG-16](https://arxiv.org/abs/1409.1556)                                                                         | [PyTorch](https://pytorch.org/hub/pytorch_vision_vgg/)                                                                                                         | models/vision/classification/vgg          |
| [VGG-19](https://arxiv.org/abs/1409.1556)                                                                         | [PyTorch](https://pytorch.org/hub/pytorch_vision_vgg/)                                                                                                         | models/vision/classification/vgg          |


#### Object Detection & Segmentation

| Model Class                                   | Model Source                                                                                   | HALO Examples                   |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------- |
| [YOLO v3](https://pjreddie.com/darknet/yolo/) | [ONNX](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3) | models/vision/detection/yolo    |
| [UNet](https://arxiv.org/abs/1505.04597)      | [PyTorch](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)                | models/vision/segmentation/unet |
<!--
| RetinaNet                                      |                                                                                                  |
| SSD                                            |                                                                                                  |
-->

<!--
#### NLP <a name="nlp"/>

| Model Class | Description |
| ----------- | ----------- |
| BERT        |             |
|             |
-->

