[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

![Build and Test](https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization/workflows/Build%20and%20Test/badge.svg?branch=master)
![API Publish](https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization/workflows/API%20Publish/badge.svg?branch=master)


HALO
===============

**H**eterogeneity-**A**ware **L**owering and **O**ptimization (**HALO**) is a heterogeneous computing acceleration platform based on compiler technology.
It exploits the powerfulness of heterogeneous computing while hiding the heterogeneity of computing resources
through the abstract, extendable interface called Open Deep Learning API (**ODLA**). 
HALO provides a unified Ahead-Of-Time compilation solution, auto tailored for cloud, edge, and IoT scenarios.

## Partners

We appreciate the support of ODLA runtimes from the following partners:

<!-- alphabetical order of partners -->

<table style="width:100%">
<tr>
<td><a href=https://www.graphcore.ai/posts/graphcore-announces-support-for-odla>
    <img src="./docs/partners/graphcore.png" width=80%></a></td>
<td><a href=https://www.intel.com><img src="./docs/partners/intel.png" width=80%>
    </a></td>
<td><a href=https://www.qualcomm.com/products/cloud-artificial-intelligence>
    <img title="Qualcomm Cloud AI 100" src="./docs/partners/qualcomm.jpg" width=80%>
</a></td>
</tr>
</table>

<!--
|[![GraphCore Logo](./docs/partners/graphcore.png)](https://www.graphcore.ai/posts/graphcore-announces-support-for-odla) | ![Intel Logo](./docs/partners/intel.png) | [![Qualcomm Logo](./docs/partners/qualcomm.jpg "Support of ODLA Runtime for Qualcomm® Cloud AI 100, a high-performance AI inference accelerator")](https://www.qualcomm.com/products/cloud-artificial-intelligence) |
:--:|:--:|:--:
-->

## What Can HALO Do?

Simply speaking, HALO compiles AI models into C/C++ code.

The generated code are based on Open Deep Learning API (**ODLA**), which allows it run on any platforms with corresponding ODLA Runtime Library.

<center>
<img src="docs/halo_odla.png" width="85%" align=middle>
</center>

So far, HALO supports the compilation of models of
- Caffe
- ONNX
- TensorFlow
- TFLite

Various ODLA runtime libraries are implemented:
- [Eigen](http://eigen.tuxfamily.org)
- [GraphCore® IPU](https://www.graphcore.ai)
- [Intel® oneAPI](https://github.com/oneapi-src)
- [Qualcomm® Cloud AI 100](https://www.qualcomm.com/products/cloud-artificial-intelligence)
- [TensorRT™](https://developer.nvidia.com/tensorrt)
- [XNNPACK](https://github.com/google/XNNPACK)

## How to Use HALO

To build HALO, please follow the instructions [here](docs/how_to_build.md).

The workflow of deploying models using HALO includes:
1. Use HALO to compile the model file(s) into an ODLA-based C/C++ source file.
2. Use a C/C++ compiler to compile the generated C/C++ file into an object file.
3. Link the object file, the weight binary, and specific ODLA runtime library together.

### A Simple Example

Let's start with a simple example of MNIST based on
[TensorFlow Tutorial](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/tutorials/mnist/beginners/index.md).
The diagram below shows the overall workflow:

<center>
<img src="docs/mnist.png" align="middle" width="75%">
</center>

Brief explanations:

HALO generates 3 files: 
* mnist.h : the header file to be used by application.
* mnist.cc : the ODLA C++ file that represents the model.
* mnist.bin : the weights in ELF format.

To application, the inference is simply viewed as a function call ``mnist()``. 

Note that, for portability purpose, HALO always exports functions in the C convention even though the output file model.cc is in the C++ format.

More detailed explanations can be found [here](docs/mnist_sample.md).
Example code can be found [here](models/vision/classification/mnist_simple)

Please refer to [HALO options list](docs/halo_cl_options.md) for all command line options.

### More Examples

#### An Object Dection Example

[This example](docs/yolo-examples.md) includes a complete end-to-end workflow of
deploying YOLOv3 object detection model. 
It demonstrates the use of HALO to compile a subgraph, to change the data layout,
to override input dimensions.

#### Using Inside Python 

HALO generated code can also be used inside Python.

[This example](docs/using-inside-python.md) shows how to do image classification
with [CaffeNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
inside Python using HALO.

#### List of Demos 

[models directory](models/) contains scripts for the following models, which download the pretrained models, compile and deploy them using HALO on X86-CPU or NVGPU.
Please refer to [Instruction.md](models/Instruction.md) for more details about how to run the examples.


#### Image Classification <a name="image-classification"/>

| Model Class                                                                                                       | Model Source                                                                                                                                                   | HALO Examples                             |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| [AlexNet](https://arxiv.org/abs/1404.5997)                                                                        | [PyTorch](https://pytorch.org/hub/pytorch_vision_alexnet/)                                                                                                     | models/vision/classification/alexnet      |
| [CaffeNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | [BVLC/Caffe](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)                                                                         | models/vision/classification/caffenet     |
| [DenseNet-121](https://arxiv.org/abs/1608.06993)                                                                  | [PyTorch](https://pytorch.org/hub/pytorch_vision_densenet/)                                                                                                    | models/vision/classification/densenet     |
| [GoogleNet](https://arxiv.org/abs/1409.4842)                                                                      | [PyTorch](https://pytorch.org/hub/pytorch_vision_googlenet/)                                                                                                   | models/vision/classification/googlenet    |
| [Inception_V1](https://arxiv.org/abs/1409.4842)                                                                   | [ONNX](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v1)                                                  | models/vision/classification/inception    |
| [Inception_V3](https://arxiv.org/abs/1512.00567)                                                                  | [PyTorch](https://pytorch.org/hub/pytorch_vision_inception_v3/)                                                                                                | models/vision/classification/inception    |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                                                                        | [TensorFlow Tutorial](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/tutorials/mnist/beginners/index.md) | models/vision/classification/mnist_simple |
| [MobileNet_V2](https://arxiv.org/abs/1801.04381)                                                                  | [PyTorch](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)                                                                                                | models/vision/classification/mobilenet    |
| [Resnet V1-18](https://arxiv.org/abs/1512.03385)                                                                  | [ONNX](https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v1-7.onnx)                                                        | models/vision/classification/resnet       |
| [ResNet V2-50](https://arxiv.org/abs/1603.05027)                                                                  | [ONNX](https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx)                                                        | models/vision/classification/resnet       |
| [ResNet V2-101](https://arxiv.org/abs/1603.05027)                                                                 | [ONNX](https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v2-7.onnx)                                                       | models/vision/classification/resnet       |
| [ShuffleNet](https://arxiv.org/abs/1707.01083)                                                                    | [ONNX](https://github.com/onnx/models/blob/master/vision/classification/shufflenet/model/shufflenet-9.onnx)                                                    | models/vision/classification/shufflenet   |
| [ShuffleNet_V2](https://arxiv.org/abs/1707.01083)                                                                 | [ONNX](https://github.com/onnx/models/blob/master/vision/classification/shufflenet/model/shufflenet-v2-10.onnx)                                                |
| [SqueezeNet_10](https://arxiv.org/abs/1602.07360)                                                                 | [PyTorch](https://pytorch.org/hub/pytorch_vision_squeezenet/)                                                                                                  | models/vision/classification/squeezenet   |
| [SqueezeNet_11](https://arxiv.org/abs/1602.07360)                                                                 | [PyTorch](https://pytorch.org/hub/pytorch_vision_squeezenet/)                                                                                                  | models/vision/classification/squeezenet   |
| [VGG-16](https://arxiv.org/abs/1409.1556)                                                                         | [PyTorch](https://pytorch.org/hub/pytorch_vision_vgg/)                                                                                                         | models/vision/classification/vgg          |
| [VGG-19](https://arxiv.org/abs/1409.1556)                                                                         | [PyTorch](https://pytorch.org/hub/pytorch_vision_vgg/)                                                                                                         | models/vision/classification/vgg          |


#### Object Detection & Segmentation <a name="object-detection--segmentation"/>

| Model Class                                    | Model Source                                                                                     | HALO Examples                      |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------- |
| [YOLO v3](https://pjreddie.com/darknet/yolo/)  | [ONNX](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3)   | models/vision/detection/yolo       |
| [UNet](https://arxiv.org/abs/1505.04597)       | [PyTorch](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)                  | models/vision/segmentation/unet    |
| RetinaNet                                      |                                                                                                  |
| SSD                                            |                                                                                                  |

#### NLP <a name="nlp"/>

| Model Class | Description |
| ----------- | ----------- |
| BERT        |             |
|             |

[HALO Command Line Options](docs/halo_cl_options.md)

## Contributions and Feedback

We're always looking for help to improve HALO.

We mainly follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) is used to enforce the coding style check.

We use GitHub issues to track bugs.

## License

HALO is licensed under the [Apache 2.0 License](LICENSE)
