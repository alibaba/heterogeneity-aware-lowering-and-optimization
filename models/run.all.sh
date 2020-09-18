#!/bin/bash

source env.src

./vision/classification/alexnet/run_alexnet.sh
./vision/classification/caffenet/run_caffenet.sh
./vision/classification/caffenet/run_caffenet_onnx.sh
make -C vision/detection/yolo run
./vision/classification/densenet/run_densenet.sh
./vision/classification/googlenet/run_googlenet.sh
./vision/classification/inception/run_inception_v1.sh
./vision/classification/inception/run_inception_v3.sh
./vision/classification/mnist_simple/run_mnist_simple.sh
./vision/classification/mobilenet/run_mobilenet_v2.sh
./vision/classification/resnet/run_resnet_v1_18.sh
./vision/classification/resnet/run_resnet_v2_50.sh
./vision/classification/resnet/run_resnet_v2_101.sh
./vision/classification/shufflenet/run_shufflenet.sh
./vision/classification/squeezenet/run_squeezenet_1_1.sh
./vision/classification/squeezenet/run_squeezenet_1_0.sh
./vision/classification/vgg/run_vgg19.sh
./vision/classification/vgg/run_vgg16.sh
./vision/segmentation/unet/run_unet.sh
