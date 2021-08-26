#!/bin/bash
source ../env.src 


bash ./classification/alexnet/run_alexnet_tensorrt.sh

# bash ./classification/caffenet/run_caffenet_tensorrt.sh # segmentation fault

bash ./classification/densenet/run_densenet_tensorrt.sh

# bash ./classification/googlenet/run_googlenet_tensorrt.sh  # segementation fault

# bash ./classification/inception/run_inception_v1_tensorrt.sh  # reshape 问题 

bash ./classification/inception/run_inception_v3_tensorrt.sh

bash ./classification/mobilenet/run_mobilenet_v2_tensorrt.sh

bash ./classification/resnet/run_resnet_v2_50_tensorrt.sh

bash ./classification/resnet/run_resnet_v1_18_tensorrt.sh

bash ./classification/resnet/run_resnet_v2_101_tensorrt.sh

# bash ./classification/shufflenet/run_shufflenet_tensorrt.sh  # reshape 问题

# bash ./classification/squeezenet/run_squeezenet_1_0_tensorrt.sh # segmentation fault

# bash ./classification/squeezenet/run_squeezenet_1_1_tensorrt.sh # segmentation fault 

bash ./classification/vgg/run_vgg16_tensorrt.sh

bash ./classification/vgg/run_vgg19_tensorrt.sh

bash ./classification/efficientnet/run_efficientnet_tensorrt.sh   

# bash ./classification/rcnn/run_rcnn_tensorrt.sh

# bash ./body_analysis/arcface/run_arcface_tensorrt.sh