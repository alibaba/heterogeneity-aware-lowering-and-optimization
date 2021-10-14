#!/bin/bash -xe

VER="0.7.4"
FLAVOR="devel"
NAMESPACE="registry-intl.us-west-1.aliyuncs.com/computation"

base_image_gpu="nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04"
base_image_cpu="ubuntu:18.04"

#docker build --build-arg BASE_IMAGE=${base_image_cpu} \
#  -t $NAMESPACE/halo:$VER-$FLAVOR-x86_64-ubuntu18.04 .

docker build --build-arg BASE_IMAGE=${base_image_gpu} \
  -t $NAMESPACE/halo:$VER-$FLAVOR-cuda11.3.1-cudnn8-ubuntu18.04 .

