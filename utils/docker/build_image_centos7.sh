#!/bin/bash -xe

VER="0.8.1"
FLAVOR="devel"
NAMESPACE="registry-intl.us-west-1.aliyuncs.com/computation"

base_image_gpu="nvidia/cuda:11.4.2-cudnn8-devel-centos7"
base_image_cpu="centos:7.2.1511"

#docker build --build-arg BASE_IMAGE=${base_image_cpu} \
#  -t $NAMESPACE/halo:$VER-$FLAVOR-x86_64-centos7  -f Dockerfile.centos .

docker build --build-arg BASE_IMAGE=${base_image_gpu} \
-t $NAMESPACE/halo:$VER-$FLAVOR-cuda11.4.2-cudnn8-centos7 -f Dockerfile.centos .

