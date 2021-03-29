#!/bin/bash

base_image_gpu="nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04"
base_image_cpu="ubuntu:18.04"

docker build --build-arg BASE_IMAGE=${base_image_cpu} \
  -t latest-x86_64-ubuntu18.04 .

docker build --build-arg BASE_IMAGE=${base_image_gpu} \
  -t latest-cuda10.0-cudnn7-ubuntu18.04 .
