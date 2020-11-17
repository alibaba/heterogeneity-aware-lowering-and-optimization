#!/bin/bash

base_image_gpu="nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04"
base_image_cpu="ubuntu:18.04"

docker build --build-arg BASE_IMAGE=${base_image_cpu} \
  -t halo:0.5-x86_64-ubuntu18.04 .

docker build --build-arg BASE_IMAGE=${base_image_gpu} \
  -t halo:0.5-cuda10.0-cudnn7-ubuntu18.04 .
