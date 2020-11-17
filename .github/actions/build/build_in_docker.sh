#!/bin/bash -xe

VER="0.5"

IMAGE="halo:$VER-cuda10.0-cudnn7-ubuntu18.04"
CONTAINER_NAME="halo.ci-$VER-GPU"
docker_run_flag="--runtime=nvidia"
  
#IMAGE="halo:$VER-x86_64-ubuntu18.04"
#  CONTAINER_NAME=$USER.halo-$VER-CPU

MOUNT_DIR="$PWD"
if [ ! -z "$1" ]; then
  MOUNT_DIR=$1
fi

DOCKER_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`

if [ -z "$DOCKER_ID" ]; then
  docker run $docker_run_flag -t -d --name $CONTAINER_NAME -v $MOUNT_DIR:/host  --tmpfs /tmp:exec --rm $IMAGE
fi

docker exec $CONTAINER_NAME bash -c 'cd /host && rm -fr build && mkdir -p build && cd build && cmake -G Ninja ../halo && ninja && ninja check-halo'
