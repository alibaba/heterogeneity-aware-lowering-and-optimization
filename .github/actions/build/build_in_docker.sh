#!/bin/bash -xe

VER="0.5"

IMAGE="halo:$VER-cuda10.0-cudnn7-ubuntu18.04"
CONTAINER_NAME=halo.ci-$VER-GPU
docker_run_flag="--runtime=nvidia"
  
#IMAGE="halo:$VER-x86_64-ubuntu18.04"
#  CONTAINER_NAME=$USER.halo-$VER-CPU

MOUNT_DIR=$PWD
if [ ! -z "$1" ]; then
  MOUNT_DIR=$1
fi

GROUP=`id -g -n`
GROUPID=`id -g`
OLD_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`

if [ -z "$OLD_ID" ]; then
  ID=`docker run $docker_run_flag --privileged -t -d --name $CONTAINER_NAME -v $MOUNT_DIR:/host  --tmpfs /tmp:exec --rm $IMAGE `
fi

docker exec -it $CONTAINER_NAME bash -c 'cd /host && mkdir build && cd build && cmake -G Ninja ../halo && ninja && ninja-check'
