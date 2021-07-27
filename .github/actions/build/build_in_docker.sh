#!/bin/bash -xe

REPO="registry-intl.us-west-1.aliyuncs.com/computation/halo"
VER="latest"
FLAVOR="devel"

MOUNT_DIR="$PWD"
if [ ! -z "$1" ]; then
  MOUNT_DIR=$1
  shift
fi

VARIANT="x86_64"
if [ ! -z "$1" ]; then
  VARIANT=$1
  shift
fi


IMAGE="$REPO:$VER-$FLAVOR-$VARIANT-ubuntu18.04"
CONTAINER_NAME="halo.ci-$VER-$VARIANT"

docker_run_flag=""
cmake_flags="-DDNNL_COMPILER=gcc-10"
check_cmds="ninja check-halo && ninja check-halo-models"

if [[ "$VARIANT" =~ cuda ]]; then
  docker_run_flag="--runtime=nvidia"
fi

cmake_flags="$cmake_flags -DHALO_USE_STATIC_PROTOBUF=ON"

DOCKER_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`

if [ -z "$DOCKER_ID" ]; then
  gid=$(id -g ${USER})
  group=$(id -g -n ${USER})
  uid=$(id -u ${USER})
  extra_mnt="-v /opt/poplar_sdk:/opt/poplar_sdk:ro"
  mkdir -p /tmp/ubuntu.cache
  extra_mnt="$extra_mnt -v /tmp/ubuntu.cache:/cache"
  docker run -e CCACHE_DIR=/cache $docker_run_flag -t -d --name $CONTAINER_NAME -v $MOUNT_DIR:/host \
    $extra_mnt --tmpfs /tmp:exec --rm $IMAGE
  docker exec $CONTAINER_NAME bash -c "groupadd -f -g $gid $group"
  docker exec $CONTAINER_NAME bash -c \
    "adduser --shell /bin/bash --uid $uid --gecos '' --gid $gid \
    --disabled-password --home /home/$USER $USER"
fi

extra_cmd="source /opt/poplar_sdk/poplar/enable.sh" # dummy command

docker exec --user $USER $CONTAINER_NAME bash -c \
  "$extra_cmd && cd /host && rm -fr build && mkdir -p build && cd build && \
  cmake -G Ninja $cmake_flags ../halo && ninja && $check_cmds && ninja package"
