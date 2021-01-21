#!/bin/bash -xe

VER="0.5"

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


IMAGE="halo:$VER-$VARIANT-ubuntu18.04"
CONTAINER_NAME="halo.ci-$VER-$VARIANT"

docker_run_flag=""
cmake_flags=""
check_cmds="ninja check-halo && ninja check-halo-models"

if [[ "$VARIANT" =~ cuda ]]; then
  docker_run_flag="--runtime=nvidia"
fi

if [[ "$VARIANT" =~ graphcore ]]; then
  cmake_flags="-DPOPLAR_SDK_ROOT=/opt/poplar_sdk-ubuntu_18_04-1.4.0+365-665f971c8f \
              -DPOPLAR_VERSION=ubuntu_18_04-1.4.0+71819-c5c0c8ebab \
              -DPOPART_VERSION=popart-ubuntu_18_04-1.4.0+5352-e86081acc9 \
	            -DODLA_BUILD_DNNL=OFF -DODLA_BUILD_TRT=OFF \
              -DODLA_BUILD_EIGEN=OFF -DODLA_BUILD_XNNPACK=OFF"
  check_cmds="ninja check-halo"
fi

DOCKER_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`

if [ -z "$DOCKER_ID" ]; then
  gid=$(id -g ${USER})
  group=$(id -g -n ${USER})
  uid=$(id -u ${USER})
  docker run $docker_run_flag -t -d --name $CONTAINER_NAME -v $MOUNT_DIR:/host \
    --tmpfs /tmp:exec --rm $IMAGE
  docker exec $CONTAINER_NAME bash -c "groupadd -f -g $gid $group"
  docker exec $CONTAINER_NAME bash -c \
    "adduser --shell /bin/bash --uid $uid --gecos '' --gid $gid \
    --disabled-password --home /home/$USER $USER"
fi

if [[ "$VARIANT" =~ graphcore ]]; then
  extra_cmd="source /opt/poplar_sdk-ubuntu_18_04-1.4.0+365-665f971c8f/poplar-ubuntu_18_04-1.4.0+71819-c5c0c8ebab/enable.sh \
         && source /opt/poplar_sdk-ubuntu_18_04-1.4.0+365-665f971c8f/popart-ubuntu_18_04-1.4.0+5352-e86081acc9/enable.sh"
fi

docker exec --user $USER $CONTAINER_NAME bash -c \
  "$extra_cmd && cd /host && rm -fr build && mkdir -p build && cd build && \
   cmake -G Ninja $cmake_flags ../halo && ninja && $check_cmds"
