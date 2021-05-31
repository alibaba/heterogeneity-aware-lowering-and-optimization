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

OS="ubuntu18.04"
if [ ! -z "$1" ]; then
  OS=$1
  shift;
fi

IMAGE="$REPO:$VER-$FLAVOR-$VARIANT-$OS"
CONTAINER_NAME="halo.ci-$VER-$VARIANT"

docker_run_flag=""
cmake_flags="-DDNNL_COMPILER=gcc-10"
check_cmds="ninja check-halo && ninja check-halo-models"

if [[ "$VARIANT" =~ cuda ]]; then
  docker_run_flag="--runtime=nvidia"
fi

cmake_flags="-DODLA_BUILD_POPART=OFF"
if [[ "$VARIANT" =~ graphcore ]]; then
  sdk_os="ubuntu_18_04"
  if [[ "$OS" == "centos7" ]];then
    sdk_os="centos_7_6"
  fi
  cmake_flags="-DPOPLAR_SDK_ROOT=/opt/poplar_sdk-$sdk_os-2.0.1+562-81b90b6055 \
              -DPOPLAR_VERSION=$sdk_os-2.0.1+130834-d32e9bc95a \
              -DPOPART_ROOT=/opt/poplar_sdk-$sdk_os-2.0.1+562-81b90b6055/popart-$sdk_os-2.0.0+130834-d32e9bc95a \
	            -DODLA_BUILD_DNNL=OFF -DODLA_BUILD_TRT=OFF \
              -DODLA_BUILD_POPART=ON \
              -DODLA_BUILD_EIGEN=OFF -DODLA_BUILD_XNNPACK=OFF \
              -DHALO_USE_TIDY_CHECK=OFF \
              -DHALO_GEN_DOCS=OFF .. "
  check_cmds="ninja check-halo"
fi

DOCKER_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`

gid=$(id -g ${USER})
uid=$(id -u ${USER})

if [ -z "$DOCKER_ID" ]; then
  docker run $docker_run_flag -t -d --name $CONTAINER_NAME -v $MOUNT_DIR:/host \
    --tmpfs /tmp:exec --user=$uid:$gid --rm $IMAGE
fi

extra_cmd="true" # dummy command
if [[ "$OS" == "centos7" ]];then
  extra_cmd="$extra_cmd;source /opt/rh/devtoolset-7/enable"
fi

if [[ "$VARIANT" =~ graphcore ]]; then
  extra_cmd="$extra_cmd;source /opt/poplar_sdk-$sdk_os-2.0.1+562-81b90b6055/poplar-$sdk_os-2.0.1+130834-d32e9bc95a/enable.sh \
             source /opt/poplar_sdk-$sdk_os-2.0.1+562-81b90b6055/popart-$sdk_os-2.0.0+130834-d32e9bc95a/enable.sh"
fi

docker exec --user=$uid:$gid $CONTAINER_NAME bash -c \
  "$extra_cmd && cd /host && rm -fr build && mkdir -p build && cd build && \
  cmake -G Ninja $cmake_flags ../halo && ninja && $check_cmds && ninja package"
