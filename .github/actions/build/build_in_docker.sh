#!/bin/bash -xe

REPO="registry-intl.us-west-1.aliyuncs.com/computation/halo"
VER="0.8.1"
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
check_cmds="ninja FileCheck && parallel -k --plus LIT_NUM_SHARDS={##}  LIT_RUN_SHARD={#}  CUDA_VISIBLE_DEVICES={} ninja check-halo ::: {0..1}"
check_cmds="$check_cmds && parallel -k --plus LIT_NUM_SHARDS={##}  LIT_RUN_SHARD={#}  CUDA_VISIBLE_DEVICES={} ninja check-halo-models ::: {0..1}"

if [[ "$VARIANT" =~ cuda ]]; then
  docker_run_flag="--runtime=nvidia"
fi

cmake_flags="$cmake_flags -DHALO_USE_STATIC_PROTOBUF=ON -DCPACK_SYSTEM_NAME=ubuntu-i686"

gid=$(id -g ${USER})
group=$(id -g -n ${USER})
uid=$(id -u ${USER})
extra_mnt="-v /opt/poplar_sdk-ubuntu_18_04-2.3.1_793:/opt/poplar_sdk:ro"
mkdir -p /tmp/ubuntu.cache
extra_mnt="$extra_mnt -v /tmp/ubuntu.cache:/cache"

rm -fr $MOUNT_DIR/output_ubuntu && mkdir -p $MOUNT_DIR/output_ubuntu
extra_cmd="source /opt/poplar_sdk/poplar/enable.sh" # dummy command
cmd="cd /build && cmake -G Ninja $cmake_flags /host/halo "
cmd="$cmd && ninja && $extra_cmd && $check_cmds && ninja package "
cmd="$cmd && cp /build/*.bz2 /host/output_ubuntu"
docker run -e CCACHE_DIR=/cache $docker_run_flag -v $MOUNT_DIR:/host \
  --tmpfs /build:exec --tmpfs /tmp:exec --entrypoint="" \
  $extra_mnt  --rm --user $uid:$gid $IMAGE bash -c "$cmd"
