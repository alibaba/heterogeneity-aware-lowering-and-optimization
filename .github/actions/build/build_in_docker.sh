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
if [[ "$VARIANT" =~ cuda ]]; then
  docker_run_flag="--runtime=nvidia"
fi
  
DOCKER_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`

if [ -z "$DOCKER_ID" ]; then
  docker run $docker_run_flag -t -d --name $CONTAINER_NAME -v $MOUNT_DIR:/host \
    --tmpfs /tmp:exec --rm $IMAGE
  docker exec groupadd -f -g $(id -g ${USER}) $(id -g -n ${USER})
  docker exec adduser --shell /bin/bash --uid $(id -u ${USER}) --gecos '' \
    --ingroup $(id -g ${USER}) --disabled-password --home /home/$USER $USER
fi

docker exec $CONTAINER_NAME bash -c 'cd /host && rm -fr build && mkdir -p build && cd build && cmake -G Ninja ../halo && ninja && ninja check-halo'
