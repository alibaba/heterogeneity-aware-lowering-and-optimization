#!/bin/bash -xe

MOUNT_DIR=/home/$USER
if [ ! -z "$1" ]; then
  MOUNT_DIR=$1
fi

VER="v1.30"
CONTAINER_NAME=$USER.halo_$VER
if [ ! -z "$2" ]; then
  TARGET=$2
  CONTAINER_NAME=$USER.$TARGET.halo_$VER
fi

IMAGE="reg.docker.alibaba-inc.com/aisml/sinianai_devl_env_cuda10.0_cudnn7_ubuntu18.04:$VER"
GROUP=`id -g -n`
GROUPID=`id -g`
OLD_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`

if [ -z "$OLD_ID" ]; then
  ID=`docker run --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d --name $CONTAINER_NAME -v $MOUNT_DIR:/host  --tmpfs /tmp:exec --rm $IMAGE `
  docker exec --user root $ID groupadd -f -g $GROUPID $GROUP;
  docker exec --user root $ID adduser --shell /bin/bash --uid $UID --gecos '' --ingroup $GROUP --disabled-password --home /home/$USER $USER
  docker exec --user $USER $ID bash -c 'mkdir ~/.ssh; mkdir /tmp/ccache; ln -s /tmp/ccache ~/.ccache'

  if [ -f ~/.ssh/authorized_keys ]; then
    docker cp ~/.ssh/authorized_keys $ID:/home/$USER/.ssh
  fi
  if [ -f ~/.ssh/config ]; then
    docker cp ~/.ssh/config $ID:/home/$USER/.ssh
  fi
  if [ -f ~/.ssh/id_rsa.pub ]; then
    docker cp ~/.ssh/id_rsa.pub $ID:/home/$USER/.ssh
  fi
  if [ -f ~/.ssh/id_rsa ]; then
    docker cp ~/.ssh/id_rsa $ID:/home/$USER/.ssh
  fi

  GIT_USER=`git config --get user.name`
  GIT_EMAIL=`git config --get user.email`
  docker exec --user $USER $ID bash -c "git config --global user.name \"$GIT_USER\""
  docker exec --user $USER $ID bash -c "git config --global user.email \"$GIT_EMAIL\""

fi
if [ -t 1 ]; then
  docker exec -it --user $USER $CONTAINER_NAME bash
fi
