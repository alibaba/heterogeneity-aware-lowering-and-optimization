#!/bin/bash -xe

VER="0.8.1"
FLAVOR="devel"
NAMESPACE="registry-intl.us-west-1.aliyuncs.com/computation"

if [[ "$0" =~ "start_docker_gpu.sh" ]]; then
  IMAGE="$NAMESPACE/halo:$VER-$FLAVOR-cuda11.4.2-cudnn8-ubuntu18.04"
  CONTAINER_NAME=$USER.halo-$VER-GPU
  docker_run_flag="--runtime=nvidia"
else
  use_gpu=0
  IMAGE="$NAMESPACE/halo:$VER-$FLAVOR-x86_64-ubuntu18.04"
  CONTAINER_NAME=$USER.halo-$VER-CPU
  docker_run_flag=""
fi

MOUNT_DIR=/home/$USER
if [ ! -z "$1" ]; then
  MOUNT_DIR=$1
fi

GROUP=`id -g -n`
GROUPID=`id -g`
OLD_ID=`docker ps -aq -f name=$CONTAINER_NAME -f status=running`

if [ -z "$OLD_ID" ]; then
  ID=`docker run $docker_run_flag --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d --name $CONTAINER_NAME -v $MOUNT_DIR:/host  --tmpfs /tmp:exec --rm $IMAGE `
  docker exec --user root $ID groupadd -f -g $GROUPID $GROUP
  docker exec --user root $ID adduser --shell /bin/bash --uid $UID --gecos '' --ingroup $GROUP --disabled-password --home /home/$USER $USER
  docker exec --user root $ID bash -c "echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER && chmod 0440 /etc/sudoers.d/$USER"
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

docker exec -it --user $USER $CONTAINER_NAME bash
