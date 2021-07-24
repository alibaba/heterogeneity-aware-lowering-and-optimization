#!/bin/bash -xe

VER="latest"
FLAVOR="devel"
NAMESPACE="registry-intl.us-west-1.aliyuncs.com/computation"

docker build -t $NAMESPACE/halo:$VER-$FLAVOR-x86_64-centos7 -f Dockerfile.centos .


