#!/bin/bash 

VER="v0.1"
NAMESPACE="registry-intl.us-west-1.aliyuncs.com/computation"
TAG=$NAMESPACE/halo:$VER-model-zoo

docker build . -t $TAG && docker run $TAG 
