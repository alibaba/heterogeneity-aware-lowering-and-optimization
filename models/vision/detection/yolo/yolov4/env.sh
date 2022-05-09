#!/bin/bash 
set -e
# RUN: %s %t.1

pip3 install --upgrade pip
pip3 install scipy==1.5.4 opencv-python==4.5.5.64 numpy  Pillow
apt-get update && apt-get install -y --no-install-recommends \
     ffmpeg \
     libsm6=2:1.2.2-1 \
     libxext6=2:1.3.3-1
