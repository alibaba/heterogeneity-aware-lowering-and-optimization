#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [HALO_BUILD_DIR]"
  exit 1
fi

build_path=$1
export HALO_BUILD_DIR=$build_path
export HALO_SRC_DIR=${PWD}/../../

python3 run_test.py
