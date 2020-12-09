#!/bin/bash
#e.g ./run_single.sh test_abs 0.0001 tensorrt /host/heterogeneity-aware-lowering-and-optimization/build
test_case=$1
error_thre=$2
device=$3
build_path=$4

export HALO_BUILD_DIR=$build_path
export HALO_SRC_DIR=${PWD}/../../

python3 run_test.py --test-mode 'single'    \
                    --enable-time-perf ''   \
                    --test-case $test_case    \
                    --error-threshold=$error_thre \
                    --device $device       \