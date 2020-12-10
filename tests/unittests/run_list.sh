#!/bin/bash
# RUN: %s %src_dir %build_dir %S

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 [HALO_SRC_DIR] [HALO_BUILD_DIR] [HALO_UNITTESTS_PATH]"
  exit 1
fi

export HALO_SRC_DIR=$1
export HALO_BUILD_DIR=$2
export HALO_UNITTESTS_PATH=$3

odla_lib_path=${HALO_BUILD_DIR}/lib

# build_ipu not install tensorrt lib, can't run the run_test.py
odla_tensorrt_file=${odla_lib_path}/libodla_tensorrt.so
if [ ! -f "${odla_tensorrt_file}" ]; then
  echo "No odla_tensorrt found. Skip tests"
  exit 0
fi

python3 ${HALO_UNITTESTS_PATH}/run_test.py --test-mode 'list'    \
                    --enable-time-perf '' \
                    --test-case ''        \
                    --error-threshold=0   \
                    --device ''           \
