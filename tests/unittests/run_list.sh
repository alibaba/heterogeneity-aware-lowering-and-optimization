#!/bin/bash
# RUN: %s

# build_ipu not install tensorrt lib, can't run the run_test.py
odla_lib_path=${HALO_BUILD_DIR}/lib
odla_tensor_file=${odla_lib_path}/libodla_tensorrt.so
if [ "${odla_tensor_file}" != "`find ${odla_lib_path} -name "libodla_tensorrt.so"`" ]; then
exit 0
fi

python3 ${HALO_UNITTESTS_PATH}/run_test.py --test-mode 'list'    \
                    --enable-time-perf '' \
                    --test-case ''        \
                    --error-threshold=0   \
                    --device ''           \
