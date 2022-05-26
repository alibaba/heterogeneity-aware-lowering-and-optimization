#!/bin/bash
# RUN: %s %t.1 %t.2

curr_dir=`dirname $0`
if [[ $# != 0 ]];then
  export TEST_TEMP_DIR=`dirname $1`
fi
out=$TEST_TEMP_DIR

model_name="mnist_simple"
model_path=$MODELS_ROOT/vision/classification/$model_name

$HALO_BIN -target cxx -o $out/mnist_simple.so $model_path/$model_name.pb -I $ODLA_INC

g++ -c $curr_dir/main.cc -I$out -o $out/main.o

if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "Using TensorRT based ODLA runtime"
  g++ -o $out/test $out/main.o $out/mnist_simple.so \
    -L$ODLA_LIB -lodla_tensorrt -Wl,-rpath=$ODLA_LIB
  $out/test $model_path/test_image $model_path/test_label | tee $1
# RUN: FileCheck --input-file %t.1 %s
fi

echo "Using DNNL-based ODLA implementation"
g++ -o $out/test $out/main.o $out/mnist_simple.so \
  -L$ODLA_LIB -lodla_dnnl -Wl,-rpath=$ODLA_LIB

$out/test $model_path/test_image $model_path/test_label | tee $2
# RUN: FileCheck --input-file %t.2 %s

# CHECK: Accuracy 91{{.*}}/10000 (91.{{.*}}%)