#!/bin/bash
# RUN: %s

curr_dir=`dirname $0`
out=$TEST_TEMP_DIR/mnist

mkdir -p $out

if [[ ! -e $out/test_image ]]; then
  wget -qO- http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  | gunzip -c > $out/test_image
  wget -qO- http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip -c > $out/test_label
fi

$HALO_BIN -target cxx -o $out/mnist_simple.cc $curr_dir/mnist_simple.pb

g++ $out/mnist_simple.cc -I$ODLA_INC -c -o $out/mnist_simple.o

g++ -c $curr_dir/main.cc -I$out -o $out/main.o

if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "Using TensorRT based ODLA runtime"
  g++ -o $out/test $out/main.o $out/mnist_simple.o $out/mnist_simple.bin \
    -L$ODLA_LIB -lodla_tensorrt -Wl,-rpath=$ODLA_LIB
  res_tensorrt_info=`$out/test $out/test_image $out/test_label`
  echo ${res_tensorrt_info} >> $out/mnist_tensorrt.txt
# RUN: FileCheck --input-file %test_temp_dir/mnist/mnist_tensorrt.txt %s
fi

echo "Using DNNL-based ODLA implementation"
g++ -o $out/test $out/main.o $out/mnist_simple.o $out/mnist_simple.bin \
  -L$ODLA_LIB -lodla_dnnl -Wl,-rpath=$ODLA_LIB

res_dnnl_info=`$out/test $out/test_image $out/test_label`
echo ${res_dnnl_info} >> $out/mnist_dnnl.txt
# RUN: FileCheck --input-file %test_temp_dir/mnist/mnist_dnnl.txt %s
# CHECK: Accuracy 9190/10000 (91.9%)