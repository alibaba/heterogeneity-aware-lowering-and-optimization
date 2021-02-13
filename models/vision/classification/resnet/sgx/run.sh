#!/bin/bash
# RUN: %s

OUT=$TEST_TEMP_DIR/resnet50_sgx
SGX_DIR=/opt/intel/sgxsdk

mkdir -p $OUT
model_url="https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx"
model_file="$TEST_TEMP_DIR/resnet50-v2-7.onnx"

curr_dir=`dirname $0`

# Download model if it is not exist
wget -nc -O $model_file $model_url

set -xe

# FIXME: remove --outputs option.
$HALO_BIN -target cxx $model_file -o $OUT/model.cc -disable-broadcasting \
  -entry-func-name=resnet50_v2 --outputs=resnetv24_batchnorm0_fwd

$SGX_DIR/bin/x64/sgx_edger8r $curr_dir/resnet50_v2.edl --search-path $SGX_DIR/include \
  --untrusted-dir $OUT --trusted-dir $OUT

# Build enclave code
e_c_flags="-nostdinc -isystem $SGX_DIR/include/tlibc -I$SGX_DIR/include/
           -fpie -fvisibility=hidden -ffunction-sections -fdata-sections"
e_cxx_flags="$e_c_flags -nostdinc++ -isystem $SGX_DIR/include/libcxx"

g++ -c $e_cxx_flags $OUT/model.cc -c -o $OUT/model.o -I$OUT -I$ODLA_INC
gcc -c $e_c_flags $OUT/resnet50_v2_t.c -c -o $OUT/resnet50_v2_t.o -Iout

DNNL_SGX="/opt/intel/sgx_dnnl"
DNNL_LINK_LIBS="-lsgx_dnnl -lsgx_omp -lsgx_pthread -lpthread"
SGX_LINK_LIBS="-lsgx_tstdc -lsgx_tcxx -lsgx_tcrypto -lsgx_tservice_sim"

security_ld_flags="-Wl,-z,relro,-z,now,-z,noexecstack"
g++ -o $OUT/model.so $OUT/model.o $OUT/resnet50_v2_t.o $OUT/model.bin \
  $security_ld_flags -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles \
  -L$SGX_DIR/lib64 -L$DNNL_SGX/lib -L$ODLA_LIB \
  -Wl,--whole-archive -lsgx_trts_sim -Wl,--no-whole-archive \
  -Wl,--start-group $DNNL_LINK_LIBS $SGX_LINK_LIBS -lodla_sgx_dnnl -Wl,--end-group \
  -Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
  -Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
  -Wl,--defsym,__ImageBase=0 -Wl,--gc-sections   \
  -Wl,--version-script=$curr_dir/link.lds

if [[ ! -f $OUT/private_test.pem ]]; then
  openssl genrsa -out $OUT/private_test.pem -3 3072
fi

$SGX_DIR/bin/x64/sgx_sign sign -key $OUT/private_test.pem -enclave $OUT/model.so \
  -out $OUT/model.signed.so -config $curr_dir/config.xml

# App
g++ -g -c $curr_dir/app.cc -o $OUT/app.o -I$OUT -I$SGX_DIR/include -I$SRC_DIR/tests/include
gcc -g -c $OUT/resnet50_v2_u.c -I$OUT -I$SGX_DIR/include -o $OUT/resnet50_v2_u.o
g++ $OUT/app.o $OUT/resnet50_v2_u.o -o $OUT/app -L$SGX_DIR/sdk_libs -lsgx_urts_sim -Wl,-rpath=$SGX_DIR/sdk_libs

(cd $OUT && ./app)