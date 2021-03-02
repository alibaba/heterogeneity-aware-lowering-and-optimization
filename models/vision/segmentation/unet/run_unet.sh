#!/bin/bash
# RUN: %s

curr_dir=`dirname $0`
model_file="$TEST_TEMP_DIR/unet.onnx"
image_dir="$TEST_TEMP_DIR/images/unet"

if [ ! -e $model_file ]; then
  python3 $curr_dir/get_unet_model.py $model_file
fi

mkdir -p $image_dir
if [ ! -e $image_dir/test.jpg ]; then
  wget -nc -P $image_dir https://github.com/zhixuhao/unet/raw/master/img/0test.png
  convert $image_dir/0test.png -resize 256x256 $image_dir/test.jpg
  rm -f $image_dir/0test.png
fi

echo "======== Testing with ODLA XNNPACK ========"
python3 $curr_dir/../../invoke_halo.py --model $model_file --image-dir $image_dir --input_h 256 --input_w 256 --output_size 65536 --odla xnnpack --convert-layout-to=nhwc
# RUN: FileCheck --input-file %test_temp_dir/unet_xnnpack.txt %s

# CHECK: [40859 40860 41114]