#!/bin/bash
# RUN: %s %t

curr_dir=`dirname $0`
if [[ $# != 0 ]];then
  export TEST_TEMP_DIR=`dirname $1`
fi

model_name="unet"
model_file="$MODELS_ROOT/vision/segmentation/$model_name/$model_name.onnx"
image_dir="$MODELS_ROOT/vision/segmentation/$model_name"

echo "======== Testing with ODLA XNNPACK ========"
python3 $curr_dir/../../invoke_halo.py --model $model_file --image-dir $image_dir \
        --input_h 256 --input_w 256 --output_size 65536 --odla xnnpack \
        --convert-layout-to=nhwc | tee $1
# RUN: FileCheck --input-file %t %s

# CHECK: [40859 40860 41114]