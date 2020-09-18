#!/bin/bash
model_url="https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx"
model_file="$TEST_TEMP_DIR/resnet50-v2-7.onnx"
image_dir="$TEST_TEMP_DIR/images"
curr_dir=`dirname $0`

# Download model if it is not exist
wget -nc -O $model_file $model_url

# Download sample images if it is not exist
$curr_dir/../../get_images.sh $image_dir

# Using HALO to compile and run inference with ODLA DNNL
echo "======== Testing with ODLA DNNL ========"
python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla dnnl

# check if GPU is enabled or not
if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla tensorrt
fi