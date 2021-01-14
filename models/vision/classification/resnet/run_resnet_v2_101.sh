#!/bin/bash
# RUN: %s

model_url="https://media.githubusercontent.com/media/onnx/models/master/vision/classification/resnet/model/resnet101-v2-7.onnx"
model_file="$TEST_TEMP_DIR/resnet101-v2-7.onnx"
image_dir="$TEST_TEMP_DIR/images"
curr_dir=`dirname $0`

# Download model if it is not exist
wget -nc -O $model_file $model_url

# Download sample images if it is not exist
$curr_dir/../../get_images.sh $image_dir

# Using HALO to compile and run inference with ODLA DNNL
echo "======== Testing with ODLA DNNL ========"
python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla dnnl
# RUN: FileCheck --input-file %test_temp_dir/resnet101-v2-7_dnnl.txt %s

# check if GPU is enabled or not
if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla tensorrt
# RUN: FileCheck --input-file %test_temp_dir/resnet101-v2-7_tensorrt.txt %s
fi

# CHECK: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-NEXT: sport.jpg ==> "ski",
# CHECK-NEXT: food.jpg ==> "plate",
# CHECK-NEXT: plane.jpg ==> "warplane, military plane",