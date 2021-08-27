#!/bin/bash
# RUN: %s %t.1 %t.2

model_name="alexnet"
model_file="$MODELS_ROOT/vision/classification/$model_name/$model_name.onnx"
image_dir="$MODELS_ROOT/vision/test_images"
curr_dir=`dirname $0`

if [[ $# != 0 ]];then
  export TEST_TEMP_DIR=`dirname $1`
fi

if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  for i in 1 2 4 8 16 32 64
  do
  python3 $curr_dir/../../invoke_halo.py --model $model_file \
          --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir \
          --odla tensorrt | tee $1
  done
# RUN: FileCheck --input-file %t.1 %s
fi

# # Using HALO to compile and run inference with ODLA XNNPACK
# echo "======== Testing with ODLA DNNL ========"
# python3 $curr_dir/../../invoke_halo.py --model $model_file \
#         --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir \
#         --odla dnnl | tee $2
# RUN: FileCheck --input-file %t.2 %s

# CHECK: dog.jpg ==> "wallaby, brush kangaroo",
# CHECK-NEXT: food.jpg ==> "ice cream, icecream",
# CHECK-NEXT: plane.jpg ==> "airliner",
# CHECK-NEXT: sport.jpg ==> "ski",