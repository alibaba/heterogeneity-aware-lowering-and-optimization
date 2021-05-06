#!/bin/bash
# RUN: %s %t.1 %t.2

model_name="densenet"
model_file="$MODELS_ROOT/vision/classification/$model_name/$model_name""121.onnx"
image_dir="$MODELS_ROOT/vision/test_images"
curr_dir=`dirname $0`
if [[ $# != 0 ]];then
  export TEST_TEMP_DIR=`dirname $1`
fi

# check if GPU is enabled or not
if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  python3 $curr_dir/../../invoke_halo.py --model $model_file \
           --label-file $curr_dir/../1000_labels.txt --image-dir \
           $image_dir --odla tensorrt | tee $1
# RUN: FileCheck --input-file %t.1 %s
fi

# Using HALO to compile and run inference with ODLA DNNL
echo "======== Testing with ODLA DNNL (NCHW) ========"
python3 $curr_dir/../../invoke_halo.py --model $model_file \
        --label-file $curr_dir/../1000_labels.txt --image-dir \
        $image_dir --odla dnnl | tee $2
# RUN: FileCheck --input-file %t.2 %s

# CHECK: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-NEXT: food.jpg ==> "ice cream, icecream",
# CHECK-NEXT: plane.jpg ==> "airliner",
# CHECK-NEXT: sport.jpg ==> "ski",
