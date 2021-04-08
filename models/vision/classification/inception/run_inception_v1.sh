#!/bin/bash
# RUN: %s %t.1 %t.2
model_name="inception-v1"
model_file="$MODELS_ROOT/vision/classification/inception/$model_name-9.onnx"
image_dir="$MODELS_ROOT/vision/test_images"
curr_dir=`dirname $0`
if [[ $# != 0 ]];then
  export TEST_TEMP_DIR=`dirname $1`
fi

# check if GPU is enabled or not
if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  python3 $curr_dir/../../invoke_halo.py --model $model_file \
    --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir \
    --odla tensorrt --img-preprocess=minus_128 | tee $1
# RUN: FileCheck --input-file %t.1 --check-prefix CHECK-TENSORRT %s
# CHECK-TENSORRT: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-TENSORRT: food.jpg ==> "jellyfish",
# CHECK-TENSORRT: plane.jpg ==> "airliner",
# CHECK-TENSORRT: sport.jpg ==> "ski",
fi

# Using HALO to compile and run inference with ODLA DNNL
echo "======== Testing with ODLA DNNL (NHWC)========"
python3 $curr_dir/../../invoke_halo.py --model $model_file \
  --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir \
  --odla dnnl --img-preprocess=minus_128 --convert-layout-to=nhwc | tee $2
# RUN: FileCheck --input-file %t.2 --check-prefix CHECK-DNNL %s
# CHECK-DNNL: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-DNNL: food.jpg ==> "bubble",
# CHECK-DNNL: plane.jpg ==> "jigsaw puzzle",
# CHECK-DNNL: sport.jpg ==> "ski",