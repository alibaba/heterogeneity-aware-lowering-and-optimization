#!/bin/bash
# RUN: %s
model_name="inception_v1"
model_file="$TEST_TEMP_DIR/$model_name.onnx"
image_dir="$TEST_TEMP_DIR/images"
curr_dir=`dirname $0`

# Download model if it is not exist
if [ ! -e $model_file ]; then
  wget -O $model_file 'https://github.com/onnx/models/blob/master/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx?raw=true'
fi

# Download sample images if it is not exist
$curr_dir/../../get_images.sh $image_dir

# check if GPU is enabled or not
if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  python3 $curr_dir/../../invoke_halo.py --model $model_file \
    --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir \
    --odla tensorrt --img-preprocess=minus_128
# RUN: FileCheck --input-file %test_temp_dir/inception_v1_tensorrt.txt --check-prefix CHECK-TENSORRT %s
# CHECK-TENSORRT: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-TENSORRT: sport.jpg ==> "ski",
# CHECK-TENSORRT: food.jpg ==> "jellyfish",
# CHECK-TENSORRT: plane.jpg ==> "airliner",
fi

# Using HALO to compile and run inference with ODLA DNNL
echo "======== Testing with ODLA DNNL (NHWC)========"
python3 $curr_dir/../../invoke_halo.py --model $model_file \
  --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir \
  --odla dnnl --img-preprocess=minus_128 --convert-layout-to=nhwc
# RUN: FileCheck --input-file %test_temp_dir/inception_v1_dnnl.txt --check-prefix CHECK-DNNL %s
# CHECK-DNNL: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-DNNL: sport.jpg ==> "ski",
# CHECK-DNNL: food.jpg ==> "bubble",
# CHECK-DNNL: plane.jpg ==> "jigsaw puzzle",