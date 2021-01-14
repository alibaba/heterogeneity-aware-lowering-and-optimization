#!/bin/bash
# RUN: %s

model_name="vgg16"
model_file="$TEST_TEMP_DIR/$model_name.onnx"
image_dir="$TEST_TEMP_DIR/images"
curr_dir=`dirname $0`

# Download model if it is not exist
if [ ! -e $model_file ]; then
   $curr_dir/../get_cls_model_from_pytorch.py $model_name $model_file
fi
#wget -nc -O $model_file $model_url

# Download sample images if it is not exist
$curr_dir/../../get_images.sh $image_dir

# check if GPU is enabled or not
if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla tensorrt
# RUN: FileCheck --input-file %test_temp_dir/vgg16_tensorrt.txt %s
fi

# Using HALO to compile and run inference with ODLA DNNL
echo "======== Testing with ODLA DNNL (NHWC) ========"
python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla dnnl --convert-layout-to=nhwc
# RUN: FileCheck --input-file %test_temp_dir/vgg16_dnnl.txt %s

# CHECK: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-NEXT: sport.jpg ==> "ski",
# CHECK-NEXT: food.jpg ==> "ice cream, icecream",
# CHECK-NEXT: plane.jpg ==> "airliner",