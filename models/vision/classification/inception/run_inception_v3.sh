#!/bin/bash
# RUN: %s

model_name="inception_v3"
model_file="$TEST_TEMP_DIR/$model_name.onnx"
image_dir="$TEST_TEMP_DIR/images"
curr_dir=`dirname $0`

# Download model if it is not exist
if [ ! -e $model_file ]; then
  $curr_dir/../get_cls_model_from_pytorch.py $model_name $model_file 299
fi

# Download sample images if it is not exist
$curr_dir/../../get_images.sh $image_dir 

# check if GPU is enabled or not
if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla tensorrt --input_h 299 --input_w 299
# RUN: FileCheck --input-file %test_temp_dir/inception_v3_tensorrt.txt %s
else
	echo "This tests uses ODLA TensorRT"
fi

# CHECK: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-NEXT: sport.jpg ==> "ski",
# CHECK-NEXT: food.jpg ==> "ice cream, icecream",
# CHECK-NEXT: plane.jpg ==> "airliner",