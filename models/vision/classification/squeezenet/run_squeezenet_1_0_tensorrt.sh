#!/bin/bash
# RUN: %s %t.1 %t.2

model_name="squeezenet1"
model_file="/models/vision/classification/squeezenet/$model_name""_0.onnx"
image_dir="/models/vision/test_images"
curr_dir=`dirname $0`
if [[ $# != 0 ]];then
  export TEST_TEMP_DIR=`dirname $1`
fi

# Using Tensorrt to compile and run inference.
echo "=======Testing squeezenet_1_0 with TensorRT======="
python3 $curr_dir/../../onnx2tensorrt.py --model $model_file \
        --label-file $curr_dir/../1000_labels.txt --image_dir $image_dir
        
# # Using HALO to compile and run inference with ODLA DNNL
# echo "======== Testing with ODLA DNNL ========"
# python3 $curr_dir/../../invoke_halo.py --model $model_file \
#         --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir \
#         --odla dnnl | tee $1
# # RUN: FileCheck --input-file %t.1 %s

# # check if GPU is enabled or not
# if [[ $TEST_WITH_GPU -eq 1 ]]; then
#   echo "======== Testing with ODLA TensorRT ========"
#   python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file \
#           $curr_dir/../1000_labels.txt --image-dir $image_dir \
#           --odla tensorrt | tee $2
# # RUN: FileCheck --input-file %t.2 %s
# fi

# CHECK: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-NEXT: food.jpg ==> "mashed potato",
# CHECK-NEXT: plane.jpg ==> "airliner",
# CHECK-NEXT: sport.jpg ==> "ski",
