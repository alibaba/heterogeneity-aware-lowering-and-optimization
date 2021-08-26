#!/bin/bash
# RUN: %s
model_name="rcnn"
docker_model_file="/models/vision/classification/$model_name"
model_file="$docker_model_file/$model_name""-ilsvrc13-9.onnx"
image_dir="/models/vision/test_images"
curr_dir=`dirname $0`

# # Download model if it is not exist
# if [ ! -e $model_file ]; then
#   $curr_dir/../get_cls_model_from_pytorch.py $model_name $model_file
# fi

# Download sample images if it is not exist
# $curr_dir/../../get_images.sh $image_dir

echo "=======Testing efficientnet with TensorRT======="
python3 $curr_dir/../../onnx2tensorrt.py --model $model_file --output_size 200


# if [[ $TEST_WITH_GPU -eq 1 ]]; then
#   echo "======== Testing with ODLA TensorRT ========"
#   python3 $curr_dir/../../invoke_halo.py --model $model_file  --image-dir $image_dir --odla tensorrt
# fi

# # Using HALO to compile and run inference with ODLA XNNPACK
# echo "======== Testing with ODLA DNNL ========"
# python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla dnnl