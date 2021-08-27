#!/bin/bash
# RUN: %s
model_name="inception"
docker_model_file="/models/vision/classification/$model_name"
model_file="$docker_model_file/$model_name""_v3.onnx"
image_dir="/models/vision/test_images"
curr_dir=`dirname $0`

# # Download model if it is not exist
# if [ ! -e $model_file ]; then
#   $curr_dir/../get_cls_model_from_pytorch.py $model_name $model_file
# fi

# Download sample images if it is not exist
# $curr_dir/../../get_images.sh $image_dir

echo "=======Testing inception_v3 with TensorRT======="
for i in 1 2 4 8 16 32 64
do 
python3 $curr_dir/../../onnx2tensorrt.py --batch_size $i --model $model_file --label-file $curr_dir/../1000_labels.txt --input_h 299 --input_w 299
done


# if [[ $TEST_WITH_GPU -eq 1 ]]; then
#   echo "======== Testing with ODLA TensorRT ========"
#   python3 $curr_dir/../../invoke_halo.py --model $model_file  --image-dir $image_dir --odla tensorrt
# fi

# # Using HALO to compile and run inference with ODLA XNNPACK
# echo "======== Testing with ODLA DNNL ========"
# python3 $curr_dir/../../invoke_halo.py --model $model_file --label-file $curr_dir/../1000_labels.txt --image-dir $image_dir --odla dnnl