#!/bin/bash
# RUN: %s

model_name="caffenet"
image_dir="$TEST_TEMP_DIR/images"
curr_dir=`dirname $0`

# Download model if it is not exist
model_file="/$TEST_TEMP_DIR/bvlc_reference_caffenet.caffemodel"
if [ ! -e $model_file ]; then
  wget -P /tmp https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt
  wget -P /tmp http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
fi

# Download sample images if it is not exist
$curr_dir/../../get_images.sh $image_dir

# check if GPU is enabled or not
if [[ $TEST_WITH_GPU -eq 1 ]]; then
  echo "======== Testing with ODLA TensorRT ========"
  python3 $curr_dir/../../invoke_halo.py --model /tmp/deploy.prototxt /tmp/bvlc_reference_caffenet.caffemodel \
    --label-file $curr_dir/../1000_labels.txt --input_h=227 --input_w=227 \
    --input-shape=data:1x3x227x227 \
    --image-dir $image_dir --odla tensorrt --img-preprocess=minus_128
# RUN: FileCheck --input-file %test_temp_dir/bvlc_reference_caffenet_tensorrt.txt %s
else
	echo "This tests uses ODLA TensorRT"
fi

# CHECK: dog.jpg ==> "Samoyed, Samoyede",
# CHECK-NEXT: sport.jpg ==> "ski",
# CHECK-NEXT: food.jpg ==> "ice cream, icecream",
# CHECK-NEXT: plane.jpg ==> "airliner",
