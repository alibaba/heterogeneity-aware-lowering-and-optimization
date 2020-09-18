#!/bin/bash

image_dir="$TEST_TEMP_DIR/images"

# Download sample images if it is not exist
mkdir -p $image_dir

imgs="http://images.cocodataset.org/test2017/000000030207.jpg:plane.jpg \
      http://images.cocodataset.org/test2017/000000228503.jpg:food.jpg \
      http://images.cocodataset.org/test2017/000000133861.jpg:sport.jpg \
      https://github.com/pytorch/hub/raw/master/dog.jpg:dog.jpg"

for img in $imgs;do
  url=${img%:*}
  file=$image_dir/${img##*:}
  if [[ ! -e $file ]] ;then
    wget -O $file $url
  fi
done
