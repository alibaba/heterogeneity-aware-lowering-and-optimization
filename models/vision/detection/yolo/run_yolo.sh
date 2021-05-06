#!/bin/bash -xv
# RUN: %s %t.1
curr_dir=`dirname $0`
if [[ $# != 0 ]];then
  export TEST_TEMP_DIR=`dirname $1`
fi

make -C $curr_dir clean
make -C $curr_dir

out_dir=${TEST_TEMP_DIR}
$out_dir/demo $TEST_TEMP_DIR/test.jpg | tee $1

# RUN: FileCheck --input-file %t.1 %s
# CHECK: Detecting for image, output: out/result.jpg
# CHECK-NEXT: [person], pos:[158.149, 204.101, 514.119, 337.473] score:0.992785
# CHECK-NEXT: [backpack], pos:[209.222, 205.724, 308.212, 257.877] score:0.854316
# CHECK-NEXT: [skis], pos:[437.568, 148.244, 581.4, 347.084] score:0.932281
