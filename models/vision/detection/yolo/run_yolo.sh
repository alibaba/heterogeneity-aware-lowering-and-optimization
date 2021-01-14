#!/bin/bash -xv
# RUN: %s
curr_dir=`dirname $0`
make -C $curr_dir

out_dir=${TEST_TEMP_DIR}/yolo
res_info=`$out_dir/demo $out_dir/test.jpg`
echo "${res_info}" >> $out_dir/yolo_xnnpack.txt

# RUN: FileCheck --input-file %test_temp_dir/yolo/yolo_xnnpack.txt %s
# CHECK: Detecting for image, output: out/result.jpg
# CHECK-NEXT: [person], pos:[158.149, 204.101, 514.119, 337.473] score:0.992785
# CHECK-NEXT: [backpack], pos:[209.222, 205.724, 308.212, 257.877] score:0.854316
# CHECK-NEXT: [skis], pos:[437.568, 148.244, 581.4, 347.084] score:0.932281