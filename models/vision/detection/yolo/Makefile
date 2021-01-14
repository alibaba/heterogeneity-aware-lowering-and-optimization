.PHONY: demo clean

out_dir := ${TEST_TEMP_DIR}/yolo
demo : ${out_dir}/demo

SHELL := /bin/bash

flags=-O3
cv_libs:=-L/opt/opencv/3.4.11/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio

$(shell mkdir -p ${out_dir})

${out_dir}/yolov3.onnx:
	wget -O $@ 'https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx?raw=true'

${out_dir}/yolo.c : ${out_dir}/yolov3.onnx
	$(HALO_BIN) -target cc -exec-mode=interpret -emit-value-id-as-int -reorder-data-layout=channel-last -remove-input-transpose -remove-output-transpose \
	       	-o $@ $<  --disable-broadcasting -outputs conv2d_59 -outputs conv2d_67 -outputs conv2d_75 -input-shape=input_1:1x3x416x416 -entry-func-name=yolo_v3

${out_dir}/yolo.o : ${out_dir}/yolo.c
	gcc -I${ODLA_INC} ${flags} -DNDEBUG -fPIC -c  -o $@ $<

${out_dir}/yolo_postproc.o : yolov3_postproc.cc
	g++ -std=c++14 -c ${flags} -o $@ $<

${out_dir}/demo.o : demo.cc
	g++ -c ${flags} -DVENDOR="CPU" -std=c++11 -o $@ $<  -I. -I/opt/opencv/3.4.11/include

${out_dir}/demo: ${out_dir}/demo.o ${out_dir}/yolo_postproc.o ${out_dir}/yolo.o ${out_dir}/yolo.bin
	g++ -DVENDOR="CPU" -std=c++11 -o $@ $^  -L${ODLA_LIB} -Wl,-rpath=${ODLA_LIB}  -lodla_xnnpack  ${cv_libs} 

${out_dir}/test.jpg:
	wget -O $@ http://images.cocodataset.org/test2017/000000133861.jpg

demo: ${out_dir}/demo ${out_dir}/test.jpg

clean:
	rm -fr ${out_dir}
