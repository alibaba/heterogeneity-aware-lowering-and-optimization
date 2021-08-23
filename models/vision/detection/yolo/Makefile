.PHONY: demo clean

out_dir := ${TEST_TEMP_DIR}
demo : ${out_dir}/demo

SHELL := /bin/bash

flags=-O3
cv_libs:=-L/opt/opencv/3.4.11/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio
LDLIBS = -lpopart -lpoplar -lpopops

$(shell mkdir -p ${out_dir})

${out_dir}/yolo.c : ${MODELS_ROOT}/vision/detection/yolo/yolov3-10.onnx
	$(HALO_BIN) -target cc -exec-mode=interpret -emit-value-id-as-int -reorder-data-layout=channel-last -remove-input-transpose -remove-output-transpose \
	       	-o $@ $<  --disable-broadcasting -outputs conv2d_59 -outputs conv2d_67 -outputs conv2d_75 -input-shape=input_1:1x3x416x416 -entry-func-name=yolo_v3
			   
${out_dir}/yolo.o : ${out_dir}/yolo.c
	gcc -I${ODLA_INC} ${flags} -DNDEBUG -fPIC -c  -o $@ $<

${out_dir}/yolo_postproc_ipu.o : yolov3_postproc_ipu.cc
	g++ -std=c++14 -c ${flags} yolov3_postproc_ipu.cc ${LDLIBS} -ldl -DONNX_NAMESPACE=onnx -o $@ -D_GLIBCXX_USE_CXX11_ABI=1

${out_dir}/test.jpg : ${MODELS_ROOT}/vision/detection/yolo/test.jpg
	ln -s $< $@
${out_dir}/demo.o : demo.cc
	g++ -c ${flags} -DVENDOR="CPU" -std=c++11 -o $@ $<  -I. -I/opt/opencv/3.4.11/include

${out_dir}/demo: ${out_dir}/demo.o ${out_dir}/yolo_postproc_ipu.o ${out_dir}/yolo.o ${out_dir}/yolo.bin
	g++ -DVENDOR="CPU" -std=c++11 -o $@ $^  -L${ODLA_LIB} -Wl,-rpath=${ODLA_LIB}  -lodla_xnnpack  ${cv_libs} ${LDLIBS} -ldl 

demo: ${out_dir}/demo ${out_dir}/test.jpg

clean:
	rm -fr ${out_dir}
