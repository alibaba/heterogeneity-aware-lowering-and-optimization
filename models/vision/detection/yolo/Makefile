.PHONY: demo clean

out_dir := ${TEST_TEMP_DIR}
demo : ${out_dir}/demo

SHELL := /bin/bash

flags=-O3
cv_libs:=-L/opt/opencv/3.4.11/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio
CXXFLAGS = -std=c++14 -fPIC -g -DONNX_NAMESPACE=onnx -D_GLIBCXX_USE_CXX11_ABI=1
LDLIBS = -lpopart -lpoplar -lpopops
THIRD_PARTY = ${SRC_DIR}/ODLA/platforms/odla_popart/custom_ops/third_party
INCLUDES = -I ${THIRD_PARTY}/include -I ${THIRD_PARTY}/onnx -Iinclude
BUILD_DIR = build
TARGET = $(BUILD_DIR)/libcustom_ops.so

$(shell mkdir -p ${out_dir})

custom_op: nms.cc
	g++ $< -shared -lpopart -lpoplar -lpopops -lpoputil $(CXXFLAGS) $(INCLUDES) -o $(TARGET)

${out_dir}/yolo.c : ${MODELS_ROOT}/vision/detection/yolo/yolov3-10.onnx
	$(HALO_BIN) -target cc -exec-mode=interpret -emit-value-id-as-int -reorder-data-layout=channel-last -remove-input-transpose -remove-output-transpose \
	       	-o $@ $<  --disable-broadcasting -outputs conv2d_59 -outputs conv2d_67 -outputs conv2d_75 -input-shape=input_1:1x3x416x416 -entry-func-name=yolo_v3
			   
${out_dir}/yolo.o : ${out_dir}/yolo.c
	gcc -I${ODLA_INC} ${flags} -DNDEBUG -fPIC -c  -o $@ $<
${out_dir}/yolo_postproc.o : yolov3_postproc.cc custom_op
	g++ -std=c++14 -c ${flags} yolov3_postproc.cc ${LDLIBS} -ldl -DONNX_NAMESPACE=onnx -o $@ -D_GLIBCXX_USE_CXX11_ABI=1
${out_dir}/test.jpg : ${MODELS_ROOT}/vision/detection/yolo/test.jpg
	ln -s $< $@
${out_dir}/demo.o : demo.cc
	g++ -c ${flags} -DVENDOR="CPU" -std=c++11 -o $@ $<  -I. -I/opt/opencv/3.4.11/include

${out_dir}/demo: ${out_dir}/demo.o ${out_dir}/yolo_postproc.o ${out_dir}/yolo.o ${out_dir}/yolo.bin
	g++ -DVENDOR="CPU" -std=c++11 -o $@ $^  -L${ODLA_LIB} -Wl,-rpath=${ODLA_LIB}  -lodla_xnnpack  ${cv_libs} ${LDLIBS} -ldl 

demo: ${out_dir}/demo ${out_dir}/test.jpg

clean:
	rm -fr ${out_dir}
