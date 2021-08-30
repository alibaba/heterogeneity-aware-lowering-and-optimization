HALO_DIR=../../..
MODELS_ROOT=/models
INCLUDES=-I ${HALO_DIR}/include -I ${HALO_DIR}/build/include -I ${HALO_DIR}/tests/include -I ${HALO_DIR}/external/llvm-project/llvm/include -I ${HALO_DIR}/build/llvm/include

all: demo_ipu demo_ipu_custom_ops

yolov3_test.cc.popart.cc yolov3_test.cc.popart.bin: ${MODELS_ROOT}/vision/detection/yolo/yolov3-10.onnx
	${HALO_DIR}/build/bin/halo --disable-broadcasting -target cxx -enable-ipu-device --ipu-num=1 $^ \
	-outputs conv2d_59 -outputs conv2d_67 -outputs conv2d_75 -input-shape=input_1:1x3x416x416 -entry-func-name=yolo_v3 -o yolov3_test.cc.popart.cc

yolov3_test.cc.popart.o: yolov3_test.cc.popart.cc
	c++ -c -o $@ $^ -I ${HALO_DIR}/ODLA/include

yolov3_test.cc.main_popart.o: yolov3_test.cc
	c++ -c -DUSE_NCHW_DATA -DCOMPARE_ERROR=5e-1 ${INCLUDES} $^ -o $@

yolov3_test.cc.pp.o: yolov3_postproc.cc
	c++ -c -o $@ $^

yolov3_test.cc.popart.exe: yolov3_test.cc.popart.o yolov3_test.cc.popart.bin yolov3_test.cc.main_popart.o yolov3_test.cc.pp.o
	c++ $^ -Wl,-rpath=${HALO_DIR}/build/lib -L ${HALO_DIR}/build/lib -lodla_popart -o $@

yolov3_test.cc.main_popart_custom_op.o: yolov3_test_custom_op.cc
	c++ -c -DUSE_NCHW_DATA -DCOMPARE_ERROR=5e-1 ${INCLUDES} $^ -o $@

yolov3_test.cc.popart_custom_op.o: yolov3_test.cc.popart_custom_op.cc
	c++ -c -o $@ $^ -I ${HALO_DIR}/ODLA/include

yolov3_test.cc.popart_custom_op.exe: yolov3_test.cc.popart_custom_op.o yolov3_test.cc.popart.bin yolov3_test.cc.main_popart_custom_op.o
	c++ $^ -Wl,-rpath=${HALO_DIR}/build/lib -L ${HALO_DIR}/build/lib -lodla_popart -o $@

demo_ipu: yolov3_test.cc.popart.exe
demo_ipu_custom_ops:yolov3_test.cc.popart_custom_op.exe

.PHONY: clean
clean:
	rm -f *.exe *.bin *.o *.h

