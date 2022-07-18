#!/bin/bash 
set -ex
# RUN: %s %t.1

model="yolov4"
cntPath=$(cd `dirname $0`;pwd)

modelPath=$MODELS_ROOT/vision/detection/yolo/$model.onnx

outputDir=$cntPath/out
if [ -n "$TEST_TEMP_DIR" ];then
	outputDir=$TEST_TEMP_DIR/$model/out
fi

setEnvVar() {
	# export INSTALL_DIR=/opt/halo
	# export HALO_BIN=$INSTALL_DIR/bin/halo
	# export ODLA_INC=$INSTALL_DIR/include
	# export ODLA_LIB=$INSTALL_DIR/lib
	export ODLA_TRT_USE_EXPLICIT_BATCH=1
	export YOLOV4_OUTPUT_PATH=$outputDir
	export YOLO_MODEL_PATH=$modelPath
	if [ -n "$ODLA_LIB" ];then
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ODLA_LIB
	fi
}

checkOutput() {
	outputPath=$outputDir
	if [ -d $outputPath ];then
		rm -rf $outputPath
	fi
	mkdir -p $outputPath
}

compileModelWithHalo() {
	# compile models with HALO
	$HALO_BIN  $modelPath \
		-target cxx \
		-disable-broadcasting \
		-entry-func-name=model \
		-o $outputPath/$model.cc \
		--inputs=input_1:0
		
	g++ $outputPath/$model.cc \
		-g -c -fPIC \
		-I$ODLA_INC \
		-o $outputPath/$model.o

	# generate so 
	g++ $outputPath/$model.o $outputPath/$model.bin \
		-shared \
		-lodla_tensorrt \
		-g \
		-Wl,-rpath=$ODLA_LIB \
		-L $ODLA_LIB  \
		-o $outputPath/$model.so
	
	cd $cntPath
}

setEnvVar
checkOutput
compileModelWithHalo
python3 detection.py | tee $1

# RUN: FileCheck --input-file %t.1 %s
# CHECK: [procesing] person.jpg
# CHECK-NEXT: ["dog"], pos:[63.3, 264.5, 200.3, 346.7] score:0.992
# CHECK-NEXT: ["person"], pos:[190.9, 99.3, 275.0, 368.4] score:0.996
# CHECK-NEXT: ["horse"], pos:[407.5, 134.0, 602.9, 346.3] score:0.995