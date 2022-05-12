#!/bin/bash 
set -ex
# RUN: %s %t.1

model="yolov5l"
cntPath=$(cd `dirname $0`;pwd)

modelPath=$MODELS_ROOT/vision/detection/yolo/$model.onnx

outputDir=$cntPath/out
if [ -n "$TEST_TEMP_DIR" ];then
	outputDir=$TEST_TEMP_DIR/$model/out
fi

setEnvVar() {
	export ODLA_TRT_USE_EXPLICIT_BATCH=1
	export YOLO5l_OUTPUT_PATH=$outputDir
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
		-entry-func-name=model \
		-o $outputPath/$model.cc \
		
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
python3 detection.py  | tee $1

# RUN: FileCheck --input-file %t.1 %s
# CHECK: [procesing] zidane.jpg
# CHECK-NEXT: ["person"], pos:[749.0, 44.0, 1139.0, 708.0] score:0.942
# CHECK-NEXT: ["person"], pos:[135.0, 200.0, 1110.0, 711.0] score:0.895
# CHECK-NEXT: ["tie"], pos:[435.0, 436.0, 526.0, 717.0] score:0.818