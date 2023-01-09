#!/bin/bash 
set -ex
# RUN: %s %t.1

model="retinanet-9"
cntPath=$(cd `dirname $0`;pwd)

modelPath=$MODELS_ROOT/vision/detection/retinanet/$model.onnx

outputDir=$cntPath/out
if [ -n "$TEST_TEMP_DIR" ];then
	outputDir=$TEST_TEMP_DIR/$model/out
fi

setEnvVar() {
	export ODLA_TRT_USE_EXPLICIT_BATCH=1
	export RETINANET_MODEL_PATH=$modelPath
	export RETINANET_OUTPUT_PATH=$outputDir
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
		--inputs=input
		
	g++ $outputPath/$model.cc \
		-g -c -fPIC \
		-I$ODLA_INC \
		-o $outputPath/$model.o

	# generate so lib
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
# CHECK: output_720_80.txt True
# CHECK-NEXT: output_720_40.txt True
# CHECK-NEXT: output_720_20.txt True
# CHECK-NEXT: output_720_10.txt True
# CHECK-NEXT: output_720_5.txt True
# CHECK-NEXT: output_36_80.txt True
# CHECK-NEXT: output_36_40.txt True
# CHECK-NEXT: output_36_20.txt True
# CHECK-NEXT: output_36_10.txt True
# CHECK-NEXT: output_36_5.txt True