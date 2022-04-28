#!/bin/bash 
set -ex

model="yolov5l"
cntPath=$(cd `dirname $0`;pwd)

modelPath=$cntPath/model/$model.onnx
outputDir=$cntPath/out

setEnvVar() {
	export INSTALL_DIR=/opt/halo
	export HALO_BIN=$INSTALL_DIR/bin/halo
	export ODLA_INC=$INSTALL_DIR/include
	export ODLA_LIB=$INSTALL_DIR/lib
	export ODLA_TRT_USE_EXPLICIT_BATCH=1
	export YOLOV_OUTPUT_PATH=$outputDir
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ODLA_LIB
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
		# -disable-broadcasting \
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
python3 detection.py