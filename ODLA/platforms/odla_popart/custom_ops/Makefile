CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g -DONNX_NAMESPACE=onnx
LDLIBS = -shared -lpopart -lpoplar -lpopops -lpoputil
INCLUDES = -Iinclude -Ithird_party/onnx/ -Ithird_party/include

BUILD_DIR = build
SOURCES = rsqrt.cc erf.cc postprocess.cc attention_mask.cc
TARGET = $(BUILD_DIR)/libcustom_ops.so

all: create_build_dir rsqrt_custom_op rsqrt_test attention_mask_test

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

rsqrt_custom_op: ${SOURCES}
	$(CXX) $(SOURCES)  $(LDLIBS) $(CXXFLAGS) $(INCLUDES) -o $(TARGET)

rsqrt_test: rsqrt_test.cc rsqrt_custom_op
	$(CXX) -std=c++14 rsqrt_test.cc -lpopart -lpoplar -lpopops -ldl -DONNX_NAMESPACE=onnx -o rsqrt_test

#erf_test: erf_test.cc rsqrt_custom_op
#	$(CXX) -std=c++14 erf_test.cc -lpopart -lpoplar -lpopops -ldl -DONNX_NAMESPACE=onnx -o erf_test

attention_mask_test: attention_mask_test.cc rsqrt_custom_op
#	$(CXX) $(LDLIBS) $(CXXFLAGS) $(INCLUDES) -o attention_mask_test
	$(CXX) -std=c++14 -fPIC -g -DONNX_NAMESPACE=onnx attention_mask_test.cc -lpopart -lpoplar -lpopops -ldl -o attention_mask_test

.PHONY: clean
clean:
	rm -r  $(BUILD_DIR) rsqrt_test attention_mask_test
