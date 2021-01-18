CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g -DONNX_NAMESPACE=onnx
LDLIBS = -shared -lpopart -lpoplar -lpopops
INCLUDES = -Iinclude

BUILD_DIR = build
SOURCES = rsqrt.cc erf.cc
TARGET = $(BUILD_DIR)/libcustom_ops.so

all: create_build_dir rsqrt_custom_op rsqrt_test erf_test

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

rsqrt_custom_op: rsqrt.cc erf.cc
	$(CXX) $(SOURCES)  $(LDLIBS) $(CXXFLAGS) $(INCLUDES) -o $(TARGET)

rsqrt_test: rsqrt_test.cc 
	$(CXX) -std=c++14 -Wl,--no-as-needed rsqrt_test.cc -lpopart -lpoplar -lpopops -ldl -DONNX_NAMESPACE=onnx -o rsqrt_test -Wno-deprecated-declarations

erf_test: erf_test.cc
	$(CXX) -std=c++14 -Wl,--no-as-needed erf_test.cc -lpopart -lpoplar -lpopops -ldl -DONNX_NAMESPACE=onnx -o erf_test -Wno-deprecated-declarations

.PHONY: clean
clean:
	rm -r  $(BUILD_DIR) rsqrt_test erf_test
