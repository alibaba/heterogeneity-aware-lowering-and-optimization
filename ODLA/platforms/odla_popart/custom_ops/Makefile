CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g -DONNX_NAMESPACE=onnx -D_GLIBCXX_USE_CXX11_ABI=1
LDLIBS = -shared -lpopart -lpoplar -lpopops -lpoputil
INCLUDES = -Iinclude -I third_party/include -I third_party/onnx

BUILD_DIR = build
SOURCES = rsqrt.cc erf.cc
TARGET = $(BUILD_DIR)/libcustom_ops.so

all: create_build_dir custom_op rsqrt_test

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

custom_op:  erf.cc rsqrt.cc nms.cc 
	$(CXX) $^ $(LDLIBS) $(CXXFLAGS) $(INCLUDES) -o $(TARGET)

rsqrt_test: rsqrt_test.cc custom_op
	$(CXX) -std=c++14 rsqrt_test.cc -lpopart -lpoplar -lpopops -ldl -DONNX_NAMESPACE=onnx -o rsqrt_test -D_GLIBCXX_USE_CXX11_ABI=1

erf_test: erf_test.cc custom_op
	$(CXX) -std=c++14 erf_test.cc -lpopart -lpoplar -lpopops -ldl -DONNX_NAMESPACE=onnx -o erf_test -D_GLIBCXX_USE_CXX11_ABI=1

.PHONY: clean
clean:
	rm -r  $(BUILD_DIR) rsqrt_test erf_test
