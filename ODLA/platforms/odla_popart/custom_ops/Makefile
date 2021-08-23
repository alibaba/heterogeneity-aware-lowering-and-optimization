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
	$(CXX) $(CXXFLAGS) rsqrt_test.cc -lpopart -lpoplar -lpopops -ldl -o rsqrt_test

erf_test: erf_test.cc custom_op
	$(CXX) $(CXXFLAGS) erf_test.cc -lpopart -lpoplar -lpopops -ldl -o erf_test

.PHONY: clean
clean:
	rm -r  $(BUILD_DIR) rsqrt_test erf_test
