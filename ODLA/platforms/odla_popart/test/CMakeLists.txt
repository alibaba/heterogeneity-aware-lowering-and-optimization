cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_FLAGS "-std=c++11 -g ${CMAKE_CXX_FLAGS}")

enable_testing()
include_directories(../../odla_popart/custom_ops/third_party/include)
include_directories(../../odla_popart)
include_directories(../../../../ODLA/include)
link_directories(../../../build/lib)

##odla_compute_test
add_executable(odla_compute_test  odla_compute_test.cc)
set_property(TARGET odla_compute_test PROPERTY CXX_STANDARD 14)
target_compile_definitions(odla_compute_test PRIVATE ONNX_NAMESPACE=onnx)
target_link_libraries(odla_compute_test PRIVATE popart poplar popops odla_popart pthread)
add_test(NAME odla_compute_test COMMAND odla_compute_test)

##popart_config_test
add_executable(popart_config_test  popart_config_test.cc)
set_property(TARGET popart_config_test PROPERTY CXX_STANDARD 14)
target_compile_definitions(popart_config_test PRIVATE ONNX_NAMESPACE=onnx)
target_link_libraries(popart_config_test PRIVATE popart poplar popops odla_popart pthread)
add_test(NAME popart_config_test COMMAND popart_config_test)

##odla_ops_math_test
add_executable(odla_ops_math_test  odla_ops_math_test.cc)
set_property(TARGET odla_ops_math_test PROPERTY CXX_STANDARD 14)
target_compile_definitions(odla_ops_math_test PRIVATE ONNX_NAMESPACE=onnx)
target_link_libraries(odla_ops_math_test PRIVATE popart poplar popops odla_popart pthread)
add_test(NAME odla_ops_math_test COMMAND odla_ops_math_test)

##odla_ops_nn_test
add_executable(odla_ops_nn_test  odla_ops_nn_test.cc)
set_property(TARGET odla_ops_nn_test PROPERTY CXX_STANDARD 14)
target_compile_definitions(odla_ops_nn_test PRIVATE ONNX_NAMESPACE=onnx)
target_link_libraries(odla_ops_nn_test PRIVATE popart poplar popops odla_popart pthread)
add_test(NAME odla_ops_nn_test COMMAND odla_ops_nn_test)

##odla_ops_process_test
add_executable(odla_ops_process_test  odla_ops_process_test.cc)
set_property(TARGET odla_ops_process_test PROPERTY CXX_STANDARD 14)
target_compile_definitions(odla_ops_process_test PRIVATE ONNX_NAMESPACE=onnx)
target_link_libraries(odla_ops_process_test PRIVATE popart poplar popops odla_popart pthread)
add_test(NAME odla_ops_process_test COMMAND odla_ops_process_test)

##odla_ops_test
add_executable(odla_ops_test  odla_ops_test.cc)
set_property(TARGET odla_ops_test PROPERTY CXX_STANDARD 14)
target_compile_definitions(odla_ops_test PRIVATE ONNX_NAMESPACE=onnx)
target_link_libraries(odla_ops_test PRIVATE popart poplar popops odla_popart pthread)
add_test(NAME odla_ops_test COMMAND odla_ops_test)

