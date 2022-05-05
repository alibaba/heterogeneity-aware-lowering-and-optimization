# ==============================================================================
# Copyright (C) 2020-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
# ==============================================================================

enable_language(CUDA)

set(CUDA_VERSION 11.3 CACHE STRING "Version of CUDA tool-kit.")
set(CUDNN_VERSION 7.6 CACHE STRING "Version of Cudnn.")
set(CUDA_ROOT /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux)
find_library(CUDART_LIB NAMES cudart PATHS ${CUDA_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
find_library(nvinfer NAMES nvinfer)
find_library(nvinfer_plugin NAMES nvinfer_plugin)
find_library(nvinfer_plugin_static NAMES nvinfer_plugin_static)


if(NOT CUDART_LIB OR NOT nvinfer OR NOT nvinfer_plugin)
  message(STATUS "CUDA library not found, skip building ODLA for TensorRT")
  add_library(odla_tensorrt INTERFACE) # pseudo target
  return()
endif()

set(PLUGIN_NAME odla_plugin)
add_subdirectory(plugins)

add_odla_library(odla_tensorrt SHARED odla_tensorrt.cc)
add_dependencies(odla_tensorrt ODLA_TRTPLUGIN)

target_include_directories(odla_tensorrt PRIVATE ${CUDA_ROOT}/include ${CMAKE_BINARY_DIR}/include/halo)
target_compile_options(odla_tensorrt PRIVATE -Wno-deprecated-declarations)
target_link_libraries(odla_tensorrt ODLA ${nvinfer} ${CUDART_LIB} ${nvinfer_plugin})
set_target_properties(odla_tensorrt PROPERTIES LINK_FLAGS
                      "-Wl,-rpath=\\\${ORIGIN}/tensorrt:\\\${ORIGIN} -L${CMAKE_BINARY_DIR}/tensorrt")
target_link_libraries(odla_tensorrt ${PLUGIN_NAME})
set_target_properties(odla_tensorrt PROPERTIES LINK_FLAGS
                      "-Wl,-rpath=${CMAKE_BINARY_DIR}/lib -L${CMAKE_BINARY_DIR}/lib")
