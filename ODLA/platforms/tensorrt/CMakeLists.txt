# ==============================================================================
# Copyright (C) 2021 Alibaba Group Holding Limited.
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

set(CUDA_VERSION 10.0)
set(TRT_ROOT "/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux"
    CACHE STRING "Path of TensorRT package")
find_library(cudart NAMES cudart PATHS ${TRT_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
find_library(nvinfer NAMES nvinfer)
if(NOT cudart OR NOT nvinfer)
  message(STATUS "CUDA library not found, skip building ODLA for TensorRT")
  add_library(odla_tensorrt INTERFACE) # pseudo target
  return()
endif()

add_odla_library(odla_tensorrt SHARED odla_tensorrt.cc)

target_include_directories(odla_tensorrt PRIVATE ${TRT_ROOT}/include)
target_compile_options(odla_tensorrt PRIVATE -Wno-deprecated-declarations)
target_link_libraries(odla_tensorrt ODLA ${nvinfer} ${cudart})
