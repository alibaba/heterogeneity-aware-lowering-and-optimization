# ==============================================================================
# Copyright (C) 2019-2020 Alibaba Group Holding Limited.
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

macro(add_odla_library)
  add_library(${ARGN})
  install(TARGETS ${ARGV0} LIBRARY DESTINATION lib)
endmacro(add_odla_library)

set(CMAKE_SKIP_BUILD_RPATH FALSE)

add_subdirectory(dnnl)
add_subdirectory(odla_popart)

set(CUDA_VERSION 10.0)
set(TRT_ROOT /usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux)
find_library(cudart NAMES cudart PATHS ${TRT_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
find_library(nvinfer NAMES nvinfer)
if(NOT cudart OR NOT nvinfer)
  message(STATUS "CUDA library not found, skip building odla for TensorRT")
  add_library(odla_tensorrt INTERFACE) # pseudo target
else()
  add_odla_library(odla_tensorrt SHARED odla_tensorrt.cc)
  target_include_directories(odla_tensorrt PRIVATE ${TRT_ROOT}/include)
  target_compile_options(odla_tensorrt PRIVATE -Wno-deprecated-declarations)
  target_link_libraries(odla_tensorrt ODLA ${nvinfer} ${cudart})
endif()

set(EIGEN_VERSION 3.3.7)
set(EIGEN_ROOT /opt/eigen-${EIGEN_VERSION})
add_odla_library(odla_eigen SHARED odla_eigen.cc)
target_include_directories(odla_eigen PRIVATE ${EIGEN_ROOT})
target_link_libraries(odla_eigen ODLA)

set(XNNPACK_ROOT /opt/XNNPACK)
add_odla_library(odla_xnnpack SHARED odla_xnnpack.c)
find_library(xnnpack NAMES XNNPACK PATHS ${XNNPACK_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
find_library(clog NAMES clog PATHS ${XNNPACK_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
find_library(cpuinfo NAMES cpuinfo PATHS ${XNNPACK_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
find_library(pthreadpool NAMES pthreadpool PATHS ${XNNPACK_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)

target_include_directories(odla_xnnpack PRIVATE ${XNNPACK_ROOT}/include)
target_compile_options(odla_xnnpack PRIVATE -Wno-deprecated-declarations)
target_link_libraries(odla_xnnpack ODLA ${xnnpack} ${cpuinfo} ${pthreadpool} ${clog} pthread)
