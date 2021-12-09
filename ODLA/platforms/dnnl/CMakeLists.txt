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

set(CMAKE_VERBOSE_MAKEFILE on)

if (ODLA_BUILD_DNNL_GPU)
  if(NOT DEFINED ENV{DNNLROOT})
     message(FATAL_ERROR "Not defined environment variable: DNNLROOT.\nPlease run: source /opt/intel/oneapi/dnnl/latest/env/vars.sh")
  endif()

  find_program(DPCPP dpcpp)
  if(NOT DPCPP)
    message(FATAL_ERROR "dpcpp not found!\nPlease run: source /opt/intel/oneapi/compiler/latest/env/vars.sh ")
  endif()

  message(${DPCPP})

  set(DNNL_COMPILER dpcpp)
  message(STATUS "BUILD odla_dnnl_gpu by ${DNNL_COMPILER} in oneAPI")
  set(DNNL_ROOT $ENV{DNNLROOT})
  set(DNNL_CXX_FLAGS ${DNNL_CXX_FLAGS} -DODLA_BUILD_DNNL_GPU -I${DNNL_ROOT}/examples)
  set(DEFAULT_COMPILER ${DNNL_COMPILER})
  set(CMAKE_CXX_COMPILER ${DNNL_COMPILER})
  message(STATUS "CMAKE_CXX_COMPILER, ${CMAKE_CXX_COMPILER}")

else()
  if(DEFINED ENV{DNNLROOT})
    set(DNNL_ROOT $ENV{DNNLROOT})
  else()
    set(DNNL_ROOT /opt/dnnl)
  endif()
  #set(DNNL_CXX_FLAGS -flax-vector-conversions -mfma -O2 -Wno-error)
  set(DNNL_CXX_FLAGS -flax-vector-conversions -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl -Wno-error)
  if(NOT DNNL_COMPILER)
    message(STATUS "GCC10 path ${DNNL_COMPILER} not found, skip enable bf16 on odla_dnnl")
    set(DEFAULT_COMPILER ${CMAKE_CXX_COMPILER})
  else()
    set(CMAKE_CXX_COMPILER_LAUNCHER ${DNNL_COMPILER})
    set(DEFAULT_COMPILER ${CMAKE_CXX_COMPILER})
    set(DNNL_CXX_FLAGS ${DNNL_CXX_FLAGS} -mavx512bf16)
    message(STATUS "Found GCC10: ${DNNL_COMPILER}, build odla_dnnl with bf16 support!")
  endif()
endif()

set(ODLA_DNNL_SRC
  odla_dnnl.cc
  odla_dnnl_binary.cc
  odla_dnnl_cast.cc
  odla_dnnl_debug.cc
  odla_dnnl_loss.cc
  odla_dnnl_rnn.cc
  odla_dnnl_quantization.cc
  odla_dnnl_statistics.cc
  odla_dnnl_unary.cc
)

add_odla_library(odla_dnnl SHARED ${ODLA_DNNL_SRC})

if (ODLA_BUILD_DNNL_GPU)
  set_target_properties(odla_dnnl PROPERTIES OUTPUT_NAME "odla_dnnl_gpu")
endif()

find_library(dnnl NAMES dnnl PATHS ${DNNL_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
target_include_directories(odla_dnnl PRIVATE ${DNNL_ROOT}/include)
target_include_directories(odla_dnnl PRIVATE ${EIGEN_ROOT})
target_link_libraries(odla_dnnl ODLA ${dnnl})
target_compile_options(odla_dnnl PRIVATE ${DNNL_CXX_FLAGS})

add_odla_library(odla_dnnl_interpreter SHARED ${ODLA_DNNL_SRC})
target_include_directories(odla_dnnl_interpreter PRIVATE ${EIGEN_ROOT})
target_include_directories(odla_dnnl_interpreter PRIVATE ${DNNL_ROOT}/include)
target_compile_definitions(odla_dnnl_interpreter PRIVATE ODLA_DNNL_BUILD_AS_INTERPRETER)
target_link_libraries(odla_dnnl_interpreter ODLA ${dnnl})
target_compile_options(odla_dnnl_interpreter PRIVATE ${DNNL_CXX_FLAGS})
set(CMAKE_CXX_COMPILER_LAUNCHER ${DEFAULT_COMPILER})
