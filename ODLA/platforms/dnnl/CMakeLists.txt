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

set(DNNL_ROOT /opt/dnnl)
set(DNNL_CXX_FLAGS -flax-vector-conversions -mavx512f -mavx512bw -mavx512dq -mavx512vl -Wno-error)
if(NOT DNNL_COMPILER)
  message(STATUS "GCC10 path ${DNNL_COMPILER} not found, skip enable bf16 on odla_dnnl")
  set(DEFAULT_COMPILER ${CMAKE_CXX_COMPILER})
else()
  set(CMAKE_CXX_COMPILER_LAUNCHER ${DNNL_COMPILER})
  set(DEFAULT_COMPILER ${CMAKE_CXX_COMPILER})
  set(DNNL_CXX_FLAGS ${DNNL_CXX_FLAGS} -mavx512bf16)
  message(STATUS "Found GCC10: ${DNNL_COMPILER}, build odla_dnnl with bf16 support!")
endif()

add_odla_library(odla_dnnl SHARED odla_dnnl.cc)
find_library(dnnl NAMES dnnl PATHS ${DNNL_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
target_include_directories(odla_dnnl PRIVATE ${DNNL_ROOT}/include)
target_link_libraries(odla_dnnl ODLA ${dnnl})
target_compile_options(odla_dnnl PRIVATE ${DNNL_CXX_FLAGS})

add_odla_library(odla_dnnl_interpreter SHARED odla_dnnl.cc)
target_include_directories(odla_dnnl_interpreter PRIVATE ${DNNL_ROOT}/include)
target_compile_definitions(odla_dnnl_interpreter PRIVATE ODLA_DNNL_BUILD_AS_INTERPRETER)
target_link_libraries(odla_dnnl_interpreter ODLA ${dnnl})
target_compile_options(odla_dnnl_interpreter PRIVATE ${DNNL_CXX_FLAGS})
set(CMAKE_CXX_COMPILER_LAUNCHER ${DEFAULT_COMPILER})
