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

set(CMAKE_CXX_STANDARD 14)
#add_odla_library(custom_ops SHARED custom_ops/rsqrt.cc custom_ops/erf.cc)
#set(CUSTOM_OP_CXX_FLAGS -std=c++14 -fPIC -DONNX_NAMESPACE=onnx)
#set(POPART_CUSTOM_OP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops)
#target_include_directories(custom_ops PRIVATE ${POPART_CUSTOM_OP_PATH}/include )
#target_link_libraries(custom_ops ODLA poplar popart popops)
#target_compile_options(custom_ops PRIVATE ${CUSTOM_OP_CXX_FLAGS})

#set(POPLAR_ROOT ${POPLAR_SDK_ENABLED})
#set(POPART_ROOT ${POPART_ROOT})
#set(ODLA_POPART_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ODLA/platforms/odla_popart)
add_odla_library(odla_popart SHARED odla_compute.cc odla_ops.cc common.cc) 
 #find_library(popart NAMES popart PATHS ${POPART_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
 #find_library(popart NAMES poplar PATHS ${POPLAR_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
 #find_library(popart NAMES popops PATHS ${POPLAR_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
find_library(custom_ops NAMES custom_ops PATHS ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops PATH_SUFFIXES lib NO_DEFAULT_PATH)
#target_include_directories(odla_popart PRIVATE ${POPART_ROOT}/include )
#target_include_directories(odla_popart PRIVATE ${POPLAR_ROOT}/include )
target_link_libraries(odla_popart ODLA poplar popart popops ${custom_ops})

set(POPART_CXX_FLAGS -std=c++14 -DONNX_NAMESPACE=onnx)
target_compile_options(odla_popart PRIVATE ${POPART_CXX_FLAGS})


set(CMAKE_CXX_COMPILER_LAUNCHER ${DEFAULT_COMPILER})
set(CMAKE_CXX_STANDARD 17)
