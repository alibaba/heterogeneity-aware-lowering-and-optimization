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

option(ODLA_BUILD_POPART_USE_CXX11ABI "Use CXX11 ABI for odla_popart" ON)
option(ODLA_BUILD_POPART_CUSTOM_OPS "Link with Popart custom ops" ON)
option(ODLA_BUILD_POPART_UNITEST "Link with Popart unitest " ON)


if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
endif()

add_odla_library(odla_popart SHARED common.cc odla_compute.cc
                               odla_ops_math.cc odla_ops_nn.cc
                               odla_ops_process.cc odla_ops.cc
                               odla_ops_custom.cc odla_pipeline.cc
                               odla_popart.cc popart_config.cc
                               )

if (NOT POPLAR_ROOT)
  set(POPLAR_ROOT "/opt/poplar_sdk/poplar" CACHE PATH "Path of poplar root")
endif()
if (NOT POPART_ROOT)
  set(POPART_ROOT "/opt/poplar_sdk/popart" CACHE PATH "Path of popart root")
endif()
set(POPLAR_ROOT ${POPLAR_ROOT} PARENT_SCOPE)
set(POPART_ROOT ${POPART_ROOT} PARENT_SCOPE)

list(APPEND CMAKE_PREFIX_PATH ${POPLAR_ROOT})
list(APPEND CMAKE_PREFIX_PATH ${POPART_ROOT})
find_package(popart 2.3.0... REQUIRED CONFIG COMPONENTS popart-only REQUIRED)

message(STATUS "Found popart, version: ${popart_VERSION}")

if (ODLA_BUILD_POPART_CUSTOM_OPS)
  message(STATUS "Build popart custom op")
  add_subdirectory(custom_ops)
endif()

set_property(TARGET odla_popart PROPERTY CXX_STANDARD 14)

target_compile_definitions(odla_popart PRIVATE ONNX_NAMESPACE=onnx)
if (NOT ODLA_BUILD_POPART_USE_CXX11ABI)
  target_compile_definitions(odla_popart PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()


target_link_libraries(odla_popart PUBLIC ODLA custom_ops popart-only)

if(ENABLE_COVERAGE)
  target_link_libraries(odla_popart PUBLIC gcov)
endif()

target_include_directories(odla_popart PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops/third_party/onnx/
   ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops/third_party/include/
)
if(ODLA_BUILD_POPART_UNITEST)
  add_subdirectory(test)
endif()
