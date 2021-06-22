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

add_odla_library(odla_popart SHARED common.cc odla_compute.cc
                               odla_ops_math.cc odla_ops_nn.cc
                               odla_ops_process.cc odla_ops.cc
                               odla_ops_custom.cc
                               )
find_library(popart NAMES popart PATHS ${POPART_ROOT} PATH_SUFFIXES lib
             NO_DEFAULT_PATH)
find_package(popart CONFIG COMPONENTS popart-only REQUIRED)

if (ODLA_BUILD_POPART_CUSTOM_OPS)
  message(INFO "Build popart custom op")
  add_subdirectory(custom_ops)
endif()

#find_package(popart CONFIG COMPONENTS popart-only REQUIRED)
set_property(TARGET odla_popart PROPERTY CXX_STANDARD 14)

target_compile_definitions(odla_popart PRIVATE ONNX_NAMESPACE=onnx)
if (NOT ODLA_BUILD_POPART_USE_CXX11ABI)
  target_compile_definitions(odla_popart PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

target_link_libraries(odla_popart PUBLIC ODLA custom_ops popart-only)

if (ODLA_BUILD_POPART_COSTOM_OPS)
endif()
