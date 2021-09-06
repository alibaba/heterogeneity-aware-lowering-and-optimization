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

add_odla_library(custom_ops SHARED
	#erf.cc
  rsqrt.cc
  postprocess.cc
  attention_mask.cc
)

set_property(TARGET custom_ops PROPERTY CXX_STANDARD 14)

target_compile_definitions(custom_ops PRIVATE ONNX_NAMESPACE=onnx)

if (NOT ODLA_BUILD_POPART_USE_CXX11ABI)
  target_compile_definitions(custom_ops PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

target_link_libraries(custom_ops PRIVATE popart-only)
target_include_directories(custom_ops PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx/
   ${CMAKE_CURRENT_SOURCE_DIR}/third_party/include/
)
