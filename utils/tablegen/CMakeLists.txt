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

add_llvm_external_project(halo_tblgen)
set(LLVM_LINK_COMPONENTS Support)

add_tablegen(halo_tblgen HALO
  attribute_emitter.cc
  converter_emitter.cc
  doc_emitter.cc
  enum_emitter.cc
  fusion_emitter.cc
  halo_tblgen.cc
  inst.cc
  inst_class_emitter.cc
  inst_info_emitter.cc
  irbuilder_emitter.cc
  modelzoo_emitter.cc
)

llvm_update_compile_flags(halo_tblgen)

include_directories(${LLVM_SRC_DIR}/include ${CMAKE_BINARY_DIR}/llvm/include)

set_target_properties(halo_tblgen PROPERTIES FOLDER "Tablegenning"
  CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
)
