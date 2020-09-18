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

set(HALO_TABLEGEN_EXE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/halo_tblgen")

macro(tblgen_instructions tds)
  set(LIST_TDS "${ARGV}")
  foreach(td IN LISTS LIST_TDS)
    get_filename_component(fname ${td} NAME_WE)
    set(LLVM_TARGET_DEFINITIONS ${td})
    halo_tblgen("${fname}.h" -gen-inst-class --ifdef=${fname})
  endforeach()
endmacro(tblgen_instructions)

# Generate "all_instructions.h" that includes all generated header files.
macro(generate_all_instr_header)
  set(LIST_TDS "${ARGV}")
  set(HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/all_instructions.h)
  file(WRITE ${HEADER_FILE})
  foreach(td IN LISTS LIST_TDS)
    get_filename_component(fname ${td} NAME_WE)
    file(APPEND ${HEADER_FILE} "#include \"halo/lib/ir/${fname}.h\"\n")
  endforeach()
endmacro(generate_all_instr_header)