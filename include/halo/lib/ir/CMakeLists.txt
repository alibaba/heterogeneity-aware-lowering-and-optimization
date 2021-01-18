# ==============================================================================
# Copyright (C) 2019-2020 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0(the "License");
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

# TableGenned files
set(LLVM_TARGET_DEFINITIONS attribute_types.td)
halo_tblgen(attribute.cc.inc -gen-attr-def)
halo_tblgen(attribute.h.inc -gen-attr-decl)
halo_tblgen(attribute_enums.h.inc -gen-attr-enum)

set(LLVM_TARGET_DEFINITIONS arg_types.td)
halo_tblgen(datatype.def -gen-datatype)

set(INST_TDS)
list(APPEND INST_TDS
  common_instructions.td
  common_cast_instructions.td
  common_reduction_instructions.td
  controlflow_instructions.td
  creator_instructions.td
  math_instructions.td
  math_binary_instructions.td
  math_unary_instructions.td
  nn_activation_instructions.td
  nn_cnn_instructions.td
  nn_instructions.td
  object_detection_instructions.td
  customize_instructions.td
)
tblgen_instructions(${INST_TDS})

set(LLVM_TARGET_DEFINITIONS instructions.td)
halo_tblgen(instructions_info.def -gen-inst-info)
halo_tblgen(ir_builder.h.inc -gen-irbuilder-decl)
halo_tblgen(ir_builder.cc.inc -gen-irbuilder-def)

set(LLVM_TARGET_DEFINITIONS tf_convert.td)
halo_tblgen(convert_tf_info.def -gen-convert-info)

set(LLVM_TARGET_DEFINITIONS onnx_convert.td)
halo_tblgen(convert_onnx_info.def -gen-convert-info)

set(LLVM_TARGET_DEFINITIONS tflite_convert.td)
halo_tblgen(convert_tflite_info.def -gen-convert-info)

set(LLVM_TARGET_DEFINITIONS caffe_convert.td)
halo_tblgen(convert_caffe_info.def -gen-convert-info)

set(LLVM_TARGET_DEFINITIONS fusion.td)
halo_tblgen(fusion.cc.inc -gen-fusion)

add_public_tablegen_target(IRGEN)

list(APPEND INST_TDS
  extension_instructions.h
)
generate_all_instr_header(${INST_TDS})