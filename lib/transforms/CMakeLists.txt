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

# name.
set(NAME TRANSFORMS)

# source files.
set(SRCS
  analyzer.cc
  caffeextension_legalizer.cc
  constant_decombine.cc
  dce.cc
  device_placement.cc
  fusion.cc
  input_legalizer.cc
  input_rewriter.cc
  inst_simplify.cc
  onnxextension_legalizer.cc
  output_rewriter.cc
  reorder_channel.cc
  splitting.cc
  tfextension_legalizer.cc
  tfliteextension_legalizer.cc
  transforms_util.cc
  type_legalizer.cc
  typecast.cc
)

# dependences which need to be built first.
set(DEPENDENCES
  IRGEN
  PARSER
)

create_halo_object(TARGET_NAME ${NAME}
  TARGET_SRCS ${SRCS}
  TARGET_DEPENDENCES ${DEPENDENCES}
)

target_include_directories(${NAME} PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/parser/onnx
  ${CMAKE_BINARY_DIR}/lib/parser/onnx
  ${LLVM_SRC_DIR}/include
  ${CMAKE_BINARY_DIR}/llvm/include
)
