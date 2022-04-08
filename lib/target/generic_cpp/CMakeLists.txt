# ==============================================================================
# Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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
set(NAME TARGET_GENERIC_CPP)

# source files.
set(SRCS
  add.cc
  argmax.cc
  batchnorm.cc
  call.cc
  cast.cc
  code_formatter.cc
  compress.cc
  concat.cc
  conv.cc
  custom.cc
  deconv.cc
  einsum.cc
  expand.cc
  gather.cc
  gemm.cc
  generic_cxx_codegen.cc
  generic_cxx_constant_writer.cc
  hardsigmoid.cc
  hardswish.cc
  if.cc
  loss.cc
  lrn.cc
  lstm.cc
  matmul.cc
  multiheadattention.cc
  nms.cc
  onehot.cc
  pad.cc
  pooling.cc
  quantization.cc
  random.cc
  reduction.cc
  relu.cc
  reshape.cc
  resize.cc
  return.cc
  shrink.cc
  sigmoid.cc
  slice.cc
  softmax.cc
  softplus.cc
  softsign.cc
  split.cc
  swish.cc
  templated_cxx_codegen.cc
  tfidf.cc
  tile.cc
  topk.cc
  transpose.cc
  hgengine.cc
  quant.cc
  dequant.cc
)

set(LLVM_LINK_COMPONENTS Support)
set(CLANG_LIBS
  clangAST
  clangBasic
  clangFormat
  clangRewrite
  clangRewriteFrontend
  clangTooling
  clangToolingCore
)

# dependences which need to be built first.
set(DEPENDENCES TARGET ${CLANG_LIBS})

create_halo_object(TARGET_NAME ${NAME}
  TARGET_SRCS ${SRCS}
  TARGET_DEPENDENCES ${DEPENDENCES}
)

target_include_directories(${NAME} SYSTEM
   PRIVATE
   ${CLANG_SRC_DIR}/include
   ${LLVM_SRC_DIR}/include
   ${CMAKE_BINARY_DIR}/llvm/tools/clang/include
   ${CMAKE_BINARY_DIR}/llvm/include
)

target_link_libraries(${NAME} PRIVATE ${CLANG_LIBS})
