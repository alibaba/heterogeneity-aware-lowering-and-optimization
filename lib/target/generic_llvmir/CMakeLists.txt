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
set(NAME TARGET_GENERIC_LLVMIR)

# source files.
set(SRCS
  batch_matmul.cc
  batchnorm.cc
  conv.cc
  gather.cc
  gemm.cc
  generic_constant_writer.cc
  generic_llvmir_codegen.cc
  math_binary.cc
  math_unary.cc
  matmul.cc
  onehot.cc
  pad.cc
  pooling.cc
  reduce_mean.cc
  relu.cc
  reshape.cc
  return.cc
  sitofp.cc
  slice.cc
  softmax.cc
  transpose.cc
)

set(LLVM_LIBS
  LLVMAnalysis
  LLVMAsmPrinter
  LLVMBinaryFormat
  LLVMBitReader
  LLVMBitWriter
  LLVMCodeGen
  LLVMCore
  LLVMInstCombine
  LLVMIRReader
  LLVMipo
  LLVMLinker
  LLVMMC
  LLVMObject
  LLVMSelectionDAG
  LLVMScalarOpts
  LLVMSupport
  LLVMTarget
  LLVMVectorize
  LLVMX86CodeGen
)

# dependences which need to be built first.
set(DEPENDENCES ${LLVM_LIBS} TARGET)

create_halo_object(TARGET_NAME ${NAME}
  TARGET_SRCS ${SRCS}
  TARGET_DEPENDENCES ${DEPENDENCES}
)

target_include_directories(${NAME}
   PUBLIC ${LLVM_SRC_DIR}/include ${CMAKE_BINARY_DIR}/llvm/include
)

target_link_libraries(${NAME} PUBLIC ${LLVM_LIBS})