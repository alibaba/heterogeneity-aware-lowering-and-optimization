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
set(NAME TARGET_JIT_COMPILER)

# source files.
set(SRCS
  cxx_jit_compiler.cc
  generic_jit_linker.cc
)

set(LLVM_LINK_COMPONENTS Support)
set(CLANG_LIBS
  clangCodeGen
  clangFrontend
  clangTooling
  clangToolingCore
  clangSerialization
)

set(LLD_LIBS
  lldCommon
  lldELF
  lldReaderWriter
)

# dependences which need to be built first.
set(DEPENDENCES TARGET ${CLANG_LIBS})

create_halo_object(TARGET_NAME ${NAME}
  TARGET_SRCS ${SRCS}
  TARGET_DEPENDENCES ${DEPENDENCES}
)

target_include_directories(${NAME} SYSTEM
  PRIVATE
  ${LLVM_SRC_DIR}/include
  ${CLANG_SRC_DIR}/include
  ${LLD_SRC_DIR}/include
  ${CMAKE_BINARY_DIR}/llvm/tools/clang/include
  ${CMAKE_BINARY_DIR}/llvm/include
)

target_link_libraries(${NAME} PRIVATE ${CLANG_LIBS} ${LLD_LIBS})
