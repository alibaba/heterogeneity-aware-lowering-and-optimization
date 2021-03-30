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

set(NAME CAFFEPARSER)

# Source files.
set(SRCS
  caffe_parser.cc
)

set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_SOURCE_DIR}/include/halo/lib/ir/caffe_convert.td
)

halo_tblgen(caffe_convert.h.inc -gen-converter-decl)
halo_tblgen(caffe_convert.cc.inc -gen-converter-def)
halo_tblgen(caffe_regist_op.h.inc -gen-register-op)
add_public_tablegen_target(CAFFE_GEN)

# Dependences which need to be built first.
set(DEPENDENCES
  PARSER
  CAFFE_GEN
)

create_halo_object(TARGET_NAME ${NAME}
  TARGET_SRCS ${SRCS}
  TARGET_DEPENDENCES ${DEPENDENCES}
)

target_include_directories(${NAME} PRIVATE
  ${LLVM_SRC_DIR}/include
  ${CMAKE_BINARY_DIR}/llvm/include
)

gen_protobuf_files(TARGET_NAME ${NAME}
  PROTO_DIR ${CMAKE_SOURCE_DIR}/external/protos/caffe
)