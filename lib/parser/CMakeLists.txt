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

add_subdirectory(caffe)
add_subdirectory(onnx)
add_subdirectory(tflite)
add_subdirectory(tensorflow)

set(NAME PARSER)

# Source files.
set(SRCS
  parser.cc
)

# Dependences which need to be built first.
set(DEPENDENCES
  IR
)

create_halo_object(TARGET_NAME ${NAME}
  TARGET_SRCS ${SRCS}
  TARGET_DEPENDENCES ${DEPENDENCES}
)

target_include_directories(${NAME} PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/tensorflow
  ${CMAKE_CURRENT_BINARY_DIR}/onnx
  ${CMAKE_CURRENT_BINARY_DIR}/caffe
  ${CMAKE_CURRENT_BINARY_DIR}/tflite
  ${LLVM_SRC_DIR}/include
  ${CMAKE_BINARY_DIR}/llvm/include
)
