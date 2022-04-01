# ==============================================================================
# Copyright (C) 2019-2022 Alibaba Group Holding Limited.
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

set(NAME EXPORTER)

# Source files.
set(SRCS
  serializer.cc
)

# Dependences which need to be built first.
set(DEPENDENCES
  ONNXPARSER
)

create_halo_object(TARGET_NAME ${NAME}
  TARGET_SRCS ${SRCS}
  TARGET_DEPENDENCES ${DEPENDENCES}
)

target_include_directories(${NAME} PRIVATE
  ${CMAKE_BINARY_DIR}/lib/parser/onnx
)

if (HALO_NO_RTTI)
    target_compile_definitions(${NAME} PUBLIC -DGOOGLE_PROTOBUF_NO_RTTI=1)
endif()
