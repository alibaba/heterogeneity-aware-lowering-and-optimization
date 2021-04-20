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

# Build protobuf.

option(HALO_USE_STATIC_PROTOBUF "Link against static protobuf library" OFF)

if (HALO_USE_STATIC_PROTOBUF)
  set(Protobuf_USE_STATIC_LIBS ON)
endif()

find_package(Protobuf REQUIRED 3.9.1)

if (Protobuf_FOUND)
  message(STATUS "Found Protobuf ${Protobuf_VERSION}: ${Protobuf_LIBRARY}")
else()
  message(FATAL_ERROR "Protobuf lib not found")
endif(Protobuf_FOUND)

macro(gen_protobuf_files)
  set(oneValueArgs TARGET_NAME PROTO_DIR)
  set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR})
  cmake_parse_arguments("" "" "${oneValueArgs}" "" ${ARGN})
  file(GLOB PROTOS ${_PROTO_DIR}/*.proto)
  set(ALL_GEN_SRCS) # all generated sources
  foreach(PROTO_FILE ${PROTOS})
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    protobuf_generate_cpp(GEN_SRC GEN_HDR ${PROTO_FILE})
    list(APPEND ALL_GEN_SRCS ${GEN_SRC})
    add_custom_command(OUTPUT  ${GEN_SRC}
      COMMAND ${PROTOC} --cpp_out=${GEN_DIR} -I${_PROTO_DIR}  ${PROTO_FILE}
      DEPENDS PROTOBUF ${PROTO_FILE}
    )
  endforeach(PROTO_FILE)

  # Add link library and include pathes for generated files.
  set(GEN_TARGET ${_TARGET_NAME}_GEN)
  add_library(${GEN_TARGET} OBJECT "")
  target_sources(${GEN_TARGET} PUBLIC ${ALL_GEN_SRCS})
  if (HALO_NO_RTTI)
    target_compile_definitions(${GEN_TARGET} PUBLIC -DGOOGLE_PROTOBUF_NO_RTTI=1)
  endif()
  target_link_libraries(${GEN_TARGET} PUBLIC ${Protobuf_LIBRARIES})
  target_include_directories(${GEN_TARGET} PUBLIC ${Protobuf_INCLUDE_DIRS} ${GEN_DIR})

  # Halo target depends on the generated files.
  target_link_libraries(${_TARGET_NAME} PRIVATE ${GEN_TARGET})

  # No clang-tidy on generated files.
  set_target_properties(${_TARGET_NAME} PROPERTIES CXX_CLANG_TIDY "")
endmacro(gen_protobuf_files)