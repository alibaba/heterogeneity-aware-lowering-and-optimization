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

# Build flatbuf.

find_package(Flatbuffers REQUIRED 1.12.0)
if (Flatbuffers_FOUND)
  message(STATUS "Found Flatbuffers ${Flatbuffers_VERSION}: ${Flatbuffers_LIBRARY}")
else()
  message(FATAL_ERROR "Flatbuffers lib not found")
endif(Flatbuffers_FOUND)

macro(gen_flatbuf_files)
  set(oneValueArgs TARGET_NAME FLAT_DIR)
  set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR})
  cmake_parse_arguments("" "" "${oneValueArgs}" "" ${ARGN})
  file(GLOB FLATS ${_FLAT_DIR}/*.fbs)

  foreach(FLAT_FILE ${FLATS})
    get_filename_component(FLAT_NAME ${FLAT_FILE} NAME_WE)
    set(GEN_HDR ${GEN_DIR}/${FLAT_NAME}_generated.h)
    # execute flatc tools to compile schema file to XXX_generated.h
    add_custom_command(OUTPUT ${GEN_HDR}
      COMMAND flatc -c -o ${GEN_DIR} ${FLAT_FILE}
      DEPENDS ${FLAT_FILE}
    )
    add_custom_target(${FLAT_NAME} DEPENDS ${GEN_HDR})
    add_dependencies(${_TARGET_NAME} ${FLAT_NAME})
  endforeach(FLAT_FILE)

  # Add link library and include pathes for generated files.
  target_link_libraries(${_TARGET_NAME} PRIVATE ${Flatbuffers_LIBRARIES})
  target_include_directories(${_TARGET_NAME} PRIVATE ${Flatbuffers_INCLUDE_DIRS} ${GEN_DIR})

  # No clang-tidy on generated files.
  set_target_properties(${_TARGET_NAME} PROPERTIES CXX_CLANG_TIDY "")
endmacro(gen_flatbuf_files)
