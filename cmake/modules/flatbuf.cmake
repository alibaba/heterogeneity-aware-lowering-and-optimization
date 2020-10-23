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
  set(ALL_GEN_SRCS) # all generated sources
  foreach(FLAT_FILE ${FLATS})
    get_filename_component(FLAT_NAME ${FLAT_FILE} NAME_WE)
  set(GEN_HDR ${GEN_DIR}/${FLAT_NAME}_generated.h)
  set(GEN_SRC ${GEN_DIR}/${FLAT_NAME}_generated.cc)
    list(APPEND ALL_GEN_SRCS ${GEN_SRC})

    # execute flatc tools to compile schema file to XXX_generated.h
    execute_process(COMMAND touch ${GEN_SRC})
    execute_process(COMMAND flatc -c -o ${GEN_DIR} ${FLAT_FILE})
  endforeach(FLAT_FILE)

  # Add link library and include pathes for generated files.
  set(GEN_TARGET ${_TARGET_NAME}_GEN)
  add_library(${GEN_TARGET} OBJECT "")
  target_sources(${GEN_TARGET} PUBLIC ${ALL_GEN_SRCS})
  target_link_libraries(${GEN_TARGET} PUBLIC ${Flatbuffers_LIBRARIES})
  target_include_directories(${GEN_TARGET} PUBLIC ${Flatbuffers_INCLUDE_DIRS} ${GEN_DIR})
  # Halo target depends on the generated files.
  target_link_libraries(${_TARGET_NAME} PRIVATE ${GEN_TARGET})

  # No clang-tidy on generated files.
  set_target_properties(${_TARGET_NAME} PROPERTIES CXX_CLANG_TIDY "")
endmacro(gen_flatbuf_files)
