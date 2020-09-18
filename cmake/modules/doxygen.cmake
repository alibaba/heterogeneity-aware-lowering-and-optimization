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

option(HALO_GEN_DOCS "Generate documentations" ON)

if(HALO_GEN_DOCS)
  # Check if Doxygen is installed.
  find_package(Doxygen)
  if(DOXYGEN_FOUND)  
    set(DOXYGEN_OUT_DIR ${CMAKE_BINARY_DIR}/docs)
    set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs/doxyfile.in)
    set(DOXYFILE_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

    set(ODLA_OUT_DIR ${CMAKE_BINARY_DIR}/odla_docs)
    set(ODLA_DOXYFILE_OUT ${ODLA_OUT_DIR}/Doxyfile)

    if(EXISTS ${DOXYGEN_OUT_DIR})
      file(REMOVE_RECURSE ${DOXYGEN_OUT_DIR})
    endif()

    file(MAKE_DIRECTORY ${DOXYGEN_OUT_DIR})
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/odla_docs)

    configure_file(${DOXYGEN_IN} ${DOXYFILE_OUT} @ONLY)
    configure_file(${PROJECT_SOURCE_DIR}/ODLA/Doxyfile ${ODLA_DOXYFILE_OUT} @ONLY)

    add_custom_target(DOCS
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
      COMMAND ${DOXYGEN_EXECUTABLE} ${ODLA_DOXYFILE_OUT}
      COMMAND make -C ${ODLA_OUT_DIR}/latex
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPENDS IRDOC
      COMMENT "Generating documentation with Doxygen"
      #VERBATIM
    )

  else()
    message(FATAL_ERROR "Doxygen is needed to generate the documentation")
  endif()
endif()