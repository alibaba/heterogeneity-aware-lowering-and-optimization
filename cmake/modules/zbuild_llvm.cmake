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

unset(CMAKE_CXX_FLAGS)
set(LLVM_SRC_DIR ${CMAKE_SOURCE_DIR}/external/llvm-project/llvm)
set(LLVM_CCACHE_BUILD ${HALO_CCACHE_BUILD})
set(LLVM_ENABLE_EH OFF)

set(LLVM_LIT_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm/bin)

# Enable build clang libraries
set(CLANG_SRC_DIR ${CMAKE_SOURCE_DIR}/external/llvm-project/clang)
set(LLVM_ENABLE_PROJECTS_USED ON)
set(LLVM_ENABLE_PROJECTS "clang")
set(LLVM_EXTERNAL_CLANG_SOURCE_DIR ${CLANG_SRC_DIR})

# Enable build lld libraries
set(LLD_BUILD_TOOLS OFF)
set(LLD_SRC_DIR ${CMAKE_SOURCE_DIR}/external/llvm-project/lld)
set(LLVM_ENABLE_PROJECTS_USED ON)
set(LLVM_ENABLE_PROJECTS "${LLVM_ENABLE_PROJECTS};lld")
set(LLVM_EXTERNAL_LLD_SOURCE_DIR ${LLD_SRC_DIR})

set(HALO_USE_SHARED_LLVM_LIBS AUTO CACHE STRING
  "Link against shared LLVM libraries")
set_property(CACHE HALO_USE_SHARED_LLVM_LIBS PROPERTY STRINGS AUTO ON OFF)
set(BUILD_SHARED_LIBS OFF)
if (HALO_USE_SHARED_LLVM_LIBS STREQUAL "AUTO")
  if (CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(BUILD_SHARED_LIBS ON)
  endif()
else()
  set(BUILD_SHARED_LIBS ${BUILD_LLVM_COMPONENTS_SHARED})
endif()

add_subdirectory(${LLVM_SRC_DIR} ${CMAKE_BINARY_DIR}/llvm EXCLUDE_FROM_ALL)
add_llvm_external_project(clang ${CLANG_SRC_DIR})
add_llvm_external_project(lld ${LLD_SRC_DIR})
set(CMAKE_MODULE_PATH ${LLVM_SRC_DIR}/cmake/modules)
include(DetermineGCCCompatible)

function(halo_tblgen ofn)
  tablegen(HALO ${ARGV} "-I${CMAKE_SOURCE_DIR}/include/halo/lib/ir")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT}
    ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE
  )
endfunction()
