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

cmake_minimum_required(VERSION 3.14.5)

set(PROJECT_NAME "HALO" HOMEPAGE_URL "https://github.com/alibaba/heterogeneity-aware-lowering-and-optimization")
set(PROJECT_VERSION "0.8.0")

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES C CXX ASM)

# ========= Set Build Type  =========
set(DEFAULT_BUILD_TYPE "RELEASE")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE)

# ========= Global CXX Flags =========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

# ========= Output Directories =========
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ========= Include CMake Modules =========
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(ccache)
include(clang_tidy)
include(create_halo_library)
include(doxygen)
include(glog)
include(protobuf)
include(flatbuf)
include(tablegen)
include(zbuild_llvm)

# ========= Include Directories =========
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ========= Global Setting and Options =========
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ========= Build Components =========
add_subdirectory(armory)
add_subdirectory(docs)
add_subdirectory(driver)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(ODLA)
add_subdirectory(runtime/templates)
add_subdirectory(tests)
add_subdirectory(models)
add_subdirectory(python)

# ========= Install Components =========
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(halo_install)
