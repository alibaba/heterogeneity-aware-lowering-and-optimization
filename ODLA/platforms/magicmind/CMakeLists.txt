# ==============================================================================
# Copyright (C) 2020-2021 Alibaba Group Holding Limited.
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

set(NEUWARE_HOME /usr/local/neuware/)
set(DEFAULT_COMPILER ${CMAKE_CXX_COMPILER})
find_library(MAGICMIND_LIB NAMES magicmind PATHS ${NEUWARE_HOME} PATH_SUFFIXES lib64 NO_DEFAULT_PATH)
if(NOT MAGICMIND_LIB)
  message(STATUS "not find MagicMind")
  return()
else()
  message(STATUS "find MagicMind")
endif()
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

add_library(odla_magicmind SHARED odla_magicmind.cc)
target_include_directories(odla_magicmind PRIVATE ${NEUWARE_HOME}/include)
target_link_libraries(odla_magicmind ODLA ${MAGICMIND_LIB})
target_compile_options(odla_magicmind PRIVATE "")
set(CMAKE_CXX_COMPILER_LAUNCHER ${DEFAULT_COMPILER})
