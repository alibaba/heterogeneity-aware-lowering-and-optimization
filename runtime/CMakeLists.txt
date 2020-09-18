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

set(PROJECT_NAME "HALO_RUNTIME_LIB")
set(PROJECT_VERSION "1.0.0")

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES C CXX ASM)

set(CMAKE_CXX_COMPILER clang++-9)

add_subdirectory(generic)
add_subdirectory(riscv)

install(TARGETS RT_GENERIC 
  ARCHIVE DESTINATION lib
)
install(TARGETS RT_RISCV
  ARCHIVE DESTINATION lib
)