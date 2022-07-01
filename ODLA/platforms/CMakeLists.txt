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

macro(add_odla_library)
  add_library(${ARGN})
  install(TARGETS ${ARGV0} LIBRARY DESTINATION lib)
endmacro(add_odla_library)

set(CMAKE_SKIP_BUILD_RPATH FALSE)

option(ODLA_BUILD_ASCEND "Build ODLA Runtime Library for ASCEND" ON)
if (ODLA_BUILD_ASCEND)
  add_subdirectory(ascend)
endif()

option(ODLA_BUILD_EIGEN "Build ODLA Runtime Library for Eigen" ON)
if (ODLA_BUILD_EIGEN)
  set(EIGEN_VERSION 3.4.0)
  set(EIGEN_ROOT "/opt/eigen-${EIGEN_VERSION}" CACHE STRING "Path of Eigen package")
  find_file(eigen NAMES Eigen PATHS ${EIGEN_ROOT} PATH_SUFFIXES Eigen NO_DEFAULT_PATH)
  if(NOT eigen)
    message(STATUS "Eigen not found")
  else()
    message(STATUS "Found Eigen")
  endif()
  add_odla_library(odla_eigen SHARED odla_eigen.cc)
  target_compile_options(odla_eigen PRIVATE "-Wno-enum-compare")
  target_include_directories(odla_eigen PRIVATE ${EIGEN_ROOT})
  target_link_libraries(odla_eigen ODLA)
endif()

option(ODLA_BUILD_DNNL "Build ODLA Runtime Library for DNNL" ON)
if (ODLA_BUILD_DNNL)
  add_subdirectory(dnnl)
endif()

option(ODLA_BUILD_POPART "Build ODLA Runtime Library for POPART" ON)
if (ODLA_BUILD_POPART)
  add_subdirectory(odla_popart)
endif()

option(ODLA_BUILD_TRT "Build ODLA Runtime Library for TensorRT" ON)
if (ODLA_BUILD_TRT)
  add_subdirectory(tensorrt)
endif()

option(ODLA_BUILD_VODLA "Build VODLA Runtime Library" ON)
if (ODLA_BUILD_VODLA)
  add_subdirectory(vodla)
endif()

option(ODLA_BUILD_MAGICMIND "Build ODLA Runtime Library for Magicmind" OFF)
if (ODLA_BUILD_MAGICMIND)
  add_subdirectory(magicmind)
endif()

option(ODLA_BUILD_XNNPACK "Build ODLA Runtime Library for XNNPACK" ON)
if (ODLA_BUILD_XNNPACK)
  set(XNNPACK_ROOT "/opt/XNNPACK" CACHE STRING "Path of XNNPACK package")
  add_odla_library(odla_xnnpack SHARED odla_xnnpack.c)
  find_library(xnnpack NAMES XNNPACK PATHS ${XNNPACK_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
  find_library(clog NAMES clog PATHS ${XNNPACK_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
  find_library(cpuinfo NAMES cpuinfo PATHS ${XNNPACK_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
  find_library(pthreadpool NAMES pthreadpool PATHS ${XNNPACK_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)

  target_include_directories(odla_xnnpack PRIVATE ${XNNPACK_ROOT}/include)
  target_compile_options(odla_xnnpack PRIVATE -Wno-deprecated-declarations)
  target_link_libraries(odla_xnnpack ODLA ${xnnpack} ${cpuinfo} ${pthreadpool} ${clog} pthread)
endif()

option(ODLA_BUILD_Profiler "Build ODLA Profiler Library" ON)
if (ODLA_BUILD_Profiler)
  add_odla_library(odla_profiler SHARED odla_profiler.cc)
  target_link_libraries(odla_profiler ODLA dl)
endif()

option(ODLA_BUILD_TF_Wrapper "Build ODLA Tensorflow Wrapper" OFF)
if (ODLA_BUILD_TF_Wrapper)
  add_subdirectory(tensorflow)
endif()
