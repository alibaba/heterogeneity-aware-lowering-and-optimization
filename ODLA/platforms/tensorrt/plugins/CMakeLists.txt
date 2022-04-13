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

add_custom_target(ODLA_TRTPLUGIN)

if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS
      53
      61
      70
      75
      80
      86
    )
  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()

#Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
  set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

set(TRT_BIN_DIR ${CMAKE_BINARY_DIR}/lib)
set(TRT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../external/tensorrt)
set(TARGET_NAME ${PLUGIN_NAME})
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_obj)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMMON_UTILS_DIR ${TRT_SRC_DIR}/plugin/common)
set(CUB_ROOT_DIR ${TRT_SRC_DIR}/third_party/cub)

set(CUDA_LIBRARIES ${CUDART_LIB})
find_library(CUDNN_LIB NAMES cudnn HINTS ${CUDA_ROOT} PATH_SUFFIXES lib)
find_library(CUBLAS_LIB cublas HINTS ${CUDA_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)

#if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#endif()

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
    batchedNMSPlugin
    oneHotPlugin
)

include_directories(${COMMON_UTILS_DIR} ${COMMON_UTILS_DIR}/kernels)# ../samples/common)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

# Add common
add_subdirectory(${COMMON_UTILS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/common)
# Remove
list(FILTER PLUGIN_SOURCES EXCLUDE REGEX "checkMacrosPlugin.cpp")
list(FILTER PLUGIN_SOURCES EXCLUDE REGEX "reducedMathPlugin.cpp")
list(FILTER PLUGIN_SOURCES EXCLUDE REGEX "cudaDriverWrapper.cpp")

# Set gencodes
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS ${GENCODES})
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/initPlugin.cc")
#list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../samples/common/logger.cpp")

if (CUDA_VERSION VERSION_GREATER "11.0")
  set(CUB_ROOT_DIR  ${CUDA_ROOT}/include)
 endif()
################################## SHARED LIBRARY #######################################

add_library(${SHARED_TARGET} SHARED
    ${PLUGIN_SOURCES}
)

target_include_directories(${SHARED_TARGET}
    PUBLIC ${CUB_ROOT_DIR}
    PRIVATE ${COMMON_UTILS_DIR}
    PUBLIC ${CUDA_ROOT}/include
    PRIVATE ${TARGET_DIR}
)

set_target_properties(${SHARED_TARGET} PROPERTIES
    CXX_STANDARD "14"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
)
set_target_properties(${SHARED_TARGET} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
#set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined")
set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL -Wl,--no-undefined")
#set_target_properties(${SHARED_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_property(TARGET ${SHARED_TARGET} PROPERTY CUDA_STANDARD 14)

target_link_libraries(${SHARED_TARGET}
    ${CUBLAS_LIB}
    ${CUDART_LIB}
    ${CUDNN_LIB}
    nvinfer
    nvinfer_plugin
)

################################## STATIC LIBRARY #######################################
add_library(${STATIC_TARGET} OBJECT
    ${PLUGIN_SOURCES}
)

target_include_directories(${STATIC_TARGET}
    PUBLIC ${CUB_ROOT_DIR}
    PRIVATE ${COMMON_UTILS_DIR}
    PUBLIC ${CUDA_ROOT}/include
    PRIVATE ${TARGET_DIR}
)

set_target_properties(${STATIC_TARGET} PROPERTIES
    CXX_STANDARD "14"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TRT_BIN_DIR}"
)
set_target_properties(${STATIC_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${STATIC_TARGET} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")

set_target_properties(${STATIC_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

#set_target_properties(${STATIC_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

#set_target_properties(${STATIC_TARGET} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${STATIC_TARGET} PROPERTY CUDA_STANDARD 14)

#########################################################################################

add_dependencies(ODLA_TRTPLUGIN ${SHARED_TARGET} ${STATIC_TARGET})

################################### INSTALLATION ########################################

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
