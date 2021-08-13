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

# TARGET_NAME: Target Name
# TARGET_SRCS: Source files
# TARGET_DEPENDENCES: Dependences which need to be built before
macro(create_halo_object)
  set(ONE_VALUE_ARGS TARGET_NAME)
  set(MULTI_VALUE_ARGS TARGET_SRCS TARGET_DEPENDENCES)
  cmake_parse_arguments("" "" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})
  add_library(${_TARGET_NAME} OBJECT ${_TARGET_SRCS})

  if(NOT DEFINED _TARGET_NAME)
    message(FATAL_ERROR "No TARGET_NAME defined for create_halo_object")
  endif()

  if(DEFINED _TARGET_DEPENDENCES)
    add_dependencies(${_TARGET_NAME} ${_TARGET_DEPENDENCES})
  endif()

  set(CORE_COMPONENTS
    ${CORE_COMPONENTS} ${_TARGET_NAME}
    PARENT_SCOPE
  )
  set(CORE_COMPONENTS_OBJECTS
    ${CORE_COMPONENTS_OBJECTS} $<TARGET_OBJECTS:${_TARGET_NAME}>
    PARENT_SCOPE
  )

  if(HALO_USE_TIDY_CHECK)
    set_target_properties(${_TARGET_NAME} PROPERTIES
      CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
  endif()
  target_compile_options(${_TARGET_NAME} PUBLIC ${HALO_COMPILE_FLAGS})
endmacro(create_halo_object)

# Use this macro instead of add_subdirectory() if the current CMakeLists.txt
# doesn't create it's own halo object. Otherwise, it won't propage halo
# internal variables to it's parent scope.
macro(add_halo_subdirectory)
  add_subdirectory(${ARGV})
  set(CORE_COMPONENTS ${CORE_COMPONENTS} PARENT_SCOPE)
  set(CORE_COMPONENTS_OBJECTS ${CORE_COMPONENTS_OBJECTS} PARENT_SCOPE)
endmacro(add_halo_subdirectory)


# NORTTI: Disable RTTI
# NOEXCEPTIONS: Disable exception support.
# WERROR: Enable all warnings and treat them as error.
set(HALO_NO_RTTI)
function(set_halo_compile_flags)
  set(options NORTTI NOEXCEPTIONS WERROR)
  cmake_parse_arguments("" "${options}" "" "" ${ARGN})
  set(HALO_NO_RTTI ${_NORTTI} PARENT_SCOPE)
  if(MSVC)
    if(_NORTTI)
      list(APPEND HALO_COMPILE_FLAGS "/GR-")
    else()
      list(APPEND HALO_COMPILE_FLAGS "/GR-")
    endif()
    if(_NOEXCEPTIONS)
       list(APPEND HALO_COMPILE_FLAGS "/EHs-c-")
    endif()
    if(_WERROR)
      list(APPEND HALO_COMPILE_FLAGS "/Wall /WX")
    endif()
  else()
    if(_NORTTI)
      list(APPEND HALO_COMPILE_FLAGS -fno-rtti)
    endif()
    if(_NOEXCEPTIONS)
      list(APPEND HALO_COMPILE_FLAGS -fno-exceptions)
    endif()
    if(_WERROR)
      list(APPEND HALO_COMPILE_FLAGS -pedantic -Wall -Werror)
    endif()
  endif()
  set(HALO_COMPILE_FLAGS ${HALO_COMPILE_FLAGS} PARENT_SCOPE)
endfunction(set_halo_compile_flags)

# Compile flags for halo objects
set(HALO_COMPILE_FLAGS)

# Set the HALO_COMPILE_flags
set_halo_compile_flags(HALO_COMPILE_FLAGS NORTTI NOEXCEPTIONS WERROR)