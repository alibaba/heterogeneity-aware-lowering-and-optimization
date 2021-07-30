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

add_subdirectory(lib)

# Identify OS name.
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  file(READ "/etc/os-release" os_content)
  string(REGEX MATCH "PRETTY_NAME=\"(.*)\".VERSION_ID" OS_NAME "${os_content}")
  set(OS_NAME "${CMAKE_MATCH_1}")
endif()

# Generate VCS info.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/version.inc")

find_first_existing_vc_file("${CMAKE_SOURCE_DIR}" halo_vc)

if (halo_vc)
  set(LLVM_SRC_DIR ${CMAKE_SOURCE_DIR}/external/llvm-project/llvm)
  set(LLVM_CMAKE_PATH ${LLVM_SRC_DIR}/cmake/modules)
  set(get_version_script "${LLVM_CMAKE_PATH}/GenerateVersionFromVCS.cmake")

  add_custom_command(OUTPUT "${version_inc}"
    DEPENDS "${halo_vc}" "${get_version_script}"
    COMMAND
    ${CMAKE_COMMAND} -DHALO_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                     -DODLA_SOURCE_DIR=${CMAKE_SOURCE_DIR}/ODLA
                     -DNAMES=HALO\;ODLA
                     -DHEADER_FILE=${version_inc}
                     -P ${get_version_script}
    COMMAND echo "#define HALO_VERSION_MAJOR ${PROJECT_VERSION_MAJOR}" >> ${version_inc}
    COMMAND echo "#define HALO_VERSION_MINOR ${PROJECT_VERSION_MINOR}" >> ${version_inc}
    COMMAND echo "#define HALO_VERSION_PATCH ${PROJECT_VERSION_PATCH}" >> ${version_inc}
    COMMAND echo "#define HALO_VERSION_STR \"${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}\"" >> ${version_inc}
    COMMAND echo "#define HALO_BUILT_OS \"${OS_NAME}\"" >> ${version_inc}
    VERBATIM
  )
  set_source_files_properties("${version_inc}"
    PROPERTIES GENERATED TRUE
    HEADER_FILE_ONLY TRUE)
else()
  message(WARNING "revision info unavailable")
  FILE(APPEND ${version_inc} "#define HALO_VERSION_MAJOR ${PROJECT_VERSION_MAJOR}")
  FILE(APPEND ${version_inc} "#define HALO_VERSION_MINOR ${PROJECT_VERSION_MINOR}")
  FILE(APPEND ${version_inc} "#define HALO_VERSION_MINOR ${PROJECT_VERSION_PATCH}")
  FILE(APPEND ${version_inc} "#define HALO_BUILT_OS \"${OS_NAME}\"")
endif()

add_custom_target(halo_version_h DEPENDS "${version_inc}")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/version.h DESTINATION include/halo)
install(FILES ${version_inc} DESTINATION include/halo)
