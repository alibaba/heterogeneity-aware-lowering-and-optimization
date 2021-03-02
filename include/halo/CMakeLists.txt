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
    ${CMAKE_COMMAND} "-DHALO_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
                     "-DODLA_SOURCE_DIR=${CMAKE_SOURCE_DIR}/ODLA"
                     "-DNAMES=\"HALO;ODLA\""
                     "-DHEADER_FILE=${version_inc}"
                     -P "${get_version_script}")
  set_source_files_properties("${version_inc}"
    PROPERTIES GENERATED TRUE
    HEADER_FILE_ONLY TRUE)
else()
  message(WARNING "revision info unavailable")
  file(WRITE "${version_inc}" "")
endif()

add_custom_target(halo_version_h DEPENDS "${version_inc}")
