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

set(HALO_TEST_DEPENDS
  FileCheck
  RTLIB
  halolib
  halo
  analyzer
  diagnostic
)

if (ODLA_BUILD_TRT)
  list(APPEND HALO_TEST_DEPENDS odla_tensorrt)
endif()

if (ODLA_BUILD_DNNL)
  list(APPEND HALO_TEST_DEPENDS odla_dnnl)
endif()

if (ODLA_BUILD_EIGEN)
  list(APPEND HALO_TEST_DEPENDS odla_eigen)
endif()

if (ODLA_BUILD_XNNPACK)
  list(APPEND HALO_TEST_DEPENDS odla_xnnpack)
endif()


configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
)

add_lit_testsuite(check-halo "Running Halo regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${HALO_TEST_DEPENDS}
)
