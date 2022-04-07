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

# Install halo

install(TARGETS halo halolib analyzer
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        INCLUDES DESTINATION include/halo
        PUBLIC_HEADER DESTINATION include/halo
)

find_program(PYTHON "python3")
set(SETUP_PY "${CMAKE_BINARY_DIR}/python/setup.py")
set(OUTPUT_PY "${CMAKE_BINARY_DIR}/python/dist")
set(ENV{PYTHONPATH} ${OUTPUT_PY})
install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install -O2 --root=${OUTPUT_PY})")
install(DIRECTORY ${OUTPUT_PY}/usr/local/lib/ DESTINATION lib)

if (HALO_BUILD_RTLIB)
install(DIRECTORY ${CMAKE_BINARY_DIR}/runtime/lib/ DESTINATION lib)
endif()
install(CODE "execute_process(COMMAND ${CMAKE_SOURCE_DIR}/demo/install.sh ${CMAKE_INSTALL_PREFIX})")
install(FILES ${CMAKE_BINARY_DIR}/docs/HaloIR.md DESTINATION docs OPTIONAL)
install(FILES ${CMAKE_SOURCE_DIR}/README.md DESTINATION docs OPTIONAL)
install(FILES ${CMAKE_SOURCE_DIR}/CHANGELOG.md DESTINATION docs OPTIONAL)
install(DIRECTORY ${CMAKE_BINARY_DIR}/docs/html DESTINATION docs OPTIONAL)
install(DIRECTORY ${CMAKE_BINARY_DIR}/odla_docs/html
        DESTINATION docs/odla OPTIONAL)
install(FILES ${CMAKE_BINARY_DIR}/odla_docs/latex/refman.pdf
        DESTINATION docs/odla RENAME ODLA_reference.pdf OPTIONAL)

set(CPACK_GENERATOR "TBZ2")
string(TOLOWER "${CPACK_SYSTEM_NAME}" os)
if (os MATCHES "centos.*")
  set(CPACK_GENERATOR "TBZ2;RPM")
endif()
set(CPACK_POST_BUILD_SCRIPTS "execute_process(COMMAND ${PYTHON} ${SETUP_PY} bdist_wheel)")

include(CPack)
