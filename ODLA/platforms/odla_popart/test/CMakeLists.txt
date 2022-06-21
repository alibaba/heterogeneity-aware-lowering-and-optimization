# ==============================================================================
# Copyright (C) 2022 Alibaba Group Holding Limited.
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
# limitations under the License.
# =============================================================================
cmake_minimum_required(VERSION 3.14)

if(ENABLE_COVERAGE)
    message("Use gcov for code coverage test")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
endif()

function(add_odla_cpp_unit_test name)
  set(oneValueArgs "VARIANTS;LABELS;TEST_UTILS")
  cmake_parse_arguments(test_args "" "FILES" "${oneValueArgs}" "FILES;${ARGN}")
  set(FILES "${test_args_FILES}")
  set(target_name ${ARGV0})

  add_executable(${target_name} ${FILES} utils.cc)
  target_link_libraries(${target_name} PUBLIC ODLA pthread gcov custom_ops odla_popart popart-only)

  target_include_directories(${target_name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
 
endfunction()

add_odla_cpp_unit_test(unitest_odla_compute odla_compute_test.cc)
add_odla_cpp_unit_test(unitest_odla_config odla_config_test.cc)
add_odla_cpp_unit_test(unitest_odla_ops_math odla_ops_math_test.cc)
add_odla_cpp_unit_test(unitest_odla_ops odla_ops_test.cc)
add_odla_cpp_unit_test(unitest_odla_ops_nn odla_ops_nn_test.cc)
add_odla_cpp_unit_test(unitest_odla_ops_process odla_ops_process_test.cc)
