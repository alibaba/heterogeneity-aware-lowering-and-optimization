cmake_minimum_required(VERSION 3.14)

if(ENABLE_COVERAGE)
    message("Use gcov for code coverage test")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
endif()

function(add_odla_cpp_unit_test name)
  set(oneValueArgs "VARIANTS;LABELS;TEST_UTILS")
  cmake_parse_arguments(test_args "" "FILES" "${oneValueArgs}" "FILES;${ARGN}")
  set(FILES "${test_args_FILES}")

  message(${FILES})
  set(target_name ${ARGV0})
  message(${target_name})

  add_executable(${target_name} ${FILES} utils.cc)
  target_link_libraries(${target_name} PUBLIC ODLA pthread gcov custom_ops odla_popart popart-only)

  target_include_directories(${target_name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
 
endfunction()

add_odla_cpp_unit_test(unittes_odla_compute odla_compute_test.cc)
add_odla_cpp_unit_test(unittes_odla_config odla_config_test.cc)
add_odla_cpp_unit_test(unittes_odla_ops_math odla_ops_math_test.cc)
