# ==============================================================================
# Copyright (C) 2019-2022 Alibaba Group Holding Limited.
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

add_odla_library(odla_tensorflow SHARED odla_tf.cc)

if (NOT TENSORFLOW_INC_DIR)
  execute_process(
      COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())"
      OUTPUT_VARIABLE TENSORFLOW_INC_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE retcode)
endif()
if (IS_DIRECTORY "${TENSORFLOW_INC_DIR}")
  message(STATUS "Tensorflow include directory: ${TENSORFLOW_INC_DIR}")
  target_include_directories(odla_tensorflow PRIVATE ${TENSORFLOW_INC_DIR})
else()
  message(FATAL_ERROR "Tensorflow not found")
endif()

if (NOT TENSORFLOW_LIB_DIR)
  execute_process(
      COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_lib())"
      OUTPUT_VARIABLE TENSORFLOW_LIB_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE retcode)
endif()

if (IS_DIRECTORY "${TENSORFLOW_LIB_DIR}")
  message(STATUS "Tensorflow library directory: ${TENSORFLOW_LIB_DIR}")
else()
  message(FATAL_ERROR "Tensorflow not found")
endif()

if (NOT TENSORFLOW_CXX11_ABI)
  set(TENSORFLOW_CXX11_ABI 1)
  execute_process(
      COMMAND python3 -c "import tensorflow; print(tensorflow.__cxx11_abi_flag__)"
      OUTPUT_VARIABLE TENSORFLOW_CXX11_ABI
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE retcode)
endif()
message(STATUS "Tensorflow CXX11 ABI: ${TENSORFLOW_CXX11_ABI}")
if (NOT ${TENSORFLOW_CXX11_ABI})
  target_compile_definitions(odla_tensorflow PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

find_library(tf_cc_lib
    NAMES tensorflow_cc
    PATHS "${TENSORFLOW_LIB_DIR}"
    NO_DEFAULT_PATH
)

if(tf_cc_lib)
    message(STATUS "tensorflow_cc : ${tf_cc_lib}")
    target_link_libraries(odla_tensorflow ${tf_cc_lib})
else()
      message(FATAL_ERROR "tensorflow_cc library not found")
endif()

find_library(tf_framework_lib
    NAMES tensorflow_framework
    PATHS "${TENSORFLOW_LIB_DIR}"
    NO_DEFAULT_PATH
)

if(tf_framework_lib)
    message(STATUS "tensorflow_framework: ${tf_framework_lib}")
    target_link_libraries(odla_tensorflow ${tf_framework_lib})
  else()
    message(FATAL_ERROR "tensorflow_framework not found")
  endif()

target_link_libraries(odla_tensorflow ODLA)
