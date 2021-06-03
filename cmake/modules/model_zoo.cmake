# ==============================================================================
# Copyright (C) 2019-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0(the "License");
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

#set(MODEL_TBLGEN ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/halo_tblgen)
set(INCFILE ${CMAKE_SOURCE_DIR}/models/benchmarks)

macro(gen_modelfile)
  add_custom_command(OUTPUT ${ARGV2}
    # to generate some files for model zoo testing
    # (test & configuration files)
    COMMAND ${HALO_TABLEGEN_EXE} ${ARGV0} -I ${INCFILE} ${ARGV1} -o ${ARGV2}
    DEPENDS ${ARGV1} ${HALO_TABLEGEN_EXE}
    COMMENT "Building ${ARGV2}..."
    )
    set(MODELGEN_OUTPUT ${MODELGEN_OUTPUT} ${ARGV2})
    set_source_files_properties(${ARGV2} 
      PROPERTIES GENERATED 1)
endmacro()

macro(add_models model_name)
  set(TDFILE ${CMAKE_SOURCE_DIR}/models/benchmarks/${model_name}.td)
  set(MODELFILE ${CMAKE_SOURCE_DIR}/models/benchmarks/${model_name}.inc)
  set(DEVICES "${ARGN}")
  gen_modelfile(-gen-test-model ${TDFILE} ${MODELFILE})
  foreach(device IN LISTS DEVICES)
    set(DEV_NAME ${CMAKE_SOURCE_DIR}/tests/benchmarks/${device}/${model_name}/test_${model_name}.cc)
    gen_modelfile(-gen-config-model ${TDFILE} ${DEV_NAME})
  endforeach()
  set(MODELGEN_OUTPUT ${MODELGEN_OUTPUT}
    ${MODELFILE} PARENT_SCOPE
  )
endmacro()

macro(add_models_target target)
  if( MODELGEN_OUTPUT )
    add_custom_target(${target} DEPENDS ${MODELGEN_OUTPUT})
  endif()
endmacro()