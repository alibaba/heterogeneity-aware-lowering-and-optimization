# ==============================================================================
# Copyright (C) 2022-2022 Alibaba Group Holding Limited.
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
if(DEFINED $ENV{ASCEND_HOME})
    set(ASCEND_PATH $ENV{ASCEND_HOME})
elseif(EXISTS "/usr/local/Ascend")
    set(ASCEND_PATH "/usr/local/Ascend")
else()
    set(ASCEND_PATH "~/Ascend")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(
    ${ASCEND_PATH}/ascend-toolkit/latest/opp/op_proto/built-in/inc
    ${ASCEND_PATH}/ascend-toolkit/latest/fwkacllib/include/graph
    ${ASCEND_PATH}/ascend-toolkit/latest/fwkacllib/include/ge
    ${ASCEND_PATH}/ascend-toolkit/latest/fwkacllib/include/
    ${ASCEND_PATH}/ascend-toolkit/latest/fwkacllib/include/parser
    inc
)

link_directories(
    ${ASCEND_PATH}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling/
    ${ASCEND_PATH}/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/
    ${ASCEND_PATH}/ascend-toolkit/latest/fwkacllib/lib64/stub/
)

add_odla_library(odla_ascend SHARED src/model_process.cc src/odla_ascend_acl.cc src/odla_ops.cc)

target_link_libraries(odla_ascend ODLA ascendcl fe optiling graph ge_runner)
