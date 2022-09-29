# ==============================================================================
# Copyright (C) 2022 Alibaba Group Holding Limited.
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
add_odla_library(odla_torch SHARED odla_torchscript.cc)

if (CMAKE_PREFIX_PATH)
  find_package(Torch REQUIRED)
endif()
if (TORCH_INSTALL_PREFIX)
  set(TORCH_PATH ${TORCH_INSTALL_PREFIX})
else()
  execute_process(
    COMMAND python3 -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode)
endif()
if (NOT TORCH_PATH)
  message(FATAL_ERROR "torch install is not found.")
else()
  message(STATUS "torch install path: ${TORCH_PATH}")
endif()

set(PYTORCH_INC_DIR
    ${TORCH_PATH}/include
    ${TORCH_PATH}/torch/csrc/api/include
    ${TORCH_PATH}/include/TH
    ${TORCH_PATH}/include/THC
)
message(STATUS "torch include dirs: ${PYTORCH_INC_DIR}")
target_include_directories(odla_torch PRIVATE ${PYTORCH_INC_DIR})

set(PYTORCH_LIBS
     -lc10
     -ltorch
     -ltorch_cpu
     -ltorch_python)

if (NOT TORCH_CXX_FLAGS)
    execute_process(
      COMMAND python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
      OUTPUT_VARIABLE TORCH_CXX11_ABI
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE retcode)
    message(STATUS "Torch CXX flags:-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
    target_compile_definitions(odla_torch PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
else()
    message(STATUS "Torch CXX flags:${TORCH_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()
target_link_options(odla_torch PUBLIC -Wl,-rpath,${TORCH_PATH}/lib -L${TORCH_PATH}/lib)
target_link_libraries(odla_torch PUBLIC ${PYTORCH_LIBS})
target_link_libraries(odla_torch PUBLIC ODLA)