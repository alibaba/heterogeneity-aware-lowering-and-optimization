# ==============================================================================
# Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

set(DEV_NAMES)
list(APPEND DEV_NAMES
  dnnl
  tensorrt
)

file(GLOB tds "*.td")
foreach(td_path IN LISTS tds)
  string(REGEX REPLACE ".+/(.+)\\..*" "\\1" td_name ${td_path})
  if (NOT ${td_name} MATCHES "models")
    add_models(${td_name} ${DEV_NAMES})
  endif()
endforeach()


add_models_target(MODELGEN)