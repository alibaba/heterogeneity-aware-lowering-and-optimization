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

option(HALO_USE_GLOG "Halo enable glog" ON)

if(HALO_USE_GLOG)
  # Check if Glog is installed.
  find_package(glog REQUIRED)
  if(glog_FOUND)
    include_directories(BEFORE ${GLOG_INCLUDE_DIRS}) 
  else()
    message(FATAL_ERROR "Glog is not found.")
  endif()
endif()