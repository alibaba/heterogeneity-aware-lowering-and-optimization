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

option(HALO_BUILD_RTLIB "Build RTLib" ON)

set(CORE_COMPONENTS)
set(CORE_COMPONENTS_OBJECTS)

add_subdirectory(compiler)
add_subdirectory(distributed)
add_subdirectory(executor)
add_subdirectory(framework)
add_subdirectory(interface)
add_subdirectory(ir)
add_subdirectory(mm)
add_subdirectory(parser)
add_subdirectory(pass)
add_subdirectory(quantizer)
if (HALO_BUILD_RTLIB)
  add_subdirectory(runtime)
endif()
add_subdirectory(serializer)
add_subdirectory(target)
add_subdirectory(threadpool)
add_subdirectory(transforms)
add_subdirectory(utils)

# ========= Build the Halo core library =========
add_library(halolib SHARED)
target_link_options(halolib PUBLIC -Wl,-no-undefined)
set_target_properties(halolib PROPERTIES OUTPUT_NAME halo)
set_target_properties(halolib PROPERTIES PUBLIC_HEADER include/halo/halo.h)
set_target_properties(halolib PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
)

set(GLOG_COMPONENT)
if(HALO_USE_GLOG)
  set(GLOG_COMPONENT glog)
endif()
target_link_libraries(halolib PUBLIC ${CORE_COMPONENTS} ${GLOG_COMPONENT})
add_dependencies(halolib halo_version_h)
