//===- odla_impl_common.h -------------------------------------------------===//
//
// Copyright (C) 2019-2022 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef _ODLA_PLATFORMS_ODLA_COMMON_H_
#define _ODLA_PLATFORMS_ODLA_COMMON_H_

#include "version.inc"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define VERSION_STR(major, minor, patch, build) \
  STR(major) "." STR(minor) "." STR(patch) "." STR(build)

#ifndef NDEBUG
#define ODLA_BUILD_TYPE "DEBUG"
#else
#define ODLA_BUILD_TYPE "RELEASE"
#endif

#define ODLA_VERSION_COMMON_INFO                                        \
  "Repo: " ODLA_REPOSITORY ", Rev:" ODLA_REVISION "\nOS:" HALO_BUILT_OS \
  "\nBuild Type: " ODLA_BUILD_TYPE "\nBuilt on: " __DATE__ " " __TIME__

#define ODLA_VERSION_STR(lib_name, major, minor, patch, build, extra_info) \
  "ODLA library name: " lib_name "\nVersion: " VERSION_STR(                \
      major, minor, patch, build) "\n" ODLA_VERSION_COMMON_INFO            \
                                  "\n" extra_info
#endif
