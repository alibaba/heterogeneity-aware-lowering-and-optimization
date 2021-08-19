//===- version.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "halo/version.h"

// Embeds version info be into binary, which can be extract via `strings`.
namespace halo {
const char* HaloVersionInfo = "HALO Version: " HALO_VERSION_STR;

#ifdef HALO_REVISION
const char* HaloRevInfo = "HALO Repo: " HALO_REPOSITORY " Rev:" HALO_REVISION;
#else
const char* HaloRevInfo = "HALO_Repo: NA";
#endif

#ifdef HALO_REVISION
const char* ODLARevInfo = "ODLA Repo: " ODLA_REPOSITORY " Rev:" ODLA_REVISION;
#else
const char* ODLARevInfo = "ODLA_Repo: NA";
#endif

#ifndef NDEBUG
#define HALO_BUILD_TYPE "(DEBUG)" // NOLINT
#else
#define HALO_BUILD_TYPE "(RELEASE)" // NOLINT
#endif

#ifdef HALO_BUILT_OS
const char* HaloBuiltInfo =
    "HALO Build info: " HALO_BUILT_OS " " HALO_BUILD_TYPE;
#else
const char* HaloBuiltInfo = "HALO Built info: Unknown " HALO_BUILD_TYPE;
#endif

} // namespace halo