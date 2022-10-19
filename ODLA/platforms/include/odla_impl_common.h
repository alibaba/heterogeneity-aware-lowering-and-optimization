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

#include <ODLA/odla_log.h>

#include <string>

#include "ODLA/odla_common.h"
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

#ifdef __PRETTY_FUNCTION__
#define FUNC_NAME __PRETTY_FUNCTION__
#else
#define FUNC_NAME __FUNCTION__
#endif

static void ODLALog(const char* file, int line, const char* func,
                    odla_log_level level, const char* msg) {
  odla_GetLogger()(file, line, func, level, msg);
}

static void ODLALog(const char* file, int line, const char* func,
                    odla_log_level level, const std::string& msg) {
  ODLALog(file, line, func, level, msg.c_str());
}

#define ODLA_LOG_TRACE(msg) \
  ODLALog(__FILE__, __LINE__, FUNC_NAME, ODLA_LOG_LEVEL_TRACE, msg)

#define ODLA_LOG_DEBUG(msg) \
  ODLALog(__FILE__, __LINE__, FUNC_NAME, ODLA_LOG_LEVEL_DEBUG, msg)

#define ODLA_LOG_WARN(msg) \
  ODLALog(__FILE__, __LINE__, FUNC_NAME, ODLA_LOG_LEVEL_WARN, msg)

#define ODLA_LOG_ERROR(msg) \
  ODLALog(__FILE__, __LINE__, FUNC_NAME, ODLA_LOG_LEVEL_ERROR, msg)

#define ODLA_LOG_FATAL(msg) \
  ODLALog(__FILE__, __LINE__, FUNC_NAME, ODLA_LOG_LEVEL_FATAL, msg)

#endif //_ODLA_PLATFORMS_ODLA_LOG_COMMON_H_
