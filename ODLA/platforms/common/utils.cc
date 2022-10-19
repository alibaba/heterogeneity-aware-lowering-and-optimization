//===- utils.cc -----------------------------------------------------------===//
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

#include <ODLA/odla_log.h>

#include <iostream>

static odla_log_level gODLALogLevel = ODLA_LOG_LEVEL_ERROR; // NOLINT
static inline void ODLAStdErrLogger(const char* file_name, int line,
                                    const char* func_name, odla_log_level level,
                                    const char* msg) {
  if (level > gODLALogLevel) {
    return;
  }
  static const char level_shorts[] = {'-', 'F', 'E', 'W', 'I', 'D', 'T'};
  if (file_name != nullptr && line > 0) {
    std::cerr << file_name << ":" << line << " ";
  }
  std::cerr << level_shorts[level] << " " << msg << "\n";
}

static odla_logging_func gODLALogger = ODLAStdErrLogger; // NOLINT

odla_status odla_SetLogLevel(odla_log_level level) { gODLALogLevel = level; }
odla_logging_func odla_GetLogger() { return gODLALogger; }

odla_status odla_SetLogger(odla_logging_func logger) {
  gODLALogger = logger;
  return ODLA_SUCCESS;
}
