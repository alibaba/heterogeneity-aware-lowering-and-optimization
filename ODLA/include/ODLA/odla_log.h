//===- odla_log.h ---------------------------------------------------------===//
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

#ifndef _ODLA_LOG_H_
#define _ODLA_LOG_H_

#include <ODLA/odla_common.h>

/*! \file
 * \details This file defines the ODLA log related APIs.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief log level enum
typedef enum {
  ODLA_LOG_LEVEL_OFF,
  ODLA_LOG_LEVEL_FATAL,
  ODLA_LOG_LEVEL_ERROR,
  ODLA_LOG_LEVEL_WARN,
  ODLA_LOG_LEVEL_INFO,
  ODLA_LOG_LEVEL_DEBUG,
  ODLA_LOG_LEVEL_TRACE,
} odla_log_level;

//! \brief Logging function type
typedef void (*odla_logging_func)(const char* file_name, int line,
                                  const char* func_name, odla_log_level level,
                                  const char* msg);
//! \brief Set current log level
/*!
  \param level the log level used

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetLogLevel(odla_log_level level);

//! \brief Set active logger
/*!
  \param logger the function pointer to logger

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetLogger(odla_logging_func logger);

//! \brief Get active logger
/*!
  \return function pointer to current logger
*/
extern ODLA_API_EXPORT odla_logging_func ODLA_API_CALL odla_GetLogger();

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_LOG_H_
