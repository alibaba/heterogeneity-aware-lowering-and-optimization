//===- odla_version.h -----------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
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

#ifndef _ODLA_VERSION_H_
#define _ODLA_VERSION_H_

/*! \file
 * \details This file defines the ODLA version number.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define ODLA_MAJOR 0 // !< ODLA major version.
#define ODLA_MINOR 5 // !< ODLA minor version.
#define ODLA_PATCH 0 // !< ODLA patch version.

//! \brief ODLA version number.
#define ODLA_VERSION_NUMBER ((ODLA_MAJOR)*100 + (ODLA_MINOR)*10 + (OLDA_PATCH))

//! \brief Get version info of runtime library.
/*!
  \return NULL-terminated string of version info.
*/
const char* odla_GetVersionString();

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_VERSION_H_