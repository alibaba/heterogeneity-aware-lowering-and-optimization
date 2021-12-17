//===- odla_task.h --------------------------------------------------------===//
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

#ifndef _ODLA_TASK_H_
#define _ODLA_TASK_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_device.h>
#include <ODLA/odla_value.h>

/*! \file
 * \details This file defines the ODLA task related APIs.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief odla task definition
typedef struct {
  void (*func)(odla_device, odla_values inputs, odla_values outputs);
  odla_values inputs;
  odla_values outputs;
} odla_task;

//! \brief odla task async executor
typedef odla_status (*odla_async_executor)(odla_status (*)(odla_device,
                                                           odla_task),
                                           odla_device, odla_task);

//! \biref set a customized async executor
odla_status odla_SetAsyncExecutor(odla_async_executor executor);

//! \brief Start a ODLA task in a synchronized manner.
/*!
  \param device The device to run the task
  \param task The task object
  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_RunTask(odla_device device, odla_task task);

//! \brief Start a ODLA task in a asynchronized manner.
/*!
  \param device The device to run the task
  \param task The task object
  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_RunTaskAsync(odla_device device, odla_task task);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_COMMON_H_