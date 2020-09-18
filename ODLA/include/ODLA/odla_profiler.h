//===- odla_profiler.h ----------------------------------------------------===//
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

#ifndef _ODLA_PROFILER_H_
#define _ODLA_PROFILER_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_device.h>

/*! \file
 * \details This file defines the ODLA profiler related APIs.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Device trace object
typedef struct _odla_device_trace* odla_device_trace;

//! \brief Device trace item object
typedef struct _odla_device_trace_item* odla_device_trace_item;

//! \brief Create a device_trace object
/*!
  \param device_trace the pointer to the created device_trace object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_CreateDeviceTrace(odla_device_trace* device_trace);

//! \brief Set the device trace with a property item
/*!
  \param device_trace the device trace object
  \param device_trace_item the item
  \param variadic

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetDeviceTraceItem(odla_device_trace device_trace,
                        odla_device_trace_item device_trace_item, ...);

//! \brief Release a created device_trace
/*!
  \param context the device_trace object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_ReleaseDeviceTrace(odla_device_trace device_trace);

//! \brief Start the profiler tracing on a device
/*!
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_StartDeviceProfiler(odla_device device);

//! \brief Asynchronously start the profiler tracing on a device
/*!
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_AsyncStartDeviceProfiler(odla_device device);

//! \brief Stop the profiler tracing on a device
/*!
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_StopDeviceProfiler(odla_device device);

//! \brief Asynchronously stop the profiler tracing on a device
/*!
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_AsyncStopDeviceProfiler(odla_device device);

//! \brief Retrieve the profiling trace from a device
/*!
  \param device the device object
  \param device_trace the pointer to the retrieved trace object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_RetrieveDeviceTrace(odla_device device, odla_device_trace* device_trace);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_PROFILER_H_