//===- odla_memory.h ------------------------------------------------------===//
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

#ifndef _ODLA_MEMORY_H_
#define _ODLA_MEMORY_H_

#include <ODLA/odla_common.h>

/*! \file
 * \details This file defines the ODLA memory related APIs.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief memory copy types
typedef enum {
  ODLA_MEMCPY_H2H,
  ODLA_MEMCPY_H2D,
  ODLA_MEMCPY_D2H,
  ODLA_MEMCPY_D2D,
} odla_memcpy_type;

//! \brief Allocate device memory
/*!
  \param devPtr the pointer to allocated device memory
  \param size the requested allocation size in bytes

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_AllocateDeviceMemory(odla_void** dev_ptr, odla_size_t size);

//! \brief Free device memory
/*!
  \param devPtr the device pointer to memory to free

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_FreeDeviceMemory(odla_void* ptr);

//! \brief Allocate host memory
/*!
  \param ptr the pointer to allocated host memory
  \param size the requested allocation size in bytes

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_AllocateHostMemory(odla_void** host_ptr, odla_size_t size);

//! \brief Free host memory
/*!
  \param ptr the host pointer to memory to free

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_FreeHostMemory(odla_void* ptr);

//! \brief Copy data between host and device.
/*!
  \param dst the destination memory address
  \param src the source memory address
  \param size the size in bytes to copy
  \param type the memory copy type

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_CopyMemory(
    odla_void* dst, odla_void* src, odla_size_t size, odla_memcpy_type type);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_MEMORY_H_
