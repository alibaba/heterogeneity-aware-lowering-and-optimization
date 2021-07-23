//===- odla_common.h ------------------------------------------------------===//
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

#ifndef _ODLA_COMMON_H_
#define _ODLA_COMMON_H_

/*! \file
 * \details This file defines the ODLA basic common types.
 */

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(_WIN32) && defined(_MSC_VER))
// Integer types
typedef signed __int8 odla_int8;      /**< 8-bit signed integer type */
typedef signed __int16 odla_int16;    /**< 16-bit signed integer type */
typedef signed __int32 odla_int32;    /**< 32-bit signed integer type */
typedef signed __int64 odla_int64;    /**< 64-bit signed integer type */
typedef unsigned __int8 odla_uint8;   /**< 8-bit unsigned integer type */
typedef unsigned __int16 odla_uint16; /**< 16-bit unsigned integer type */
typedef unsigned __int32 odla_uint32; /**< 32-bit unsigned integer type */
typedef unsigned __int64 odla_uint64; /**< 64-bit unsigned integer type */

// brief Quantized integer types
typedef signed __int8 odla_qint8;   /**< 8-bit signed quantized integer type */
typedef signed __int16 odla_qint16; /**< 16-bit signed quantized integer type */
typedef signed __int32 odla_qint32; /**< 32-bit signed quantized integer type */
typedef signed __int64 odla_qint64; /**< 64-bit signed quantized integer type */
typedef unsigned __int8
    odla_quint8; /**< 8-bit unsigned quantized integer type */
typedef unsigned __int16
    odla_quint16; /**< 16-bit unsigned quantized integer type */
typedef unsigned __int32
    odla_quint32; /**< 32-bit unsigned quantized integer type */
typedef unsigned __int64
    odla_quint64; /**< 64-bit unsigned quantized integer type */

// Floating point types
typedef unsigned __int16 odla_float16;  /**< 16-bit floating point type */
typedef unsigned __int16 odla_bfloat16; /**< 16-bit brain floating point type */
typedef float odla_float32;             /**< 32-bit floating point type */
typedef double odla_float64;            /**< 64-bit floating point type */

#else

// Integer types
typedef __INT8_TYPE__ odla_int8;     /**< 8-bit signed integer type */
typedef __INT16_TYPE__ odla_int16;   /**< 16-bit signed integer type */
typedef __INT32_TYPE__ odla_int32;   /**< 32-bit signed integer type */
typedef __INT64_TYPE__ odla_int64;   /**< 64-bit signed integer type */
typedef __UINT8_TYPE__ odla_uint8;   /**< 8-bit unsigned integer type */
typedef __UINT16_TYPE__ odla_uint16; /**< 16-bit unsigned integer type */
typedef __UINT32_TYPE__ odla_uint32; /**< 32-bit unsigned integer type */
typedef __UINT64_TYPE__ odla_uint64; /**< 64-bit unsigned integer type */

// Quantized integer types
typedef __INT8_TYPE__ odla_qint8;   /**< 8-bit signed quantized integer type */
typedef __INT16_TYPE__ odla_qint16; /**< 16-bit signed quantized integer type */
typedef __INT32_TYPE__ odla_qint32; /**< 32-bit signed quantized integer type */
typedef __INT64_TYPE__ odla_qint64; /**< 64-bit signed quantized integer type */
typedef __UINT8_TYPE__
    odla_quint8; /**< 8-bit unsigned quantized integer type */
typedef __UINT16_TYPE__
    odla_quint16; /**< 16-bit unsigned quantized integer type */
typedef __UINT32_TYPE__
    odla_quint32; /**< 32-bit unsigned quantized integer type */
typedef __UINT64_TYPE__
    odla_quint64; /**< 64-bit unsigned quantized integer type */

// Floating point types
typedef __UINT16_TYPE__ odla_float16;  /**< 16-bit floating point type */
typedef __UINT16_TYPE__ odla_bfloat16; /**< 16-bit brain floating point type */
typedef float odla_float32;            /**< 32-bit floating point type */
typedef double odla_float64;           /**< 64-bit floating point type */
#endif

#ifndef NULL
#define NULL ((void*)0)
#endif

typedef odla_uint32 odla_bool; /**< boolean type */

//! \enum Enum of element types
typedef enum {
  ODLA_INT8,
  ODLA_INT16,
  ODLA_INT32,
  ODLA_INT64,
  ODLA_UINT8,
  ODLA_UINT16,
  ODLA_UINT32,
  ODLA_UINT64,

  ODLA_QINT8,
  ODLA_QINT16,
  ODLA_QINT32,
  ODLA_QINT64,
  ODLA_QUINT8,
  ODLA_QUINT16,
  ODLA_QUINT32,
  ODLA_QUINT64,

  ODLA_FLOAT16,
  ODLA_BFLOAT16,
  ODLA_FLOAT32,
  ODLA_FLOAT64,

  ODLA_BOOL,
} odla_element_type;

//! \brief char
typedef char odla_char;

//! \brief size_t
typedef __SIZE_TYPE__ odla_size_t;

//! \brief void
typedef void odla_void;

//! \brief Return status
typedef enum {
  ODLA_SUCCESS,
  //! \brief dlopen a shared library error
  ODLA_DL_ERROR,
  ODLA_FILE_NOT_EXIST,
  //! \brief illegal input argument, such as nullptr
  ODLA_INVALID_PARAM,
  //! \brief allocate/deallocate memory error, out of memory error
  ODLA_MEM_ERROR,
  ODLA_UNSUPPORTED_DATATYPE,
  ODLA_UNSUPPORTED_DEVICE_TYPE,
  //! \brief odla op is not implemented yet
  ODLA_UNSUPPORTED_OP,
  //! \brief process timeout
  ODLA_TIMEOUT,
  //! \brief internal error
  INTERNAL_LOGIC_ERR,
  //! auto recoverable error
  RECOVERABLE_ERR,
  //! manual recoverable error, include partition reset and full reset type
  PARTITION_RESET,
  FULL_RESET,
  //! unrecoverable error
  UNRECOVERABLE_ERR,
  ODLA_FAILURE,
} odla_status;

//! \brief API export directives
#if defined(_WIN32)
#define ODLA_API_EXPORT __declspec(dllexport)
#define ODLA_API_CALL __stdcall
#else
#define ODLA_API_EXPORT
#define ODLA_API_CALL
#endif

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_COMMON_H_
