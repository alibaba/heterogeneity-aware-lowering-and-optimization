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

#include <stddef.h>

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

#include <stdint.h>
// Integer types
typedef int8_t odla_int8;     /**< 8-bit signed integer type */
typedef int16_t odla_int16;   /**< 16-bit signed integer type */
typedef int32_t odla_int32;   /**< 32-bit signed integer type */
typedef int64_t odla_int64;   /**< 64-bit signed integer type */
typedef uint8_t odla_uint8;   /**< 8-bit unsigned integer type */
typedef uint16_t odla_uint16; /**< 16-bit unsigned integer type */
typedef uint32_t odla_uint32; /**< 32-bit unsigned integer type */
typedef uint64_t odla_uint64; /**< 64-bit unsigned integer type */

// Quantized integer types
typedef int8_t odla_qint8;     /**< 8-bit signed quantized integer type */
typedef int16_t odla_qint16;   /**< 16-bit signed quantized integer type */
typedef int32_t odla_qint32;   /**< 32-bit signed quantized integer type */
typedef int64_t odla_qint64;   /**< 64-bit signed quantized integer type */
typedef uint8_t odla_quint8;   /**< 8-bit unsigned quantized integer type */
typedef uint16_t odla_quint16; /**< 16-bit unsigned quantized integer type */
typedef uint32_t odla_quint32; /**< 32-bit unsigned quantized integer type */
typedef uint64_t odla_quint64; /**< 64-bit unsigned quantized integer type */

// Floating point types
typedef uint16_t odla_float16;  /**< 16-bit floating point type */
typedef uint16_t odla_bfloat16; /**< 16-bit brain floating point type */
typedef float odla_float32;     /**< 32-bit floating point type */
typedef double odla_float64;    /**< 64-bit floating point type */
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
typedef size_t odla_size_t;

//! \brief void
typedef void odla_void;

//! \brief Return status
typedef enum {
  ODLA_SUCCESS,
  ODLA_FAILURE,
} odla_status;

//! \brief BF16 mode
typedef enum {
  BF16_DISABLE,
  BF16_ACCURACY_MODE,
  BF16_PERFORMACE_MODE,
  BF16_AUTO_MODE,
} odla_bf16_mode;

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