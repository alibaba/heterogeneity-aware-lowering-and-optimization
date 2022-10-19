//===- odla_value.h -------------------------------------------------------===//
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

#ifndef _ODLA_VALUE_H_
#define _ODLA_VALUE_H_

#include <ODLA/odla_common.h>

/*! \file
 * \details This file defines the ODLA value related APIs.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Supported maximum dimension size
#define ODLA_MAX_DIMENSION 10

//! \brief Supported maximum output size
#define ODLA_MAX_OUTPUTS 64

typedef struct {
  odla_element_type data_type;
  union {
    odla_int32 val_int32;
    odla_uint32 val_uint32;
    odla_int64 val_int64;
    odla_uint64 val_uint64;
    odla_float32 val_fp32;
    odla_float64 val_fp64;
    odla_string val_str;
  };
} odla_scalar_value;

//! \brief Shape of value
typedef struct {
  // size = -1: undefined size
  odla_int32 size;
  // dims[i] = -1: undefined dimension
  odla_int64 dims[ODLA_MAX_DIMENSION];
} odla_value_shape;

//! \brief Type of value
typedef struct {
  odla_element_type element_type;
  odla_value_shape shape;
} odla_value_type;

//! \brief Value definition
typedef struct _odla_value* odla_value;

//! \brief Multiple values
typedef struct {
  odla_size_t size;
  odla_value values[ODLA_MAX_OUTPUTS];
} odla_values;

//! \brief Unique id of each value
typedef struct _odla_value_id* odla_value_id;

//! \brief Multiple value ids
typedef struct {
  odla_size_t size;
  odla_value_id value_ids[ODLA_MAX_OUTPUTS];
} odla_value_ids;

//! \brief Set a value data
/*!
  \param value the pointer to the value
  \param data_ptr the pointer to the raw data buffer

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetValueData(odla_value value, const odla_void* data_ptr);

//! \brief Set a value data by id
/*!
  \param value_id the value id
  \param data_ptr the pointer to the raw data buffer

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetValueDataById(const odla_value_id value_id, const odla_void* data_ptr);

//! \brief Get the raw data ptr from value
/*!
  \param value the value
  \param data_ptr the raw data pointer of the value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetRawDataPtr(const odla_value value, odla_void** data_ptr);
//! \brief Get a value data by id
/*!
  \param value_id the value id
  \param data_ptr the pointer to the retrieved data buffer

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetValueDataById(const odla_value_id value_id, odla_void* data_ptr);

//! \brief Get the type of a value
/*!
  \param value the value
  \param value_type the pointer to the retrieved value type

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetValueType(const odla_value value, odla_value_type* value_type);

//! \brief Get the type of a value by id
/*!
  \param value_id the value id
  \param value_type the pointer to the retrieved value type

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_GetValueTypeById(
    const odla_value_id value_id, odla_value_type* value_type);

//! \brief Get the id of a value
/*!
  \param value the value
  \param value_id the pointer to the retrieved value id

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetValueId(const odla_value value, odla_value_id* value_id);

//! \brief Return the value by id
/*!
  \param value_id the value id
  \param value the pointer to the value with the id

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_FindValueById(const odla_value_id value_id, odla_value* value);

//! \brief Set a value as a computation output
/*!
  \param value the value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetValueAsOutput(odla_value value);

//! \brief Set multi values as a computation outputs
/*!
  \param value the values

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetValuesAsOutput(odla_values values);

//! \brief Set a value by id as a computation output
/*!
  \param value_id the value id

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetValueAsOutputById(const odla_value_id value_id);

//! \brief Release a value
/*!
  \param value the value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_ReleaseValue(odla_value value);

//! \brief Release a value by id
/*!
  \param value_id the value id

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_ReleaseValueById(odla_value_id value_id);

//! \brief Dump the data of the odla_value for debugging purpose.
/*!
  \param value the value to be dumpped
*/
extern ODLA_API_EXPORT void ODLA_API_CALL odla_Dump(odla_value value);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_VALUE_H_
