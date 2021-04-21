//===- odla_ops_basic.h ---------------------------------------------------===//
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

#ifndef _ODLA_OPERATOR_OPS_BASIC_H_
#define _ODLA_OPERATOR_OPS_BASIC_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_value.h>

/*! \file
 * \details This file defines the ODLA basic operators.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \def CreateArgument
//! \brief Create an argument value
/*!
  \param value_type argument value type
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_CreateArgument(
    const odla_value_type value_type, const odla_value_id value_id);

//! \brief Create a regular value
/*!
  \param value_type value type
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_CreateValue(
    const odla_value_type value_type, const odla_value_id value_id);

//! \def CreateConstant
//! \brief Create a constant value
/*!
  \param value_type constant value type
  \param data_ptr the pointer to the raw data
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_CreateConstant(const odla_value_type value_type, const odla_void* data_ptr,
                    const odla_value_id value_id);

//! \def CloneValue
//! \brief Clone a value with the data copy
/*!
  \param src_value the src value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_CloneValue(const odla_value src_value, const odla_value_id value_id);

//! \def CloneValueById
//! \brief Clone a value by id with the data copy
/*!
  \param src_value_id the src value id
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_CloneValueById(
    const odla_value_id src_value_id, const odla_value_id value_id);

//! \def CreateVariable
//! \brief Create a trainable variable value
/*!
  \param value_type variable value type
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_CreateVariable(
    const odla_value_type value_type, const odla_value_id value_id);

//! \def Assign
//! \brief Assign a value to a variable
/*!
  \param dst a variable to be assigned to
  \param src a source value
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_Assign(odla_value dst,
                                                             odla_value src);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_OPERATOR_OPS_BASIC_H_