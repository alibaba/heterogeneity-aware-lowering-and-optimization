//===- odla_ops_custom.h --------------------------------------------------===//
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

#ifndef _ODLA_OPERATOR_OPS_CUSTOM_H_
#define _ODLA_OPERATOR_OPS_CUSTOM_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_value.h>

/*! \file
 * \details This file defines the ODLA custom operator.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief CustomOp
/*!
  \param inputs the inputs value
  \param op_name a pointer to the name string of the custom operator
  \param function_name a pointer to the name string of the impl function
  \param value_ids unique value ids
  \param Variadic

  \return odla_values
*/
extern ODLA_API_EXPORT odla_values ODLA_API_CALL
odla_CustomOp(odla_values inputs, const odla_char* op_name,
              const odla_char* function_name, const odla_value_ids ids, ...);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_OPERATOR_OPS_CUSTOM_H_