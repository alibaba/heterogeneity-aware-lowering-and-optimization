//===- odla_ops_loop.h ----------------------------------------------------===//
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

#ifndef _ODLA_OPERATOR_OPS_LOOP_H_
#define _ODLA_OPERATOR_OPS_LOOP_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_value.h>

/*! \file
 * \details This file defines the ODLA loop operators.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Types of loop output
typedef enum {
  ODLA_LOOP_LAST_VALUE, /*!< The value for the last iteration */
  ODLA_LOOP_FWD_BEFORE, /* !< Concatenated values for each iteration:
                           [init_value...last_value) */
  ODLA_LOOP_FWD_AFTER,  /* !< Concatenated values for each iteration:
                           [state_1...last_value]) */
  ODLA_LOOP_REV_BEFORE, /* !< Concatenated values for each iteration,
                           in reverse order: (last_value...init_value]) */
  ODLA_LOOP_REV_AFTER,  /* !< Concatenated values for each iteration, in
                           reverse order: [last_value...init_value) */
} odla_loop_output_mode;

//! \brief Create a loop variable which has an initial value and recurrent
//! values.
/*!
  \param init_value initial value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_CreateLoopVariable(odla_value init_value, odla_value_id value_id);

//! \brief Create a loop. All subsequent loop related APIs are specific to
//! this one until the correspoding odla_EndLoop is called.
/*!
  \param trip_count the iteration count for the loop. The type of the value must
  be a scalar of integer.
  \param value_id a unique value id (can be NULL)

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_BeginLoop(odla_value trip_count, odla_value_id value_id);

//! \def End a loop scope.
//! \brief End a loop scope and define output values for the loop.
/*!
  \param condition The boolean condition to end the loop. If it is NULL, the
  condition is assumed as True, therefore the loop will be solely controlled by
  the `trip count` defined in `odla_BeginLoop`.
  \param output_values The values as loop outputs. These values must be created
  by `odla_CreateLoopVariable`.
  \param flags Each output value has a corresponding flag. If the flag is not
  `ODLA_LOOP_LAST_VALUE`, an extra concatenated value will be returned.
  \param value_ids unique value ids (can be NULL)

  \return odla_values loop output values.
*/
extern ODLA_API_EXPORT odla_values ODLA_API_CALL
odla_EndLoop(odla_value condition, odla_values output_values,
             const odla_loop_output_mode* flags, odla_value_ids);

extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_BeginIf(odla_value condition, odla_value_id);

extern ODLA_API_EXPORT odla_status odla_EnterBranchBody(odla_bool true_branch);

extern ODLA_API_EXPORT odla_values ODLA_API_CALL odla_EndIf(odla_value_ids);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_OPERATOR_OPS_LOOP_H_