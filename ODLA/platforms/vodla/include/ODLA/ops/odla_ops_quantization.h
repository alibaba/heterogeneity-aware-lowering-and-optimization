//===- odla_ops_quantization.h --------------------------------------------===//
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

#ifndef _ODLA_OPERATOR_OPS_QUANTIZATION_H_
#define _ODLA_OPERATOR_OPS_QUANTIZATION_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_value.h>

/*! \file
 * \details This file defines the ODLA quantization related operators.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Quantization info for each odla value
typedef struct {
  // value id of odla_value
  odla_value_id value_id;
  int ch_idx;
  odla_float32 scale;
  odla_float32 offset;
  odla_float32 min;
  odla_float32 max;
} odla_value_quant_info;

//! \brief Dequantize a tensor.
/*!
  Converts a quantized tensor to the full precision tensor.
  \param input the input value
  \param scale scale for input
  \param zero_point zero point for input
  \param axis the axis of the dequantizing. Ignored if scale is a scalar
  \param target_data_type the data type of full precision output
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_Dequantize(
    odla_value input, odla_value scale, odla_value zero_point, odla_int32 axis,
    odla_element_type target_data_type, const odla_value_id value_id);

//! \brief Quantize a tensor.
/*!
  Converts a tensor to low precision tensor.
  \param input the input value
  \param scale scale for input
  \param zero_point zero point for input
  \param axis the axis of the quantizing. Ignored if scale is a scalar
  \param target_data_type the data type of low precision output
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_Quantize(
    odla_value input, odla_value scale, odla_value zero_point, odla_int32 axis,
    odla_element_type target_data_type, const odla_value_id value_id);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_OPERATOR_OPS_QUANTIZATION_H_