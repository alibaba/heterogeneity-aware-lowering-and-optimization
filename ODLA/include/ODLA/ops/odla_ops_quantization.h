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

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_OPERATOR_OPS_QUANTIZATION_H_