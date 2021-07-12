//===- odla_ops_custom.cc -------------------------------------------------===//
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

#include <ODLA/odla.h>

#include <popart/builder.hpp>
#include <string>

#include "common.h"
#include "odla_popart.h"

extern thread_local odla_computation g_comp;

odla_values odla_CustomOp(odla_values inputs, const odla_char* op_name,
                          const odla_char* function_name,
                          const odla_value_ids ids, ...) {
  odla_values ret;
  if (std::string(op_name) == "custom_IpuGelu") {
    assert(ids.size == 1);
    assert(inputs.size == 1);
    const char* id = reinterpret_cast<const char*>(ids.value_ids[0]);
    const auto& name = id != nullptr ? std::string(id) : "IpuGelu";
    auto input = inputs.values[0];
    popart::TensorId result =
        g_comp->builder->aiGraphcoreOpset1().gelu({input->tensor_id});
    auto val = new _odla_value(result,
                               {g_comp->builder->getTensorDataType(result),
                                g_comp->builder->getTensorShape(result)},
                               name);
    ret.size = 1;
    ret.values[0] = val;
  } else {
    assert(0);
  }
  return ret;
}
