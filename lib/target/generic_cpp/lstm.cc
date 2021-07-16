//===- conv.cc ------------------------------------------------------------===//
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

#include "halo/lib/framework/common.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(LSTMInst* inst) {
  const Def& x = inst->GetOperand(0);
  const Def& w = inst->GetOperand(1);
  const Def& r = inst->GetOperand(2);
  const Def& b = inst->GetOperand(3);

  CXXValue op_x = ir_mapping_[x];
  CXXValue op_w = ir_mapping_[w];
  CXXValue op_r = ir_mapping_[r];
  CXXValue op_b = ir_mapping_[b];

  uint32_t seq_len = x.GetType().GetNumOfElementsInDim(0);
  uint32_t hidden_size = r.GetType().GetNumOfElementsInDim(2);

  std::vector<CXXValue> rets;
  rets.emplace_back(inst->GetName(),
                    TensorTypeToCXXType(inst->GetResultsTypes()[0], false));
  rets.emplace_back(inst->GetName() + "_h",
                    TensorTypeToCXXType(inst->GetResultsTypes()[1], false));
  rets.emplace_back(inst->GetName() + "_c",
                    TensorTypeToCXXType(inst->GetResultsTypes()[2], false));

  std::string op_direction("ODLA_RNN_FORWARD");
  Direction direction = inst->GetDirection();

  if (direction == Direction::REVERSE) {
    op_direction = "ODLA_RNN_REVERSE";
  } else if (direction == Direction::BIDIRECTIONAL) {
    op_direction = "ODLA_RNN_BIDIRECTIONAL";
  } else {
    HLCHECK(direction == Direction::FORWARD);
  }

  std::string outputs = "ODLA_RNN_HIDDEN_CELL_STATE";

  EmitODLACall(rets, "odla_LSTM", op_x, EmitShape(w.GetType()), op_w, op_r,
               op_b, seq_len, hidden_size, op_direction, outputs);

  ir_mapping_[Def(inst, 0)] = rets[0];
  ir_mapping_[Def(inst, 1)] = rets[1];
  ir_mapping_[Def(inst, 2)] = rets[2];
}

} // namespace halo
