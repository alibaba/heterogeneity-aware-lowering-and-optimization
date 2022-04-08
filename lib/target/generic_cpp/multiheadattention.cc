//===- multiheadattention.cc ----------------------------------------------===//
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

#include "halo/lib/ir/nn_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(MultiHeadAttentionInst* inst) {
  constexpr int q_idx = 2;
  constexpr int k_idx = 4;
  constexpr int v_idx = 6;
  CXXValue input = ir_mapping_[inst->GetOperand(0)];
  CXXValue mask = ir_mapping_[inst->GetOperand(1)];
  CXXValue q_w = ir_mapping_[inst->GetOperand(q_idx)];
  CXXValue q_b = ir_mapping_[inst->GetOperand(q_idx + 1)];
  CXXValue k_w = ir_mapping_[inst->GetOperand(k_idx)];
  CXXValue k_b = ir_mapping_[inst->GetOperand(k_idx + 1)];
  CXXValue v_w = ir_mapping_[inst->GetOperand(v_idx)];
  CXXValue v_b = ir_mapping_[inst->GetOperand(v_idx + 1)];

  CXXValue ret(inst->GetName(), input.type);

  ir_mapping_[*inst] = ret;
  EmitODLACall(ret, "odla_MultiHeadAttention", input, mask, q_w, q_b, k_w, k_b,
               v_w, v_b);
}

} // namespace halo
