//===------------------------== split.cc ------------------------------------=//
//
// Copyright (C) 2019-2022 Alibaba Group Holding Limited.
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

#include <cstdio>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(SplitInst* inst) {
  const Def& split_dim = inst->GetOperand(0);
  const Def& input = inst->GetOperand(1);
  const int num_split = inst->GetNumSplit();

  CXXValue op0 = ir_mapping_[split_dim];
  CXXValue op1 = ir_mapping_[input];

  std::vector<CXXValue> rets;
  for (int i = 0, e = num_split; i != e; ++i) {
    rets.emplace_back(inst->GetName() + "_" + std::to_string(i),
                      TensorTypeToCXXType(inst->GetResultType(i), false));
    ir_mapping_[Def(inst, i)] = rets[i];
  }

  EmitODLACall(rets, "odla_Split", op1, op0, inst->GetNumSplit());
}

} // namespace halo
