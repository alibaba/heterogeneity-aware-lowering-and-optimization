//===- custom.cc ----------------------------------------------------------===//
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

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(CustomInst* inst) {
  const Def& lhs = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[lhs];

  std::vector<CXXValue> rets;
  for (int i = 0; i < static_cast<int>(inst->GetNumOfResults()); i++) {
    rets.push_back({inst->GetName() + std::to_string(i),
                    TensorTypeToCXXType(inst->GetResultType(i), false)});
    ir_mapping_[Def(inst, i)] = rets[i];
  }

  std::vector<CXXValue> inputs{op0};
  const std::string op_name = "\"" + inst->GetOpname() + "\"";
  EmitODLACall(rets, "odla_CustomOp", inputs, op_name, op_name);
}

} // namespace halo