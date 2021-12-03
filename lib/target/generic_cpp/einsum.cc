//===- einsum.cc ----------------------------------------------------------===//
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

#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(EinsumInst* inst) {
  const auto& equation = inst->GetEquation();

  CXXValue op0 = ir_mapping_[inst->GetOperand(0)];
  CXXValue ret(inst->GetName(), op0.type);

  ir_mapping_[*inst] = ret;
  const halo::Type& ret_shape = inst->GetResultType();
  const auto num = inst->GetNumOfOperands();
  std::vector<CXXValue> inputs;
  for (size_t i = 0; i < num; ++i) {
    const Def& op = inst->GetOperand(i);
    CXXValue op_v = ir_mapping_[op];
    inputs.push_back(op_v);
  }
  const std::string equation_str = "\"" + equation + "\"";
  EmitODLACall(ret, "odla_Einsum", inputs, equation_str, EmitShape(ret_shape));
}

} // namespace halo
