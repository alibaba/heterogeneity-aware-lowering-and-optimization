//===- relu.cc ------------------------------------------------------------===//
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

#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(ReluInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_Relu", op0);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(Relu6Inst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];

  CXXValue ret(inst->GetName(), op0.type);

  constexpr int hi = 6;
  EmitODLACall(ret, "odla_Clamp", op0, 0, hi);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(LeakyReluInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];

  CXXValue ret(inst->GetName(), op0.type);

  EmitODLACall(ret, "odla_LeakyRelu", op0, inst->GetAlpha());

  ir_mapping_[*inst] = ret;
}

} // namespace halo
