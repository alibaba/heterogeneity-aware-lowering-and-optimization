//===- if.cc --------------------------------------------------------------===//
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

#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnBranchBody(const IfInst& inst, bool taken) {
  auto bb = taken ? inst.GetThenBranch() : inst.GetElseBranch();
  int i = 1;
  for (auto& arg : bb->Args()) {
    ir_mapping_[*arg] = ir_mapping_[inst.GetOperand(i++)];
  }

  os_ << "  {\n";
  os_ << "    odla_EnterBranchBody(" << (taken ? "true" : "false") << ");\n";
  for (auto& c : bb->Constants()) {
    RunOnConstant(*c, true);
    RunOnConstant(*c, false);
  }
  for (auto& inst : *bb) {
    RunOnBaseInstruction(inst.get());
  }
  os_ << "  }\n";
  visited_.insert(bb);
} // namespace halo

void GenericCXXCodeGen::RunOnInstruction(IfInst* inst) {
  HLCHECK(inst->GetNumOfOperands() >= 1);
  const auto& cond = inst->GetOperand(0);
  CXXValue op_cond = ir_mapping_[cond];
  std::vector<CXXValue> outs;
  for (int i = 0, e = inst->GetNumOfResults(); i < e; ++i) {
    CXXValue cv(inst->GetName() + "_out_" + std::to_string(i), CXXType("void"));
    outs.push_back(cv);
    ir_mapping_[Def{inst, i}] = cv;
  }

  CXXValue null("", CXXType("void"));

  EmitODLACall(null, "odla_BeginIf", op_cond);
  RunOnBranchBody(*inst, true);
  RunOnBranchBody(*inst, false);

  EmitODLACall<2, true>(outs, "odla_EndIf");
}

} // namespace halo
