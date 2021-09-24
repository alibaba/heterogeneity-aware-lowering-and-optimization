//===- gather.cc ----------------------------------------------------------===//
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

#include <cstdio>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(GatherInst* inst) {
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[lhs];
  CXXValue op1 = ir_mapping_[rhs];

  const auto& ret_type = inst->GetResultType();

  CXXValue ret(inst->GetName(), op0.type);

  EmitODLACall(ret, "odla_Gather", op0, op1, inst->GetAxis(),
               EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(GatherElementsInst* inst) {
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[lhs];
  CXXValue op1 = ir_mapping_[rhs];

  const auto& ret_type = inst->GetResultType();

  CXXValue ret(inst->GetName(), op0.type);

  EmitODLACall(ret, "odla_GatherElements", op0, op1, inst->GetAxis(),
               EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

} // namespace halo