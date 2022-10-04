//===- reshape.cc ---------------------------------------------------------===//
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

#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(ReshapeInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];

  CXXValue ret(inst->GetName(), op0.type);
  const Def& out_shape = inst->GetOperand(1);
  CXXValue op1 = ir_mapping_[out_shape];

  EmitODLACall(ret, "odla_ReshapeDynamic", op0, op1);

  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(ReshapeDynamicInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];

  CXXValue ret(inst->GetName(), op0.type);
  const Def& out_shape = inst->GetOperand(1);
  CXXValue op1 = ir_mapping_[out_shape];

  EmitODLACall(ret, "odla_ReshapeDynamic", op0, op1);

  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(ShapeInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];

  const auto& ret_type = inst->GetResultType();
  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_Shape", op0, EmitShape(ret_type));

  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(UnsqueezeInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];

  CXXValue ret(inst->GetName(), op0.type);

  const Def& axes = inst->GetOperand(1);
  CXXValue op1 = ir_mapping_[axes];
  EmitODLACall(ret, "odla_Unsqueeze", op0, op1);

  ir_mapping_[*inst] = ret;
}

} // namespace halo
