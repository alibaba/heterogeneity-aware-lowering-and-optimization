//===- batchnorm.cc -------------------------------------------------------===//
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

#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(BatchNormInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& mean = inst->GetOperand(3);
  const Def& var = inst->GetOperand(4);

  CXXValue op0 = ir_mapping_[input];
  CXXValue op1 = ir_mapping_[mean];
  CXXValue op2 = ir_mapping_[var];

  std::string scale_name = "nullptr";
  std::string offset_name = "nullptr";

  {
    CXXValue op = ir_mapping_[inst->GetOperand(1)];
    scale_name = op.name;
  }

  {
    CXXValue op = ir_mapping_[inst->GetOperand(2)];
    offset_name = op.name;
  }

  CXXValue ret(inst->GetName(), op0.type);

  EmitODLACall(ret, "odla_BatchNormalization", op0, inst->GetDataFormat(), op1,
               op2, inst->GetEpsilon(), scale_name, offset_name, 1, 0);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(GroupNormInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& scale = inst->GetOperand(1);
  const Def& bias = inst->GetOperand(2);

  CXXValue op0 = ir_mapping_[input];
  CXXValue op1 = ir_mapping_[scale];
  CXXValue op2 = ir_mapping_[bias];
  HLCHECK(inst->GetGroups() > 0 && "Invalid Group");
  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_GroupNormalization", op0, inst->GetDataFormat(),
               inst->GetGroups(), inst->GetEpsilon(), op1, op2, 1, 0);

  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(InstanceNormInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& scale = inst->GetOperand(1);
  const Def& bias = inst->GetOperand(2);

  CXXValue op0 = ir_mapping_[input];
  CXXValue op1 = ir_mapping_[scale];
  CXXValue op2 = ir_mapping_[bias];

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_InstanceNormalization", op0, inst->GetDataFormat(),
               inst->GetEpsilon(), op1, op2, 1, 0);

  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(LpNormalizeInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& scale = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[input];
  CXXValue op1 = ir_mapping_[scale];

  CXXValue ret(inst->GetName(), op0.type);
  int p = inst->GetP();
  HLCHECK(p > 0 && "Invalid exponent value in the norm formulation.");
  EmitODLACall(ret, "odla_LpNormalize", op0, inst->GetP(),
               inst->GetDataFormat(), (inst->GetAxis()).size(), inst->GetAxis(),
               inst->GetEpsilon(), op1);

  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(LayerNormInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& scale = inst->GetOperand(1);
  const Def& offset = inst->GetOperand(2);

  CXXValue op0 = ir_mapping_[input];
  CXXValue op1 = ir_mapping_[scale];
  CXXValue op2 = ir_mapping_[offset];

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_LayerNorm", op0, (inst->GetAxis()).size(),
               inst->GetAxis(), inst->GetEpsilon(), op1, op2);
  ir_mapping_[*inst] = ret;
}

} // end namespace halo
