//===- add.cc -------------------------------------------------------------===//
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

#include <cstddef>
#include <cstdio>

#include "halo/api/halo_data.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnUnaryInstruction(Instruction* inst) {
  static const std::unordered_map<OpCode, const char*> names{
      {OpCode::ERF, "odla_Erf"},     {OpCode::EXP, "odla_Exp"},
      {OpCode::FLOOR, "odla_Floor"}, {OpCode::RSQRT, "odla_Rsqrt"},
      {OpCode::SQRT, "odla_Sqrt"},   {OpCode::SIN, "odla_Sin"},
      {OpCode::SINH, "odla_Sinh"},   {OpCode::COS, "odla_Cos"},
      {OpCode::COSH, "odla_Cosh"},   {OpCode::ROUND, "odla_Round"}};

  auto it = names.find(inst->GetOpCode());
  HLCHECK(it != names.end());

  const Def& lhs = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[lhs];

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, it->second, op0);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnBinaryInstruction(Instruction* inst) {
  static const std::unordered_map<OpCode, const char*> names{
      {OpCode::ADD, "odla_Add"},
      {OpCode::SUB, "odla_Sub"},
      {OpCode::MUL, "odla_Mul"},
      {OpCode::DIV, "odla_Div"}};
  auto it = names.find(inst->GetOpCode());
  HLCHECK(it != names.end());
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[lhs];
  CXXValue op1 = ir_mapping_[rhs];

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, it->second, op0, op1);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(AddInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(SubInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(MulInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(DivInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(FloorInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(RoundInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ErfInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ExpInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(SqrtInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(RsqrtInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(SinInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(SinhInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(CosInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(CoshInst* inst) {
  RunOnUnaryInstruction(inst);
}

} // namespace halo
