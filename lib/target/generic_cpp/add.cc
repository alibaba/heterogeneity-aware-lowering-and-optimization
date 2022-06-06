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
      {OpCode::ABS, "odla_Abs"},        {OpCode::ACOS, "odla_ACos"},
      {OpCode::ACOSH, "odla_ACosh"},    {OpCode::ASIN, "odla_ASin"},
      {OpCode::ASINH, "odla_ASinh"},    {OpCode::ATAN, "odla_ATan"},
      {OpCode::ATANH, "odla_ATanh"},    {OpCode::COS, "odla_Cos"},
      {OpCode::COSH, "odla_Cosh"},      {OpCode::ERF, "odla_Erf"},
      {OpCode::EXP, "odla_Exp"},        {OpCode::FLOOR, "odla_Floor"},
      {OpCode::RSQRT, "odla_Rsqrt"},    {OpCode::SQRT, "odla_Sqrt"},
      {OpCode::SIN, "odla_Sin"},        {OpCode::SINH, "odla_Sinh"},
      {OpCode::ROUND, "odla_Round"},    {OpCode::NEG, "odla_Neg"},
      {OpCode::RCP, "odla_Reciprocal"}, {OpCode::NOT, "odla_Not"},
      {OpCode::CEIL, "odla_Ceil"},      {OpCode::LOG, "odla_Log"},
      {OpCode::TANH, "odla_Tanh"},      {OpCode::TAN, "odla_Tan"},
      {OpCode::ISINF, "odla_IsInf"},    {OpCode::SIGN, "odla_Sign"},
      {OpCode::ISNAN, "odla_IsNaN"},    {OpCode::NOT, "odla_Not"}};

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
      {OpCode::AND, "odla_And"},
      {OpCode::DIV, "odla_Div"},
      {OpCode::MAXIMUM, "odla_Max"},
      {OpCode::MEAN, "odla_Mean"},
      {OpCode::MINIMUM, "odla_Min"},
      {OpCode::MUL, "odla_Mul"},
      {OpCode::SUB, "odla_Sub"},
      {OpCode::POW, "odla_Pow"},
      {OpCode::OR, "odla_Or"},
      {OpCode::XOR, "odla_Xor"},
      {OpCode::SQUAREDDIFFERENCE, "odla_SquaredDifference"}};
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

void GenericCXXCodeGen::RunOnInstruction(MaximumInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(MinimumInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(MulInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(DivInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(AndInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(OrInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(XorInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(SquaredDifferenceInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ModInst* inst) {
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[lhs];
  CXXValue op1 = ir_mapping_[rhs];

  CXXValue ret(inst->GetName(), op0.type);
  auto mod = inst->GetFmod();
  EmitODLACall(ret, "odla_Mod", op0, op1, mod);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(CeilInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(FloorInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(LogInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(RcpInst* inst) {
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

void GenericCXXCodeGen::RunOnInstruction(AbsInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ACosInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ACoshInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ASinInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ASinhInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ATanInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(ATanhInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(NegInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(NotInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(PowInst* inst) {
  RunOnBinaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(TanhInst* inst) {
  RunOnUnaryInstruction(inst);
}
void GenericCXXCodeGen::RunOnInstruction(TanInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(IsNaNInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(IsInfInst* inst) {
  const Def& lhs = inst->GetOperand(0);
  CXXValue op0 = ir_mapping_[lhs];
  CXXValue ret(inst->GetName(), op0.type);
  auto check_pos = inst->GetDetectPositive();
  auto check_neg = inst->GetDetectNegative();
  EmitODLACall(ret, "odla_IsInf", op0, check_pos, check_neg);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(SignInst* inst) {
  RunOnUnaryInstruction(inst);
}

void GenericCXXCodeGen::RunOnInstruction(CmpInst* inst) {
  CXXValue op0 = ir_mapping_[inst->GetOperand(0)];
  CXXValue op1 = ir_mapping_[inst->GetOperand(1)];
  CXXValue ret(inst->GetName(), op0.type);
  auto pred = inst->GetPredicator();
  static const std::unordered_map<KindPredicate, const char*> odla_funcs{
      {KindPredicate::LT, "odla_Less"},
      {KindPredicate::LE, "odla_LessOrEqual"},
      {KindPredicate::GT, "odla_Greater"},
      {KindPredicate::GE, "odla_GreaterOrEqual"},
      {KindPredicate::NE, "odla_NotEqual"},
      {KindPredicate::EQ, "odla_Equal"}};
  auto it = odla_funcs.find(pred);
  HLCHECK(it != odla_funcs.end());

  EmitODLACall(ret, it->second, op0, op1);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(DetInst* inst) {
  CXXValue op0 = ir_mapping_[inst->GetOperand(0)];

  CXXValue ret(inst->GetName(), op0.type);
  const auto& ret_type = inst->GetResultType();
  EmitODLACall(ret, "odla_Det", op0, EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(ShiftInst* inst) {
  CXXValue op0 = ir_mapping_[inst->GetOperand(0)];
  CXXValue op1 = ir_mapping_[inst->GetOperand(1)];

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_Shift", op0, op1, inst->GetIsLeftShift());
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(MeanInst* inst) {
  CXXValue op0 = ir_mapping_[inst->GetOperand(0)];
  CXXValue ret(inst->GetName(), op0.type);

  const auto num = inst->GetNumOfOperands();
  std::vector<CXXValue> inputs;
  for (size_t i = 0; i < num; ++i) {
    const Def& op = inst->GetOperand(i);
    CXXValue op_v = ir_mapping_[op];
    inputs.push_back(op_v);
  }
  EmitODLACall(ret, "odla_Mean", inputs);
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(SelectInst* inst) {
  CXXValue op0 = ir_mapping_[inst->GetOperand(0)];
  CXXValue op1 = ir_mapping_[inst->GetOperand(1)];
  CXXValue op2 = ir_mapping_[inst->GetOperand(2)];

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_Select", op0, op1, op2,
               EmitShape(inst->GetResultType()));
  ir_mapping_[*inst] = ret;
}

} // namespace halo
