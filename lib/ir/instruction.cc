//===- instruction.cc ------------------------------------------------------==//
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

#include "halo/lib/ir/instruction.h"

namespace halo {

bool Instruction::ComputeResultTypes() {
  if (HasValidResultTypes()) {
    return true;
  }

  // By default, returns the 1st operand's type.
  const auto& ops = GetOperands();
  if (ops.empty()) {
    return false;
  }
  auto& results = GetResultsTypes();
  Type ty = ops[0].GetType();
  if (!ty.IsValid()) {
    return false;
  }
  results[0] = ty;
  return true;
}

void Instruction::CopyAttrsFrom(const Instruction& src) {
  HLCHECK(op_code_ == src.GetOpCode());
  auto& attrs = GetAttributes();
  attrs.clear();
  for (const auto& attr : src.GetAttributes()) {
    attrs.push_back(attr->Clone());
  }
}

void Instruction::CopyAttrsFrom(const Def& def) {
  HLCHECK(IsA<Instruction>(def));
  CopyAttrsFrom(*DynCast<Instruction>(def));
}

void Instruction::Print(std::ostream& os) const {
  os << "Inst: " << GetName() << "(";
  bool is_first = true;
  for (auto& type : GetResultsTypes()) {
    if (!is_first) {
      os << ", ";
    }
    is_first = false;
    type.Print(os);
  }
  os << ") = ";
  PrintOpcode(os);
  os << "(";
  PrintOperands(os);
  os << ")";
  if (GetNumOfAttributes() > 0) {
    os << " {Attrs: ";
    PrintAttributes(os);
    os << "}";
  }
  os << "\n";
}

void Instruction::PrintOpcode(std::ostream& os) const {
  os << OpCodeToString(op_code_);
}

std::string Instruction::OpCodeToString(OpCode op_code) {
  std::ostringstream os;
  switch (op_code) {
#define GET_INST_INFO_OPCODE_PRINT
#include "halo/lib/ir/instructions_info.def"
    default:
      os << "invalid";
#undef GET_INST_INFO_OPCODE_PRINT
  }
  return os.str();
}

} // namespace halo