//===- inst.cc --------------------------------------------------*- C++ -*-===//
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

#include "inst.h"

namespace halo {

namespace tablegen {

// Initialize type flags
Type::Type(const llvm::Record* record) {
  Width = record->getValueAsInt("width_");
  IsInt = record->getValueAsBit("is_integer_");
  IsBool = (Width == 1) && IsInt;
  IsString = record->getValueAsBit("is_string_");
  IsPointer = record->getValueAsBit("is_pointer_");
  IsFloat = !IsInt && !IsString;
  IsArray = record->getValueAsBit("is_array_");
  Is2DArray = record->getValueAsBit("is_2d_array_");
  IsUnsigned = record->getValueAsBit("is_unsigned_");
  IsQuantized = record->getValueAsBit("is_quantized_");
  AltName = record->getValueAsString("alt_name");
}

void Type::EmitDoc(llvm::raw_ostream& o) const {
  if (IsString) {
    o << "string";
  } else if (IsBool) {
    o << "bool";
  } else {
    if (IsFloat) {
      o << "fp";
    } else {
      if (IsQuantized) {
        o << "q";
      }
      if (IsUnsigned) {
        o << "u";
      }
      o << "int";
    }
    o << Width;
  }
  if (IsArray || Is2DArray) {
    o << " list";
  }
}

// Setup access function name by converting variable name to function
// naming convention, e.g., trans_a to TransA
std::string Attr::SetAccessName(const std::string& attr_name) {
  std::string access_name;
  llvm::SmallVector<llvm::StringRef, 4> segs;
  llvm::StringRef name(attr_name);
  name.split(segs, '_');
  for (size_t i = 0; i < segs.size(); ++i) {
    std::string s = segs[i].str();
    s[0] = toupper(s[0]);
    access_name += s;
  }
  return access_name;
}

// Construct Attr
Attr::Attr(const llvm::Record* record, llvm::raw_ostream& o, int index)
    : record_(record), os_(o), index_(index) {
  name_ = record_->getValueAsString("attr_name_").str();
  type_ = record_->getValueAsDef("type_")->getName().str();
  cpp_type_ = record_->getValueAsDef("type_")->getValueAsString("cpp_type_");
  init_ = record->getValueAsString("init_value_").str();
  desc_ = record->getValueAsString("description_").str();
  access_name_ = SetAccessName(name_);
}

// Emit Set/Get functions
// e.g.,
// const bool GetTransA() const {..}
// void SetTransA(const bool trans_a){..}
void Attr::EmitAccess() {
  const std::string attr_ptr =
      "GetAttributes()[" + std::to_string(index_) + "]";
  const std::string type_ref =
      record_->getValueAsDef("type_")->getValueAsBit("is_array_") ? "&" : "";
  bool is_ptr = record_->getValueAsDef("type_")->getValueAsBit("is_pointer_");
  os_.indent(2);
  os_ << cpp_type_ << " Get" << access_name_ << "() const {\n";
  os_.indent(4);
  os_ << "return " << attr_ptr << "->GetValueAs" << type_ << "();\n";
  os_.indent(2);
  os_ << "}\n";
  os_.indent(2);
  if (is_ptr) {
    os_ << "void Set" << access_name_ << "(" << cpp_type_ << " " << name_
        << ") {\n";
  } else {
    os_ << "void Set" << access_name_ << "(const " << cpp_type_ << type_ref
        << " " << name_ << ") {\n";
  }
  os_.indent(4);
  os_ << attr_ptr << "->SetValueAs" << type_ << "(" << name_ << ")"
      << ";\n";
  os_.indent(2);
  os_ << "}\n";
}

// Emit attribute constructor function with attribute's name and init values.
// e.g.,
// Attribute::CreateBool("trans_a", false)
void Attr::EmitInit() {
  os_ << "Attribute::Create" << type_ << "(\"" << name_ << "\", ";
  auto type_record = record_->getValueAsDef("type_");
  if (!init_.empty()) {
    if (type_record->isSubClassOf("EnumValueType")) {
      os_ << cpp_type_ << "::";
    }
    os_ << init_;
  } else if (type_record->isSubClassOf("EnumValueType")) {
    os_ << cpp_type_ << "::INVALID";
  } else {
    std::string init_value;
    Type vt(type_record);
    if (vt.IsArray || vt.Is2DArray) {
      init_value = "{}";
    } else if (vt.IsBool) {
      init_value = "false";
    } else if (vt.IsInt) {
      init_value = "0";
    } else if (vt.IsFloat) {
      init_value = "0.0f";
    } else if (vt.IsString) {
      init_value = "\"\"";
    } else {
      llvm_unreachable("unsupported value type");
    }
    os_ << init_value;
  }
  os_ << ")";
}

// Emit doc
// attribute name: type, default value, description.
void Attr::EmitDoc() {
  os_ << name_ << ": ";
  auto type_record = record_->getValueAsDef("type_");
  if (type_record->isSubClassOf("EnumValueType")) {
    os_ << "Enum " << cpp_type_;
    // TODO(s.ding): emit enum value list
  } else {
    Type(type_record).EmitDoc(os_);
  }
  if (!init_.empty()) {
    os_ << ", default to " << init_;
  }
  os_ << ", " << desc_;
}

// Construct Arg
Arg::Arg(const llvm::Record* record, llvm::raw_ostream& o, bool is_in)
    : record_(record), os_(o), is_input_(is_in) {
  min_rank_ = record->getValueAsDef("dim_")->getValueAsInt("min_rank_");
  max_rank_ = record->getValueAsDef("dim_")->getValueAsInt("max_rank_");

  auto types_r = record_->getValueAsDef("types_");
  if (types_r->isSubClassOf("MatchArgType")) {
    match_type_index_ = types_r->getValueAsInt("match_arg_id_");
  } else {
    std::vector<llvm::Record*> types =
        types_r->getValueAsListOfDefs("prime_types_");
    for (auto ic = types.begin(), ec = types.end(); ic != ec; ++ic) {
      types_.emplace_back(*ic);
    }
  }
  desc_ = record_->getValueAsString("description_").str();
  is_optional_ = record_->isSubClassOf("OptionalArg");
  is_var_length_ = record_->isSubClassOf("VarArg");
}

// Emit Arg(operand & result) doc
void Arg::EmitDoc(const int arg_id, int& unique_types) {
  if (NeedEmitTypeConstraints()) {
    ++unique_types;
  }
  if (is_input_) {
    os_ << "X";
  } else {
    os_ << "Y";
  }
  os_ << arg_id;
  if (is_input_ && is_optional_) {
    os_ << "(OPTIONAL)";
  }
  // TODO(s.ding): var length operands and results
  os_ << ": ";
  os_ << "(T" << unique_types << ")";
  if (min_rank_ == max_rank_) {
    if (min_rank_ == 0) {
      os_ << ", scalar";
    } else {
      os_ << ", " << min_rank_ << "D";
    }
  }
  os_ << ". " << desc_;
}

bool Arg::EmitTypeConstraints(int id) {
  if (!NeedEmitTypeConstraints()) {
    return false;
  }
  os_ << "T" << id << ": ";
  int i = 0;
  for (const auto& type : types_) {
    type.EmitDoc(os_);
    if (++i < types_.size()) {
      os_ << ", ";
    }
  }
  return true;
}

bool Arg::NeedEmitTypeConstraints() { return (match_type_index_ == -1); }

// Construct Inst
Inst::Inst(const llvm::Record* record, llvm::raw_ostream& o)
    : record_(record), os_(o) {
  op_ = record_->getName().str();
  name_ = op_ + "Inst";
  std::string big_op(op_);
  std::transform(big_op.begin(), big_op.end(), big_op.begin(), toupper);
  opcode_ = "OpCode::" + big_op;
  std::vector<llvm::Record*> attrs = record_->getValueAsListOfDefs("attrs_");
  std::vector<llvm::Record*> ins = record_->getValueAsListOfDefs("ins_");
  std::vector<llvm::Record*> outs = record_->getValueAsListOfDefs("outs_");
  num_max_in_ = num_min_in_ = ins.size();
  num_out_ = outs.size();

  int attr_index = 0;
  for (const auto& attr : attrs) {
    attrs_.emplace_back(attr, os_, attr_index++);
  }

  for (const auto& in : ins) {
    args_.emplace_back(in, os_, true);
    if (args_.back().IsOptional()) {
      num_min_in_--;
    }
  }
  for (auto arg : outs) {
    Arg a(arg, os_, false);
    if (a.IsVarArg()) {
      num_out_ = kMaxOutNum;
    }
    args_.emplace_back(a);
  }
  desc_ = record_->getValueAsString("description_").str();
}

// Emit attribute access functions
void Inst::EmitAccessAttributes() {
  for (Attr& attr : attrs_) {
    attr.EmitAccess();
  }
}

// Emit static info
void Inst::EmitStaticInstInfo() {
  os_ << "  static constexpr int num_min_ins_ = " << num_min_in_ << ";\n";
  os_ << "  static constexpr int num_max_ins_ = " << num_max_in_ << ";\n";
  os_ << "  static constexpr int num_outs_ = " << num_out_ << ";\n";
}

void Inst::EmitConstructorCommon() {
  if (!attrs_.empty()) {
    os_.indent(4);
    os_ << "InitAttributes();\n";
  }
  if (opcode_ == "OpCode::CALL" || opcode_ == "OpCode::IF" ||
      opcode_ == "OpCode::RETURN" || opcode_ == "OpCode::KVPARSER") {
    os_.indent(4);
    os_ << "SetVariadicReturns(true);\n";
    if (opcode_ == "OpCode::RETURN") {
      os_.indent(4);
      os_ << "SetNumOfResults(GetNumOfOperands());\n";
    } else if (opcode_ == "OpCode::KVPARSER") {
      os_.indent(4);
      os_ << "auto DenseColumnNames_vec = GetDenseColumnNames();\n";
      os_ << "SetNumOfResults(DenseColumnNames_vec.size());\n";
    }
  }
}

// Emit instruction constructor functions
void Inst::EmitConstructor() {
  auto num_indent = name_.size() + 1;
  if (num_min_in_ != -1) {
    // individual operand style, e.g.,
    // GemmInst(GlobalContext& context, const std::string& name,
    //          const Def& op1, const Def& op2, const Def& op3)
    //	   : Instruction(context, name, num_outs, OpCode::GEMM) {
    //   AddOperands({op1, op2, op3});
    //   InitAttributes();
    // }
    for (int i = num_min_in_; i <= num_max_in_; ++i) {
      os_.indent(2);
      os_ << name_ << "(GlobalContext& context, const std::string& name";
      if (i > 0) {
        os_ << ",";
      }
      for (int j = 0; j < i; ++j) {
        if (j % 3 == 0) {
          os_ << "\n";
          os_.indent(2);
          os_.indent(num_indent);
        } else {
          os_ << " ";
        }
        os_ << "const Def& op" << j;
        if (j < i - 1) {
          os_ << ",";
        }
      }
      os_ << ")\n";
      os_.indent(6); // NOLINT.
      os_ << ": Instruction(context, name, " << num_out_ << ", " << opcode_
          << ") {\n";
      if (i > 0) {
        os_.indent(4);
        if (i == 1) {
          os_ << "AddOneOperand(op0";
        } else if (i > 1) {
          os_ << "AddOperands({op0";
          for (int j = 1; j < i; ++j) {
            os_ << ", op" << j;
          }
          os_ << "}";
        }
        os_ << ");\n";
      }
      EmitConstructorCommon();
      os_.indent(2);
      os_ << "}\n";
    }
  }
  // operand list style, e.g.,
  // GemmInst(GlobalContext& context, const std::string& name,
  //          const std::vector<Def>& operands)
  //	  : Instruction(context, name, num_outs, OpCode::GEMM) {
  //   AddOperands(operands);
  //   InitAttributes();
  // }
  os_.indent(2);
  os_ << name_ << "(GlobalContext& context, const std::string& name,\n";
  os_.indent(2);
  os_.indent(num_indent);
  os_ << "const std::vector<Def>& operands)\n";
  os_.indent(6); // NOLINT.
  os_ << ": Instruction(context, name, " << num_out_ << ", " << opcode_
      << ") {\n";
  os_.indent(4);
  os_ << "AddOperands(operands);\n";
  EmitConstructorCommon();
  os_.indent(2);
  os_ << "}\n";
}

void Inst::EmitClone() {
  const std::string cls_name = op_ + "Inst";
  os_ << "  std::unique_ptr<Instruction> Clone() const override {\n";
  os_ << "    auto inst = std::make_unique<" << cls_name
      << ">(GetGlobalContext(), GetName(), "
         "std::vector<Def>(GetNumOfOperands(), "
         "Def::GetUndefined()));\n";
  os_ << "  inst->CopyAttrsFrom(*this);\n";
  os_ << "  inst->SetNumOfResults(this->GetNumOfResults());\n";
  os_ << "    return std::move(inst);\n";
  os_ << "  }\n";
}
// Emit a private function to constructor the attributes.
// e.g.,
// void InitAttributes() {
//   AddOneAttribute(Attribute::CreateFloat("alpha", 0.0));
//   AddOneAttribute(Attribute::CreateFloat("beta", 0.0));
//   AddOneAttribute(Attribute::CreateBool("transA", true));
//   AddOneAttribute(Attribute::CreateBool("transB", false));
// }
void Inst::EmitInitAttributes() {
  os_.indent(2);
  os_ << "void InitAttributes() {\n";
  for (Attr& attr : attrs_) {
    os_.indent(4);
    os_ << "AddOneAttribute(";
    attr.EmitInit();
    os_ << ");\n";
  }
  os_.indent(2);
  os_ << "}\n";
}

/// Main entry to emit an instruction class
void Inst::Run() {
  os_ << "/// " << desc_ << "\n";
  os_ << "class " + name_ + " : public Instruction {\n";
  os_ << " public:\n";
  EmitConstructor();
  EmitClone();
  EmitAccessAttributes();
  EmitVerify();
  EmitOperandOptional();
  EmitClassof();

  os_ << "\n";
  os_ << " private:\n";
  if (!attrs_.empty()) {
    EmitInitAttributes();
  }
  // EmitStaticInstInfo();
  os_ << "};\n\n";
}

void Inst::EmitDoc() {
  // emit inst name
  os_ << "## " << op_ << "  \n";
  // emit inst description
  os_ << desc_ << "  \n";
  // emit attributes
  if (!attrs_.empty()) {
    os_ << "\n**Attributes:**  \n";
  }
  for (auto attr : attrs_) {
    attr.EmitDoc();
    os_ << "  \n";
  }
  // emit operands and results
  int arg_id = 0;
  int num_constraints = 0;
  bool last_is_input = false;

  if (num_max_in_ != 0) {
    os_ << "\n**Operands:**  \n";
    last_is_input = true;
  }
  for (auto ib = args_.begin(), ie = args_.end(); ib != ie; ++ib) {
    if (ib->IsInput() != last_is_input) {
      arg_id = 0;
      last_is_input = false;
      os_ << "\n**Results:**  \n";
    }
    ib->EmitDoc(++arg_id, num_constraints);
    os_ << "  \n";
  }
  // Emit type constraints
  if (num_constraints != 0) {
    int i = 1;
    os_ << "\n**Type Constraints:**  \n";
    for (auto ib = args_.begin(), ie = args_.end(); ib != ie; ++ib) {
      if (ib->EmitTypeConstraints(i)) {
        ++i;
        os_ << "  \n";
      }
    }
  }
}

void Inst::EmitOperandOptional() {
  os_ << "  bool IsOperandOptional(size_t idx) const noexcept override {\n";
  for (int i = 0; i < args_.size(); ++i) {
    if (args_[i].IsOptional()) {
      os_ << "    if (idx == " << i << ") { return true;}\n";
    }
  }
  os_ << "    return false;\n";
  os_ << "  }";
}

void Inst::EmitVerify() {
  os_ << "  bool Verify(bool before_infer_shape) const override {\n";
  os_ << "    bool broken = false;\n";
  if (num_min_in_ != -1) {
    os_ << "    size_t num_ins = GetNumOfOperands();\n";
    os_ << "    broken |= CheckFailed(num_ins >= " << num_min_in_ << " && "
        << "num_ins <= " << num_max_in_ << ",\n";
    os_.indent(8); // NOLINT.
    os_ << "\"operand number is expected to be (" << num_min_in_ << ", "
        << num_max_in_ << ").\", this);\n";
  }

  os_ << "    if (!before_infer_shape) {\n";
  int i = 0;
  for (auto& arg : args_) {
    arg.EmitVerify(i++);
  }
  os_ << "    }\n";
  // call custom verification code.
  os_ << "    broken |= CustomVerify();\n";
  os_ << "    return broken;\n";
  os_ << "  }\n";
}

void Inst::EmitClassof() {
  os_ << "  static inline bool Classof(const IRObject* obj) {\n";
  os_ << "    if (!Instruction::Classof(obj)) {\n";
  os_ << "      return false;\n";
  os_ << "    }\n";
  os_ << "    const Instruction* inst = DynCast<Instruction>(obj);\n";
  os_ << "    return inst->GetOpCode() == " << opcode_ << ";\n";
  os_ << "  }\n";
}

bool Arg::NeedVerify() {
  // verify only input
  if (!is_input_) {
    return false;
  }
  // verify specified num of dims
  if (min_rank_ == max_rank_) {
    return true;
  }
  // verify data type match
  if (match_type_index_ != -1) {
    return true;
  }
  return false;
}

void Arg::EmitVerify(int i) {
  if (!NeedVerify()) {
    return;
  }
  int indent = 6; // NOLINT.
  if (is_optional_) {
    os_.indent(indent);
    os_ << "if (" << i << " < num_ins) {\n";
    indent += 2;
  }
  os_.indent(indent);
  os_ << "const Type& ty_" << i << " = GetOperand(" << i << ").GetType();\n";
  if (min_rank_ == max_rank_) {
    os_.indent(indent);
    os_ << "broken |= CheckFailed(ty_" << i
        << ".GetNumOfDims() == " << min_rank_ << ",\n";
    os_.indent(indent + 4); // NOLINT.
    os_ << "\"dim = " << min_rank_ << " is expected at operand " << i
        << ".\", this);\n";
  }
  if (match_type_index_ != -1) {
    os_.indent(indent);
    os_ << "DataType dt_" << i << " = ty_" << i << ".GetDataType();\n";
    os_.indent(indent);
    os_ << "DataType dt_match_" << i << " = GetOperand(" << match_type_index_
        << ").GetType().GetDataType();\n";
    os_.indent(indent);
    os_ << "broken |= CheckFailed(dt_" << i << " == dt_match_" << i << ",\n";
    os_.indent(indent + 4); // NOLINT.
    os_ << "\"type match is expected at operand " << i << " and "
        << match_type_index_ << ".\", this);\n";
  }
  if (is_optional_) {
    indent -= 2;
    os_.indent(indent);
    os_ << "}\n";
  }
}

} // namespace tablegen

} // end namespace halo
