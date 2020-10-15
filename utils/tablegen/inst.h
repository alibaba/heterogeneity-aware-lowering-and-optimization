//===- inst.h ---------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_UTIL_TABLEGEN_HALO_INST_H_
#define HALO_UTIL_TABLEGEN_HALO_INST_H_

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

namespace halo {

namespace tablegen {

const int kMaxOutNum = 16;

/// ValueType class
class Type {
 public:
  Type(const llvm::Record* record);
  /// Emit doc
  void EmitDoc(llvm::raw_ostream& o) const;

  /// bit-width
  int Width;
  /// integer type flag
  bool IsInt;
  /// float type flag
  bool IsFloat;
  /// bool type flag
  bool IsBool;
  /// string type flag
  bool IsString;
  /// list type flag, always false for an arg type.
  bool IsArray;
  /// whether an unsigned integer
  bool IsUnsigned;
  /// whether a quantized type
  bool IsQuantized;
};

/// Attr class and attribute access functions emitter.
class Attr {
 public:
  Attr(const llvm::Record* record, llvm::raw_ostream& o, int index);
  /// Emit attribute Set/Get functions
  void EmitAccess();
  /// Emit attribute constructor function
  /// with attribute's name and init value.
  void EmitInit();
  /// Emit document
  void EmitDoc();

  const std::string& GetAccessName() const noexcept { return access_name_; }

  /// normalize Set/Get function name.
  static std::string SetAccessName(const std::string& attr_name);

 private:
  const llvm::Record* record_;
  llvm::raw_ostream& os_;

  // attribute name
  std::string name_;
  // attribute ValueType def name.
  std::string type_;
  // attribute ValueType cpp_type.
  std::string cpp_type_;
  // index to Instruction attribute container.
  int index_;
  // normalized access function name
  std::string access_name_;
  // init value
  std::string init_;
  // attribute description
  std::string desc_;
};

/// Arg class
class Arg {
 public:
  Arg(const llvm::Record* record, llvm::raw_ostream& o, bool is_in);
  /// Whether an optional arg
  bool IsOptional() const { return is_optional_; }
  /// Whether an variable length arg
  bool IsVarArg() const { return is_var_length_; }
  /// Whether an input
  bool IsInput() const { return is_input_; }
  /// Return the matched arg index. -1 means no contraints.
  int GetMatchTypeIndex() const { return match_type_index_; }
  /// Emit markdown document
  void EmitDoc(const int id, int& unique_types);
  /// Emit type constraint
  bool EmitTypeConstraints(int id);
  /// Whether need to emit verify code
  bool NeedVerify();
  /// Emit verify code
  void EmitVerify(int i);

 private:
  bool NeedEmitTypeConstraints();

  const llvm::Record* record_;
  llvm::raw_ostream& os_;

  // doc description
  std::string desc_;
  // supported ValueType def name.
  std::vector<Type> types_;
  // type dimension
  int min_rank_;
  int max_rank_;
  // whether input or output
  bool is_input_;
  // whether input is optional
  bool is_optional_ = false;
  bool is_var_length_ = false;
  // type matched arg index; -1 for no such constraint.
  int match_type_index_ = -1;
};

/// Inst class and sub-instruction class emitter
class Inst {
 public:
  Inst(const llvm::Record* record, llvm::raw_ostream& o);
  /// Main entry to emit a sub-instruction class declaration
  void Run();
  /// Setup instruction corresponding OpCode enum
  void SetOpcode();
  /// Get op name as the original one in td
  const std::string& GetOpName() const { return op_; }
  /// Emit class constructor functions
  void EmitConstructor();
  /// Emit clone function
  void EmitClone();
  /// Emit attributes set/get functions
  void EmitAccessAttributes();
  /// Emit class static info
  void EmitStaticInstInfo();
  /// Emit a private function 'InitAttributes'
  /// to construct the attributes.
  void EmitInitAttributes();
  /// Emit a public function to copy attributes from the same Instr class.
  void EmitVerify();
  /// Emit formated (markdown) document
  void EmitDoc();

 private:
  void EmitConstructorCommon();

  const llvm::Record* record_;
  llvm::raw_ostream& os_;

  // operator name like AddN, Batchnorm
  std::string op_;
  // instruction sub-class name such as AddNInst, BatchnormInst
  std::string name_;
  // opcode name like OPCODE_ADDN
  std::string opcode_ = "OPCODE_INVALID";
  // attribute list
  std::vector<Attr> attrs_;
  // minimal number of operands, -1 if dynamic.
  int num_min_in_ = 0;
  // minimal number of operands.
  int num_max_in_ = 0;
  // num of results, -1 if dynamic.
  int num_out_ = -1;
  // operand and output list
  std::vector<Arg> args_;
  // instruction description
  std::string desc_;
};

} // end namespace tablegen

} // end namespace halo

#endif // HALO_UTIL_TABLEGEN_INST_H_