//===- instruction.h --------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_INSTRUCTION_H_
#define HALO_LIB_IR_INSTRUCTION_H_

#include <vector>

#include "halo/api/halo_data.h"
#include "halo/lib/ir/values.h"

namespace halo {

class BasicBlock;
class Function;

/// This class represents the base for all IR instructions.
class Instruction : public IRObject {
 public:
  Instruction() = delete;
  virtual ~Instruction() {}

  explicit Instruction(GlobalContext& context, const std::string& name,
                       const int num_outputs, const OpCode op)
      : IRObject(context, name, num_outputs), op_code_(op) {
    if (name.empty()) SetName("inst_" + std::to_string(GetId()));
  }

  // Disable the copy and assignment constructors.
  Instruction(const Instruction&) = delete;
  Instruction& operator=(const Instruction&) = delete;

  /// Return the parent basic block to which this instruction belongs.
  BasicBlock* GetParent() const noexcept { return parent_basic_block_; }

  bool ComputeResultTypes() override;

  OpCode GetOpCode() const noexcept { return op_code_; }
  static std::string OpCodeToString(OpCode op);

  Kind GetKind() const noexcept override { return Kind::Instruction; }

  static inline bool Classof(const Instruction* inst) { return true; }
  static inline bool Classof(const IRObject* obj) {
    return obj->GetKind() == Kind::Instruction;
  }

  void Print(std::ostream& os) const override;
  virtual void PrintOpcode(std::ostream& os) const;
  virtual std::unique_ptr<Instruction> Clone() const = 0;
  virtual void CopyAttrsFrom(const Instruction& src);
  virtual void CopyAttrsFrom(const Def& def);

  /// Interface to verify the various constraints.
  /// Return false if verification passes, return true otherwise.
  virtual bool Verify(bool before_infer_shape) const { return false; };
  /// Interface to implement custom verification code.
  /// It is called through Verify().
  /// Return false if custom verification passes, return true otherwise.
  virtual bool CustomVerify() const { return false; };

 private:
  // The parent basic block to which this instruction belongs
  BasicBlock* parent_basic_block_ = nullptr;

  // Op related info
  OpCode op_code_ = OpCode::INVALID;

  friend class IRBuilder;
};

template <typename T_TO, typename T_FROM>
T_TO* DynCast(Instruction* inst) {
  if (inst->GetOpCode() != T_TO::GetOpCode()) {
    HLCHECK(0 && "Invalid cast");
    return nullptr;
  }
  return Downcast<T_TO>(inst);
}

} // namespace halo

#endif // HALO_LIB_IR_INSTRUCTION_H_