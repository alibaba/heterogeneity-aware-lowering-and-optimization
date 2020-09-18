//===- basic_block.h --------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_BASIC_BLOCK_H_
#define HALO_LIB_IR_BASIC_BLOCK_H_

#include <list>

#include "halo/lib/ir/instruction.h"

namespace halo {

class Function;

/// This class defines a basic block in IR.
/// A basic block is simply a list of instructions without terminating in
/// the middle. It is formed as "Single-Enter and Single-Exit".
class BasicBlock final : public IRObject {
 public:
  BasicBlock() = delete;
  explicit BasicBlock(GlobalContext& ctx, const std::string& name)
      : IRObject(ctx, name) {
    if (name.empty()) {
      SetName("bb_" + std::to_string(GetId()));
    }
  }
  ~BasicBlock() = default;

  // Disable the copy and assignment constructors.
  BasicBlock(const BasicBlock&) = delete;
  BasicBlock(BasicBlock&&) = delete;
  BasicBlock& operator=(const BasicBlock&) = delete;
  BasicBlock operator=(BasicBlock&&) = delete;

  /// Iteration over the instructions in this block.
  using InstructionList = std::list<std::unique_ptr<Instruction>>;
  using iterator = InstructionList::iterator;
  using const_iterator = InstructionList::const_iterator;
  using reverse_iterator = InstructionList::reverse_iterator;
  using const_reverse_iterator = InstructionList::const_reverse_iterator;

  /// BasicBlock iterator methods.
  ///
  inline iterator begin() noexcept { return instructions_.begin(); }
  inline const_iterator begin() const noexcept { return instructions_.begin(); }
  inline iterator end() noexcept { return instructions_.end(); }
  inline const_iterator end() const noexcept { return instructions_.end(); }
  inline reverse_iterator rbegin() noexcept { return instructions_.rbegin(); }
  inline const_reverse_iterator rbegin() const noexcept {
    return instructions_.crbegin();
  }
  inline reverse_iterator rend() noexcept { return instructions_.rend(); }
  inline const_reverse_iterator rend() const noexcept {
    return instructions_.crend();
  }

  inline size_t size() const noexcept { return instructions_.size(); }
  inline bool empty() const noexcept { return instructions_.empty(); }
  inline Instruction* front() { return instructions_.front().get(); }
  inline Instruction* back() { return instructions_.back().get(); }

  /// Get the instruction list.
  InstructionList& Instructions() noexcept { return instructions_; }
  const InstructionList& Instructions() const noexcept { return instructions_; }

  /// Return the parent function to which this basic block belongs.
  Function* GetParent() noexcept { return parent_function_; }

  Kind GetKind() const noexcept override { return Kind::BasicBlock; }

  static inline bool Classof(const BasicBlock* bb) { return true; }
  static inline bool Classof(const IRObject* obj) {
    return obj->GetKind() == Kind::BasicBlock;
  }

  /// Print the content of this basic block.
  void Print(std::ostream& os) const override;

 private:
  Function* parent_function_ = nullptr;
  InstructionList instructions_;

  friend class BasicBlockBuilder;
};

} // namespace halo

#endif // HALO_LIB_IR_BASIC_BLOCK_H_