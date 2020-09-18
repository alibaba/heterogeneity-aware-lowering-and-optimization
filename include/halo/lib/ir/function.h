//===- function.h -----------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_FUNCTION_H_
#define HALO_LIB_IR_FUNCTION_H_

#include "halo/lib/ir/argument.h"
#include "halo/lib/ir/basic_block.h"
#include "halo/lib/ir/constant.h"

namespace halo {

class Module;
class Argument;
class ReturnInst;

/// This class defines a function in IR.
class Function final : public IRObject {
 public:
  Function() = delete;
  explicit Function(GlobalContext& ctx, const std::string& name)
      : IRObject(ctx, name) {
    if (name.empty()) SetName("func_" + std::to_string(GetId()));
  }
  ~Function() = default;

  /// Iteration over the basic blocks in this function.
  using BasicBlockList = std::list<std::unique_ptr<BasicBlock>>;
  using iterator = BasicBlockList::iterator;
  using const_iterator = BasicBlockList::const_iterator;
  using reverse_iterator = BasicBlockList::reverse_iterator;
  using const_reverse_iterator = BasicBlockList::const_reverse_iterator;

  /// Function iterator methods.
  iterator begin() noexcept { return basic_blocks_.begin(); }
  const_iterator begin() const noexcept { return basic_blocks_.begin(); }
  iterator end() noexcept { return basic_blocks_.end(); }
  const_iterator end() const noexcept { return basic_blocks_.end(); }

  size_t size() const noexcept { return basic_blocks_.size(); }
  bool empty() const noexcept { return basic_blocks_.empty(); }
  BasicBlock* front() { return basic_blocks_.front().get(); }
  BasicBlock* back() { return basic_blocks_.back().get(); }

  /// Get the list of basic blocks.
  BasicBlockList& BasicBlocks() noexcept { return basic_blocks_; }
  const BasicBlockList& BasicBlocks() const noexcept { return basic_blocks_; }

  /// Arguments
  using ArgumentList = std::list<std::unique_ptr<Argument>>;
  using arg_iterator = ArgumentList::iterator;
  using const_arg_iterator = ArgumentList::const_iterator;
  arg_iterator arg_begin() noexcept { return args_.begin(); }
  const_arg_iterator arg_begin() const noexcept { return args_.begin(); }
  arg_iterator arg_end() noexcept { return args_.end(); }
  const_arg_iterator arg_end() const noexcept { return args_.end(); }
  ArgumentList& Args() noexcept { return args_; }
  const ArgumentList& Args() const noexcept { return args_; }
  Argument* arg_front() const { return args_.front().get(); }
  Argument* arg_back() const { return args_.back().get(); }

  /// Constants
  using ConstantList = std::list<std::unique_ptr<Constant>>;
  using constant_iterator = ConstantList::iterator;
  using const_constant_iterator = ConstantList::const_iterator;
  constant_iterator constant_begin() noexcept { return constants_.begin(); }
  const_constant_iterator constant_begin() const noexcept {
    return constants_.begin();
  }
  constant_iterator constant_end() noexcept { return constants_.end(); }
  const_constant_iterator constant_end() const noexcept {
    return constants_.end();
  }
  ConstantList& Constants() noexcept { return constants_; }
  const ConstantList& Constants() const noexcept { return constants_; }

  /// Returns the parent module this function belongs to.
  Module* GetParent() const noexcept { return parent_module_; }

  /// Print the content of this function.
  void Print(std::ostream& os) const override;

  Kind GetKind() const noexcept override { return Kind::Function; }

  bool IsEntryFunction() const noexcept { return is_entry_func_; }
  void SetAsEntryFunction(bool flag) noexcept { is_entry_func_ = flag; }

  const std::string& GetDeviceName() const noexcept { return device_; }
  void SetDeviceName(const std::string& name) noexcept { device_ = name; }
  ReturnInst* GetReturnInst() const;

  static inline bool Classof(const Function* func) { return true; }
  static inline bool Classof(const IRObject* obj) {
    return obj->GetKind() == Kind::Function;
  }

 private:
  // TODO: Function Proto
  Module* parent_module_ = nullptr;
  ArgumentList args_;
  ConstantList constants_;
  BasicBlockList basic_blocks_;
  bool is_entry_func_ = false;
  std::string device_ = "";
  friend class FunctionBuilder;
};

} // namespace halo

#endif // HALO_LIB_IR_FUNCTION_H_