//===- module.h -------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_MODULE_H_
#define HALO_LIB_IR_MODULE_H_

#include <string>

#include "halo/lib/ir/function.h"

namespace halo {

/// This class defines a module in IR.
class Module final : public IRObject {
 public:
  Module() = delete;
  explicit Module(GlobalContext& context, const std::string& name)
      : IRObject(context, name) {
    if (name.empty()) SetName("module_" + std::to_string(GetId()));
  }
  ~Module() = default;

  // Iteration over the functions in the module.
  using FunctionList = std::list<std::unique_ptr<Function>>;
  using iterator = FunctionList::iterator;
  using const_iterator = FunctionList::const_iterator;
  using reverse_iterator = FunctionList::reverse_iterator;
  using const_reverse_iterator = FunctionList::const_reverse_iterator;

  // Module iterator methods.
  iterator begin() noexcept { return functions_.begin(); }
  const_iterator begin() const noexcept { return functions_.begin(); }
  iterator end() noexcept { return functions_.end(); }
  const_iterator end() const noexcept { return functions_.end(); }

  size_t size() const noexcept { return functions_.size(); }
  bool empty() const noexcept { return functions_.empty(); }
  Function* front() { return functions_.front().get(); }
  Function* back() { return functions_.back().get(); }

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

  /// Get the functions in this module.
  FunctionList& Functions() noexcept { return functions_; }
  const FunctionList& Functions() const noexcept { return functions_; }

  /// Print the content of this module.
  void Print(std::ostream& os) const override;

  Kind GetKind() const noexcept override { return Kind::Module; }

  static inline bool Classof(const Module* m) { return true; }
  static inline bool Classof(const IRObject* obj) {
    return obj->GetKind() == Kind::Module;
  }

 private:
  // TODO: Module Proto
  // TODO: args
  FunctionList functions_;
  ConstantList constants_;
};

} // namespace halo

#endif // HALO_LIB_IR_MODULE_H_