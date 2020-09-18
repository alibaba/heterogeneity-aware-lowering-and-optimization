//===- pass.h ---------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_PASS_PASS_H_
#define HALO_LIB_PASS_PASS_H_

#include <iostream>
#include <string>

#include "halo/lib/ir/basic_block.h"
#include "halo/lib/ir/function.h"
#include "halo/lib/ir/module.h"

namespace halo {

// An abstract interface for run optimization.
class Pass {
 public:
  /// Pass types.
  enum class PassType {
    MODULE,
    FUNCTION,
    BASICBLOCK,
  };

  explicit Pass(const std::string& name) : name_(name) {}

  virtual ~Pass() = default;

  virtual bool IsPassManager() const noexcept { return false; }
  const std::string& Name() const noexcept { return name_; }
  virtual void Print(std::ostream& os) const { os << name_ << "\n"; }

 private:
  const std::string name_;
};

/// A base class for all module-level passes.
class ModulePass : public Pass {
 public:
  static constexpr PassType Type = PassType::MODULE;
  explicit ModulePass(const std::string& name) : Pass(name) {}
  virtual bool RunOnModule(Module* module) = 0;
};

class FunctionPass : public Pass {
 public:
  static constexpr PassType Type = PassType::FUNCTION;
  explicit FunctionPass(const std::string& name) : Pass(name) {}
  virtual bool RunOnFunction(Function* function) = 0;
};

class BasicBlockPass : public Pass {
 public:
  static constexpr PassType Type = PassType::BASICBLOCK;
  explicit BasicBlockPass(const std::string& name) : Pass(name) {}
  virtual bool RunOnBasicBlock(BasicBlock* bb) = 0;
};

} // namespace halo.

#endif // HALO_LIB_PASS_PASS_H_