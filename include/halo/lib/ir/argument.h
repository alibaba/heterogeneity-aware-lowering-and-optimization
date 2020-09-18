//===- argument.h -----------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_ARGUMENT_H_
#define HALO_LIB_IR_ARGUMENT_H_

#include "halo/lib/ir/values.h"

namespace halo {

class Function;

/// This class represents an arugment of a function.
class Argument final : public IRObject {
 public:
  Argument() = delete;
  Argument(const Argument&) = delete;
  Argument(Argument&&) = delete;
  Argument operator=(const Argument&) = delete;
  Argument operator=(Argument&&) = delete;
  ~Argument() = default;

  explicit Argument(GlobalContext& context, const std::string& name);

  explicit Argument(GlobalContext& context, const std::string& name,
                    const Type& type);

  /// Get the parent function of this argument.
  Function* GetParent() const noexcept { return parent_function_; }

  /// Set the type of this argument.
  void SetType(const Type& type);

  Kind GetKind() const noexcept override { return Kind::Argument; }

  static inline bool Classof(const Argument* arg) { return true; }
  static inline bool Classof(const IRObject* obj) {
    return obj->GetKind() == Kind::Argument;
  }

  void Print(std::ostream& os) const override;

 private:
  Function* parent_function_ = nullptr;

  friend class ArgumentBuilder;
};

} // namespace halo

#endif // HALO_LIB_IR_ARGUMENT_H_