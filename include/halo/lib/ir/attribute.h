//===- attribute.h ----------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_ATTRIBUTE_H_
#define HALO_LIB_IR_ATTRIBUTE_H_

#include <memory>
#include <string>
#include <vector>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/global_context.h"

namespace halo {

class Function;
class BasicBlock;

/// Attribute represents a known-contant value of an IRObject.
/// This ia an abstract base class. The concrete subclasses contain the
/// storage of various data types.
class Attribute {
 public:
#include "halo/lib/ir/attribute.h.inc"
  Attribute(const std::string& name);
  virtual ~Attribute() {}
  const std::string& GetName() const noexcept { return name_; };
  void SetName(const std::string& name) noexcept { name_ = name; }
  virtual AttrKind GetKind() const = 0;
  virtual const void* GetDataImpl() const = 0;
  virtual void* GetDataImpl() = 0;
  virtual std::unique_ptr<Attribute> Clone() const = 0;

  /// Print out the info
  virtual void Print(std::ostream& os) const = 0;
  /// Print the info to the debug output.
  virtual void Dump() const { Print(GlobalContext::Dbgs()); };

 private:
  std::string name_;
};

} // namespace halo

#endif // HALO_LIB_IR_ATTRIBUTE_H_