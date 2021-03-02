//===- values.cc ----------------------------------------------------------===//
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

#include "halo/lib/ir/values.h"

#include <algorithm>

namespace halo {

/// Equal operator
bool Value::operator==(const Value& other) const noexcept {
  return owner_ == other.GetOwner() && idx_ == other.GetIdx();
}
/// Not-Equal operator
bool Value::operator!=(const Value& other) const noexcept {
  return owner_ != other.GetOwner() || idx_ != other.GetIdx();
}

/// Print out the info
void Value::Print(std::ostream& os) const {
  os << "<";
  if (IsNull()) {
    HLCHECK(0 && "Null Operand");
  } else {
    os << GetOwner()->GetName();
  }
  os << ", " << GetIdx() << ">:";
  GetOwner()->GetResultsTypes()[GetIdx()].Print(os);
}

/// Return the list iterator of a use.
UseList::iterator UseList::GetIterator(const Use& u) noexcept {
  auto it = uses_.begin();
  for (; it != uses_.end(); ++it) {
    if (*it == u) {
      break;
    }
  }
  return it;
}

/// Return true if it removes a use successfully.
bool UseList::RemoveUse(const Use& u) noexcept {
  auto it = GetIterator(u);
  if (it != uses_.end()) {
    uses_.erase(it);
    return true;
  }
  return false;
}

/// Return true if it contains a specific use.
bool UseList::HasUse(const Use& u) noexcept {
  auto it = GetIterator(u);
  return (it != uses_.end());
}

/// Return the uses of the def object
UseList& Def::GetUses() const {
  IRObject* def = GetDef();
  UseList& uses = def->GetIthResultUses(GetDefResultIdx());
  return uses;
}

/// Return the type of the def object.
const Type& Def::GetType() const {
  return GetOwner()->GetResultsTypes()[GetIdx()];
}

void Def::SetType(const Type& type) {
  GetOwner()->GetResultsTypes()[GetIdx()] = type;
}

/// Explicit constructor of the Object class.
IRObject::IRObject(GlobalContext& context, const std::string& name)
    : context_(context), name_(name) {
  id_ = context.GetGlobalCounter();
}

IRObject::IRObject(GlobalContext& context, const std::string& name,
                   const int num_of_results)
    : IRObject(context, name) {
  if (num_of_results > 0) {
    // Resize the results' types vectors.
    results_types_.resize(num_of_results);
    // Resize the results' uses vectors.
    results_uses_.resize(num_of_results);
  }
}

IRObject::IRObject(GlobalContext& context, const std::string& name,
                   const std::vector<Def>& operands, int num_of_results)
    : IRObject(context, name, num_of_results) {
  for (auto& operand : operands) {
    AddOneOperand(operand);
  }
}

IRObject::IRObject(GlobalContext& context, const std::string& name,
                   const std::vector<Def>& operands)
    : IRObject(context, name) {
  for (auto& operand : operands) {
    AddOneOperand(operand);
  }
}

void IRObject::SetNumOfResults(unsigned int num_of_results) {
  results_types_.resize(num_of_results);
  results_uses_.resize(num_of_results);
}

UseList& IRObject::GetIthResultUses(size_t i) {
  if (HasVariadicReturns() && i >= results_uses_.size()) {
    results_types_.resize(i + 1);
    // Resize the results' uses vectors.
    results_uses_.resize(i + 1);
  } else {
    HLCHECK(i < results_uses_.size() && "access uselist out of bound.");
  }
  return results_uses_[i];
}

bool IRObject::HasValidResultTypes() const {
  if (has_dynamic_type_) {
    return false;
  }
  for (const Type& type : results_types_) {
    if (!type.IsValid()) {
      return false;
    }
  }
  return true;
}

bool IRObject::ComputeResultTypes() {
  HLCHECK(0 && "Unreachable");
  return false;
}

/// Append one operand to the last of operand list
/// and update operand's uselist
void IRObject::AddOneOperand(const Def& one) {
  if (!one.IsNull()) {
    size_t use_idx = operands_.size();
    UseList& uses = one.GetUses();
    Use use(this, use_idx);
    uses.AddUse(use);
  }
  operands_.push_back(one);
}

/// Destructor of the IRObject class
IRObject::~IRObject() {}

bool IRObject::CheckFailed(bool cond, const std::string& message) const {
  if (!(cond)) {
    CheckFailed(message);
    return true;
  }
  return false;
}

/// Print operands
void IRObject::PrintOperands(std::ostream& os) const {
  size_t num_of_operands = GetNumOfOperands();
  if (num_of_operands == 0) {
    return;
  }
  GetOperand(0).Print(os);
  for (size_t i = 1; i < num_of_operands; ++i) {
    os << ", ";
    GetOperand(i).Print(os);
  }
}

/// Print attributes
void IRObject::PrintAttributes(std::ostream& os) const {
  size_t num_of_attributes = GetNumOfAttributes();
  if (num_of_attributes == 0) {
    return;
  }
  attributes_.at(0)->Print(os);
  for (size_t i = 1; i < num_of_attributes; ++i) {
    os << ", ";
    attributes_.at(i)->Print(os);
  }
}

size_t IRObject::GetNumberOfUses() const {
  size_t n = 0;
  for (auto const& uses : GetResultsUses()) {
    n += uses.GetNumOfUses();
  }
  return n;
}

void IRObject::ResetOperand(size_t idx) {
  ReplaceOperandWith(idx, Def::GetUndefined());
}

void IRObject::DropAllOperands() {
  for (size_t i = 0; i < operands_.size(); ++i) {
    ResetOperand(i);
  }
  operands_.clear();
}

void IRObject::ReplaceOperandWith(size_t idx, const Def& new_def) {
  Def old_op = GetOperand(idx);
  if (old_op == new_def) {
    return;
  }
  Use use(this, idx);
  if (!old_op.IsNull()) {
    // Drop the use.
    UseList& uses = old_op.GetUses();
    HLCHECK(uses.HasUse(use));
    uses.RemoveUse(use);
  }
  if (!new_def.IsNull()) {
    new_def.GetUses().AddUse(use);
  }
  operands_[idx] = new_def;
}

void IRObject::ReplaceAllUsesWith(size_t result_idx, const Def& new_def) {
  // Copy the use list because the list will be updated by ReplaceOperandWith()
  // function.
  auto uses = GetIthResultUses(result_idx).GetUses();
  for (Use& use : uses) {
    if (use != new_def) {
      use.GetOwner()->ReplaceOperandWith(use.GetUseOperandIdx(), new_def);
    }
  }
}

void IRObject::ReplaceAllUsesWith(const std::vector<Def>& new_defs) {
  for (size_t i = new_defs.size(); i < results_types_.size(); ++i) {
    HLCHECK(GetResultsUses()[i].empty());
  }
  for (size_t i = 0, e = new_defs.size(); i < e; ++i) {
    ReplaceAllUsesWith(i, new_defs[i]);
  }
}

} // namespace halo