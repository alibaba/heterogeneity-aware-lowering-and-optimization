//===- values.h -------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_VALUE_H_
#define HALO_LIB_IR_VALUE_H_

#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "halo/lib/framework/global_context.h"
#include "halo/lib/framework/type.h"
#include "halo/lib/ir/attribute.h"

namespace halo {

class IRObject;

/// This class defines a Value, a pair of <IRObject*, int>.
/// Since one object could include multiple values, an idx is required to
/// indicate the indice.
class Value {
 public:
  Value() = default;
  explicit Value(IRObject* obj, int idx) : owner_(obj), idx_(idx) {
    HLCHECK(obj != nullptr);
  }
  Value(const Value&) = default;
  Value(Value&&) = default;
  Value& operator=(const Value&) = default;
  Value& operator=(Value&&) = default;
  virtual ~Value() = default;

  /// Return true if this value is NULL.
  bool IsNull() const noexcept { return owner_ == nullptr; }
  /// Return the owner.
  IRObject* GetOwner() const noexcept { return owner_; }
  /// Return the index.
  int GetIdx() const noexcept { return idx_; }

  /// Equal operator
  bool operator==(const Value& other) const noexcept;
  /// Not-Equal operator
  bool operator!=(const Value& other) const noexcept;

  /// Print out the info
  void Print(std::ostream& os) const;

  /// Print the info to the debug output.
  virtual void Dump() const { Print(GlobalContext::Dbgs()); };

 private:
  IRObject* owner_ = nullptr;
  int idx_ = 0;
};

/// This class defines a value use. It is simple a wrapper of
/// of the base Value class.
/// A use instance indicates the use value, and its operand indice,
/// indicating as <IRObject* use_, int use_operand_idx_>.
class Use final : public Value {
 public:
  Use() = delete;
  explicit Use(IRObject* obj, int idx) : Value(obj, idx) {}

  ~Use() = default;

  /// Return the use object.
  IRObject* GetUse() const noexcept { return GetOwner(); }
  /// Return the use operand index.
  int GetUseOperandIdx() const noexcept { return GetIdx(); }
};

/// This class defines a use list of a value.
/// It is typically a very short list (std::list is fine).
class UseList {
 public:
  UseList() = default;
  ~UseList() = default;

  /// Return the uses list.
  std::list<Use>& GetUses() noexcept { return uses_; }
  const std::list<Use>& GetUses() const noexcept { return uses_; }

  /// Return the number of uses.
  size_t GetNumOfUses() const noexcept { return uses_.size(); }

  /// Return true if the list has some uses.
  bool HasUses() const noexcept { return !uses_.empty(); }

  /// Return true if the list has only one use.
  bool HasOneUse() const noexcept { return uses_.size() == 1; }

  /// Add a use.
  void AddUse(const Use& u) { uses_.push_back(u); }

  /// Return true if it removes a use successfully.
  bool RemoveUse(const Use& u) noexcept;

  /// Return true if it contains a specifi use.
  bool HasUse(const Use& u) noexcept;

  // Iteration over the uses in the list.
  using iterator = std::list<Use>::iterator;
  using const_iterator = std::list<Use>::const_iterator;
  using reverse_iterator = std::list<Use>::reverse_iterator;
  using const_reverse_iterator = std::list<Use>::const_reverse_iterator;

  /// UseList iterator methods.
  ///
  inline iterator begin() noexcept { return uses_.begin(); }
  inline const_iterator begin() const noexcept { return uses_.begin(); }
  inline iterator end() noexcept { return uses_.end(); }
  inline const_iterator end() const noexcept { return uses_.end(); }
  inline reverse_iterator rbegin() noexcept { return uses_.rbegin(); }
  inline const_reverse_iterator rbegin() const noexcept {
    return uses_.rbegin();
  }
  inline reverse_iterator rend() noexcept { return uses_.rend(); }
  inline const_reverse_iterator rend() const noexcept { return uses_.rend(); }

  inline size_t size() const noexcept { return uses_.size(); }
  inline bool empty() const noexcept { return uses_.empty(); }
  inline const Use& front() const { return uses_.front(); }
  inline Use& front() { return uses_.front(); }
  inline const Use& back() const { return uses_.back(); }
  inline Use& back() { return uses_.back(); }

 private:
  std::list<Use> uses_;

  /// Return the list iterator of a use.
  iterator GetIterator(const Use& u) noexcept;
};

/// This class defines a value def. It is simple a wrapper of
/// of the base Value class.
/// An operand is a result item of a def Value, indicating as
/// <IRObject* def_, int def_result_idx_>.
class Def final : public Value {
 public:
  explicit Def(IRObject* obj, int idx) : Value(obj, idx) {}
  ~Def() = default;

  /// Return the def object.
  IRObject* GetDef() const noexcept { return GetOwner(); }
  /// Return the def result index.
  int GetDefResultIdx() const noexcept { return GetIdx(); }
  /// Return the uselist of the def
  UseList& GetUses() const;
  /// Return the type of the def.
  const Type& GetType() const;
  /// Set the type of the def.
  void SetType(const Type&);

  /// Get the undefined value.
  inline static Def GetUndefined() { return Def(); }

 private:
  Def() : Value() {}
};

/// This class defines a base class for all computable objects in the IR system.
/// It is the base for instruciton, basic block, and function.
class IRObject {
 public:
  // constructors
  IRObject() = delete;
  explicit IRObject(GlobalContext& context, const std::string& name);
  explicit IRObject(GlobalContext& context, const std::string& name,
                    int num_of_results);
  explicit IRObject(GlobalContext& context, const std::string& name,
                    const std::vector<Def>& operands);
  explicit IRObject(GlobalContext& context, const std::string& name,
                    const std::vector<Def>& operands, int num_of_results);

  // destructor
  virtual ~IRObject();

  enum class Kind {
    Module,
    Function,
    Argument,
    BasicBlock,
    Constant,
    Instruction,
    Invalid
  };

  /// Get the global context in which this data type was uniqued.
  GlobalContext& GetGlobalContext() const noexcept { return context_; }

  /// Set the name.
  void SetName(const std::string& name) noexcept { name_ = name; }
  /// Get the name.
  std::string GetName() const noexcept { return name_; }

  /// Get the kind of the IR object.
  virtual Kind GetKind() const noexcept = 0;

  /// Get the Id.
  uint64_t GetId() const noexcept { return id_; }

  /// Return the operands list.
  std::vector<Def>& GetOperands() noexcept { return operands_; }
  const std::vector<Def>& GetOperands() const noexcept { return operands_; }

  /// Return the i-th operand.
  Def GetOperand(size_t idx) const {
    HLCHECK(idx < operands_.size() && "access operand list out of bound.");
    return operands_[idx];
  }

  /// Return the IRObject that produces value for the i-th operand.
  IRObject* GetOperandDefObj(size_t idx) const {
    return GetOperand(idx).GetDef();
  }

  /// Return the number of operands.
  size_t GetNumOfOperands() const noexcept { return operands_.size(); }

  /// Append one operand and update its uselist.
  void AddOneOperand(const Def& one);

  /// Append list of operands.
  void AddOperands(const std::vector<Def>& defs) {
    for (const Def& one : defs) {
      AddOneOperand(one);
    }
  }

  /// Reset `idx`-th operand to null. The number of operands remains unchanged.
  void ResetOperand(size_t idx);

  /// Drop all operands and reset the operand counter.
  void DropAllOperands();

  /// Set the has_dynamic_type to true.
  void SetDynamicType() noexcept { has_dynamic_type_ = true; }

  /// Return true if some result type is dynamic.
  bool HasDynamicType() const noexcept { return has_dynamic_type_; }

  /// Return the number of results.
  size_t GetNumOfResults() const noexcept { return results_types_.size(); }

  /// Return the results' types.
  std::vector<Type>& GetResultsTypes() noexcept { return results_types_; }
  const std::vector<Type>& GetResultsTypes() const noexcept {
    return results_types_;
  }

  /// Return the first result type.
  const Type& GetResultType() const {
    HLCHECK(!results_types_.empty());
    return results_types_[0];
  }

  /// Return the i-th result type.
  const Type& GetResultType(size_t idx) const {
    HLCHECK(idx < results_types_.size());
    return results_types_[idx];
  }

  /// Returns true if the object holds valid result types.
  bool HasValidResultTypes() const;

  /// Retuns true if the object has variadic returns.
  bool HasVariadicReturns() const noexcept { return has_variadic_returns_; }

  /// Compute the result type. Returns false if failed.
  virtual bool ComputeResultTypes();

  /// Return the results' use_list.
  std::vector<UseList>& GetResultsUses() noexcept { return results_uses_; }
  const std::vector<UseList>& GetResultsUses() const noexcept {
    return results_uses_;
  }

  void SetNumOfResults(unsigned num_of_results);

  /// Return ith result uselist
  UseList& GetIthResultUses(size_t i) {
    HLCHECK(i < results_uses_.size() && "access uselist out of bound.");
    return results_uses_[i];
  }

  /// Return the number of attributes.
  size_t GetNumOfAttributes() const noexcept { return attributes_.size(); }

  /// Return the total number of uses for all results.
  size_t GetNumberOfUses() const;

  /// Return the attributes list.
  std::vector<std::unique_ptr<Attribute>>& GetAttributes() noexcept {
    return attributes_;
  }

  /// Return the attributes list.
  const std::vector<std::unique_ptr<Attribute>>& GetAttributes() const
      noexcept {
    return attributes_;
  }

  /// Append one attribute.
  void AddOneAttribute(std::unique_ptr<Attribute> one) {
    attributes_.push_back(std::move(one));
  }

  /// Append multiple attributes.
  void AddAttributes(std::vector<std::unique_ptr<Attribute>>& attributes) {
    for (std::unique_ptr<Attribute>& one : attributes) {
      AddOneAttribute(std::move(one));
    }
  }

  /// Replace all uses of this instruction's i-th output with `new_def`.
  void ReplaceAllUsesWith(size_t result_idx, const Def& new_def);

  /// Replace all uses of this instruction's outputs with `new_defs`.
  void ReplaceAllUsesWith(const std::vector<Def>& new_defs);

  /// Replace `idx-th` operand with `new_def`.
  void ReplaceOperandWith(size_t idx, const Def& new_def);

  /// Convert IRObject* to Def(IRObject*, 0).
  operator Def() { return Def{this, 0}; }
  Def operator[](int idx) { return Def{this, idx}; }

  /// Print out the info
  virtual void Print(std::ostream& ss) const = 0;

  /// Print operands
  virtual void PrintOperands(std::ostream& os) const;

  /// Print attributes
  virtual void PrintAttributes(std::ostream& os) const;

  /// Print the info to the debug output.
  virtual void Dump() const {
    std::ostream& os = context_.Dbgs();
    Print(os);
  };

  /// Check IRObject, if check fails, print an error message.
  bool CheckFailed(bool cond, const std::string& message) const;
  /// Check IRObject, if check fails, print an error message
  /// and dump relevant IR
  template <typename T1, typename... Ts>
  bool CheckFailed(bool cond, const std::string& message, const T1& v1,
                   const Ts&... vs) const {
    if (!(cond)) {
      CheckFailed(message);
      DumpTs(v1, vs...);
      return true;
    }
    return false;
  }

  static inline bool Classof(const IRObject* obj) { return true; }

 protected:
  void SetVariadicReturns(bool flag) noexcept { has_variadic_returns_ = flag; }

 private:
  void CheckFailed(const std::string& message) const {
    std::ostream& os = context_.Dbgs();
    os << message << '\n';
  }

  template <typename T1, typename... Ts>
  void DumpTs(const T1& v1, const Ts&... vs) const {
    v1->Dump();
    DumpTs(vs...);
  }

  template <typename... Ts>
  void DumpTs() const {}

  GlobalContext& context_;
  std::string name_ = "";

  // Unique Id
  uint64_t id_ = -1;

  // Operands
  std::vector<Def> operands_;

  // Dynamic type flag
  bool has_dynamic_type_ = false;

  // Variadic returns
  bool has_variadic_returns_ = false;

  // Results' types
  std::vector<Type> results_types_;

  // Results' use_list
  std::vector<UseList> results_uses_;

  // Attribute list
  std::vector<std::unique_ptr<Attribute>> attributes_;
};

template <typename T_TO>
bool IsA(const halo::Def& def) {
  return IsA<T_TO>(def.GetOwner());
}

template <typename T_TO>
T_TO* DynCast(const Def& def) {
  return DynCast<T_TO, IRObject>(def.GetOwner());
}

} // namespace halo

namespace std {
template <>
struct hash<halo::Value> {
  size_t operator()(const halo::Value& v) const {
    return hash<void*>()(v.GetOwner()) ^ hash<int>()(v.GetIdx());
  }
};

template <>
struct hash<halo::Def> {
  size_t operator()(const halo::Def& v) const {
    return hash<void*>()(v.GetOwner()) ^ hash<int>()(v.GetIdx());
  }
};

} // namespace std

#endif // HALO_LIB_IR_VALUE_H_