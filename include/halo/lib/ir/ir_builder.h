//===- ir_builder.h ---------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_IR_IR_BUILDER_H_
#define HALO_LIB_IR_IR_BUILDER_H_

#include <algorithm>
#include <type_traits>
#include <unordered_map>

#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/ir/basic_block.h"
#include "halo/lib/ir/function.h"
#include "halo/lib/ir/module.h"

namespace halo {

/// This is the base class for IRObject Builder classes.
/// It inserts objects into it's parent's list.
/// For efficient insertion, it maintains the correspoding iterators of each
/// object in the list indexed by the object's id. This requires the iterators
/// be stable with respect to insert/delete.
template <typename ParentObjectType, typename ContainerType>
class IRObjectBuilder {
 public:
  using ElementType = typename ContainerType::value_type;
  using Iterator = typename ContainerType::iterator;
  using Pointer = typename ElementType::pointer;
  using ConstPointer =
      typename std::add_pointer<const typename ElementType::element_type>::type;

  // The itertor cannot be invalidated after erase/insert because we cache the
  // iterators. C++17 has no traits for this property, so we just check if it
  // is a list or not. If other container is used, we can define a trait class.
  static_assert(std::is_same<ContainerType, std::list<ElementType>>::value,
                "Iterator of the container must be stable");

  explicit IRObjectBuilder(GlobalContext& ctx, ParentObjectType* parent,
                           ContainerType& list)
      : context_(ctx), parent_(parent), list_(list), insert_pt_(list.end()) {}
  explicit IRObjectBuilder(GlobalContext& ctx, ParentObjectType* parent,
                           ContainerType& list, ConstPointer insert_pt,
                           bool insert_before)
      : context_(ctx), parent_(parent), list_(list) {}

  ~IRObjectBuilder() = default;
  IRObjectBuilder(const IRObjectBuilder&) = delete;
  IRObjectBuilder(IRObjectBuilder&&) = delete;
  IRObjectBuilder operator=(const IRObjectBuilder&) = delete;
  IRObjectBuilder operator=(IRObjectBuilder&&) = delete;

  /// Set the insert point to the point before `position`.
  void SetInsertBefore(ConstPointer position) {
    insert_pt_ = FindIterator(position);
  }

  /// Set the insert point to the point after `position`.
  void SetInsertAfter(ConstPointer position) {
    SetInsertBefore(position);
    insert_pt_ = std::next(insert_pt_);
  }

  /// Insert `ptr` to the current insert position.
  Pointer Insert(Pointer ptr) {
    ElementType obj(ptr);
    Insert(std::move(obj));
    return ptr;
  }

  /// Insert `ptr` to the current insert position.
  Pointer Insert(ElementType val) {
    Pointer raw_ptr = val.get();
    insert_pt_ = std::next(InsertAt(insert_pt_, val));
    return raw_ptr;
  }

  GlobalContext& GetContext() const noexcept { return context_; }

  ParentObjectType* GetParent() const noexcept { return parent_; }

 private:
  /// Find the iterator of the pointer. To speedup the lookup, a cache is used.
  Iterator FindIterator(ConstPointer pos) {
    auto id = pos->GetId();
    if (id2it_.count(id)) {
      return id2it_[id];
    }
    // cache missed.
    Iterator it = std::find_if(list_.begin(), list_.end(), [id](const auto& i) {
      return i->GetId() == id;
    });
    HLCHECK(it != list_.end());
    id2it_[id] = it;
    return it;
  }

  Iterator InsertAt(Iterator it, ElementType& val) {
    Pointer raw_ptr = val.get();
    // Make sure the parent-son relationship is correct.
    HLCHECK(val->GetParent() == parent_);
    it = list_.insert(it, std::move(val));
    auto ret = id2it_.insert(std::make_pair(raw_ptr->GetId(), it));
    (void)ret;
    HLCHECK(ret.second && "No duplicated adding");
    return it;
  }

  GlobalContext& context_;
  ParentObjectType* parent_ = nullptr;
  ContainerType& list_;
  Iterator insert_pt_; // The insert point.
  std::unordered_map<int, Iterator> id2it_;
};

/// This class provides the APIs to create functions and insert them into
/// the current module.
class FunctionBuilder final
    : public IRObjectBuilder<Module, Module::FunctionList> {
 public:
  /// Constructs a IRBuilder for an existing module object.
  explicit FunctionBuilder(Module* module)
      : IRObjectBuilder(module->GetGlobalContext(), module,
                        module->Functions()) {}
  // New objects will be inserted before the `insert_point`.
  explicit FunctionBuilder(Module* module, const Function* insert_point)
      : IRObjectBuilder(module->GetGlobalContext(), module, module->Functions(),
                        insert_point, true) {}
  // New objects will be inserted before or after the `insert_point` based on
  // the value of `before_insert_pt`.
  explicit FunctionBuilder(Module* module, const Function* insert_point,
                           bool before_insert_pt)
      : IRObjectBuilder(module->GetGlobalContext(), module, module->Functions(),
                        insert_point, false) {}

  /// Create a new function and insert into current module.
  Function* CreateFunction(const std::string& name);

 private:
};

/// This class provides the APIs to create arguments for the current function.
class ArgumentBuilder final
    : public IRObjectBuilder<IRObject, Function::ArgumentList> {
 public:
  explicit ArgumentBuilder(Function* function)
      : IRObjectBuilder(function->GetGlobalContext(), function,
                        function->Args()) {}
  explicit ArgumentBuilder(BasicBlock* bb)
      : IRObjectBuilder(bb->GetGlobalContext(), bb, bb->Args()) {}

  /// Create a new argument and insert it to the end of current argument list.
  Argument* CreateArgument(const std::string& name);
  Argument* CreateArgument(const std::string& name, const Type& type);
};

/// This class provides the APIs to create basic blocks add them into current
/// function.
class BasicBlockBuilder final
    : public IRObjectBuilder<Function, Function::BasicBlockList> {
 public:
  explicit BasicBlockBuilder(Function* function)
      : IRObjectBuilder(function->GetGlobalContext(), function,
                        function->BasicBlocks()) {}

  /// Create a new basic block and append it to the end of current function.
  BasicBlock* CreateBasicBlock(const std::string& name);
};

/// This class provides the APIs to create constants add them into current
/// function or module.
class ConstantBuilder final
    : public IRObjectBuilder<IRObject, Function::ConstantList> {
 public:
  explicit ConstantBuilder(Function* parent)
      : IRObjectBuilder(parent->GetGlobalContext(), parent,
                        parent->Constants()) {}
  explicit ConstantBuilder(Module* parent)
      : IRObjectBuilder(parent->GetGlobalContext(), parent,
                        parent->Constants()) {}
  explicit ConstantBuilder(BasicBlock* parent)
      : IRObjectBuilder(parent->GetGlobalContext(), parent,
                        parent->Constants()) {}

  /// Create a new constant and append it to the end of current function or
  /// module using default data layout.
  Constant* CreateConstant(const std::string& name, const Type& type,
                           const void* data_ptr);

  /// Create a new constant with specified data layout and append it to the end
  /// of current function or module.
  Constant* CreateConstant(const std::string& name, const Type& type,
                           const DataLayout& data_layout, const void* data_ptr);

  /// Create a new constant from a vector of trivial types.
  template <typename T>
  Constant* CreateConstant(const std::string& name, const Type& type,
                           const std::vector<T>& v) {
    static_assert(std::is_trivial<T>(),
                  "container element must be trivial type");
    HLCHECK(Type::HasNativeType<T>(type));
    return CreateConstant(name, type, v.data());
  }

  Constant* Clone(const Constant& from);

  /// Create a new constant from a scalar by splating the value
  Constant* SplatConstant(const std::string& name, const Type& type,
                          const void* data_ptr);
  Constant* SplatConstantZero(const std::string& name, const Type& type);
};

/// This class provides the APIs to create Halo IRs and add them
/// into a basic block.
class IRBuilder final
    : public IRObjectBuilder<BasicBlock, BasicBlock::InstructionList> {
 public:
  explicit IRBuilder(BasicBlock* basic_block)
      : IRObjectBuilder(basic_block->GetGlobalContext(), basic_block,
                        basic_block->Instructions()) {}

  CustomInst* CreateCustom(const std::string& name, const std::vector<Def>& ops,
                           const int num_outs, const std::string& opcode);
  DummyInst* CreateDummy(const std::string& name, const std::vector<Def>& ops,
                         const int max_num_outputs, const std::string& opcode);
  TFExtensionInst* CreateTFExtension(const std::string& name,
                                     const std::vector<Def>& ops,
                                     const int num_outs,
                                     const std::string& opcode);
  ONNXExtensionInst* CreateONNXExtension(const std::string& name,
                                         const std::vector<Def>& ops,
                                         const int num_outs,
                                         const std::string& opcode);
  TFLITEExtensionInst* CreateTFLITEExtension(const std::string& name,
                                             const std::vector<Def>& ops,
                                             const int num_outs,
                                             const std::string& opcode);
  CAFFEExtensionInst* CreateCAFFEExtension(const std::string& name,
                                           const std::vector<Def>& ops,
                                           const int num_outs,
                                           const std::string& opcode);
  Instruction* CreateBinary(const std::string& name, const Def& op0,
                            const Def& op1, OpCode opcode,
                            KindPredicate pred = KindPredicate::INVALID);
  Instruction* Clone(const Instruction& from, const std::vector<Def>& ops);
#include "halo/lib/ir/ir_builder.h.inc"

 private:
};

} // namespace halo

#endif // HALO_LIB_IR_IR_BUILDER_H_
