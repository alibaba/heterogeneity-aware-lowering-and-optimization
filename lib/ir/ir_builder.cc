//===- ir_builder.cc ------------------------------------------------------===//
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

#include "halo/lib/ir/ir_builder.h"

#include <algorithm>

#include "halo/lib/framework/data_layout.h"

namespace halo {

Function* FunctionBuilder::CreateFunction(const std::string& name) {
  auto func = std::make_unique<Function>(GetContext(), name);
  func->parent_module_ = GetParent();
  // By default, the first function is entry function
  func->SetAsEntryFunction(func->parent_module_->Functions().empty());
  return Insert(std::move(func));
}

BasicBlock* BasicBlockBuilder::CreateBasicBlock(const std::string& name) {
  auto bb = std::make_unique<BasicBlock>(GetContext(), name);
  bb->parent_function_ = GetParent();
  return Insert(std::move(bb));
}

Argument* ArgumentBuilder::CreateArgument(const std::string& name) {
  return CreateArgument(name, Type{DataType::INVALID});
}

Argument* ArgumentBuilder::CreateArgument(const std::string& name,
                                          const Type& type) {
  auto arg = std::make_unique<Argument>(GetContext(), name, type);
  arg->parent_function_ = GetParent();
  return Insert(std::move(arg));
}

Constant* ConstantBuilder::CreateConstant(const std::string& name,
                                          const Type& type,
                                          const DataLayout& data_layout,
                                          const void* data_ptr) {
  auto c = std::make_unique<Constant>(GetContext(), name, type, data_layout,
                                      data_ptr);
  c->parent_ = GetParent();
  return Insert(std::move(c));
}

Constant* ConstantBuilder::CreateConstant(const std::string& name,
                                          const Type& type,
                                          const void* data_ptr) {
  return CreateConstant(name, type, GetContext().GetDefaultDataLayout(),
                        data_ptr);
}

Constant* ConstantBuilder::SplatConstant(const std::string& name,
                                         const Type& type,
                                         const void* data_ptr) {
  auto c = std::make_unique<Constant>(GetContext(), name, type,
                                      GetContext().GetDefaultDataLayout(),
                                      data_ptr, true /*do_splat */);
  c->parent_ = GetParent();
  return Insert(std::move(c));
}

Constant* ConstantBuilder::SplatConstantZero(const std::string& name,
                                             const Type& type) {
  DataType dt = type.GetDataType();
  switch (dt) {
    case DataType::INT32: {
      int v = 0;
      return SplatConstant(name, type, &v);
    }
    case DataType::INT64: {
      int64_t v = 0;
      return SplatConstant(name, type, &v);
    }
    default: {
      float v = 0.0;
      return SplatConstant(name, type, &v);
    }
  }
}

Constant* ConstantBuilder::Clone(const Constant& from) {
  auto c = std::make_unique<Constant>(from);
  c->parent_ = GetParent();
  return Insert(std::move(c));
}

CustomInst* IRBuilder::CreateCustom(const std::string& name,
                                    const std::vector<Def>& ops,
                                    const int num_outs,
                                    const std::string& opcode) {
  auto inst =
      std::make_unique<CustomInst>(GetContext(), name, ops, num_outs, opcode);
  inst->parent_basic_block_ = GetParent();
  CustomInst* ret = inst.get();
  Insert(std::move(inst));
  return ret;
}

TFExtensionInst* IRBuilder::CreateTFExtension(const std::string& name,
                                              const std::vector<Def>& ops,
                                              const int num_outs,
                                              const std::string& opcode) {
  auto inst = std::make_unique<TFExtensionInst>(GetContext(), name, ops,
                                                num_outs, opcode);
  inst->parent_basic_block_ = GetParent();
  TFExtensionInst* ret = inst.get();
  Insert(std::move(inst));
  return ret;
}

ONNXExtensionInst* IRBuilder::CreateONNXExtension(const std::string& name,
                                                  const std::vector<Def>& ops,
                                                  const int num_outs,
                                                  const std::string& opcode) {
  auto inst = std::make_unique<ONNXExtensionInst>(GetContext(), name, ops,
                                                  num_outs, opcode);
  inst->parent_basic_block_ = GetParent();
  ONNXExtensionInst* ret = inst.get();
  Insert(std::move(inst));
  return ret;
}

TFLITEExtensionInst* IRBuilder::CreateTFLITEExtension(
    const std::string& name, const std::vector<Def>& ops, const int num_outs,
    const std::string& opcode) {
  auto inst = std::make_unique<TFLITEExtensionInst>(GetContext(), name, ops,
                                                    num_outs, opcode);
  inst->parent_basic_block_ = GetParent();
  TFLITEExtensionInst* ret = inst.get();
  Insert(std::move(inst));
  return ret;
}

CAFFEExtensionInst* IRBuilder::CreateCAFFEExtension(const std::string& name,
                                                    const std::vector<Def>& ops,
                                                    const int num_outs,
                                                    const std::string& opcode) {
  auto inst = std::make_unique<CAFFEExtensionInst>(GetContext(), name, ops,
                                                   num_outs, opcode);
  inst->parent_basic_block_ = GetParent();
  CAFFEExtensionInst* ret = inst.get();
  Insert(std::move(inst));
  return ret;
}

Instruction* IRBuilder::Clone(const Instruction& from,
                              const std::vector<Def>& ops) {
  auto inst = from.Clone();
  inst->DropAllOperands();
  inst->parent_basic_block_ = GetParent();
  auto ret = inst.get();
  inst->AddOperands(ops);
  Insert(std::move(inst));
  return ret;
}

Instruction* IRBuilder::CreateBinary(const std::string& name, const Def& op0,
                                     const Def& op1, OpCode opcode,
                                     KindPredicate pred) {
  switch (opcode) {
    case OpCode::ADD: {
      return CreateAdd(name, op0, op1);
    }
    case OpCode::SUB: {
      return CreateSub(name, op0, op1);
    }
    case OpCode::MUL: {
      return CreateMul(name, op0, op1);
    }
    case OpCode::DIV: {
      return CreateDiv(name, op0, op1);
    }
    case OpCode::MAXIMUM: {
      return CreateMaximum(name, op0, op1);
    }
    case OpCode::MINIMUM: {
      return CreateMinimum(name, op0, op1);
    }
    case OpCode::POW: {
      return CreatePow(name, op0, op1);
    }
    case OpCode::CMP: {
      auto new_inst = CreateCmp(name, op0, op1);
      new_inst->SetPredicator(pred);
      return new_inst;
    }
    default: {
      HLCHECK(false && "Unsupported binary opcode");
      return nullptr;
    }
  }
}

DummyInst* IRBuilder::CreateDummy(const std::string& name,
                                  const std::vector<Def>& ops,
                                  const int max_num_outputs,
                                  const std::string& opcode) {
  auto inst = std::make_unique<DummyInst>(GetContext(), name, ops,
                                          max_num_outputs, opcode);
  inst->parent_basic_block_ = GetParent();
  DummyInst* ret = inst.get();
  Insert(std::move(inst));
  return ret;
}

#include "halo/lib/ir/ir_builder.cc.inc"

} // namespace halo