//===- transforms_util.cc -------------------------------------------------===//
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

#include "halo/lib/transforms/transforms_util.h"

namespace halo {

bool AppendReturnInst(BasicBlock* bb) {
  IRBuilder builder(bb);
  if (bb->empty() || bb->back()->GetOpCode() == OpCode::RETURN) {
    return false;
  }
  std::vector<Def> outputs;
  for (auto& inst_t : *bb) {
    Instruction* inst = inst_t.get();
    if (inst->GetNumberOfUses() == 0 && inst->GetOpCode() != OpCode::RETURN &&
        inst->GetDependents().empty()) {
      outputs.push_back(*inst);
    }
  }
  if (!outputs.empty()) {
    builder.SetInsertAfter(bb->back());
    builder.CreateReturn("output", outputs);
    return true;
  }
  return false;
}

std::vector<int64_t> GetExtends(const std::vector<int64_t>& dims) {
  std::vector<int64_t> extends(dims.size(), 1);
  for (int64_t i = dims.size() - 2; i >= 0; --i) {
    extends[i] = extends[i + 1] * dims[i + 1];
  }
  return extends;
}

void SplitStringToInt64List(const std::string& src, std::vector<int64_t>* dst,
                            const std::string& delimiter) {
  std::string::size_type pos2 = src.find(delimiter);
  std::string::size_type pos1 = 0;

  while (std::string::npos != pos2) {
    dst->push_back(std::stol(src.substr(pos1, pos2 - pos1)));

    pos1 = pos2 + delimiter.size();
    pos2 = src.find(delimiter, pos1);
  }
  if (pos1 != src.length()) {
    dst->push_back(std::stol(src.substr(pos1)));
  }
}

bool HasAttribute(const Instruction& inst, const std::string& name) {
  for (const auto& attr : inst.GetAttributes()) {
    if (attr->GetName() == name) {
      return true;
    }
  }
  return false;
}

std::pair<bool, int64_t> GetAvailIntegerResult(const IRObject& inst,
                                               int64_t idx) {
  return {false, -1};
}

static std::pair<bool, int64_t> GetAvailIntegerResult(const Constant& inst,
                                                      int64_t idx) {
  return {true, inst.GetDataAsInt64(idx)};
}

static std::pair<bool, int64_t> GetAvailIntegerResultForConcats(
    const std::vector<Def>& operands, int64_t idx) {
  // For stack(x0, x1, x2, ..), some component might be constant.
  int64_t start = 0;
  for (auto& op : operands) {
    const auto& ty = op.GetType();
    if (!ty.IsValid() || !ty.IsStaticShape()) {
      return {false, -1};
    }
    int64_t end = start + ty.GetTotalNumOfElements();
    if (idx >= start && idx < end) {
      return GetAvailIntegerResult(op, idx - start);
    }
    start = end;
  }
  return {false, -1};
}

static std::pair<bool, int64_t> GetAvailIntegerResult(const ConcatInst& inst,
                                                      int64_t idx) {
  return GetAvailIntegerResultForConcats(inst.GetOperands(), idx);
}

static std::pair<bool, int64_t> GetAvailIntegerResult(const StackInst& inst,
                                                      int64_t idx) {
  return GetAvailIntegerResultForConcats(inst.GetOperands(), idx);
}

static std::pair<bool, int64_t> GetAvailIntegerResult(const ShapeInst& inst,
                                                      int64_t idx) {
  // shape(x) returns [d0, d1, d2, ...]. If the element is non-negative, it is a
  // contant.
  const auto& value_type = inst.GetOperand(0).GetType();
  if (!value_type.IsValid() || idx < 0 ||
      idx >= static_cast<int64_t>(value_type.GetNumOfDims())) {
    return {false, -1};
  }
  auto x = value_type.GetNumOfElementsInDim(idx);
  return {x >= 0, x};
}

std::pair<bool, int64_t> GetAvailIntegerResult(const Def& op, int64_t idx) {
  if (IsA<Constant>(op)) {
    return GetAvailIntegerResult(*DynCast<Constant>(op), idx);
  }
  if (IsA<StackInst>(op)) {
    return GetAvailIntegerResult(*DynCast<StackInst>(op), idx);
  }
  if (IsA<ConcatInst>(op)) {
    return GetAvailIntegerResult(*DynCast<ConcatInst>(op), idx);
  }

  if (IsA<ShapeInst>(op)) {
    return GetAvailIntegerResult(*DynCast<ShapeInst>(op), idx);
  }
  return {false, -1};
}

} // end namespace halo
