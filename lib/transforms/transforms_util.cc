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

#include "halo/lib/ir/ir_builder.h"

namespace halo {

bool AppendReturnInst(BasicBlock* bb) {
  IRBuilder builder(bb);
  if (bb->empty() || bb->back()->GetOpCode() == OpCode::RETURN) {
    return false;
  }
  std::vector<Def> outputs;
  for (auto& inst_t : *bb) {
    Instruction* inst = inst_t.get();
    if (inst->GetNumberOfUses() == 0 && inst->GetOpCode() != OpCode::RETURN) {
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

} // end namespace halo