//===- input_writer.cc ----------------------------------------------------===//
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

#include "halo/lib/transforms/input_rewriter.h"

#include <unordered_set>

#include "halo/lib/ir/ir_builder.h"

namespace halo {

bool InputRewriter::RunOnBasicBlock(BasicBlock* bb) {
  if (inputs_.empty()) {
    return false;
  }
  bool changed = false;
  std::unordered_set<std::string> names(inputs_.begin(), inputs_.end());
  std::vector<Def> ops;
  ArgumentBuilder arg_builder(bb->GetParent());
  for (auto& inst : *bb) {
    auto it = names.find(inst->GetName());
    if (it != names.end()) {
      for (int i = 0, e = inst->GetNumOfResults(); i != e; ++i) {
        if (const auto& type = inst->GetResultsTypes()[i]; type.IsValid()) {
          auto arg = arg_builder.CreateArgument(
              inst->GetName() + (e == 1 ? "" : std::to_string(i)), type);
          inst->ReplaceAllUsesWith(i, Def(arg, 0));
          changed |= true;
        }
      }
    }
  }
  return changed;
}

} // end namespace halo