//===- output_writer.cc ---------------------------------------------------===//
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

#include "halo/lib/transforms/output_rewriter.h"

#include <unordered_set>

#include "halo/lib/ir/ir_builder.h"

namespace halo {

bool OutputRewriter::RunOnFunction(Function* func) {
  if (outputs_.empty()) {
    return false;
  }
  auto ret = func->GetReturnInst();
  if (ret == nullptr) {
    return false;
  }
  // names are in form of xyz or xyz:0
  std::unordered_map<std::string, std::unordered_set<int>> names;
  for (const auto& output : outputs_) {
    if (auto pos = output.find_last_of(':'); pos == std::string::npos) {
      names[output] = {};
    } else {
      constexpr int base = 10;
      auto idx = std::strtol(&output[pos + 1], nullptr, base);
      names[output.substr(0, pos)].insert(idx);
    }
  }

  std::vector<Def> ops;
  for (auto& bb : *func) {
    for (auto& inst : *bb) {
      auto it = names.find(inst->GetName());
      if (it != names.end()) {
        if (it->second.empty()) {
          for (int i = 0, e = inst->GetNumOfResults(); i < e; ++i) {
            ops.push_back(Def(inst.get(), i));
          }
        } else {
          for (auto idx : it->second) {
            ops.push_back(Def(inst.get(), idx));
          }
        }
        names.erase(it);
      }
    }
  }
  // Sanity check.
  for (const auto& kv : names) {
    std::cerr << "Unrecognized output name: " << kv.first << "\n";
  }
  HLCHECK(names.empty());
  ret->DropAllOperands();
  ret->AddOperands(ops);
  outputs_.clear(); // prevent reentry
  return true;
}

} // end namespace halo