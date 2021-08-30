//===- sharding.cc --------------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "halo/lib/transforms/sharding.h"

#include <unordered_map>
#include <unordered_set>

#include "halo/lib/ir/ir_builder.h"

namespace halo {

static unsigned CountNonConstOps(const Function& func) {
  unsigned ret = func.Args().size();
  for (auto& bb : func) {
    for (auto& ir : *bb) {
      ret += IsA<Constant>(*ir) ? 0 : 1;
    }
  }
  return ret;
}

// Simple sharding scheme: try to get equal shards of op.
static std::unordered_map<IRObject*, int> GetSimpleSharding(
    const Function& func, unsigned num_shards) {
  std::unordered_map<IRObject*, int> shardings;
  auto num_ops = CountNonConstOps(func);
  const unsigned threshold = (num_shards == 0)
                                 ? std::numeric_limits<unsigned>::max()
                                 : (num_ops + num_shards - 1) / num_shards;
  unsigned curr_shard = 0;
  unsigned allocated = 0;
  auto is_avail = [&shardings](const IRObject* n) {
    for (auto& op : n->GetOperands()) {
      if (!IsA<Constant>(op) && shardings.count(op.GetOwner()) == 0) {
        return false;
      }
    }
    return true;
  };

  // BFS visit.
  std::unordered_set<IRObject*> workset;
  for (auto& arg : func.Args()) {
    workset.insert(arg.get());
  }
  for (auto& bb : func) {
    for (auto& ir : *bb) {
      if (!IsA<Constant>(ir.get()) && is_avail(ir.get())) {
        workset.insert(ir.get());
      }
    }
  }

  while (!workset.empty()) {
    std::unordered_set<IRObject*> next;
    HLCHECK(curr_shard < num_shards);
    for (auto& node : workset) {
      HLCHECK(shardings.count(node) == 0);
      shardings[node] = curr_shard;
      if (!IsA<Constant>(node)) {
        ++allocated;
      }
      for (auto& op : node->GetOperands()) {
        if (shardings.count(op.GetOwner()) == 0 && IsA<Constant>(op)) {
          shardings[op.GetOwner()] = curr_shard;
        }
      }
    }
    // check successors
    for (auto& node : workset) {
      for (auto& uses : node->GetResultsUses()) {
        for (auto& user : uses) {
          auto n = user.GetUse();
          if (is_avail(n) && shardings.count(n) == 0) {
            next.insert(n);
          }
        }
      }
    }
    if (allocated >= threshold) {
      ++curr_shard;
      allocated = 0;
      for (auto& n : next) {
        std::cout << " ---- cut:" << n->GetName() << std::endl;
      }
    }
    workset.swap(next);
  }

  for (auto& bb : func) {
    for (auto& ir : *bb) {
      auto inst = ir.get();
      HLCHECK(shardings.count(inst) != 0);
      if (shardings.count(inst) == 0) {
        std::cerr << "[sharding] unallocated: " << inst->GetName() << "\n";
        shardings[inst] = num_shards - 1;
      }
    }
  }
  return shardings;
}

static void ApplySharding(const std::unordered_map<IRObject*, int>& shardings) {
  for (const auto& kv : shardings) {
    std::cout << kv.second << ":" << kv.first->GetName() << ":" << std::endl;
  }
}

bool Sharding::RunOnFunction(Function* func) {
  if (shards_ <= 0) {
    return false;
  }
  auto sharding_scheme = GetSimpleSharding(*func, shards_);
  ApplySharding(sharding_scheme);
  return false;
}

} // end namespace halo