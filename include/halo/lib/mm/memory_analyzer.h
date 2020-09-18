//===- liveness_analyzer.h --------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_MM_MEMORY_ANALYZER_H_
#define HALO_LIB_MM_MEMORY_ANALYZER_H_

#include <unordered_map>
#include <vector>

#include "halo/lib/ir/instruction.h"
#include "halo/lib/ir/module.h"

namespace halo {

/// This class analyzes the memory usage of a module.
class MemoryAnalyzer {
 public:
  explicit MemoryAnalyzer(const Module& m);

  virtual ~MemoryAnalyzer() = default;

  size_t GetWeightsSize() const noexcept { return weights_; }
  size_t GetNonWeightsSize() const noexcept { return non_weights_; }
  size_t GetCurrNonWeigtsSize() const noexcept { return curr_non_weights_; }
  size_t GetPeak() const noexcept { return peak_; }

  void Reset() {
    curr_use_cnts_ = use_cnts_;
    curr_non_weights_ = 0;
    peak_ = 0;
  }
  // Returns dead defs after the execution of `inst`.
  std::vector<Def> Executed(const Instruction* inst);

 private:
  void RunOnFunction(const Function& func);

  const Module& module_;
  const GlobalContext& ctx_;
  std::unordered_map<uint64_t, int> use_cnts_;
  std::unordered_map<uint64_t, int> curr_use_cnts_;

  size_t weights_;
  size_t non_weights_;
  size_t curr_non_weights_;
  size_t peak_;
};

} // namespace halo

#endif // HALO_LIB_MM_MEMORY_ANALYZER_H_