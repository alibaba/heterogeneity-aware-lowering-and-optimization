//===- analyzer.h ---------------------------------------------------------===//
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

#ifndef HALO_LIB_TRANSFORM_ANALYZER_H_
#define HALO_LIB_TRANSFORM_ANALYZER_H_

#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/pass/pass.h"

namespace halo {

/// This pass do graph analysis, generate op type / op num / input shape /
/// output shape / computational estimator, etc.
class Analyzer final : public ModulePass {
 public:
  /// Record analysis result
  struct NodeInfo {
    size_t id = 0;
    std::string name;
    halo::OpCode type;
    halo::DataType data_type;

    std::vector<std::vector<int64_t>> input_shape;
    std::vector<int64_t> output_shape;

    float io_mem = 0;
    float weight_mem = 0;

    // Note that FLOPS and FLOPs are different:
    // FLOPS means floating point operations per second, measure hardware
    // performance. FLOPs means floating point operations, measure the
    // complexity of the model. a multiply-add counts as two flops, thus macc =
    // 2 * flops
    float flops = 0;
    float percent = 0;
  };

  Analyzer() : ModulePass("Analyzer") {}

  bool RunOnModule(Module* m) override;

  void WriteCSVReport(std::ostream& os) const;

 private:
  std::vector<Analyzer::NodeInfo> node_infos_;
};

} // namespace halo

#endif // HALO_LIB_TRANSFORM_ANALYZER_H_
