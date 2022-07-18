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

#include <map>
#include <unordered_map>

#include "halo/api/halo_data.h"
#include "halo/halo.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/data_layout.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
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

    size_t io_mem = 0;
    float weight_mem = 0;

    // Note that FLOPS and FLOPs are different:
    // FLOPS means floating point operations per second, measure hardware
    // performance. FLOPs means floating point operations, measure the
    // complexity of the model. a multiply-add counts as two flops, thus macc =
    // 2 * flops
    float flops = 0;
    float percent = 0;
  };

  struct TensorInfo {
    size_t liveness = 0;
    size_t size = 0;
  };

  struct HWInfo {
    float conv_time;      // ms/Gflops
    float conv_knl_init;  // per kernel init time (ms)
    float mm_time;        // ms/Gflops
    float mm_knl_init;    // per kernel init time (ms)
    float other_time;     // ms/Gflops
    float other_knl_init; // per kernel init time (ms)
    float max_mem;        // MB
  };

  Analyzer(std::ostream* os, const AnalyzerOpts& opts)
      : ModulePass("Analyzer"), os_(os), opts_(opts) {}

  bool RunOnModule(Module* m) override;

  void GenerateRscInfo(std::ostream& os);

  std::string& GetReourceEst(int& bsz) {
    bsz = adaptive_bsz_;
    return rsc_req_;
  }

 private:
  static float GetNumOfOperators(const Instruction* inst);
  NodeInfo& GenerateCommonInfo(const Instruction* inst);
  template <class T>
  void CalPoolingInst(const T* inst);
  template <class T>
  void CalConvInst(const T* inst);
  void RunOnInstruction(Conv2DInst* inst);
  void RunOnInstruction(Conv2DTransposeInst* inst);
  void RunOnInstruction(Instruction* inst);
  void RunOnInstruction(BatchMatMulInst* inst);
  void RunOnInstruction(GemmInst* inst);
  void RunOnInstruction(MatMulInst* inst);
  void RunOnInstruction(NonMaxSuppressionInst* inst);
  void RunOnInstruction(PoolingMaxInst* inst);
  void RunOnInstruction(PoolingAvgInst* inst);
  void RunOnInstruction(BatchNormInst* inst);
  void RunOnInstruction(GatherInst* inst);
  void RunOnInstruction(ReduceMeanInst* inst);
  void RunOnInstruction(ReluInst* inst);
  void RunOnInstruction(Relu6Inst* inst);
  void RunOnInstruction(SoftmaxInst* inst);
  void RunOnInstruction(ArgmaxInst* inst);
  void RunOnInstruction(ConcatInst* inst);
  void RunOnInstruction(OneHotInst* inst);
  void RunOnInstruction(PadInst* inst);
  void RunOnInstruction(RangeInst* inst);
  void RunOnInstruction(RandomUniformInst* inst);
  void RunOnInstruction(ReduceMaxInst* inst);
  void RunOnInstruction(ReduceProductInst* inst);
  void RunOnInstruction(ReshapeInst* inst);
  void RunOnInstruction(SetDiff1DInst* inst);
  void RunOnInstruction(SliceInst* inst);
  void RunOnInstruction(StackInst* inst);
  void RunOnInstruction(TransposeInst* inst);
  void RunOnInstruction(ZExtInst* inst);
  void RunOnInstruction(ReturnInst* inst);
  void RunOnInstruction(ResizeInst* inst);
  void RunOnInstruction(LSTMInst* inst);
  void RunOnInstruction(KvParserInst* inst);

 private:
  std::ostream* os_;
  std::vector<Analyzer::NodeInfo> node_infos_;
  AnalyzerOpts opts_;
  // alive tensor buffer
  std::unordered_map<std::string, TensorInfo> alive_tensor_;
  std::string rsc_req_;
  int adaptive_bsz_ = 1;
  std::map<std::string, HWInfo> hw_paras_ = {
      {"GPU_t4", {1.476, 0.03, 0.35, 0.06, 26.8, 0.01, 16000}}};
};

} // namespace halo

#endif // HALO_LIB_TRANSFORM_ANALYZER_H_
