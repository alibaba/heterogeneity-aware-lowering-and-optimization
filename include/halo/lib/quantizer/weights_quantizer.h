//===- quantize_weights.h -------------------------------------------------===//
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
#ifndef HALO_LIB_QUANTIZER_WEIGHTS_QUANTIZER_H_
#define HALO_LIB_QUANTIZER_WEIGHTS_QUANTIZER_H_

#include <unordered_map>

#include "halo/lib/pass/pass.h"
#include "halo/lib/target/codegen.h"
namespace halo {

/// This pass quantizes weights.
class WeightsQuantizer final : public ModulePass {
 public:
  WeightsQuantizer(CodeGen::Quantization quant, const std::string& file)
      : ModulePass("Quantize Weights"), quant_(quant), pgq_file_(file) {}

  bool RunOnModule(Module* m) override;

 private:
  void RunOnConstant(Constant* val);

  typedef struct {
    float min_val;
    float max_val;
    float scale;
    int zp;
  } QuantInfo;
  CodeGen::Quantization quant_;
  const std::string pgq_file_;
  std::unordered_map<std::string, QuantInfo> quant_info_;
};

} // end namespace halo.
#endif // HALO_LIB_QUANTIZER_WEIGHTS_QUANTIZER_H_
