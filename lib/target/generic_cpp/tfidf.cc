//===- pad.cc -------------------------------------------------------------===//
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

#include <cstdio>
#include <string>

#include "halo/lib/framework/common.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/constant.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/ir/nn_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(TFIDFVectorizeInst* inst) {
  static const std::unordered_map<TFIDFMode, std::string> mode_strings{
      {TFIDFMode::TF, "ODLA_TFIDF_TF"},
      {TFIDFMode::IDF, "ODLA_TFIDF_IDF"},
      {TFIDFMode::TFIDF, "ODLA_TFIDF_TFIDF"}};

  CXXValue input = ir_mapping_[inst->GetOperand(0)];
  CXXValue pool = ir_mapping_[inst->GetOperand(1)];
  CXXValue cnt = ir_mapping_[inst->GetOperand(2)];
  CXXValue idx = ir_mapping_[inst->GetOperand(3)];
  CXXValue weights = ir_mapping_[inst->GetOperand(4)];

  CXXValue ret(inst->GetName(), input.type);
  const auto& mode_it = mode_strings.find(inst->GetMode());
  const std::string& mode =
      mode_it == mode_strings.end() ? "INVALID" : mode_it->second;
  EmitODLACall(ret, "odla_TFIDFVectorize", input, inst->GetMinGramLength(),
               inst->GetMaxGramLength(), inst->GetMaxSkip(), mode, pool, cnt,
               idx, weights, EmitShape(inst->GetResultType()));
  ir_mapping_[*inst] = ret;
}

} // namespace halo