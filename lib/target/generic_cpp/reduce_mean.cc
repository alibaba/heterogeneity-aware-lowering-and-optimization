//===- reduce_mean.cc -----------------------------------------------------===//
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

#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(ReduceMeanInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];
  const auto& ret_type = inst->GetResultType();

  CXXValue ret(inst->GetName(), op0.type);
  std::vector<uint32_t> axis;
  for (auto x : inst->GetAxis()) {
    axis.push_back(x);
  }

  EmitODLACall(ret, "odla_ReduceMean", op0, axis.size(), axis,
               inst->GetKeepDims(), EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

} // namespace halo
