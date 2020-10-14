//===- transpose.cc -------------------------------------------------------===//
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

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(TransposeInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];

  const auto& ret_type = inst->GetResultType();
  std::vector<int64_t> perm_axis;
  perm_axis.reserve(inst->GetPermutation().size());
  for (auto v : inst->GetPermutation()) {
    perm_axis.push_back(v);
  }
  halo::Type perm{DataType::INVALID, perm_axis};
  CXXValue ret(inst->GetName(), op0.type);

  EmitODLACall(ret, "odla_Transpose", op0, EmitShape(perm),
               EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

} // namespace halo