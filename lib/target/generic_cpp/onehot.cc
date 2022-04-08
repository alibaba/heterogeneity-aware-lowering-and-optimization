//===- onehot.cc ----------------------------------------------------------===//
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

void GenericCXXCodeGen::RunOnInstruction(OneHotInst* inst) {
  CXXValue op_idx = ir_mapping_[inst->GetOperand(0)];
  CXXValue op_depth = ir_mapping_[inst->GetOperand(1)];
  // We assume the IR is transformed to match with ODLA def: on_off is a [2]
  // shaped data.
  CXXValue on_off_val = ir_mapping_[inst->GetOperand(2)];
  const auto& ret_type = inst->GetResultType();
  const Constant* depth_c = DynCast<Constant>(inst->GetOperand(1));
  HLCHECK(depth_c != nullptr &&
          depth_c->GetResultType().GetTotalNumOfElements() == 1 &&
          "Depth must be constant scalar");

  CXXValue ret(inst->GetName(), op_idx.type);
  EmitODLACall(ret, "odla_OneHot", op_idx, depth_c->GetDataAsInt64(0),
               on_off_val, inst->GetAxis(), EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

} // namespace halo