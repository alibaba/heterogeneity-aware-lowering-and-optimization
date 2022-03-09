//===- tile.cc ----------------------------------------------------------===//
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

void GenericCXXCodeGen::RunOnInstruction(TileInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];
  const auto& input_type = input.GetType();
  const auto& ret_type = inst->GetResultType();

  const Def& repeat = inst->GetOperand(1);
  size_t dims = repeat.GetType().GetTotalNumOfElements();
  HLCHECK(dims == input_type.GetNumOfDims());

  HLCHECK(IsA<Constant>(repeat));
  const Constant* repeat_c = DynCast<Constant>(repeat);
  std::vector<uint32_t> repeat_v(dims, 1);
  for (size_t i = 0; i < dims; ++i) {
    if (repeat_c->GetResultType().GetDataType() == DataType::INT32) {
      repeat_v[i] = static_cast<uint32_t>(repeat_c->GetData<int32_t>(i));
    } else if (repeat_c->GetResultType().GetDataType() == DataType::INT64) {
      repeat_v[i] = static_cast<uint32_t>(repeat_c->GetData<int64_t>(i));
    }
  }

  CXXValue ret(inst->GetName(), op0.type);
  EmitODLACall(ret, "odla_Tile", op0, repeat_v, EmitShape(ret_type));

  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(TileDynamicInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& repeat = inst->GetOperand(1);
  const auto& input_type = input.GetType();
  size_t dims = repeat.GetType().GetTotalNumOfElements();
  HLCHECK(dims == input_type.GetNumOfDims());

  HLCHECK(!IsA<Constant>(repeat)); // repeat shape is dynamic
  CXXValue op0 = ir_mapping_[input];
  CXXValue op1 = ir_mapping_[repeat];

  CXXValue ret(inst->GetName(), op0.type);
  const auto& ret_type = inst->GetResultType();

  EmitODLACall(ret, "odla_TileDynamic", op0, op1, EmitShape(ret_type));

  ir_mapping_[*inst] = ret;
}

} // namespace halo