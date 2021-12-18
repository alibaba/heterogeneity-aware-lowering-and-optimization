//===- resize.cc ----------------------------------------------------------===//
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

#include <cstdio>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

static const std::string& GetInterpolationName(const ResizeInst& inst) {
  const static std::unordered_map<Interpolation, std::string> names{
      {Interpolation::NEAREST, "ODLA_NEAREST"},
      {Interpolation::LINEAR, "odla_interpolation::LINEAR"},
      {Interpolation::CUBIC, "odla_interpolation::CUBIC"}};
  const static std::string inv = "odla_interpolation::INVALID";
  auto it = names.find(inst.GetInterpolationMode());
  if (it == names.end()) {
    HLCHECK(0);
    return inv;
  }
  return it->second;
}

static const std::string& GetCoordModeName(const ResizeInst& inst) {
  const static std::unordered_map<ResizeMode, std::string> names{
      {ResizeMode::HALF_PIXEL, "ODLA_HALF_PIXEL"},
      {ResizeMode::HALF_PIXEL_TF, "odla_resize_coordinate_mode::HALF_PIXEL_TF"},
      {ResizeMode::ALIGN_CORNERS, "odla_resize_coordinate_mode::ALIGN_CORNERS"},
      {ResizeMode::ASYMMETRIC, "odla_resize_coordinate_mode::ASYMMETRIC"},

  };
  const static std::string inv = "odla_resize_coordinate_mode::INVALID";

  auto it = names.find(inst.GetMode());
  if (it == names.end()) {
    HLCHECK(0);
    return inv;
  }
  return it->second;
}

void GenericCXXCodeGen::RunOnInstruction(ResizeInst* inst) {
  const Def& lhs = inst->GetOperand(0);
  const Def& scales = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[lhs];
  CXXValue op_scales = ir_mapping_[scales];

  const auto& ret_type = inst->GetResultType();

  CXXValue ret(inst->GetName(), op0.type);
  ir_mapping_[*inst] = ret;
  if (!inst->GetResultType().IsDynamicShape()) {
    EmitODLACall(ret, "odla_Resize", op0, GetInterpolationName(*inst),
                 GetCoordModeName(*inst), inst->GetAxesMask(),
                 EmitShape(ret_type));
    return;
  }
  auto op_sizes = ir_mapping_[Def::GetUndefined()];
  EmitODLACall(ret, "odla_ResizeDynamic", op0, op_scales, op_sizes,
               GetInterpolationName(*inst), GetCoordModeName(*inst),
               EmitShape(ret_type));
}

} // namespace halo