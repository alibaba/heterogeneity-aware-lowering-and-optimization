//===- pooling.cc ---------------------------------------------------------===//
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
#include "halo/lib/transforms/type_legalizer.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(PoolingMaxInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];
  const auto& ret_type = inst->GetResultType();

  CXXValue ret(inst->GetName(), op0.type);

  const auto& info = ImageAxisInfo::GetImageAxisInfo(
      inst->GetDataFormat(), inst->GetDataFormat() /* filter format */);

  const auto& pad_b = inst->GetPaddingsBefore();
  const auto& pad_a = inst->GetPaddingsAfter();
  unsigned spatial_dims = ret_type.GetNumOfDims() - 2;

  const auto& strs = inst->GetStrides();
  const auto& wins = inst->GetKsize();

  std::vector<uint32_t> strides(
      strs.begin() + info.data_spatial_axis,
      strs.begin() + info.data_spatial_axis + spatial_dims);
  std::vector<uint32_t> window(
      wins.begin() + info.kernel_spatial_axis,
      wins.begin() + info.kernel_spatial_axis + spatial_dims);
  std::vector<uint32_t> paddings_front(pad_b.begin(), pad_b.end());
  std::vector<uint32_t> paddings_back(pad_a.begin(), pad_a.end());

  EmitODLACall(ret, "odla_MaxPool", op0, inst->GetDataFormat(), window, strides,
               paddings_front, paddings_back, EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

void GenericCXXCodeGen::RunOnInstruction(PoolingAvgInst* inst) {
  const Def& input = inst->GetOperand(0);

  CXXValue op0 = ir_mapping_[input];
  const auto& ret_type = inst->GetResultType();

  CXXValue ret(inst->GetName(), op0.type);

  const auto& info = ImageAxisInfo::GetImageAxisInfo(
      inst->GetDataFormat(), inst->GetDataFormat() /* filter format */);

  const auto& pad_b = inst->GetPaddingsBefore();
  const auto& pad_a = inst->GetPaddingsAfter();
  unsigned spatial_dims = ret_type.GetNumOfDims() - 2;

  const auto& strs = inst->GetStrides();
  const auto& wins = inst->GetKsize();

  std::vector<uint32_t> strides(
      strs.begin() + info.data_spatial_axis,
      strs.begin() + info.data_spatial_axis + spatial_dims);
  std::vector<uint32_t> window(
      wins.begin() + info.kernel_spatial_axis,
      wins.begin() + info.kernel_spatial_axis + spatial_dims);
  std::vector<uint32_t> paddings_front(pad_b.begin(), pad_b.end());
  std::vector<uint32_t> paddings_back(pad_a.begin(), pad_a.end());

  EmitODLACall(ret, "odla_AveragePool", op0, inst->GetDataFormat(), window,
               strides, paddings_front, paddings_back,
               inst->GetPaddingIncluded(), EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

} // namespace halo
