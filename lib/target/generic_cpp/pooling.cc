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

  auto padding_left = static_cast<uint32_t>(inst->GetPaddingLeft());
  auto padding_right = static_cast<uint32_t>(inst->GetPaddingRight());
  auto padding_top = static_cast<uint32_t>(inst->GetPaddingTop());
  auto padding_bottom = static_cast<uint32_t>(inst->GetPaddingBottom());
  auto stride_h =
      static_cast<uint32_t>(inst->GetStrides()[info.data_height_axis]);
  auto stride_w =
      static_cast<uint32_t>(inst->GetStrides()[info.data_width_axis]);
  auto kernel_h =
      static_cast<uint32_t>(inst->GetKsize()[info.kernel_height_axis]);
  auto kernel_w =
      static_cast<uint32_t>(inst->GetKsize()[info.kernel_width_axis]);

  std::vector<uint32_t> strides{stride_h, stride_w};
  std::vector<uint32_t> window{kernel_h, kernel_w};
  std::vector<uint32_t> paddings_front{padding_top, padding_left};
  std::vector<uint32_t> paddings_back{padding_bottom, padding_right};

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

  auto padding_left = static_cast<uint32_t>(inst->GetPaddingLeft());
  auto padding_right = static_cast<uint32_t>(inst->GetPaddingRight());
  auto padding_top = static_cast<uint32_t>(inst->GetPaddingTop());
  auto padding_bottom = static_cast<uint32_t>(inst->GetPaddingBottom());
  auto stride_h =
      static_cast<uint32_t>(inst->GetStrides()[info.data_height_axis]);
  auto stride_w =
      static_cast<uint32_t>(inst->GetStrides()[info.data_width_axis]);
  auto kernel_h =
      static_cast<uint32_t>(inst->GetKsize()[info.kernel_height_axis]);
  auto kernel_w =
      static_cast<uint32_t>(inst->GetKsize()[info.kernel_width_axis]);

  std::vector<uint32_t> strides{stride_h, stride_w};
  std::vector<uint32_t> window{kernel_h, kernel_w};
  std::vector<uint32_t> paddings_front{padding_top, padding_left};
  std::vector<uint32_t> paddings_back{padding_bottom, padding_right};

  EmitODLACall(ret, "odla_AveragePool", op0, inst->GetDataFormat(), window,
               strides, paddings_front, paddings_back, EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

} // namespace halo
