//===- conv.cc ------------------------------------------------------------===//
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

#include "halo/lib/framework/common.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(Conv2DInst* inst) {
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[lhs];
  CXXValue op1 = ir_mapping_[rhs];

  const halo::Type& ret_type = inst->GetResultType();

  const auto& info = ImageAxisInfo::GetImageAxisInfo(inst->GetDataFormat(),
                                                     inst->GetFilterFormat());
  auto padding_left = static_cast<uint32_t>(inst->GetPaddingLeft());
  auto padding_right = static_cast<uint32_t>(inst->GetPaddingRight());
  auto padding_top = static_cast<uint32_t>(inst->GetPaddingTop());
  auto padding_bottom = static_cast<uint32_t>(inst->GetPaddingBottom());
  auto stride_h =
      static_cast<uint32_t>(inst->GetStrides()[info.data_height_axis]);
  auto stride_w =
      static_cast<uint32_t>(inst->GetStrides()[info.data_width_axis]);
  auto dilation_h =
      static_cast<uint32_t>(inst->GetDilations()[info.data_height_axis]);
  auto dilation_w =
      static_cast<uint32_t>(inst->GetDilations()[info.data_width_axis]);
  auto group = inst->GetGroup();

  CXXValue ret(inst->GetName(), op0.type);
  std::vector<uint32_t> strides{stride_h, stride_w};
  std::vector<uint32_t> dilations{dilation_h, dilation_w};
  std::vector<uint32_t> paddings_front{padding_top, padding_left};
  std::vector<uint32_t> paddings_back{padding_bottom, padding_right};

  const std::string& enum_ns_layout = "odla_memory_layout::";
  const std::string& enum_prefix = "ODLA_";
  const std::string& enum_ns =
      opts_.dialect == Dialect::CXX_11 ? enum_ns_layout : "";
  const std::string& data_layout =
      enum_ns + enum_prefix +
      (inst->GetDataFormat() == DataFormat::NHWC ? "CHANNELS_LAST"
                                                 : "CHANNELS_FIRST");
  std::string kernel_layout =
      enum_ns + enum_prefix +
      (inst->GetFilterFormat() == DataFormat::HWCN ? std::string("SIO")
                                                   : std::string("OIS"));

  std::string bias_name = EmitNull();
  if (inst->GetOperands().size() == 3) {
    const Def& bias = inst->GetOperand(2);
    CXXValue op2 = ir_mapping_[bias];
    bias_name = op2.name;
  }
  EmitODLACall(ret, "odla_Conv", op0, data_layout, group, op1, kernel_layout,
               std::vector<uint32_t>{stride_h, stride_w},
               std::vector<uint32_t>{dilation_h, dilation_w},
               std::vector<uint32_t>{padding_top, padding_left},
               std::vector<uint32_t>{padding_bottom, padding_right}, bias_name,
               EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

} // namespace halo
