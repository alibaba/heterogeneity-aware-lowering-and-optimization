//===- deconv.cc ----------------------------------------------------------===//
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

void GenericCXXCodeGen::RunOnInstruction(Conv2DTransposeInst* inst) {
  const Def& lhs = inst->GetOperand(0);
  const Def& rhs = inst->GetOperand(1);

  CXXValue op0 = ir_mapping_[lhs];
  CXXValue op1 = ir_mapping_[rhs];

  const halo::Type& ret_type = inst->GetResultType();

  const auto& info = ImageAxisInfo::GetImageAxisInfo(inst->GetDataFormat(),
                                                     inst->GetFilterFormat());
  auto padding_left = inst->GetPaddingLeft();
  auto padding_right = inst->GetPaddingRight();
  auto padding_top = inst->GetPaddingTop();
  auto padding_bottom = inst->GetPaddingBottom();
  auto stride_h = inst->GetStrides()[info.data_height_axis];
  auto stride_w = inst->GetStrides()[info.data_width_axis];
  auto dilation_h = inst->GetDilations()[info.data_height_axis];
  auto dilation_w = inst->GetDilations()[info.data_width_axis];
  auto group = inst->GetGroup();

  CXXValue ret(inst->GetName(), op0.type);
  std::string strides = "(const unsigned[]){" + Join(stride_h, stride_w) + "}";
  std::string dilations =
      "(const unsigned[]){" + Join(dilation_h, dilation_w) + "}";
  std::string paddings_front =
      "(const unsigned[]){" + Join(padding_top, padding_left) + "}";
  std::string paddings_back =
      "(const unsigned[]){" + Join(padding_bottom, padding_right) + "}";

  const std::string& enum_ns_layout = "odla_memory_layout::";
  const std::string& enum_prefix = "ODLA_";

  const std::string& enum_ns =
      opts_.dialect == Dialect::CXX_11 ? enum_ns_layout : "";

  const std::string data_layout =
      enum_ns + enum_prefix +
      (inst->GetDataFormat() == DataFormat::NHWC ? "CHANNELS_LAST"
                                                 : "CHANNELS_FIRST");
  const std::string soi = enum_ns + enum_prefix + "SOI";
  const std::string ois = enum_ns + enum_prefix + "OIS";
  const std::string ios = enum_ns + enum_prefix + "IOS";
  // the shape of deconv's filter is [height, width, output_channels,
  // input_channels] that is diffrent from conv's filter [height, width,
  // input_channels, output_channels] that means C is output_channels, N is
  // input_channels in deconv's filter, but C is input_channels, N is
  // output_channels in conv's filter
  const std::string kernel_layout =
      inst->GetFilterFormat() == DataFormat::HWCN
          ? soi
          : (inst->GetFilterFormat() == DataFormat::CNHW ? ois : ios);
  std::string bias_name = EmitNull();
  halo::Type bias_ty;
  if (inst->GetOperands().size() == 3) {
    const Def& bias = inst->GetOperand(2);
    CXXValue op2 = ir_mapping_[bias];
    bias_name = op2.name;
    bias_ty = bias.GetType();
  }

  EmitODLACall(ret, "odla_DeConv", op0, data_layout, group, op1, kernel_layout,
               strides, dilations, paddings_front, paddings_back, bias_name,
               EmitShape(ret_type));
  ir_mapping_[*inst] = ret;
}

} // namespace halo
