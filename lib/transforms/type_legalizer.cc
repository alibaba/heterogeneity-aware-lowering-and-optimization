//===- type_legalizer.cc --------------------------------------------------===//
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

#include "halo/lib/transforms/type_legalizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/constant.h"
#include "halo/lib/ir/math_instructions.h"

namespace halo {

const ImageAxisInfo& ImageAxisInfo::GetImageAxisInfo(DataFormat format,
                                                     DataFormat filter_format) {
  // assume NCHW uses OIHW and NHWC uses HWIO
  // fields: batch data_ch, dimensions,  kernel inputs, ...
  const static ImageAxisInfo nchw_oihw{0, 1, -1, 2, 3, 2, 0, 1, -1, 2, 3, 2};
  const static ImageAxisInfo nchw_iohw{0, 1, -1, 2, 3, 2, 1, 0, -1, 2, 3, 2};
  const static ImageAxisInfo nhwc_hwio{0, 3, -1, 1, 2, 1, 3, 2, -1, 0, 1, 0};
  const static ImageAxisInfo nhwc_ohwi{0, 3, -1, 1, 2, 1, 0, 3, -1, 1, 2, 1};
  const static ImageAxisInfo nhwc_invalid{0,  3,  -1, 1,  2,  1,
                                          -1, -1, -1, -1, -1, -1};
  const static ImageAxisInfo ncdhw{0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 2};
  const static ImageAxisInfo ndhwc{0, 4, 1, 2, 3, 1, 4, 3, 0, 1, 2, 0};
  const static ImageAxisInfo invalid{-1, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1};

  switch (format) {
    case DataFormat::NCHW:
      if (filter_format == DataFormat::CNHW) {
        return nchw_iohw;
      }
      return nchw_oihw;
    case DataFormat::NCDHW:
      return ncdhw;
    case DataFormat::NHWC:
      switch (filter_format) {
        case DataFormat::HWCN:
          return nhwc_hwio;
        case DataFormat::NHWC:
          return nhwc_ohwi;
        case DataFormat::INVALID:
          return nhwc_invalid;
        default:
          HLCHECK(0 && "Invalid format");
          return invalid;
      }
    case DataFormat::NDHWC:
      return ndhwc;
    default:
      HLCHECK(0 && "Invalid format");
      return invalid;
  }
}

static bool IsSameType(const Type& lhs, const Type& rhs) {
  if (lhs.GetNumOfDims() != rhs.GetNumOfDims()) {
    return false;
  }
  for (size_t d = 0; d < lhs.GetNumOfDims(); d++) {
    if (lhs.GetNumOfElementsInDim(d) != rhs.GetNumOfElementsInDim(d)) {
      return false;
    }
  }
  return true;
}

static void RunOnMathBinaryInstruction(Instruction* inst) {
  auto op0 = inst->GetOperand(0);
  auto op1 = inst->GetOperand(1);
  const Type& type0 = op0.GetType();
  const Type& type1 = op1.GetType();
  bool is_cmp = (inst->GetOpCode() == OpCode::CMP);
  if (!type0.IsValid() || !type1.IsValid()) {
    return;
  }
  if (IsSameType(type0, type1)) {
    inst->GetResultsTypes()[0] =
        is_cmp ? Type{DataType::BOOL, type0.GetDimSizes()} : type0;
    return;
  }
  // According to broadcasting rules,
  // Two dimensions are compatible when
  // they are equal, or one of them is 1
  // eg:
  // A      (4d array):  8 x 1 x 6 x 1
  // B      (3d array):      7 x 1 x 5
  // Result (4d array):  8 x 7 x 6 x 5
  std::vector<int64_t> ret_shape;
  int op1_rank = type0.GetNumOfDims();
  int op2_rank = type1.GetNumOfDims();
  auto rank = (op1_rank > op2_rank) ? op1_rank : op2_rank;

  int diff = std::abs(op1_rank - op2_rank);
  ret_shape.reserve(diff);
  for (int i = 0; i < diff; i++) {
    ret_shape.push_back((op1_rank > op2_rank) ? type0.GetNumOfElementsInDim(i)
                                              : type1.GetNumOfElementsInDim(i));
  }
  for (int i = diff; i < rank; i++) {
    auto index1 = (op1_rank > op2_rank) ? i : i - diff;
    auto index2 = (op1_rank < op2_rank) ? i : i - diff;

    auto size_i_a = type0.GetNumOfElementsInDim(index1);
    auto size_i_b = type1.GetNumOfElementsInDim(index2);

    if (size_i_a != size_i_b) {
      // one of them must be 1
      if (size_i_a != 1) {
        HLCHECK(size_i_b == 1);
        ret_shape.push_back(size_i_a);
      } else if (size_i_b != 1) {
        HLCHECK(size_i_a == 1);
        ret_shape.push_back(size_i_b);
      } else {
        ret_shape.push_back(1);
      }
    } else {
      ret_shape.push_back(size_i_a);
    }
  }
  inst->GetResultsTypes()[0] =
      halo::Type{is_cmp ? DataType::BOOL : type0.GetDataType(), ret_shape};
}

static void RunOnCastInstruction(Instruction* inst, DataType dst_dt) {
  auto op0 = inst->GetOperand(0);
  if (!op0.GetType().IsValid()) {
    return;
  }
  Type new_type{dst_dt, op0.GetType().GetDimSizes()};
  inst->GetResultsTypes()[0] = new_type;
}

static void RunOnInstruction(Instruction* inst) {
  switch (inst->GetOpCode()) {
    case OpCode::ADD:
    case OpCode::DIV:
    case OpCode::MAXIMUM:
    case OpCode::MINIMUM:
    case OpCode::MUL:
    case OpCode::POW:
    case OpCode::SUB:
    case OpCode::SHIFTL:
    case OpCode::SHIFTR:
    case OpCode::AND:
    case OpCode::OR:
    case OpCode::CMP: {
      RunOnMathBinaryInstruction(inst);
      break;
    }
    default: {
      inst->ComputeResultTypes();
      break;
    }
  }
}

static void RunOnInstruction(ZExtInst* inst) {
  RunOnCastInstruction(inst, inst->GetDataType());
}

static void RunOnInstruction(FPtoSIInst* inst) {
  RunOnCastInstruction(inst, inst->GetDataType());
}

static void RunOnInstruction(SItoFPInst* inst) {
  RunOnCastInstruction(inst, inst->GetDataType());
}

static void RunOnInstruction(ReshapeInst* inst) {
  auto& op0_type = inst->GetOperand(0).GetType();
  Def op1 = inst->GetOperand(1);
  if (!IsA<Constant>(op1)) {
    return;
  }
  const Constant* shape_c = DynCast<Constant>(op1.GetOwner());
  std::vector<int64_t> new_shape;
  for (size_t i = 0, e = shape_c->GetResultType(0).GetTotalNumOfElements();
       i != e; ++i) {
    int64_t dim = 0;
    if (op1.GetType().GetDataType() == DataType::INT64) {
      dim = shape_c->GetData<int64_t>(i);
    } else {
      dim = shape_c->GetData<int32_t>(i);
    }
    new_shape.push_back(dim);
  }

  size_t product = 1;
  size_t elements_num = 1;
  int neg_dim = -1;
  if (op0_type.IsDynamicBatch()) {
    new_shape[0] = op0_type.GetNumOfElementsInDim(0);
    for (size_t i = 1; i < op0_type.GetNumOfDims(); ++i) {
      elements_num *= op0_type.GetNumOfElementsInDim(i);
    }
    for (int i = 1, e = new_shape.size(); i != e; ++i) {
      if (new_shape[i] == 0) {
        if (!op0_type.IsValid()) {
          return;
        }
        if (i < static_cast<int>(op0_type.GetNumOfDims())) {
          new_shape[i] = op0_type.GetNumOfElementsInDim(i);
        } else {
          HLCHECK(0 && "Invalid reshape");
        }
      }
      if (new_shape[i] == -1) {
        if (neg_dim > 0) {
          HLCHECK(0 && "Invalid reshape operand");
          break;
        }
        neg_dim = i;
      } else {
        product *= new_shape[i];
      }
    }
    if (neg_dim > 0 && !op0_type.IsValid()) {
      return;
    }
    if (neg_dim > 0) {
      HLCHECK(elements_num % product == 0 && "Invalid reshape operand");
      new_shape[neg_dim] = elements_num / product;
    }
  } else {
    for (int i = 0, e = new_shape.size(); i != e; ++i) {
      if (new_shape[i] == 0) {
        if (!op0_type.IsValid()) {
          return;
        }
        if (i < static_cast<int>(op0_type.GetNumOfDims())) {
          new_shape[i] = op0_type.GetNumOfElementsInDim(i);
        } else {
          HLCHECK(0 && "Invalid reshape");
        }
      }
      if (new_shape[i] == -1) {
        if (neg_dim >= 0) {
          HLCHECK(0 && "Invalid reshape operand");
          break;
        }
        neg_dim = i;
      } else {
        product *= new_shape[i];
      }
    }
    if (neg_dim >= 0 && !op0_type.IsValid()) {
      return;
    }
    if (neg_dim >= 0) {
      HLCHECK(op0_type.GetTotalNumOfElements() % product == 0 &&
              "Invalid reshape operand");
      new_shape[neg_dim] = op0_type.GetTotalNumOfElements() / product;
    }
  }

  halo::Type new_type{op0_type.GetDataType(), new_shape};
  inst->GetResultsTypes()[0] = new_type;
}

static void RunOnInstruction(PadInst* inst) {
  Def op0 = inst->GetOperand(0);
  Def op1 = inst->GetOperand(1);

  if (!IsA<Constant>(op1.GetOwner())) {
    return;
  }

  const Constant* pad_amt = DynCast<Constant>(op1.GetOwner());
  auto& input_type = op0.GetType();
  std::vector<int64_t> result_dims;

  for (size_t i = 0, dims = input_type.GetNumOfDims(); i < dims; ++i) {
    int32_t before = pad_amt->GetDataPtr<int32_t>()[i * 2];    // NOLINT
    int32_t after = pad_amt->GetDataPtr<int32_t>()[i * 2 + 1]; // NOLINT
    result_dims.push_back(input_type.GetNumOfElementsInDim(i) + before + after);
  }
  halo::Type result_type{op0.GetType().GetDataType(), result_dims};
  inst->GetResultsTypes()[0] = result_type;
}

static Type ComputeKernelWiseType(
    const Type& data_type, const std::vector<int64_t>& kernel_shape,
    const std::vector<int>& strides, Padding padding_mode,
    std::vector<int>* input_paddings, const std::vector<int>& dilations,
    DataFormat data_format, DataFormat kernel_format, int group, OpCode op) {
  std::vector<int>& explicit_paddings = *input_paddings;
  switch (op) {
    case OpCode::CONV2D:
    case OpCode::CONV2DTRANSPOSE:
      break;
    case OpCode::POOLINGAVG:
    case OpCode::POOLINGMAX:
      kernel_format = data_format;
      break;
    default:
      kernel_format = DataFormat::INVALID;
      break;
  }

  const auto& info =
      ImageAxisInfo::GetImageAxisInfo(data_format, kernel_format);
  auto& data_shape = data_type.GetDimSizes();
  auto ret_shape = data_shape;

  int kernel_h = kernel_shape[info.kernel_height_axis];
  int kernel_w = kernel_shape[info.kernel_width_axis];
  if (op != OpCode::POOLINGMAX && op != OpCode::POOLINGAVG) {
    // for depthwise, for NHWC, the kernel is H, W, in_ch, multiplier
    // for NCHW, the kernel is output, in/<group>, H, W
    int kernel_output = kernel_shape[info.kernel_output_axis];
    int kernel_input = kernel_shape[info.kernel_input_axis];
    int input_ch = data_shape[info.data_channel_axis];
    HLCHECK(input_ch % group == 0);
    // The meanings of kernel dimension are different with groups.
    // Here we recompute the.
    int per_group_ch_in = input_ch / group;
    int per_group_ch_out =
        (kernel_output * kernel_input) / (group * per_group_ch_in);
    HLCHECK(per_group_ch_in * per_group_ch_out * group ==
            kernel_output * kernel_input);
    ret_shape[info.data_channel_axis] = group * per_group_ch_out;
  }

  auto index_h = info.data_spatial_axis;
  auto index_w = index_h + 1;
  switch (padding_mode) {
    case Padding::SAME: {
      ret_shape[index_h] =
          (data_shape[index_h] + strides[index_h] - 1) / strides[index_h];
      ret_shape[index_w] =
          (data_shape[index_w] + strides[index_w] - 1) / strides[index_w];
      break;
    }
    case Padding::VALID: {
      ret_shape[index_h] = (data_shape[index_h] - kernel_h + strides[index_h]) /
                           strides[index_h];
      ret_shape[index_w] = (data_shape[index_w] - kernel_w + strides[index_w]) /
                           strides[index_w];
      break;
    }
    case Padding::EXPLICIT: {
      if (op == OpCode::CONV2DTRANSPOSE) {
        ret_shape[index_h] = (data_shape[index_h] - 1) * strides[index_h] +
                             kernel_h - explicit_paddings[0] -
                             explicit_paddings[1];
        ret_shape[index_w] = (data_shape[index_w] - 1) * strides[index_w] +
                             kernel_w - explicit_paddings[2] -
                             explicit_paddings[3];
      } else {
        ret_shape[index_h] = (data_shape[index_h] + explicit_paddings[0] +
                              explicit_paddings[1] - kernel_h) /
                                 strides[index_h] +
                             1;
        ret_shape[index_w] = (data_shape[index_w] + explicit_paddings[2] +
                              explicit_paddings[3] - kernel_w) /
                                 strides[index_w] +
                             1;
      }
      break;
    }
    default: {
      HLCHECK(0 && "unsupported padding type");
    }
  }
  auto paddings = std::max(
      0L, (ret_shape[index_h] - 1) * strides[index_h] + kernel_h +
              (dilations[index_h] - 1) * (kernel_h - 1) - data_shape[index_h]);

  explicit_paddings[0] = paddings / 2;
  explicit_paddings[1] = paddings - explicit_paddings[0];
  paddings = std::max(
      0L, (ret_shape[index_w] - 1) * strides[index_w] + kernel_w +
              (dilations[index_w] - 1) * (kernel_w - 1) - data_shape[index_w]);
  explicit_paddings[2] = paddings / 2;
  explicit_paddings[3] = paddings - explicit_paddings[2];
  if (padding_mode == Padding::SAME_LOWER) {
    std::swap(explicit_paddings[0], explicit_paddings[1]);
    std::swap(explicit_paddings[2], explicit_paddings[3]);
  }

  return Type{data_type.GetDataType(), ret_shape};
}

// TODO (unknown): move to per-inst file.
static void RunOnInstruction(Conv2DInst* inst) {
  std::vector<int> paddings;
  auto& data_type = inst->GetOperand(0).GetType();
  auto& kernel_type = inst->GetOperand(1).GetType();
  if (!data_type.IsValid() || !kernel_type.IsValid()) {
    return;
  }
  if (inst->GetGroup() == -1) {
    auto input = inst->GetOperand(0);
    int in_channel = static_cast<int>(input.GetType().GetNumOfElementsInDim(
        input.GetType().GetNumOfDims() - 1));
    // TODO(unkonwn) support depth_multipiler
    inst->SetGroup(in_channel);
  }

  std::vector<int> explicit_paddings{
      inst->GetPaddingTop(), inst->GetPaddingBottom(), inst->GetPaddingLeft(),
      inst->GetPaddingRight()};
  auto ret_type = ComputeKernelWiseType(
      data_type, kernel_type.GetDimSizes(), inst->GetStrides(),
      inst->GetPadding(), &explicit_paddings, inst->GetDilations(),
      inst->GetDataFormat(), inst->GetFilterFormat(), inst->GetGroup(),
      inst->GetOpCode());
  inst->GetResultsTypes()[0] = ret_type;
  if (inst->GetPadding() != Padding::EXPLICIT) {
    inst->SetPaddingTop(explicit_paddings[0]);
    inst->SetPaddingBottom(explicit_paddings[1]);
    inst->SetPaddingLeft(explicit_paddings[2]);
    inst->SetPaddingRight(explicit_paddings[3]);
  }
}

static void RunOnInstruction(Conv2DTransposeInst* inst) {
  std::vector<int> paddings;
  auto& data_type = inst->GetOperand(0).GetType();
  auto& kernel_type = inst->GetOperand(1).GetType();
  if (!data_type.IsValid() || !kernel_type.IsValid()) {
    return;
  }
  std::vector<int> explicit_paddings{
      inst->GetPaddingTop(), inst->GetPaddingBottom(), inst->GetPaddingLeft(),
      inst->GetPaddingRight()};
  auto ret_type = ComputeKernelWiseType(
      data_type, kernel_type.GetDimSizes(), inst->GetStrides(),
      inst->GetPadding(), &explicit_paddings, inst->GetDilations(),
      inst->GetDataFormat(), inst->GetFilterFormat(), inst->GetGroup(),
      inst->GetOpCode());
  inst->GetResultsTypes()[0] = ret_type;
  if (inst->GetPadding() != Padding::EXPLICIT) {
    inst->SetPaddingTop(explicit_paddings[0]);
    inst->SetPaddingBottom(explicit_paddings[1]);
    inst->SetPaddingLeft(explicit_paddings[2]);
    inst->SetPaddingRight(explicit_paddings[3]);
  }
}

static void RunOnCommonReductionInstruction(Instruction* inst,
                                            std::vector<int32_t> axis,
                                            bool keep_dims) {
  const auto& input_type = inst->GetOperand(0).GetType();
  if (!input_type.IsValid()) {
    return;
  }
  if (inst->GetNumOfOperands() == 2) {
    const auto& op1 = inst->GetOperand(1);
    if (!IsA<Constant>(op1)) {
      return;
    }

    const Constant* axis_c = DynCast<const Constant>(op1);
    for (size_t i = 0, e = axis_c->GetResultType(0).GetTotalNumOfElements();
         i != e; ++i) {
      if (op1.GetType().GetDataType() == DataType::INT32) {
        axis.push_back(axis_c->GetData<int32_t>(i));
      } else if (op1.GetType().GetDataType() == DataType::INT64) {
        axis.push_back(static_cast<int32_t>(axis_c->GetData<int64_t>(i)));
      }
    }
  }
  for (size_t i = 0; i < axis.size(); ++i) {
    if (axis[i] < 0) {
      axis[i] += input_type.GetNumOfDims();
    }
  }

  if (axis.empty()) {
    for (size_t i = 0, e = input_type.GetNumOfDims(); i < e; ++i) {
      axis.push_back(i);
    }
  }

  std::vector<int64_t> ret_shape;
  ret_shape.reserve(input_type.GetNumOfDims());
  for (size_t i = 0, e = input_type.GetNumOfDims(); i < e; ++i) {
    if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
      if (keep_dims) {
        ret_shape.push_back(1);
      }
    } else {
      ret_shape.push_back(input_type.GetNumOfElementsInDim(i));
    }
  }
  DataType dt = input_type.GetDataType();

  if (inst->GetOpCode() == OpCode::ARGMAX ||
      inst->GetOpCode() == OpCode::ARGMIN) {
    dt = DataType::INT32;
  }
  if (!keep_dims && ret_shape.size() == 1) {
    inst->GetResultsTypes()[0] = halo::Type{dt};
  } else {
    inst->GetResultsTypes()[0] = halo::Type{dt, ret_shape};
  }
}

static void RunOnInstruction(ReduceL1Inst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceL2Inst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceLogSumInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceLogSumExpInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceMeanInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceMaxInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceMinInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceProductInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceSumSquareInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ArgmaxInst* inst) {
  std::vector<int32_t> axis;
  if (inst->GetNumOfOperands() < 2) {
    axis.push_back(inst->GetAxis());
  }
  RunOnCommonReductionInstruction(inst, axis, inst->GetKeepDims());
}

static void RunOnInstruction(PoolingMaxInst* inst) {
  std::vector<int> paddings;
  auto& data_type = inst->GetOperand(0).GetType();
  if (!data_type.IsValid()) {
    return;
  }
  std::vector<int> explicit_paddings{
      inst->GetPaddingTop(), inst->GetPaddingBottom(), inst->GetPaddingLeft(),
      inst->GetPaddingRight()};
  std::vector<int64_t> kernel_shape;
  kernel_shape.reserve(4);
  for (auto dim : inst->GetKsize()) {
    kernel_shape.push_back(dim);
  }
  auto ret_type = ComputeKernelWiseType(
      data_type, kernel_shape, inst->GetStrides(), inst->GetPadding(),
      &explicit_paddings, {1, 1, 1, 1}, inst->GetDataFormat(),
      inst->GetDataFormat(), 1 /*group*/, inst->GetOpCode());
  inst->GetResultsTypes()[0] = ret_type;
  if (inst->GetPadding() != Padding::EXPLICIT) {
    inst->SetPaddingTop(explicit_paddings[0]);
    inst->SetPaddingBottom(explicit_paddings[1]);
    inst->SetPaddingLeft(explicit_paddings[2]);
    inst->SetPaddingRight(explicit_paddings[3]);
  }
}

static void RunOnInstruction(PoolingAvgInst* inst) {
  std::vector<int> paddings;
  auto& data_type = inst->GetOperand(0).GetType();
  if (!data_type.IsValid()) {
    return;
  }
  std::vector<int> explicit_paddings{
      inst->GetPaddingTop(), inst->GetPaddingBottom(), inst->GetPaddingLeft(),
      inst->GetPaddingRight()};
  std::vector<int64_t> kernel_shape;
  kernel_shape.reserve(4);
  for (auto dim : inst->GetKsize()) {
    kernel_shape.push_back(dim);
  }
  auto ret_type = ComputeKernelWiseType(
      data_type, kernel_shape, inst->GetStrides(), inst->GetPadding(),
      &explicit_paddings, {1, 1, 1, 1}, inst->GetDataFormat(),
      inst->GetDataFormat(), 1 /*group*/, inst->GetOpCode());
  inst->GetResultsTypes()[0] = ret_type;
  if (inst->GetPadding() != Padding::EXPLICIT) {
    inst->SetPaddingTop(explicit_paddings[0]);
    inst->SetPaddingBottom(explicit_paddings[1]);
    inst->SetPaddingLeft(explicit_paddings[2]);
    inst->SetPaddingRight(explicit_paddings[3]);
  }
}

static void RunOnMatrixMultiplyInstruction(Instruction* inst, bool trans_a,
                                           bool trans_b) {
  Def op0 = inst->GetOperand(0);
  Def op1 = inst->GetOperand(1);
  if (!op0.GetType().IsValid() || !op1.GetType().IsValid()) {
    return;
  }
  const auto& input_type = op0.GetType();
  std::vector<int64_t> ret_shape(input_type.GetDimSizes());
  auto lhs_dims = input_type.GetNumOfDims();

  const auto& rhs_type = op1.GetType();
  auto rhs_dims = rhs_type.GetNumOfDims();

  auto row = trans_a ? input_type.GetNumOfElementsInDim(lhs_dims - 1)
                     : input_type.GetNumOfElementsInDim(lhs_dims - 2);
  auto col = trans_b ? rhs_type.GetNumOfElementsInDim(rhs_dims - 2)
                     : rhs_type.GetNumOfElementsInDim(rhs_dims - 1);
  ret_shape.pop_back();
  ret_shape.pop_back();
  ret_shape.push_back(row);
  ret_shape.push_back(col);
  inst->GetResultsTypes()[0] = halo::Type{input_type.GetDataType(), ret_shape};
}

static void RunOnInstruction(GemmInst* inst) {
  RunOnMatrixMultiplyInstruction(inst, inst->GetTransposeA(),
                                 inst->GetTransposeB());
}

static void RunOnInstruction(BatchMatMulInst* inst) {
  RunOnMatrixMultiplyInstruction(inst, inst->GetTransposeA(),
                                 inst->GetTransposeB());
}

static void RunOnInstruction(MatMulInst* inst) {
  RunOnMatrixMultiplyInstruction(inst, inst->GetTransposeA(),
                                 inst->GetTransposeB());
}

static void RunOnInstruction(ConcatInst* inst) {
  size_t num_inputs = inst->GetN();
  int axis = inst->GetAxis();
  auto& input_type = inst->GetOperand(0).GetType();
  if (!input_type.IsValid()) {
    return;
  }

  if (num_inputs != 0 && inst->GetNumOfOperands() > num_inputs) {
    HLCHECK(num_inputs + 1 == inst->GetNumOfOperands());
    auto op1 = inst->GetOperand(num_inputs);
    if (!IsA<Constant>(op1)) {
      return;
    }
    Constant* c_axis = DynCast<Constant>(op1);
    axis = c_axis->GetDataAsInt64(0);
    auto ops = inst->GetOperands();
    ops.pop_back(); // Drop the last one.
    inst->DropAllOperands();
    inst->AddOperands(ops);
  } else {
    num_inputs = inst->GetNumOfOperands();
  }

  if (axis < 0) {
    axis += input_type.GetNumOfDims();
  }
  inst->SetN(num_inputs);
  inst->SetAxis(axis);

  int new_dim = 0;
  for (size_t i = 0, e = num_inputs; i != e; ++i) {
    auto& type = inst->GetOperand(i).GetType();
    if (!type.IsValid()) {
      return;
    }
    new_dim += type.GetNumOfDims() > 0
                   ? static_cast<int>(type.GetNumOfElementsInDim(axis))
                   : 1;
  }
  std::vector<int64_t> ret_shape(input_type.GetDimSizes());
  ret_shape.at(axis) = new_dim;
  inst->GetResultsTypes()[0] = halo::Type{input_type.GetDataType(), ret_shape};
}

static void RunOnInstruction(OneHotInst* inst) {
  int axis = inst->GetAxis();
  auto indices = inst->GetOperand(0);
  auto depth = inst->GetOperand(1);
  auto on_value = inst->GetOperand(2);
  if (!indices.GetType().IsValid() || !IsA<Constant>(depth.GetOwner())) {
    return;
  }
  Constant* c_depth = DynCast<Constant>(depth.GetOwner());
  int depth_c = c_depth->GetData<int32_t>(0);
  std::vector<int64_t> ret_shape(indices.GetType().GetDimSizes());
  if (axis < 0) {
    axis += static_cast<int>(indices.GetType().GetNumOfDims()) + 1;
  }
  ret_shape.insert(ret_shape.begin() + axis, depth_c);
  inst->GetResultsTypes()[0] =
      halo::Type{on_value.GetType().GetDataType(), ret_shape};
}

static void RunOnInstruction(GatherInst* inst) {
  int32_t axis = inst->GetAxis();
  auto op0 = inst->GetOperand(0);
  auto op1 = inst->GetOperand(1);
  if (inst->GetNumOfOperands() > 2) {
    auto op2 = inst->GetOperand(2);
    if (!IsA<Constant>(op2.GetOwner())) {
      return;
    }
    Constant* c_axis = DynCast<Constant>(op2.GetOwner());
    if (op2.GetType().GetDataType() == DataType::INT32) {
      axis = c_axis->GetData<int32_t>(0);
    } else {
      axis = c_axis->GetData<int64_t>(0);
    }
  }

  if (!op0.GetType().IsValid() || !op1.GetType().IsValid()) {
    return;
  }

  auto& param_type = op0.GetType();
  auto& indices_type = op1.GetType();
  std::vector<int64_t> ret_shape;
  if (axis < 0) {
    axis += param_type.GetNumOfDims();
  }
  inst->SetAxis(axis);
  for (size_t i = 0, e = param_type.GetNumOfDims(); i != e; ++i) {
    if (i != static_cast<size_t>(axis)) {
      ret_shape.push_back(param_type.GetNumOfElementsInDim(i));
    }
  }
  ret_shape.insert(ret_shape.begin() + axis, indices_type.GetDimSizes().begin(),
                   indices_type.GetDimSizes().end());
  inst->GetResultsTypes()[0] = halo::Type{param_type.GetDataType(), ret_shape};
}

static void RunOnInstruction(SliceInst* inst) {
  auto op0 = inst->GetOperand(0);
  auto op1 = inst->GetOperand(1);
  auto op2 = inst->GetOperand(2);
  auto& input_type = op0.GetType();

  if (!input_type.IsValid() || !IsA<Constant>(op2)) {
    return;
  }

  auto dims = input_type.GetNumOfDims();
  bool specified_axes = inst->GetNumOfOperands() > 4;
  std::unordered_set<int32_t> axes;
  if (specified_axes) {
    auto op4 = inst->GetOperand(4);
    if (!IsA<Constant>(op4)) {
      return;
    }
    const Constant* axes_op = DynCast<Constant>(op4);
    if (op4.GetType().GetDataType() == DataType::INT32) {
      for (int i = 0, e = op4.GetType().GetTotalNumOfElements(); i != e; ++i) {
        axes.insert(axes_op->GetData<int32_t>(i));
      }
    } else if (op4.GetType().GetDataType() == DataType::INT64) {
      for (int i = 0, e = op4.GetType().GetTotalNumOfElements(); i != e; ++i) {
        axes.insert(static_cast<int32_t>(axes_op->GetData<int64_t>(i)));
      }
    }
  } else {
    for (unsigned i = 0; i < dims; ++i) {
      axes.insert(i);
    }
  }
  HLCHECK(op2.GetType().GetTotalNumOfElements() ==
          static_cast<int64_t>(axes.size()));

  Constant* c_sizes = DynCast<Constant>(op2);
  Constant* c_begins = DynCast<Constant>(op1);
  std::vector<int64_t> ret_shape;
  for (size_t i = 0, j = 0; i < dims; ++i) {
    auto size_i = input_type.GetNumOfElementsInDim(i);
    if (axes.count(i) != 0) {
      if (op2.GetType().GetDataType() == DataType::INT32) {
        if (auto t = c_sizes->GetData<int32_t>(j); t >= 0) {
          size_i = t;
        } else if (IsA<Constant>(op1.GetOwner())) {
          size_i -= c_begins->GetData<int32_t>(j);
        } else {
          return;
        }
      } else if (op2.GetType().GetDataType() == DataType::INT64) {
        if (auto t = c_sizes->GetData<int64_t>(j); t >= 0) {
          size_i = t;
        } else if (IsA<Constant>(op1.GetOwner())) {
          size_i -= c_begins->GetData<int64_t>(j);
        } else {
          return;
        }
      }
      j++;
    }
    ret_shape.push_back(size_i);
  }
  inst->GetResultsTypes()[0] =
      halo::Type{op0.GetType().GetDataType(), ret_shape};
}

static void RunOnInstruction(StackInst* inst) {
  int axis = inst->GetAxis();
  int num_inputs = inst->GetNumOfOperands();
  auto& input0_type = inst->GetOperand(0).GetType();
  if (!input0_type.IsValid()) {
    return;
  }
  if (axis < 0) {
    axis += static_cast<int>(input0_type.GetNumOfDims()) + 1;
  }
  std::vector<int64_t> ret_shape(input0_type.GetDimSizes());
  ret_shape.insert(ret_shape.begin() + axis, num_inputs);
  inst->GetResultsTypes()[0] = halo::Type{input0_type.GetDataType(), ret_shape};
}

static void RunOnInstruction(RandomUniformInst* inst) {
  std::vector<int64_t> ret_shape;
  if (inst->GetNumOfOperands() > 0) {
    auto op0 = inst->GetOperand(0);
    if (!IsA<Constant>(op0.GetOwner())) {
      return;
    }
    Constant* c_shape = DynCast<Constant>(op0.GetOwner());
    for (size_t i = 0, e = op0.GetType().GetTotalNumOfElements(); i != e; ++i) {
      ret_shape.push_back(c_shape->GetData<int32_t>(i));
    }
  } else {
    const auto& attr_shape = inst->GetShape();
    for (size_t i = 0, e = attr_shape.size(); i != e; ++i) {
      ret_shape.push_back(attr_shape.at(i));
    }
  }
  inst->GetResultsTypes()[0] = halo::Type{DataType::FLOAT32, ret_shape};
}

static void RunOnInstruction(RangeInst* inst) {
  auto op0 = inst->GetOperand(0);
  auto op1 = inst->GetOperand(1);
  auto op2 = inst->GetOperand(2);
  if (!IsA<Constant>(op0.GetOwner()) || !IsA<Constant>(op1.GetOwner()) ||
      !IsA<Constant>(op2.GetOwner())) {
    return;
  }
  DataType dt = op0.GetType().GetDataType();
  int64_t num_elements = 0;
  Constant* c_op0 = DynCast<Constant>(op0.GetOwner());
  Constant* c_op1 = DynCast<Constant>(op1.GetOwner());
  Constant* c_op2 = DynCast<Constant>(op2.GetOwner());
  if (dt == DataType::INT32) {
    int begin = c_op0->GetData<int32_t>(0);
    int end = c_op1->GetData<int32_t>(0);
    int step = c_op2->GetData<int32_t>(0);
    num_elements = std::max(0, (end - begin) / step);
  } else if (dt == DataType::FLOAT32) {
    float begin = c_op0->GetData<float>(0);
    float end = c_op1->GetData<float>(0);
    float step = c_op2->GetData<float>(0);
    num_elements = std::max(0, static_cast<int>((end - begin) / step));
  } else {
    HLCHECK(false && "unsupported data type in RangeInst");
  }
  inst->GetResultsTypes()[0] = halo::Type{dt, {num_elements}};
}

static void RunOnInstruction(TransposeInst* inst) {
  auto op0 = inst->GetOperand(0);
  auto& shape0 = op0.GetType();
  std::vector<int32_t> perm(inst->GetPermutation());
  if (!shape0.IsValid()) {
    return;
  }
  if (inst->GetNumOfOperands() > 1) {
    HLCHECK(perm.empty() && "Expect empty attribute perm.");
    auto op1 = inst->GetOperand(1);
    if (!IsA<Constant>(op1.GetOwner())) {
      return;
    }
    Constant* c_perm = DynCast<Constant>(op1.GetOwner());
    for (int64_t i = 0, e = op1.GetType().GetTotalNumOfElements(); i != e;
         ++i) {
      perm.push_back(c_perm->GetData<int>(i));
    }
  }
  std::vector<int64_t> ret_shape;
  if (perm.empty()) {
    // empty perm means reverse
    ret_shape.resize(shape0.GetNumOfDims());
    const std::vector<int64_t>& shape0_dims = shape0.GetDimSizes();
    std::reverse_copy(shape0_dims.begin(), shape0_dims.end(),
                      ret_shape.begin());
  } else {
    for (size_t i = 0; i != perm.size(); ++i) {
      ret_shape.push_back(shape0.GetNumOfElementsInDim(perm.at(i)));
    }
  }
  inst->GetResultsTypes()[0] = halo::Type{shape0.GetDataType(), ret_shape};
}

static void RunOnInstruction(ResizeInst* inst) {
  HLCHECK(inst->GetNumOfOperands() == 2);
  const auto& op1 = inst->GetOperand(1);
  if (!IsA<Constant>(op1)) {
    return;
  }
  const Constant* shape_c = DynCast<Constant>(op1);

  const auto& input_type = inst->GetOperand(0).GetType();
  std::vector<int64_t> new_shape = input_type.GetDimSizes();
  unsigned mask = inst->GetAxesMask();
  for (int i = 0, j = 0, e = new_shape.size(); i < e; ++i) {
    if (mask != 0 && (mask & (1 << (e - 1 - i))) == 0) {
      continue;
    }
    int64_t dim = 0;
    if (op1.GetType().GetDataType() == DataType::INT64) {
      dim = shape_c->GetData<int64_t>(j++);
    } else if (op1.GetType().GetDataType() == DataType::INT32) {
      dim = shape_c->GetData<int32_t>(j++);
    } else if (op1.GetType().GetDataType() == DataType::FLOAT32) {
      HLCHECK(inst->GetExplicitShape() == false);
      dim = std::floor(new_shape[i] * shape_c->GetData<float>(j++));
    }
    new_shape[i] = dim;
  }

  inst->GetResultsTypes()[0] = Type{input_type.GetDataType(), new_shape};
}

static void RunOnInstruction(SetDiff1DInst* inst) {
  // an invalid shape util both operands are constant.
  ;
}

static void RunOnInstruction(ExpandDimsInst* inst) {
  const auto& idx_op = inst->GetOperand(1);
  const auto& idx_type = idx_op.GetType();
  const auto& input_type = inst->GetOperand(0).GetType();
  if (!IsA<Constant>(idx_op)) {
    return;
  }
  std::vector<int64_t> shape;
  shape.reserve(idx_type.GetTotalNumOfElements());
  Constant* c = DynCast<Constant>(idx_op);
  for (int i = 0, e = idx_type.GetTotalNumOfElements(); i < e; ++i) {
    shape.push_back(c->GetDataAsInt64(i));
  }
  inst->GetResultsTypes()[0] = Type{input_type.GetDataType(), shape};
}

static void RunOnInstruction(NonMaxSuppressionInst* inst) {
  if (inst->GetNumOfOperands() > 4) {
    const auto& op2 = inst->GetOperand(2);
    if (!IsA<Constant>(op2)) {
      return;
    }

    // return shape [num_selected_indices, 3], the selected index format is
    // [batch_index, class_index, box_index].
    std::vector<int64_t> ret_shape;
    const Constant* max_output_boxes = DynCast<Constant>(op2);
    const auto& op2_type = op2.GetType();
    HLCHECK(op2_type.GetTotalNumOfElements() == 1);
    if (op2_type.GetDataType() == DataType::INT32) {
      ret_shape.push_back(
          static_cast<int64_t>(max_output_boxes->GetData<int32_t>(0)));
    } else if (op2_type.GetDataType() == DataType::INT64) {
      ret_shape.push_back(max_output_boxes->GetData<int64_t>(0));
    }
    ret_shape.push_back(3);
    inst->GetResultsTypes()[0] = Type{inst->GetIndexType(), ret_shape};
  }
}

static void RunOnInstruction(TopKInst* inst) {
  HLCHECK(inst->GetNumOfOperands() == 2);
  const auto& op1 = inst->GetOperand(1);
  if (!IsA<Constant>(op1)) {
    return;
  }

  const Constant* c_k = DynCast<Constant>(op1);
  const auto& op1_type = op1.GetType();
  HLCHECK(op1_type.GetTotalNumOfElements() == 1);
  int64_t k = 0;
  if (op1_type.GetDataType() == DataType::INT32) {
    k = static_cast<int64_t>(c_k->GetData<int32_t>(0));
  } else if (op1_type.GetDataType() == DataType::INT64) {
    k = c_k->GetData<int64_t>(0);
  }

  const auto& input_type = inst->GetOperand(0).GetType();
  const auto dims = input_type.GetNumOfDims();
  // Normalize axis.
  auto axis = inst->GetAxis();
  if (axis < 0) {
    axis += dims;
    inst->SetAxis(axis);
  }

  HLCHECK(axis >= 0 && axis < static_cast<int>(dims));

  auto ret_shape = input_type.GetDimSizes();
  ret_shape[axis] = k;

  inst->GetResultsTypes()[0] = Type{input_type.GetDataType(), ret_shape};
  inst->GetResultsTypes()[1] = Type{inst->GetIndexType(), ret_shape};
}

static void RunOnInstruction(TileInst* inst) {
  auto& op0_type = inst->GetOperand(0).GetType();
  Def op1 = inst->GetOperand(1);
  if (!op0_type.IsValid() || !IsA<Constant>(op1)) {
    return;
  }
  const Constant* repeats_value = DynCast<Constant>(op1.GetOwner());
  std::vector<int64_t> new_shape;
  size_t dims_cnt = repeats_value->GetResultType(0).GetTotalNumOfElements();
  int64_t dim = 0;
  for (size_t i = 0; i < dims_cnt; ++i) {
    if (op1.GetType().GetDataType() == DataType::INT64) {
      dim = repeats_value->GetData<int64_t>(i) *
            op0_type.GetNumOfElementsInDim(i);
    } else {
      dim = repeats_value->GetData<int32_t>(i) *
            op0_type.GetNumOfElementsInDim(i);
    }
    new_shape.push_back(dim);
  }

  halo::Type new_type{op0_type.GetDataType(), new_shape};
  inst->GetResultsTypes()[0] = new_type;
}

bool TypeLegalizer::RunOnBasicBlock(BasicBlock* bb) {
  bool changed = false;
  for (auto& it : *bb) {
    Instruction* inst = it.get();
    if (inst->HasValidResultTypes()) {
      continue;
    }
    switch (inst->GetOpCode()) {
#define GET_INST_DOWNCAST_SWITCH
#include "halo/lib/ir/instructions_info.def"
#undef GET_INST_DOWNCAST_SWITCH
      default: {
        if (!relaxed_) {
          // HLCHECK(0 && "Unreachable");
        }
      }
    }
    if (inst->HasValidResultTypes()) {
      changed = true;
    }
  }
  return changed;
}

} // end namespace halo
