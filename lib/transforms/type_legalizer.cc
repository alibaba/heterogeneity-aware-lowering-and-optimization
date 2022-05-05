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
#include <type_traits>
#include <unordered_set>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/type.h"
#include "halo/lib/ir/constant.h"
#include "halo/lib/ir/loss_instructions.h"
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/transforms/transforms_util.h"

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

static void RunOnMathUnaryInstruction(Instruction* inst) {
  auto op = inst->GetOperand(0);
  const Type& type = op.GetType();
  if (!type.IsValid()) {
    return;
  }
  bool ret_bool = (inst->GetOpCode() == OpCode::ISINF) ||
                  (inst->GetOpCode() == OpCode::ISNAN);
  inst->GetResultsTypes()[0] =
      ret_bool ? Type{DataType::BOOL, type.GetDimSizes()} : type;
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
  ret_shape.reserve(rank);
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
      if (size_i_a == 1) {
        ret_shape.push_back(size_i_b);
      } else if (size_i_a == -1) {
        if (size_i_b == 1) {
          ret_shape.push_back(-1);
        } else {
          ret_shape.push_back(size_i_b);
        }
      } else {
        HLCHECK(((size_i_b == 1) || (size_i_b == -1)) &&
                "Violation of the broadcasting rules");
        ret_shape.push_back(size_i_a);
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
    case OpCode::SQUAREDDIFFERENCE:
    case OpCode::AND:
    case OpCode::OR:
    case OpCode::XOR:
    case OpCode::MOD:
    case OpCode::CMP: {
      RunOnMathBinaryInstruction(inst);
      break;
    }
    case OpCode::ACOS:
    case OpCode::ACOSH:
    case OpCode::ASIN:
    case OpCode::ASINH:
    case OpCode::ATAN:
    case OpCode::ATANH:
    case OpCode::SIN:
    case OpCode::COS:
    case OpCode::TAN:
    case OpCode::CEIL:
    case OpCode::FLOOR:
    case OpCode::ABS:
    case OpCode::EXP:
    case OpCode::LOG:
    case OpCode::NOT:
    case OpCode::SIGN:
    case OpCode::ISNAN:
    case OpCode::ISINF: {
      RunOnMathUnaryInstruction(inst);
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

static void RunOnInstruction(FPtoFPInst* inst) {
  RunOnCastInstruction(inst, inst->GetDataType());
}

static void RunOnInstruction(ConvertFromStringInst* inst) {
  RunOnCastInstruction(inst, inst->GetDataType());
}

static void RunOnInstruction(ConvertToStringInst* inst) {
  RunOnCastInstruction(inst, inst->GetDataType());
}

static void RunOnInstruction(CompressInst* inst) {
  auto num_ops = inst->GetNumOfOperands();
  HLCHECK(num_ops == 2);
  auto input = inst->GetOperand(0);
  const auto& input_type = input.GetType();
  auto mask = inst->GetOperand(1);
  const auto& mask_type = mask.GetType();
  if (!input_type.IsValid() || !mask_type.IsValid()) {
    return;
  }
  HLCHECK(mask_type.GetNumOfDims() == 1);
  int n = mask_type.GetNumOfElementsInDim(0);

  const int inv = std::numeric_limits<int32_t>::max();
  int64_t axis = inst->GetAxis();
  const auto& dt = input_type.GetDataType();
  // FIXME: here we assume all values in mask are true.
  if (axis == inv) {
    inst->GetResultsTypes()[0] = Type{dt, {n}};
    return;
  }
  axis = axis < 0 ? axis + input_type.GetNumOfDims() : axis;
  auto ret_dims = input_type.GetDimSizes();
  ret_dims[axis] = n;
  inst->GetResultsTypes()[0] = Type{dt, ret_dims};
}

static void RunOnInstruction(ReshapeInst* inst) {
  auto& op0_type = inst->GetOperand(0).GetType();
  Def op1 = inst->GetOperand(1);
  const auto& op1_type = op1.GetType();
  if (op0_type.IsValid() && !op0_type.IsStaticShape() && !IsA<Constant>(op1) &&
      op1_type.IsValid()) {
    int rank = op1_type.GetTotalNumOfElements();
    std::vector<int64_t> new_shape(rank);
    int neg_dim_cnt = 0;
    for (int i = 0; i < rank; ++i) {
      const auto& r = GetAvailIntegerResult(op1, i);
      new_shape[i] = r.second;
      if (r.second == -1) {
        neg_dim_cnt += 1;
      }
    }
    if ((neg_dim_cnt == 2) && (rank == 2) &&
        (op0_type.GetNumOfElementsInDim(0) == -1)) {
      auto new_shape_dim1 = std::accumulate(op0_type.GetDimSizes().begin() + 1,
                                            op0_type.GetDimSizes().end(), 1,
                                            std::multiplies<int64_t>());
      HLCHECK((new_shape_dim1 > 0) && "Invalid reshape");
      new_shape[1] = new_shape_dim1;
    }
    inst->GetResultsTypes()[0] = halo::Type{op0_type.GetDataType(), new_shape};
    return;
  }

  const Constant* shape_c = DynCast<Constant>(op1);
  if (shape_c == nullptr) {
    return;
  }
  const auto& shape_type = shape_c->GetResultType();
  if (shape_type.IsScalar()) {
    HLCHECK(shape_c->IsScalarOne()); // reshape to a scalar.
    inst->GetResultsTypes()[0] = halo::Type{op0_type.GetDataType(), {}};
    return;
  }
  std::vector<int64_t> new_shape;
  for (size_t i = 0, e = shape_c->GetResultType().GetTotalNumOfElements();
       i != e; ++i) {
    new_shape.push_back(shape_c->GetDataAsInt64(i));
  }

  size_t product = 1;
  size_t elements_num = 1;
  int neg_dim = -1;
  if (op0_type.IsDynamicShape()) {
    halo::Type new_type{op0_type.GetDataType(), new_shape};
    inst->GetResultsTypes()[0] = new_type;
  } else if (op0_type.IsDynamicBatch()) {
    if (new_shape[0] > 0) {
      return;
    }
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

    halo::Type new_type{op0_type.GetDataType(), new_shape};
    inst->GetResultsTypes()[0] = new_type;

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

    halo::Type new_type{op0_type.GetDataType(), new_shape};
    inst->GetResultsTypes()[0] = new_type;
  }
}

static void RunOnInstruction(DequantizeInst* inst) {
  auto op0 = inst->GetOperand(0);
  if (!op0.GetType().IsValid()) {
    return;
  }
  Type new_type{DataType::FLOAT32, op0.GetType().GetDimSizes()};
  inst->GetResultsTypes()[0] = new_type;
}

static void RunOnInstruction(DetInst* inst) {
  const auto& input_type = inst->GetOperand(0).GetType();
  auto dt = input_type.GetDataType();
  auto ret_shape = input_type.GetDimSizes();
  ret_shape.pop_back();
  ret_shape.pop_back();
  inst->GetResultsTypes()[0] = Type{dt, ret_shape};
}

static void RunOnInstruction(NegativeLogLikelihoodLossInst* inst) {
  auto& input_type = inst->GetOperand(0).GetType();
  if (!input_type.IsValid()) {
    return;
  }
  HLCHECK(input_type.GetNumOfDims() >= 2);
  if (const auto& mode = inst->GetReduction();
      mode != ReductionMode::None && mode != ReductionMode::INVALID) {
    inst->GetResultsTypes()[0] = Type{input_type.GetDataType(), {1}};
    return;
  }
  auto dims = input_type.GetDimSizes();
  dims.erase(dims.begin() + 1);
  inst->GetResultsTypes()[0] = Type{input_type.GetDataType(), dims};
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

static void FixupForDynamicShape(const Type& data_type,
                                 std::vector<int64_t>* ret_shape) {
  for (int i = 0, e = data_type.GetNumOfDims(); i < e; ++i) {
    if (data_type.GetNumOfElementsInDim(i) == kDynamicShapeSize) {
      ret_shape->at(i) = kDynamicShapeSize;
    }
  }
}

static Type ComputeKernelWiseType(
    const Type& data_type, const std::vector<int64_t>& kernel_shape,
    const std::vector<int>& strides, Padding padding_mode,
    std::vector<int>* input_paddings, const std::vector<int>& dilations,
    DataFormat data_format, DataFormat kernel_format, int group, OpCode op,
    bool ceil_mode = false) {
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
  HLCHECK(data_type.GetNumOfDims() == 4);

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
  auto dilation_h = dilations[index_h];
  auto dilation_w = dilations[index_w];
  auto kernel_h_dilated = dilation_h * (kernel_h - 1) + 1;
  auto kernel_w_dilated = dilation_w * (kernel_w - 1) + 1;

  switch (padding_mode) {
    case Padding::SAME: {
      ret_shape[index_h] =
          (data_shape[index_h] + strides[index_h] - 1) / strides[index_h];
      ret_shape[index_w] =
          (data_shape[index_w] + strides[index_w] - 1) / strides[index_w];
      break;
    }
    case Padding::VALID: {
      ret_shape[index_h] =
          (data_shape[index_h] - kernel_h_dilated + strides[index_h]) /
          strides[index_h];
      ret_shape[index_w] =
          (data_shape[index_w] - kernel_w_dilated + strides[index_w]) /
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
                              explicit_paddings[1] - kernel_h_dilated);
        ret_shape[index_w] = (data_shape[index_w] + explicit_paddings[2] +
                              explicit_paddings[3] - kernel_w_dilated);
        if (ceil_mode) {
          ret_shape[index_h] =
              (ret_shape[index_h] + strides[index_h] - 1) / strides[index_h] +
              1;
          ret_shape[index_w] =
              (ret_shape[index_w] + strides[index_w] - 1) / strides[index_w] +
              1;
        } else {
          ret_shape[index_h] = ret_shape[index_h] / strides[index_h] + 1;
          ret_shape[index_w] = ret_shape[index_w] / strides[index_w] + 1;
        }
      }
      break;
    }
    default: {
      HLCHECK(0 && "unsupported padding type");
    }
  }

  auto paddings =
      std::max(0L, (ret_shape[index_h] - 1) * strides[index_h] + kernel_h +
                       (dilation_h - 1) * (kernel_h - 1) - data_shape[index_h]);

  explicit_paddings[0] = paddings / 2;
  explicit_paddings[1] = paddings - explicit_paddings[0];
  paddings =
      std::max(0L, (ret_shape[index_w] - 1) * strides[index_w] + kernel_w +
                       (dilation_w - 1) * (kernel_w - 1) - data_shape[index_w]);
  explicit_paddings[2] = paddings / 2;
  explicit_paddings[3] = paddings - explicit_paddings[2];

  if (padding_mode == Padding::SAME_LOWER) {
    std::swap(explicit_paddings[0], explicit_paddings[1]);
    std::swap(explicit_paddings[2], explicit_paddings[3]);
  }
  FixupForDynamicShape(data_type, &ret_shape);
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
    // TODO(unknown) support depth_multiplier
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

template <typename T>
static void RunOnCommonReductionInstruction(T* inst, std::vector<int32_t> axis,
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
    dt = DataType::INT64;
  }

  constexpr bool is_arg_inst =
      std::is_same<T, ArgmaxInst>() || std::is_same<T, ArgminInst>();
  if constexpr (!is_arg_inst) { // NOLINT
    inst->SetAxis(axis);
  }
  inst->GetResultsTypes()[0] = halo::Type{dt, ret_shape};
}

static void RunOnInstruction(ShapeInst* inst) {
  const Type& input_type = inst->GetOperand(0).GetType();

  if (!input_type.IsValid()) {
    return;
  }
  int rank = input_type.GetNumOfDims();
  inst->GetResultsTypes()[0] = halo::Type{inst->GetDataType(), {rank}};
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

static void RunOnInstruction(ReduceSumInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

static void RunOnInstruction(ReduceSumSquareInst* inst) {
  RunOnCommonReductionInstruction(inst, inst->GetAxis(), inst->GetKeepDims());
}

template <typename T>
static void RunOnArgMinMaxInst(T* inst) {
  std::vector<int32_t> axis;
  if (inst->GetNumOfOperands() < 2) {
    axis.push_back(inst->GetAxis());
  }
  RunOnCommonReductionInstruction(inst, axis, inst->GetKeepDims());
}

static void RunOnInstruction(ArgmaxInst* inst) {
  return RunOnArgMinMaxInst(inst);
}

static void RunOnInstruction(ArgminInst* inst) {
  return RunOnArgMinMaxInst(inst);
}

template <typename T>
static void RunOnPoolingInstruction(Instruction* pooling_inst) {
  std::vector<int> paddings;
  auto inst = DynCast<T>(pooling_inst);
  auto& data_type = inst->GetOperand(0).GetType();
  if (!data_type.IsValid()) {
    return;
  }

  bool ceil_mode = inst->GetRoundMode() == 1;

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
      inst->GetDataFormat(), 1 /*group*/, inst->GetOpCode(), ceil_mode);
  inst->GetResultsTypes()[0] = ret_type;
  if (inst->GetPadding() != Padding::EXPLICIT) {
    inst->SetPaddingTop(explicit_paddings[0]);
    inst->SetPaddingBottom(explicit_paddings[1]);
    inst->SetPaddingLeft(explicit_paddings[2]);
    inst->SetPaddingRight(explicit_paddings[3]);
  }
}

static void RunOnInstruction(PoolingMaxInst* inst) {
  RunOnPoolingInstruction<PoolingMaxInst>(inst);
}

static void RunOnInstruction(PoolingAvgInst* inst) {
  RunOnPoolingInstruction<PoolingAvgInst>(inst);
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
  for (size_t i = 0, e = inst->GetNumOfOperands(); i != e; ++i) {
    const auto& type = inst->GetOperand(i).GetType();
    if (!type.IsValid()) {
      return;
    }
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
    HLCHECK(type.GetNumOfDims() == input_type.GetNumOfDims());
    new_dim += type.GetNumOfDims() > 0
                   ? static_cast<int>(type.GetNumOfElementsInDim(axis))
                   : 1;
  }
  std::vector<int64_t> ret_shape(input_type.GetDimSizes());
  if (ret_shape.empty()) { // concating scalars.
    ret_shape.resize(1);
  }
  ret_shape.at(axis) = new_dim;
  inst->GetResultsTypes()[0] = halo::Type{input_type.GetDataType(), ret_shape};
}

static void RunOnInstruction(OneHotInst* inst) {
  int axis = inst->GetAxis();
  auto indices = inst->GetOperand(0);
  auto depth = inst->GetOperand(1);
  auto on_value = inst->GetOperand(2);
  Constant* c_depth = DynCast<Constant>(depth);
  if (!indices.GetType().IsValid() || c_depth == nullptr) {
    return;
  }
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

static void RunOnInstruction(GatherElementsInst* inst) {
  const auto& data_type = inst->GetOperand(0).GetType().GetDataType();
  const auto& index_type = inst->GetOperand(1).GetType();
  if (data_type == DataType::INVALID || !index_type.IsValid()) {
    return;
  }

  inst->GetResultsTypes()[0] = halo::Type{data_type, index_type.GetDimSizes()};
}

static void RunOnInstruction(SliceInst* inst) {
  auto op0 = inst->GetOperand(0);
  auto op_start = inst->GetOperand(1);
  auto op_len = inst->GetOperand(2);
  auto& input_type = op0.GetType();

  if (!input_type.IsValid()) {
    return;
  }

  auto dims = input_type.GetNumOfDims();
  if (!input_type.IsStaticShape() && !IsA<Constant>(op_len)) {
    auto ret_shape = input_type.GetDimSizes();
    bool is_constant_len = true;
    for (unsigned i = 0; i < dims && is_constant_len; ++i) {
      if (ret_shape[i] == kDynamicBatchSize ||
          ret_shape[i] == kDynamicShapeSize) {
        continue;
      }
      const auto& c = GetAvailIntegerResult(op_len, i);
      is_constant_len &= c.first;
      ret_shape[i] = c.second;
    }
    if (is_constant_len) {
      inst->GetResultsTypes()[0] =
          halo::Type{input_type.GetDataType(), ret_shape};
    }
    return;
  }

  if (!IsA<Constant>(op_len)) {
    return;
  }

  std::unordered_set<int32_t> axes;

  bool specified_axes = inst->GetNumOfOperands() > 4;

  if (specified_axes) {
    auto op_axes = inst->GetOperand(4);
    if (!IsA<Constant>(op_axes)) {
      return;
    }
    const Constant* axes_c = DynCast<Constant>(op_axes);
    for (int i = 0, e = op_axes.GetType().GetTotalNumOfElements(); i != e;
         ++i) {
      axes.insert(axes_c->GetDataAsInt64(i));
    }
  } else {
    for (size_t i = 0; i < dims; ++i) {
      axes.insert(i);
    }
  }
  Constant* c_sizes = DynCast<Constant>(op_len);
  std::vector<int64_t> ret_shape(dims);
  for (size_t i = 0, j = 0; i < dims; ++i) {
    ret_shape[i] = axes.count(i) != 0 ? c_sizes->GetDataAsInt64(j++)
                                      : input_type.GetNumOfElementsInDim(i);
    if (ret_shape[i] == -1) {
      ret_shape[i] = input_type.GetNumOfElementsInDim(i);
    }
  }
  inst->GetResultsTypes()[0] =
      halo::Type{op0.GetType().GetDataType(), ret_shape};
  // Some exported ONNX model might have empty result (consumers like concat
  // will ignore it)
  if (inst->GetResultType().GetTotalNumOfElements() == 0) {
    LOG(WARNING) << inst->GetName() << " has no elements";
  }
  HLCHECK(inst->GetResultType().IsValid());
}

static void RunOnInstruction(SliceDynamicInst* inst) {
  auto const& input_type = inst->GetOperand(0).GetType();
  auto const& slice_size = inst->GetOperand(2);

  auto dims = input_type.GetNumOfDims();
  if (!input_type.IsStaticShape() && !IsA<Constant>(slice_size)) {
    auto ret_shape = input_type.GetDimSizes();
    for (unsigned i = 0; i < dims; ++i) {
      if (ret_shape[i] == kDynamicBatchSize ||
          ret_shape[i] == kDynamicShapeSize) {
        continue;
      }
      const auto& c = GetAvailIntegerResult(slice_size, i);
      ret_shape[i] = c.second;
    }

    inst->GetResultsTypes()[0] =
        halo::Type{input_type.GetDataType(), ret_shape};
    return;
  }
}

static void RunOnInstruction(SplitInst* inst) {
  auto input = inst->GetOperand(1);
  auto split_dim = inst->GetOperand(0);
  auto num_split = inst->GetNumSplit();

  const Type& input_type = input.GetType();
  if (!input_type.IsValid()) {
    return;
  }
  auto dt = input_type.GetDataType();
  auto const& input_shape = input_type.GetDimSizes();

  int rank = input_type.GetNumOfDims();
  const Constant* split_dim_c = DynCast<Constant>(split_dim);
  HLCHECK(split_dim_c != nullptr && "split_dim is not a constant");
  HLCHECK(split_dim_c->GetResultType().IsScalar() &&
          "split_dim is not a scalar");

  int axis = split_dim_c->GetDataAsInt64(0);
  HLCHECK(axis >= 0 && axis < rank && "Invalid split dim");

  auto orig_size = input_type.GetNumOfElementsInDim(axis);
  HLCHECK(num_split > 0 && (orig_size % num_split == 0) && "Invalid num_split");
  int len = static_cast<int>(orig_size / num_split);

  std::vector<int64_t> ret_shape(input_shape);
  ret_shape[axis] = static_cast<int64_t>(len);

  inst->GetResultsTypes().resize(num_split);
  for (size_t i = 0, e = num_split; i != e; ++i) {
    inst->GetResultsTypes()[i] = Type{dt, ret_shape};
  }
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

static void RunOnInstruction(QuantizeInst* inst) {
  auto op0 = inst->GetOperand(0);
  if (!op0.GetType().IsValid()) {
    return;
  }
  DataType dt = DataType::INVALID;
  constexpr int char_bits = 8;
  if (inst->GetBits() == char_bits) {
    dt = inst->GetSignBit() ? DataType::INT8 : DataType::UINT8;
  } else if (inst->GetBits() == 2 * char_bits) {
    dt = inst->GetSignBit() ? DataType::INT16 : DataType::UINT16;
  } else if (inst->GetBits() == 4 * char_bits) {
    dt = inst->GetSignBit() ? DataType::INT32 : DataType::UINT32;
  }
  HLCHECK(dt != DataType::INVALID);
  Type new_type{dt, op0.GetType().GetDimSizes()};
  inst->GetResultsTypes()[0] = new_type;
  if (inst->GetNumOfOperands() == 1) {
    // Output scale & zero point
    inst->GetResultsTypes()[1] = Type{DataType::FLOAT32, {0}};
    inst->GetResultsTypes()[2] = Type{dt, {0}};
  }
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
  auto op_num = inst->GetNumOfOperands();
  // If op_num is 3, the operands are (x, ROI, scales).
  // If op_num is 2, the operands are (x, scales)
  // TODO(unknown): ROI operand is not handled now.
  HLCHECK(op_num == 2 || op_num == 3);
  const auto& op_scale = inst->GetOperand(op_num - 1);
  if (!IsA<Constant>(op_scale)) {
    return;
  }
  const Constant* shape_c = DynCast<Constant>(op_scale);

  const auto& input_type = inst->GetOperand(0).GetType();
  std::vector<int64_t> new_shape = input_type.GetDimSizes();
  unsigned mask = inst->GetAxesMask();
  for (int i = 0, j = 0, e = new_shape.size(); i < e; ++i) {
    if (mask != 0 && (mask & (1 << (e - 1 - i))) == 0) {
      continue;
    }
    int64_t dim = 0;
    if (op_scale.GetType().GetDataType() == DataType::INT64) {
      dim = shape_c->GetData<int64_t>(j++);
    } else if (op_scale.GetType().GetDataType() == DataType::INT32) {
      dim = shape_c->GetData<int32_t>(j++);
    } else if (op_scale.GetType().GetDataType() == DataType::FLOAT32) {
      HLCHECK(inst->GetExplicitShape() == false);
      dim = std::floor(new_shape[i] * shape_c->GetData<float>(j++));
    }
    new_shape[i] = dim;
  }
  FixupForDynamicShape(input_type, &new_shape);

  inst->GetResultsTypes()[0] = Type{input_type.GetDataType(), new_shape};
}

static void RunOnInstruction(SetDiff1DInst* inst) {
  // an invalid shape util both operands are constant.
  ;
}

static void RunOnInstruction(EinsumInst* inst) {
  int n = inst->GetNumOfOperands();
  for (int i = 0; i < n; ++i) {
    if (!inst->GetOperand(i).GetType().IsValid()) {
      return;
    }
  }
  HLCHECK(n >= 1);
  std::vector<std::string> terms;
  const std::string equ = inst->GetEquation();
  std::unordered_map<char, int64_t> char2dim;
  bool has_output = false;
  for (int i = 0, e = equ.size(), new_term = 1; i < e; ++i) {
    auto c = equ[i];
    if (new_term == 1) {
      terms.push_back("");
      new_term = 0;
    }
    if (c == ' ') {
      continue;
    }
    if (c == '.') {
      HLCHECK(i + 2 < e && equ[i + 1] == '.' && equ[i + 2] == '.');
      terms.back().push_back('.');
      i += 2;
    }
    if (c == ',') {
      new_term = 1;
      continue;
    }
    if (c == '-') {
      HLCHECK(i + 1 < e && equ[i + 1] == '>');
      HLCHECK(!has_output);
      i += 1;
      has_output = true;
      new_term = 1;
      continue;
    }
    if (std::isalpha(c) != 0) {
      std::string& term = terms.back();
      term.push_back(c);
    }
  }

  int num_terms = terms.size();
  auto elem_ty = inst->GetOperand(0).GetType().GetDataType();
  if (!has_output) {
    HLCHECK(num_terms == n);
    inst->GetResultsTypes()[0] = Type{elem_ty, {}};
    return;
  }

  HLCHECK(num_terms == n + 1);
  // Setup character to dimension mapping for inputs.
  std::vector<int64_t> ellipsis_dims;
  for (int i = 0; i < n; ++i) {
    const auto& ty = inst->GetOperand(i).GetType();
    unsigned rank = ty.GetNumOfDims();
    const auto& term = terms[i];
    HLCHECK(term.size() <= rank);
    for (unsigned j = 0, s = term.size(), dim_idx = 0; j < s; ++j, ++dim_idx) {
      char c = terms[i][j];
      if (c == '.') {
        bool init = ellipsis_dims.empty();
        for (unsigned k = 0; k < rank - term.size(); ++k) {
          auto v = ty.GetNumOfElementsInDim(dim_idx++);
          if (init) {
            ellipsis_dims.push_back(v);
          } else {
            HLCHECK(ellipsis_dims[k] == v);
          }
        }
        continue;
      }
      int64_t d = ty.GetNumOfElementsInDim(dim_idx);
      if (char2dim.count(c) == 0) {
        char2dim[c] = d;
      } else {
        HLCHECK(char2dim[c] == d);
      }
    }
  }
  const auto& out_term = terms.back();
  std::vector<int64_t> out_shape;
  out_shape.reserve(out_term.size());
  for (auto c : out_term) {
    if (c == '.') {
      out_shape.insert(out_shape.end(), ellipsis_dims.begin(),
                       ellipsis_dims.end());
    } else {
      HLCHECK(char2dim.count(c) > 0);
      out_shape.push_back(char2dim[c]);
    }
  }
  inst->GetResultsTypes()[0] = Type{elem_ty, out_shape};
}

static void RunOnInstruction(ExpandDimsInst* inst) {
  const auto& shape = DynCast<Constant>(inst->GetOperand(1));
  const auto& input_type = inst->GetOperand(0).GetType();
  if (shape == nullptr) {
    return;
  }
  std::vector<int64_t> output_shape;

  int shape_rank = shape->GetResultType().GetTotalNumOfElements();
  int input_rank = input_type.GetNumOfDims();
  for (int i = 0, e = std::max(shape_rank, input_rank); i < e; ++i) {
    int input_idx = input_rank - 1 - i;
    int shape_idx = shape_rank - 1 - i;
    int64_t dim0 =
        (input_idx < 0) ? 1 : input_type.GetNumOfElementsInDim(input_idx);
    int64_t dim1 = (shape_idx < 0) ? 1 : shape->GetDataAsInt64(shape_idx);
    HLCHECK(dim0 == dim1 || dim0 == 1 || dim1 == 1);
    output_shape.push_back((dim0 == 1) ? dim1 : dim0);
  }
  std::reverse(output_shape.begin(), output_shape.end());

  halo::Type ret_type{input_type.GetDataType(), output_shape};

  inst->GetResultsTypes()[0] = Type{input_type.GetDataType(), output_shape};
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
  const auto& input_type = inst->GetOperand(0).GetType();
  const auto& op1 = inst->GetOperand(1);
  if (!input_type.IsValid() || !IsA<Constant>(op1)) {
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

static void RunOnInstruction(TileDynamicInst* inst) {
  auto const& op0 = inst->GetOperand(0);
  auto& op0_type = op0.GetType();

  if (!op0_type.IsValid() && !op0_type.IsStaticShape()) {
    return;
  }

  auto const& repeats = inst->GetOperand(1);
  auto& repeats_type = repeats.GetType();

  if (!IsA<Constant>(repeats) && repeats_type.IsValid()) { // shape is dynamic
    auto ret_shape = op0_type.GetDimSizes();
    auto rank = op0_type.GetNumOfDims();
    for (size_t i = 0; i < rank; ++i) {
      const auto& c = GetAvailIntegerResult(repeats, i);
      auto repeats_i = c.second;
      if (repeats_i == -1) {
        ret_shape[i] = -1;
        continue;
      }
      int64_t dim_i = op0_type.GetNumOfElementsInDim(i) * repeats_i;
      ret_shape[i] = dim_i;
    }
    halo::Type ret_type{op0_type.GetDataType(), ret_shape};
    inst->GetResultsTypes()[0] = ret_type;
  }
}

static void RunOnInstruction(HgEngineInst* inst) {
  std::vector<std::string> out_type_list = inst->GetOutTypeList();
  std::vector<std::vector<int64_t>> output_shapes = inst->GetOutputShapes();

  auto type = DataType::INT8;
  HLCHECK(output_shapes.size() == out_type_list.size());

  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    if (out_type_list[i] == "int8") {
      type = DataType::INT8;
    } else if (out_type_list[i] == "int16") {
      type = DataType::INT16;
    } else {
      HLCHECK(false && "Wrong output type");
    }
    for (auto dim : output_shapes[i]) {
      new_shape.push_back(static_cast<int64_t>(dim));
    }

    halo::Type new_type{type, new_shape};
    inst->GetResultsTypes()[i] = new_type;
  }
}

static void RunOnInstruction(HgQuantInst* inst) {
  HLCHECK(inst->GetNumOfOperands() == 3);
  const auto& input_type = inst->GetOperand(0).GetType();
  auto qtype = inst->GetQtype();
  auto type = DataType::INT8;
  if (qtype == "int8") {
    type = DataType::INT8;
  } else if (qtype == "int16") {
    type = DataType::INT16;
  } else {
    HLCHECK(false && "Wrong output type");
  }
  halo::Type new_type{type, input_type.GetDimSizes()};
  inst->GetResultsTypes()[0] = new_type;
}

static void RunOnInstruction(HgDequantInst* inst) {
  HLCHECK(inst->GetNumOfOperands() == 3);
  const auto& input_type = inst->GetOperand(0).GetType();
  auto dqtype = inst->GetOutType();
  auto type = DataType::FLOAT32;
  if (dqtype == "float32") {
    type = DataType::FLOAT32;
  } else {
    HLCHECK(false && "Wrong output type");
  }
  halo::Type new_type{type, input_type.GetDimSizes()};
  inst->GetResultsTypes()[0] = new_type;
}

static void RunOnInstruction(IfInst* inst) {
  auto& ret_types = inst->GetResultsTypes();
  if (inst->GetNumOfOperands() == 1) {
    auto ret0 = inst->GetElseBranch()->GetReturnInst();
    auto ret1 = inst->GetThenBranch()->GetReturnInst();
    if (ret0 != nullptr && ret0->HasValidResultTypes()) {
      ret_types = ret0->GetResultsTypes();
    } else if (ret1 != nullptr && ret1->HasValidResultTypes()) {
      ret_types = ret1->GetResultsTypes();
    }
    return;
  }
  const auto& data_type = inst->GetOperand(1).GetType();
  if (data_type.IsValid()) {
    ret_types[0] = data_type;
    ret_types[1] = data_type;
  }
}

static void RunOnInstruction(LoopInst* inst) {
  auto& ret_types = inst->GetResultsTypes();
  auto ret_inst = inst->GetBody()->GetReturnInst();
  if (ret_inst != nullptr) {
    HLCHECK(ret_types.size() == ret_inst->GetNumOfOperands());
    for (int i = 0, e = ret_types.size(); i < e; ++i) {
      ret_types[i] = ret_inst->GetOperand(i).GetType();
    }
  }
}

enum LSTMArgIndex {
  LSTM_ARG_X_IDX = 0,
  LSTM_ARG_W_IDX = 1,
  LSTM_ARG_R_IDX = 2,
  LSTM_ARG_B_IDX = 3,
  LSTM_ARG_SEQUENCE_LENGTH_IDX = 4,
  LSTM_ARG_INITIAL_H_IDX = 5,
  LSTM_ARG_INITIAL_C_IDX = 6,
  LSTM_ARG_P_IDX = 7
};

enum LSTMOutputIndex {
  LSTM_OUPTUT_Y = 0,
  LSTM_OUPTUT_Y_H = 1,
  LSTM_OUPTUT_Y_C = 2
};

static void RunOnRNNBase(Instruction* inst, int layout) {
  bool is_lstm = inst->GetOpCode() == OpCode::LSTM;
  const Def& op_x = inst->GetOperand(LSTM_ARG_X_IDX);
  const Type& x_type = op_x.GetType();

  if (!x_type.IsValid()) {
    return;
  }

  HLCHECK(layout == 0 || layout == 1);

  int64_t seq_length = x_type.GetNumOfElementsInDim(0);
  int64_t batch_size = x_type.GetNumOfElementsInDim(1);

  if (layout == 1) {
    std::swap(seq_length, batch_size);
  }

  const Def& op_r = inst->GetOperand(2);
  const Type& r_type = op_r.GetType();

  int64_t num_directions = r_type.GetNumOfElementsInDim(0);
  int64_t hidden_size = r_type.GetNumOfElementsInDim(2);

  if (layout == 0) {
    inst->GetResultsTypes()[0] =
        Type{x_type.GetDataType(),
             {seq_length, num_directions, batch_size, hidden_size}};
  } else {
    inst->GetResultsTypes()[0] =
        Type{x_type.GetDataType(),
             {batch_size, seq_length, num_directions, hidden_size}};
  }

  Type common_type{x_type.GetDataType(),
                   {num_directions, batch_size, hidden_size}};

  inst->GetResultsTypes()[1] = common_type;
  if (is_lstm) {
    inst->GetResultsTypes()[2] = common_type;
  }
}

static void RunOnInstruction(LSTMInst* inst) {
  RunOnRNNBase(inst, inst->GetLayout());
}

static void RunOnInstruction(GRUInst* inst) {
  RunOnRNNBase(inst, inst->GetLayout());
}

static void RunOnInstruction(RNNInst* inst) {
  RunOnRNNBase(inst, inst->GetLayout());
}

static void RunOnInstruction(TFIDFVectorizeInst* inst) {
  const auto& type = inst->GetOperand(0).GetType();
  if (!type.IsValid()) {
    return;
  }
  auto rank = type.GetNumOfDims();
  if (rank == 1) {
    inst->GetResultsTypes()[0] =
        Type{DataType::FLOAT32, {inst->GetMaxIdx() + 1}};
  } else {
    inst->GetResultsTypes()[0] = Type{
        DataType::FLOAT32, {static_cast<int64_t>(rank), inst->GetMaxIdx() + 1}};
  }
}

static void RunOnInstruction(ReturnInst* inst) {
  std::vector<Type> types;
  types.reserve(inst->GetNumOfOperands());
  for (auto& op : inst->GetOperands()) {
    types.push_back(op.GetType());
  }
  inst->GetResultsTypes() = types;
}

static void RunOnInstruction(SelectInst* inst) {
  const auto& ty_cond = inst->GetOperand(0).GetType();
  const auto& ty_x = inst->GetOperand(1).GetType();
  const auto& ty_y = inst->GetOperand(2).GetType();
  if (!ty_cond.IsValid() || !ty_x.IsValid() || !ty_y.IsValid()) {
    return;
  }
  // Result shape is 3-way broadcasted with ty_x, ty_y and ty_cond.
  auto rank_x = ty_x.GetNumOfDims();
  auto rank_y = ty_y.GetNumOfDims();
  auto rank_c = ty_cond.GetNumOfDims();
  auto rank_r = std::max(rank_x, rank_y);
  std::vector<int64_t> ret_shape(rank_r);

  // Broadcast between x and y according to the general broadcasting rule
  for (int64_t i = rank_r - 1, idx_y = rank_y - 1, idx_x = rank_x - 1; i >= 0;
       --i) {
    auto dim_x = idx_x < 0 ? 1 : ty_x.GetNumOfElementsInDim(idx_x--);
    auto dim_y = idx_y < 0 ? 1 : ty_y.GetNumOfElementsInDim(idx_y--);
    ret_shape[i] = std::max(dim_x, dim_y);
  }

  // A special case where TF1.x Select requires cond
  // to be broadcasted in a different way:
  // cond: [512]    x/y: [512, 3, 4] -->
  // cond: [512, 1, 1]
  if (rank_r >= rank_c && rank_c == 1 &&
      ret_shape[0] == ty_cond.GetNumOfElementsInDim(0)) {
    inst->GetResultsTypes()[0] = Type{ty_x.GetDataType(), ret_shape};
    return;
  }

  auto rank_xy = rank_r;
  rank_r = std::max(rank_r, rank_c);
  ret_shape.resize(rank_r);
  for (int64_t i = rank_r - 1, idx_c = rank_c - 1, idx_xy = rank_xy - 1; i >= 0;
       --i) {
    auto dim_xy = idx_xy < 0 ? 1 : ret_shape[idx_xy--];
    auto dim_c = idx_c < 0 ? 1 : ty_cond.GetNumOfElementsInDim(idx_c--);
    ret_shape[i] = std::max(dim_xy, dim_c);
  }
  inst->GetResultsTypes()[0] = Type{ty_x.GetDataType(), ret_shape};
}

static void RunOnInstruction(BitcastInst* inst) {
  const auto& type = inst->GetOperand(0).GetType();
  if (!type.IsValid()) {
    return;
  }
  auto dtype = inst->GetDataType();
  HLCHECK(dtype != DataType::INVALID);
  auto from_dtype = type.GetDataType();
  DefaultDataLayout layout;
  auto from_bits = layout.Bits(from_dtype);
  auto to_bits = layout.Bits(dtype);
  auto new_shape(type.GetDimSizes());
  if (from_bits < to_bits) {
    auto last_dim = type.GetNumOfElementsInDim(type.GetNumOfDims() - 1);
    HLCHECK(to_bits == from_bits * last_dim);
    new_shape.pop_back();
  } else if (from_bits > to_bits) {
    HLCHECK(from_bits % to_bits == 0);
    new_shape.push_back(from_bits / to_bits);
  }
  Type result_type{dtype, new_shape};
  inst->GetResultsTypes()[0] = result_type;
}

static void RunOnInstruction(TFExtensionInst* inst) {
  if (inst->GetExtOpCode() == TFExtOpCode::MERGE) {
    for (auto& op : inst->GetOperands()) {
      if (op.GetType().IsValid()) {
        inst->GetResultsTypes()[0] = op.GetType();
        return;
      }
    }
    return;
  }
  if (inst->GetExtOpCode() == TFExtOpCode::SWITCH) {
    const auto& ty = inst->GetOperand(0).GetType();
    if (ty.IsValid()) {
      inst->GetResultsTypes() = {ty, ty};
    }
    return;
  }
}

static void RunOnInstruction(UniqueInst* inst) {
  const auto& type0 = inst->GetOperand(0).GetType();
  if (!type0.IsValid()) {
    return;
  }
  auto rank = type0.GetNumOfDims();
  HLCHECK(rank == 1);
  auto idx_dt = inst->GetOutIdxType();
  HLCHECK(idx_dt == DataType::INT64 || idx_dt == DataType::INT32);
  // FIXME: result 0 has at most the same number of
  // elements of the input
  inst->GetResultsTypes()[0] = type0;
  inst->GetResultsTypes()[1] = Type{idx_dt, type0.GetDimSizes()};
}

bool TypeLegalizer::RunOnBasicBlock(BasicBlock* bb) {
  bool changed = false;
  // Dedup names.
  std::unordered_map<std::string, int> names;
  for (auto& it : *bb) {
    auto inst = it.get();
    const auto& name = inst->GetName();
    int n = names[name];
    if (n > 0) {
      inst->SetName(name + "_" + std::to_string(n));
    }
    ++names[name];
  }

  for (auto& it : *bb) {
    Instruction* inst = it.get();
    auto orig_types = inst->GetResultsTypes();
    bool fixed = true;
    for (auto& ty : orig_types) {
      fixed &= (ty.IsValid() && !ty.IsDynamicShape());
    }
    if (fixed) {
      continue;
    }
    switch (inst->GetOpCode()) {
#define GET_INST_DOWNCAST_SWITCH
#include "halo/lib/ir/instructions_info.def"
#undef GET_INST_DOWNCAST_SWITCH
      case OpCode::EXTENSION: {
        TFExtensionInst* ext = DynCast<TFExtensionInst>(inst);
        if (ext != nullptr) {
          RunOnInstruction(ext);
        }
        break;
      }
      default: {
        if (!relaxed_) {
          // HLCHECK(0 && "Unreachable");
        }
      }
    }
    const auto& new_types = inst->GetResultsTypes();
    if (new_types.size() != orig_types.size()) {
      changed |= true;
    } else {
      for (int i = 0, e = orig_types.size(); i < e; ++i) {
        const auto& orig_ty = orig_types[i];
        const auto& new_ty = new_types[i];
        changed |= (!orig_ty.IsValid() && new_ty.IsValid()) ||
                   (orig_ty.IsValid() && new_ty.IsValid() && orig_ty != new_ty);
      }
    }
  }
  return changed;
}

} // end namespace halo
