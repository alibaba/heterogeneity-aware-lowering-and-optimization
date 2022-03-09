//===- slice.cc -----------------------------------------------------------===//
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
#include <string>
#include <unordered_set>

#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/constant.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/ir/nn_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

namespace {

template <typename T>
static void NormalizerOperands(const Constant& operand,
                               const std::unordered_set<int32_t>& axes,
                               const size_t dims, std::vector<int32_t>* value) {
  bool onnx_mode = axes.size() != dims;
  for (size_t i = 0, j = 0; i < dims; ++i) {
    if (axes.count(i) != 0) {
      (*value)[i] = static_cast<int32_t>(operand.GetData<T>(j++));
    } else {
      if (!onnx_mode) {
        (*value)[i] = static_cast<int32_t>(operand.GetData<T>(i));
      }
    }
  }
}

} // end namespace

void GenericCXXCodeGen::RunOnInstruction(SliceInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& start = inst->GetOperand(1);
  const Def& size = inst->GetOperand(2);
  const Def& strides =
      inst->GetNumOfOperands() > 3 ? inst->GetOperand(3) : Def::GetUndefined();

  CXXValue op0 = ir_mapping_[input];
  CXXValue ret(inst->GetName(), op0.type);
  ir_mapping_[*inst] = ret;

  if (!IsA<Constant>(start) || !IsA<Constant>(size) ||
      ((inst->GetNumOfOperands() > 3) && !IsA<Constant>(strides))) {
    auto op1 = ir_mapping_[start];
    auto op2 = ir_mapping_[size];
    auto op3 = ir_mapping_[strides];
    EmitODLACall(ret, "odla_SliceDynamic", op0, op1, op2, op3,
                 EmitShape(inst->GetResultType()));

    return;
  }
  size_t dims = input.GetType().GetNumOfDims();
  std::unordered_set<int32_t> axes;
  if (inst->GetNumOfOperands() > 4) {
    const Def& op4 = inst->GetOperand(4);
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

  std::vector<int32_t> start_v(dims, 0);
  HLCHECK(start.GetType().GetTotalNumOfElements() ==
          static_cast<int64_t>(axes.size()));
  HLCHECK(IsA<Constant>(start));
  const Constant* start_c = DynCast<Constant>(start);
  if (start_c->GetResultType().GetDataType() == DataType::INT32) {
    NormalizerOperands<int32_t>(*start_c, axes, dims, &start_v);
  } else if (start_c->GetResultType().GetDataType() == DataType::INT64) {
    NormalizerOperands<int64_t>(*start_c, axes, dims, &start_v);
  }

  std::vector<int32_t> size_v(input.GetType().GetDimSizes().begin(),
                              input.GetType().GetDimSizes().end());
  HLCHECK(size.GetType().GetTotalNumOfElements() ==
          static_cast<int64_t>(axes.size()));
  HLCHECK(IsA<Constant>(size));
  const Constant* size_c = DynCast<Constant>(size);
  if (size_c->GetResultType().GetDataType() == DataType::INT32) {
    NormalizerOperands<int32_t>(*size_c, axes, dims, &size_v);
  } else if (size_c->GetResultType().GetDataType() == DataType::INT64) {
    NormalizerOperands<int64_t>(*size_c, axes, dims, &size_v);
  }

  std::vector<int32_t> strides_v(dims, 1);
  if (inst->GetNumOfOperands() > 3) {
    const Constant* strides_c = DynCast<Constant>(strides);
    HLCHECK(strides.GetType().GetTotalNumOfElements() ==
            static_cast<int64_t>(axes.size()));
    if (strides_c->GetResultType().GetDataType() == DataType::INT32) {
      NormalizerOperands<int32_t>(*strides_c, axes, dims, &strides_v);
    } else if (strides_c->GetResultType().GetDataType() == DataType::INT64) {
      NormalizerOperands<int64_t>(*strides_c, axes, dims, &strides_v);
    }

    // stride is provided, calculate ends = starts + sizes * strides
    std::for_each(strides_v.begin(), strides_v.end(),
                  [=](int32_t& s) { s = s >= 0 ? s : dims + s; });
    std::transform(strides_v.begin(), strides_v.end(), size_v.begin(),
                   size_v.begin(), std::multiplies<int32_t>());
    std::transform(start_v.begin(), start_v.end(), size_v.begin(),
                   size_v.begin(), std::plus<int32_t>());
  } else {
    // stride is omitted, set to [1,1,...,1], calculate ends = starts + sizes
    std::transform(size_v.begin(), size_v.end(), start_v.begin(),
                   size_v.begin(), std::plus<int32_t>());
  }

  EmitODLACall(ret, "odla_Slice", op0, start_v, size_v, strides_v,
               EmitShape(inst->GetResultType()));
}

void GenericCXXCodeGen::RunOnInstruction(SliceDynamicInst* inst) {
  const Def& input = inst->GetOperand(0);
  const Def& start = inst->GetOperand(1);
  const Def& size = inst->GetOperand(2);
  const Def& strides =
      inst->GetNumOfOperands() > 3 ? inst->GetOperand(3) : Def::GetUndefined();

  CXXValue op0 = ir_mapping_[input];
  CXXValue ret(inst->GetName(), op0.type);
  ir_mapping_[*inst] = ret;

  if (!IsA<Constant>(start) || !IsA<Constant>(size) ||
      ((inst->GetNumOfOperands() > 3) && !IsA<Constant>(strides))) {
    auto op1 = ir_mapping_[start];
    auto op2 = ir_mapping_[size];
    auto op3 = ir_mapping_[strides];
    EmitODLACall(ret, "odla_SliceDynamic", op0, op1, op2, op3,
                 EmitShape(inst->GetResultType()));

    return;
  }
}

} // namespace halo
