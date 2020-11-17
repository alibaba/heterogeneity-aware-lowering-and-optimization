//===- reorder_channel.cc -------------------------------------------------===//
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

#include "halo/lib/transforms/reorder_channel.h"

#include <iterator>
#include <unordered_set>

#include "halo/lib/framework/common.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/ir/nn_instructions.h"

namespace halo {

const std::vector<int>& GetPerm(
    bool is_2d, bool target_is_ch_first, bool is_weight,
    DataFormat weight_format = DataFormat::INVALID) {
  static const std::vector<int> nhwc2nchw{0, 3, 1, 2};
  static const std::vector<int> nchw2nhwc{0, 2, 3, 1};
  static const std::vector<int> ndhwc2ncdhw{0, 4, 1, 2, 3};
  static const std::vector<int> ncdhw2ndhwc{0, 2, 3, 4, 1};

  static const std::vector<int> hwio2oihw{3, 2, 0, 1};
  static const std::vector<int> iohw2hwio{2, 3, 0, 1};

  static const std::vector<int> oihw2hwio{2, 3, 1, 0};
  static const std::vector<int> dhwio2oidhw{4, 3, 0, 1, 2};
  static const std::vector<int> oidhw2dhwio{2, 3, 4, 1, 0};

  if (is_weight) {
    if (is_2d) {
      return target_is_ch_first
                 ? hwio2oihw
                 : (weight_format == DataFormat::CNHW ? iohw2hwio : oihw2hwio);
    }
    return target_is_ch_first ? dhwio2oidhw : oidhw2dhwio;
  }
  if (is_2d) {
    return target_is_ch_first ? nhwc2nchw : nchw2nhwc;
  }
  return target_is_ch_first ? ndhwc2ncdhw : ncdhw2ndhwc;
}

static std::unordered_map<IRObject*, IRObject*>
    Cache; // FIXME: use def,idx pair

static void UpdateFilterFormat(Conv2DTransposeInst* inst, DataFormat format) {
  if (format == DataFormat::NCHW) {
    inst->SetFilterFormat(DataFormat::CNHW);
  } else {
    inst->SetFilterFormat(DataFormat::HWCN);
  }
}

static void UpdateFilterFormat(Conv2DInst* inst, DataFormat format) {
  if (format == DataFormat::NCHW) {
    inst->SetFilterFormat(DataFormat::NCHW);
  } else {
    inst->SetFilterFormat(DataFormat::HWCN);
  }
}

template <typename T>
static void UpdateKernel(T* pooling_inst, const std::vector<int>& perm) {
  const auto& orig_kernel = pooling_inst->GetKsize();
  auto new_kernel = orig_kernel;
  HLCHECK(new_kernel.size() == perm.size());
  for (int i = 0, e = perm.size(); i < e; ++i) {
    new_kernel[i] = orig_kernel[perm[i]];
  }
  pooling_inst->SetKsize(new_kernel);
}

template <typename T>
static void UpdateStrides(T* inst, const std::vector<int>& perm) {
  const auto& orig_strides = inst->GetStrides();
  auto new_strides = orig_strides;
  HLCHECK(new_strides.size() == perm.size());
  for (size_t i = 0, e = perm.size(); i < e; ++i) {
    new_strides[i] = inst->GetStrides()[perm[i]];
  }
  inst->SetStrides(new_strides);
}

// This templated function is for instructions that has SetDataFormat().
template <typename T>
static bool Reorder(T* inst, bool target_is_ch_first) {
  // Get traits for inst
  constexpr bool has_kernels =
      std::is_same_v<T, PoolingAvgInst> || std::is_same_v<T, PoolingMaxInst>;
  constexpr bool has_filter =
      std::is_same_v<T, Conv2DInst> || std::is_same_v<T, Conv2DTransposeInst>;
  constexpr bool has_strides = has_kernels || has_filter;
  constexpr bool is_bn = std::is_same_v<T, BatchNormInst>;

  if (is_bn) {
    // Do transform after the instruction is legalized.
    if (inst->GetNumOfOperands() != 5) { // NOLINT.
      return false;
    }
  }

  const auto src_format = inst->GetDataFormat();
  bool src_is_ch_first =
      src_format == DataFormat::NCHW || src_format == DataFormat::NCDHW;
  if (src_is_ch_first ^ target_is_ch_first) {
    DataFormat dst_format;
    bool is_2d =
        src_format == DataFormat::NHWC || src_format == DataFormat::NCHW;
    if (target_is_ch_first) {
      dst_format = is_2d ? DataFormat::NCHW : DataFormat::NCDHW;
    } else {
      dst_format = is_2d ? DataFormat::NHWC : DataFormat::NDHWC;
    }
    inst->SetDataFormat(dst_format);

    IRBuilder builder(inst->GetParent());
    builder.SetInsertBefore(inst);

    auto get_prefix = [](bool ch_first) {
      return ch_first ? std::string("ncs_") : std::string("nsc_");
    };

    for (size_t i = 0, e = inst->GetNumOfOperands(); i < e; ++i) {
      const auto& op = inst->GetOperand(i);
      if (const auto& ty = op.GetDef()->GetResultType();
          ty.IsValid() && ty.GetNumOfDims() == 1) {
        continue;
      }
      if (Cache.count(op.GetDef())) {
        inst->ReplaceOperandWith(i, *Cache[op.GetDef()]);
      } else {
        TransposeInst* trans = builder.CreateTranspose(
            get_prefix(target_is_ch_first) + op.GetOwner()->GetName(), {op});
        auto weight_format = DataFormat::INVALID;
        if constexpr (has_filter) { // NOLINT
          weight_format = inst->GetFilterFormat();
        }
        const auto& perm =
            GetPerm(is_2d, target_is_ch_first, i == 1, weight_format);
        trans->SetPermutation(perm);
        inst->ReplaceOperandWith(i, *trans);
        Cache[op.GetDef()] = trans;
      }
      // For BN, only insert transpose on the first input.
      if (is_bn) {
        break;
      }
    }
    const auto& perm = GetPerm(is_2d, target_is_ch_first, false);
    if constexpr (has_strides) { // NOLINT
      UpdateStrides(inst, perm);
    }
    if constexpr (has_filter) { // NOLINT
      UpdateFilterFormat(inst, dst_format);
    }
    if constexpr (has_kernels) { // NOLINT
      UpdateKernel(inst, perm);
    }

    if (auto type = inst->GetResultType(); type.IsValid()) {
      // permute the result shape
      const auto& dims = type.GetDimSizes();
      auto new_dims = dims;
      for (size_t i = 0, e = dims.size(); i < e; ++i) {
        new_dims[i] = dims[perm[i]];
      }
      inst->GetResultsTypes()[0] = Type{type.GetDataType(), new_dims};
    }

    builder.SetInsertAfter(inst);
    // Transpose back to original result layout.
    TransposeInst* trans = builder.CreateTranspose(
        get_prefix(!target_is_ch_first) + inst->GetName(), {*inst});
    trans->SetPermutation(GetPerm(is_2d, !target_is_ch_first, false));
    inst->ReplaceAllUsesWith(0, *trans);
    return true;
  }
  return false;
}

bool ReorderChannel::RunOnBasicBlock(BasicBlock* bb) {
  bool changed = false;
  for (auto& inst_it : *bb) {
    Instruction* inst = inst_it.get();
    switch (inst->GetOpCode()) {
      case OpCode::CONV2D: {
        Conv2DInst* conv = DynCast<Conv2DInst>(inst);
        changed |= Reorder(conv, channel_first_);
        break;
      }
      case OpCode::CONV2DTRANSPOSE: {
        Conv2DTransposeInst* deconv = DynCast<Conv2DTransposeInst>(inst);
        changed |= Reorder(deconv, channel_first_);
        break;
      }
      case OpCode::POOLINGMAX: {
        PoolingMaxInst* pool = DynCast<PoolingMaxInst>(inst);
        changed |= Reorder(pool, channel_first_);
        break;
      }
      case OpCode::POOLINGAVG: {
        PoolingAvgInst* pool = DynCast<PoolingAvgInst>(inst);
        changed |= Reorder(pool, channel_first_);
        break;
      }
      case OpCode::BATCHNORM: {
        BatchNormInst* bn = DynCast<BatchNormInst>(inst);
        changed |= Reorder(bn, channel_first_);
        break;
      }
      default: {
        continue;
      }
    }
  }

  return changed;
}

} // end namespace halo
