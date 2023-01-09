//===- dynamic_type_legalizer.cc ------------------------------------------===//
//
// Copyright (C) 2019-2022 Alibaba Group Holding Limited.
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
#include "halo/lib/transforms/type_legalizer.h"

namespace halo {
static void RunOnInstruction(Instruction* inst) {}
static void RunOnInstruction(ReshapeInst* inst) {
  auto& op0_type = inst->GetOperand(0).GetType();
  Def op1 = inst->GetOperand(1);
  const auto& op1_type = op1.GetType();
  std::vector<int64_t> new_shape;
  if (op1_type.IsValid() && !IsA<Constant>(op1)) {
    for (size_t i = 0; i < op1_type.GetNumOfDims(); ++i) {
      new_shape.push_back(-1);
    }
    halo::Type new_type{op0_type.GetDataType(), new_shape};
    inst->GetResultsTypes()[0] = new_type;
  }
}

static void RunOnInstruction(UnsqueezeInst* inst) {
  auto op0 = inst->GetOperand(0);
  const auto& op0_type = op0.GetType();
  auto op1 = inst->GetOperand(1);
  const auto& op1_type = op1.GetType();

  if (!op0.GetType().IsValid()) {
    return;
  }
  std::vector<int64_t> ret_shape(
      (op1_type.GetNumOfElementsInDim(0) + op0_type.GetNumOfDims()), -1);
  Type new_type{op0_type.GetDataType(), ret_shape};
  inst->GetResultsTypes()[0] = new_type;
}

bool DynamicTypeLegalizer::RunOnBasicBlock(BasicBlock* bb) {
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
