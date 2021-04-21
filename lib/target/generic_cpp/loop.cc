//===- loop.cc ------------------------------------------------------------===//
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

#include <bits/stdint-uintn.h>

#include "halo/lib/ir/creator_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/transforms/transforms_util.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(LoopInst* inst) {
  HLCHECK(inst->GetNumOfOperands() >= 2);
  const auto& op_max_loop_cnt = inst->GetOperand(0);
  const auto& op_loop_cond = inst->GetOperand(1);

  bool cond_is_true = false;
  if (IsA<Constant>(op_loop_cond)) {
    auto c = DynCast<Constant>(op_loop_cond);
    if (c->GetResultType().GetDataType() == DataType::BOOL &&
        c->GetResultType().GetTotalNumOfElements() == 1 &&
        c->GetData<bool>(0)) {
      cond_is_true = true;
    }
  }

  auto body = inst->GetBody();
  auto return_inst = body->GetReturnInst();
  int total_output_nr = return_inst->GetNumOfOperands();
  unsigned scan_outputs =
      FindAttributeValue(*return_inst, "halo_scan_output_cnt", 0);
  std::vector<CXXValue> loop_regular_outs;
  std::vector<CXXValue> loop_scan_outs;

  loop_regular_outs.reserve(total_output_nr - scan_outputs);
  loop_scan_outs.reserve(scan_outputs);
  for (unsigned i = 0; i < total_output_nr - scan_outputs; ++i) {
    CXXValue cv(inst->GetName() + "_out_" + std::to_string(i), CXXType("void"));
    loop_regular_outs.push_back(cv);
    os_ << "  odla_value " << cv.name << " = " << EmitNull() << ";\n";
  }
  for (unsigned i = 0; i < scan_outputs; ++i) {
    CXXValue cv(inst->GetName() + "_out_s_" + std::to_string(i),
                CXXType("void"));
    loop_scan_outs.push_back(cv);
    os_ << "  odla_value " << cv.name << " = " << EmitNull() << ";\n";
  }
  os_ << "{\n";

  // Multiple loop body may use same node names.
  for (const auto& c : body->Constants()) {
    if (cond_is_true && c.get() == op_loop_cond.GetDef()) {
      continue;
    }
    RunOnConstant(*c, true);
    RunOnConstant(*c, false);
  }
  CXXValue null("", CXXType("void"));
  null.str_id = inst->GetName();

  EmitODLACall(null, "odla_BeginLoop", ir_mapping_[op_max_loop_cnt]);
  auto arg_nr = body->Args().size();
  HLCHECK(arg_nr == inst->GetNumOfOperands());
  int idx = 0;
  std::vector<CXXValue> loop_vars;
  for (const auto& arg : body->Args()) {
    auto val = ir_mapping_[inst->GetOperands()[idx]];
    if (idx >= 2) {
      // Various loops may use the same arg name.
      loop_vars.push_back({inst->GetName() + "_" + arg->GetName(), val.type});
      EmitODLACall(loop_vars.back(), "odla_CreateLoopVariable", val);
      val = loop_vars.back();
    }
    ir_mapping_[*arg] = val;
    ++idx;
  }
  auto in_return = [&return_inst](const Instruction& inst) {
    for (int idx = 0, e = return_inst->GetNumOfOperands(); idx < e; ++idx) {
      if (return_inst->GetOperand(idx).GetDef() == &inst) {
        return idx;
      }
    }
    return -1;
  };

  for (auto& inst : *body) {
    if (inst.get() == return_inst) {
      continue;
    }
    RunOnBaseInstruction(inst.get());
    if (int idx = in_return(*inst); idx >= 0) {
      EmitODLACall<2, false>(null, "odla_Assign", loop_vars[idx],
                             ir_mapping_[*inst]);
      ir_mapping_[*inst] = loop_vars[idx];
    } else if (HasAttribute(*inst, "halo_loop")) {
      EmitODLACall<2, false>(null, "odla_Assign",
                             loop_vars.back(), // FIXME(unknown):
                             ir_mapping_[*inst]);
      ir_mapping_[*inst] = loop_vars.back();
    }
  }
  std::string cond_var = cond_is_true || op_loop_cond.GetUses().empty()
                             ? EmitNull()
                             : ir_mapping_[op_loop_cond].GetName();
  // The output inside loop body used by EndLoop.
  std::vector<CXXValue> loop_internal_outs(loop_regular_outs.size());
  for (int64_t i = 0, e = loop_regular_outs.size(); i < e; ++i) {
    loop_internal_outs[i] = ir_mapping_[return_inst->GetOperand(i)];
  }

  std::string scan_masks = "(const odla_loop_output_mode[]){";
  for (unsigned i = 0; i < loop_regular_outs.size(); ++i) {
    if (i < scan_outputs) {
      scan_masks += "ODLA_LOOP_FWD_BEFORE, ";
      int64_t idx = i + loop_regular_outs.size();
      Def op{inst, static_cast<int>(idx)};
      ir_mapping_[op] = loop_scan_outs[i];
    } else {
      scan_masks += "ODLA_LOOP_LAST_VALUE, ";
    }
  }
  scan_masks += "}";
  EmitODLACall<2, true, false>(loop_scan_outs, "odla_EndLoop", cond_var,
                               loop_internal_outs, scan_masks);
  for (int64_t i = 0, e = loop_regular_outs.size(); i < e; ++i) {
    os_ << "  " << loop_regular_outs[i].name << " = "
        << loop_internal_outs[i].GetName() << ";\n";
    loop_regular_outs[i].str_id = loop_internal_outs[i].GetName();
    ir_mapping_[Def{inst, static_cast<int>(i)}] = loop_regular_outs[i];
  }

  os_ << "}\n";
}

} // namespace halo
