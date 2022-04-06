//===- extension_instructions.cc --------------------------------*- C++ -*-===//
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

#include "halo/lib/ir/extension_instructions.h"

namespace halo {
ExtensionInst::ExtensionInst(GlobalContext& context, const std::string& name,
                             const std::vector<Def>& operands, int num_outs,
                             const std::string& opname, OpCode opcode)
    : Instruction(context, name, std::max(0, num_outs), opcode) {
  AddOperands(operands);
  if (num_outs < 0) {
    SetVariadicReturns(true);
  }
  opname_ = opname;
}

bool ExtensionInst::IsOperandOptional(size_t idx) const noexcept {
  return optional_args_.count(idx) > 0;
}

void ExtensionInst::MarkOperandOptional(size_t idx) noexcept {
  optional_args_.insert(idx);
}

template <typename T>
static Instruction* CloneInst(const ExtensionInst& inst,
                              const std::string& opname_prefix) {
  auto opname = inst.GetOpName();
  if (!opname_prefix.empty()) {
    opname = opname.substr(opname.find_first_of(opname_prefix) +
                           opname_prefix.size());
  }
  auto new_inst =
      new T(inst.GetGlobalContext(), inst.GetName(),
            std::vector<Def>(inst.GetNumOfOperands(), Def::GetUndefined()),
            inst.GetNumOfResults(), opname);
  new_inst->CopyAttrsFrom(inst);
  return new_inst;
}

std::unique_ptr<Instruction> CustomInst::Clone() const {
  return std::unique_ptr<Instruction>(CloneInst<CustomInst>(*this, ""));
}

std::unique_ptr<Instruction> DummyInst::Clone() const {
  return std::unique_ptr<Instruction>(CloneInst<DummyInst>(*this, ""));
}

TFExtensionInst::TFExtensionInst(GlobalContext& context,
                                 const std::string& name,
                                 const std::vector<Def>& operands, int num_outs,
                                 const std::string& opname)
    : ExtensionInst(context, name, operands, num_outs, "tf_" + opname,
                    OpCode::EXTENSION) {
  auto it = TFMap.find(opname);
  if (it != TFMap.end()) {
    ext_op_code_ = it->second;
  } else {
    SetOpName("!tf_" + opname);
  }
}

std::unique_ptr<Instruction> TFExtensionInst::Clone() const {
  return std::unique_ptr<Instruction>(CloneInst<TFExtensionInst>(*this, "tf_"));
}

const TFExtensionInst::NameToTFOpMap TFExtensionInst::TFMap({
#define GET_TF_NAME_OPCODE_MAPPING
#include "halo/lib/ir/convert_tf_info.def"
#undef GET_TF_NAME_OPCODE_MAPPING
});

ONNXExtensionInst::ONNXExtensionInst(GlobalContext& context,
                                     const std::string& name,
                                     const std::vector<Def>& operands,
                                     int num_outs, const std::string& opname)
    : ExtensionInst(context, name, operands, num_outs, "onnx_" + opname,
                    OpCode::EXTENSION) {
  auto it = ONNXMap.find(opname);
  if (it != ONNXMap.end()) {
    ext_op_code_ = it->second;
  } else {
    SetOpName("!onnx_" + opname);
  }
}

std::unique_ptr<Instruction> ONNXExtensionInst::Clone() const {
  return std::unique_ptr<Instruction>(
      CloneInst<ONNXExtensionInst>(*this, "onnx_"));
}

const ONNXExtensionInst::NameToOpMap ONNXExtensionInst::ONNXMap({
#define GET_ONNX_NAME_OPCODE_MAPPING
#include "halo/lib/ir/convert_onnx_info.def"
#undef GET_ONNX_NAME_OPCODE_MAPPING
});

TFLITEExtensionInst::TFLITEExtensionInst(GlobalContext& context,
                                         const std::string& name,
                                         const std::vector<Def>& operands,
                                         int num_outs,
                                         const std::string& opname)
    : ExtensionInst(context, name, operands, num_outs, "tflite_" + opname,
                    OpCode::EXTENSION) {
  auto it = TFLITEMap.find(opname);
  if (it != TFLITEMap.end()) {
    ext_op_code_ = it->second;
  } else {
    SetOpName("!tflite_" + opname);
  }
}

const TFLITEExtensionInst::NameToOpMap TFLITEExtensionInst::TFLITEMap({
#define GET_TFLITE_NAME_OPCODE_MAPPING
#include "halo/lib/ir/convert_tflite_info.def"
#undef GET_TFLITE_NAME_OPCODE_MAPPING
});

std::unique_ptr<Instruction> TFLITEExtensionInst::Clone() const {
  return std::unique_ptr<Instruction>(
      CloneInst<TFLITEExtensionInst>(*this, "tflite_"));
}

CAFFEExtensionInst::CAFFEExtensionInst(GlobalContext& context,
                                       const std::string& name,
                                       const std::vector<Def>& operands,
                                       int num_outs, const std::string& opname)
    : ExtensionInst(context, name, operands, num_outs, "caffe_" + opname,
                    OpCode::EXTENSION) {
  auto it = CAFFEMap.find(opname);
  if (it != CAFFEMap.end()) {
    ext_op_code_ = it->second;
  } else {
    SetOpName("!caffe_" + opname);
  }
}
std::unique_ptr<Instruction> CAFFEExtensionInst::Clone() const {
  return std::unique_ptr<Instruction>(
      CloneInst<CAFFEExtensionInst>(*this, "caffe_"));
}

const CAFFEExtensionInst::NameToOpMap CAFFEExtensionInst::CAFFEMap({
#define GET_CAFFE_NAME_OPCODE_MAPPING
#include "halo/lib/ir/convert_caffe_info.def"
#undef GET_CAFFE_NAME_OPCODE_MAPPING
});

} // end namespace halo