//===- analyzer.cc --------------------------------------------------------===//
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

#include "halo/lib/transforms/analyzer.h"

#include <iostream>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/data_layout.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/ir/nn_cnn_instructions.h"

namespace halo {

namespace {

float GetNumOfOperators(halo::Instruction* inst) {
  switch (inst->GetOpCode()) {
    case OpCode::SIGMOID:
    case OpCode::SOFTMAX:
    case OpCode::BATCHNORM:
      return 4.0; // NOLINT
    case OpCode::POOLINGAVG:
    case OpCode::POOLINGMAX:
    case OpCode::RELU:
      return 1.0; // NOLINT
    default:
      return 2.0; // NOLINT
  }
}

Analyzer::NodeInfo& GenerateCommonInfo(
    const Instruction* inst, std::vector<Analyzer::NodeInfo>* node_infos) {
  (*node_infos).emplace_back();
  Analyzer::NodeInfo& node_info = node_infos->back();
  node_info.id = node_infos->size();
  node_info.name = inst->GetName();
  node_info.type = inst->GetOpCode();
  node_info.data_type = inst->GetOperand(0).GetType().GetDataType();
  node_info.input_shape = inst->GetOperand(0).GetType().GetDimSizes();
  node_info.output_shape = inst->GetResultType().GetDimSizes();
  return node_info;
}

// Pool computational estimator: Kh * Kw * Hout * Wout * Cout
Analyzer::NodeInfo& CalcPoolFLOPs(const std::vector<int>& kernel_shape,
                                  const DataFormat& data_format,
                                  const float operator_num,
                                  Analyzer::NodeInfo* node_info) {
  // Pooling window size format [1, Kh, Kw, 1]
  switch (data_format) {
    case DataFormat::NCHW: {
      node_info->flops = operator_num * kernel_shape.back() *
                         kernel_shape[kernel_shape.size() - 1];
      break;
    }
    case DataFormat::NHWC: {
      node_info->flops = operator_num * kernel_shape[kernel_shape.size() - 2] *
                         kernel_shape[kernel_shape.size() - 3];
      break;
    }
    default: {
      HLCHECK(0 && "Invalid format");
    }
  }
  return *node_info;
}

} // anonymous namespace

static void RunOnInstruction(Instruction* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
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
      auto& node_info = GenerateCommonInfo(inst, node_infos);
      node_info.flops = inst->GetResultType().GetTotalNumOfElements();
      DefaultDataLayout dl; 
      node_info.activation = node_info.flops * dl.Bytes(inst->GetResultType().GetDataType());
      break;
    }
    default: {
      HLCHECK(0 && "Unimplemented");
    }
  }
}

static void RunOnInstruction(ArgmaxInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(BatchMatMulInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);

  // batch matmul computational estimator: batch * ((2 * Cin - 1) * Cout)
  const auto& input_type = inst->GetOperand(0).GetType();
  const size_t batch =
      input_type.GetNumOfElementsInDim(input_type.GetNumOfDims());

  const auto& weight_inst = inst->GetOperand(1);
  HLCHECK(IsA<Constant>(weight_inst));
  const auto& weight_type = weight_inst.GetType();
  node_info.weight = static_cast<float>(weight_type.GetTotalNumOfElements());

  const Constant* weight = DynCast<Constant>(weight_inst);
  node_info.sizeof_dt = weight->GetElementSizeInBytes();

  const size_t dims = weight_type.GetNumOfDims();
  const int64_t col = inst->GetTransposeB()
                          ? weight_type.GetNumOfElementsInDim(dims - 2)
                          : weight_type.GetNumOfElementsInDim(dims - 1);

  node_info.flops = batch * (GetNumOfOperators(inst) * node_info.weight - col);
  node_info.activation = 
      node_info.sizeof_dt * node_info.weight + inst->GetResultType().GetTotalNumOfElements();
}

static void RunOnInstruction(BatchNormInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  // z = gamma * (y - mean) / sqrt(variance + epsilon) + beta
  // BatchNorm computational estimator: Cin * 4
  const auto& input_type = inst->GetOperand(0).GetType();
  node_info.flops = GetNumOfOperators(inst) * input_type.GetTotalNumOfElements();
  node_info.activation = 
      static_cast<float>(node_info.sizeof_dt * inst->GetResultType().GetTotalNumOfElements());
}

static void RunOnInstruction(ConcatInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(Conv2DInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);

  const auto& weight_inst = inst->GetOperand(1);
  HLCHECK(IsA<Constant>(weight_inst));
  node_info.weight =
      static_cast<float>(weight_inst.GetType().GetTotalNumOfElements());
  const Constant* weight = DynCast<Constant>(weight_inst);
  node_info.sizeof_dt = weight->GetElementSizeInBytes();

  // TODO(unkonwn) process group and bias
  // Conv computational estimator: 2 * Kh * Kw * Cin * Hout * Wout * Cout
  const auto& out_type = inst->GetResultType();
  const size_t dims = out_type.GetNumOfDims();
  HLCHECK(dims == 4);
  node_info.flops = GetNumOfOperators(inst) * node_info.weight;
  switch (inst->GetDataFormat()) {
    case DataFormat::NCHW: {
      node_info.flops *=
          static_cast<float>(out_type.GetNumOfElementsInDim(dims - 2) *
                             out_type.GetNumOfElementsInDim(dims - 1));
      break;
    }
    case DataFormat::NHWC: {
      node_info.flops *=
          static_cast<float>(out_type.GetNumOfElementsInDim(dims - 3) *
                             out_type.GetNumOfElementsInDim(dims - 2));
      break;
    }
    default: {
      HLCHECK(0 && "Invalid format");
    }
  }
  node_info.activation = 
     node_info.sizeof_dt * (node_info.weight + inst->GetResultType().GetTotalNumOfElements());
}

static void RunOnInstruction(GatherInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  //HLCHECK(0 && "Unimplemented");
  auto& node_info = GenerateCommonInfo(inst, node_infos); 
  DefaultDataLayout dl; 
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
  node_info.activation = node_info.flops * dl.Bytes(inst->GetResultType().GetDataType());
}

static void RunOnInstruction(GemmInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) { 
  auto& node_info = GenerateCommonInfo(inst, node_infos);

  //const auto& matrix_a = inst->GetOperand(0); 
  const auto& matrix_b = inst->GetOperand(1); 
  const auto& matrix_c = inst->GetOperand(2);

  //const size_t matrixa_sz = static_cast<float>(matrix_a.GetType().GetTotalNumOfElements()); 
  const size_t matrixb_sz = static_cast<float>(matrix_b.GetType().GetTotalNumOfElements()); 
  const size_t matrixc_sz = static_cast<float>(matrix_c.GetType().GetTotalNumOfElements());   
  
  const Constant* b = DynCast<Constant>(matrix_b);
  node_info.sizeof_dt = b->GetElementSizeInBytes();

  // GEMM computational estimator: out = alpha * A' * B' + beta * C
  const size_t dims = matrix_b.GetType().GetNumOfDims();
  const int64_t row_a = matrix_c.GetType().GetNumOfElementsInDim(0); 
  const int64_t col_b = inst->GetTransposeB()
                          ? matrix_b.GetType().GetNumOfElementsInDim(dims - 2)
                          : matrix_b.GetType().GetNumOfElementsInDim(dims - 1);
  
  node_info.flops = static_cast<float>(row_a * (2 * matrixb_sz - col_b) + 2 * matrixc_sz);
  node_info.activation = node_info.sizeof_dt * static_cast<float>(matrixb_sz + matrixc_sz);
}

static void RunOnInstruction(MatMulInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);

  const auto& weight_inst = inst->GetOperand(1);
  HLCHECK(IsA<Constant>(weight_inst));
  const auto& weight_type = weight_inst.GetType();
  node_info.weight = static_cast<float>(weight_type.GetTotalNumOfElements());

  const Constant* weight = DynCast<Constant>(weight_inst);
  node_info.sizeof_dt = weight->GetElementSizeInBytes();

  // matmul computational estimator: (2 * Cin - 1) * Cout
  size_t dims = weight_type.GetNumOfDims();
  const int64_t col = inst->GetTransposeB()
                          ? weight_type.GetNumOfElementsInDim(dims - 2)
                          : weight_type.GetNumOfElementsInDim(dims - 1);
  const auto& matrixa_type = inst->GetOperand(0).GetType();
  dims = matrixa_type.GetNumOfDims();
  const int64_t row = inst->GetTransposeA()
                          ? matrixa_type.GetNumOfElementsInDim(dims - 1)
                          : matrixa_type.GetNumOfElementsInDim(dims - 2);

  node_info.activation = node_info.sizeof_dt * (node_info.weight + inst->GetResultType().GetTotalNumOfElements());
  node_info.flops = row * (2 * node_info.weight - col);
}

static void RunOnInstruction(OneHotInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(PadInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(PoolingMaxInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  CalcPoolFLOPs(inst->GetKsize(), inst->GetDataFormat(),
                GetNumOfOperators(inst), &node_info);
}

static void RunOnInstruction(PoolingAvgInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  CalcPoolFLOPs(inst->GetKsize(), inst->GetDataFormat(),
                GetNumOfOperators(inst), &node_info);
}

static void RunOnInstruction(RangeInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(RandomUniformInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(ReduceMaxInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(ReduceMeanInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  node_info.flops = GetNumOfOperators(inst) *
                    inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

static void RunOnInstruction(ReduceProductInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(ReluInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  node_info.flops = inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

static void RunOnInstruction(Relu6Inst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  node_info.flops = inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

static void RunOnInstruction(ReshapeInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(ReturnInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  // do nothing
}

static void RunOnInstruction(SetDiff1DInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(SliceInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(SoftmaxInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  node_info.flops = GetNumOfOperators(inst) *
                    inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

static void RunOnInstruction(StackInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(TransposeInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

static void RunOnInstruction(ZExtInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  HLCHECK(0 && "Unimplemented");
}

bool Analyzer::RunOnBasicBlock(BasicBlock* bb) {
  node_infos_.clear();
  for (const auto& it : *bb) {
    Instruction* inst = it.get();
    switch (inst->GetOpCode()) {
#define GET_INST_DOWNCAST_SWITCH_TAKE_EXTRA_PARAM
#include "halo/lib/ir/instructions_info.def"
#undef GET_INST_DOWNCAST_SWITCH_TAKE_EXTRA_PARAM
      default: {
        continue;
      }
    }
  }
  return false;
}

void Analyzer::WriteCSVReport(std::ostream& os) const {
  static constexpr float mflops = 1000000.0F;
  static constexpr float gflops = 1000 * mflops;
  static constexpr float mb = 1024 * 1024.0F;
  static constexpr int ratio = 100;

  float total_flops = 0;
  float total_weights = 0;
  float total_activations = 0;
  for (const auto& it : node_infos_) {
    total_flops += it.flops;
    total_weights += it.weight;
    total_activations += it.activation;
  }

  os << "Analysis Report\n"
     << "layerID, "
     << "layerName, "
     << "opName, "
     << "type, "
     << "input-shape, "
     << "output-shape, "
     << "MFLOPs, "
     << "weight(MB), "
     << "activation(MB), "
     << "percent(%)\n";

  for (const auto& it : node_infos_) {
    os << it.id << ", " << it.name << ", "
       << Instruction::OpCodeToString(it.type) << ", "
       << Type::DataTypeToString(it.data_type) << ", (";
    for (const auto& ii : it.input_shape) {
      os << ii << " ";
    }
    os << "), (";
    for (const auto& ii : it.output_shape) {
      os << ii << " ";
    }
    os << "), " << it.flops / mflops << ", " << it.sizeof_dt * it.weight / mb
       << ", " << it.activation / mb << ", " << ratio * it.flops / total_flops
       << "\n";
  }

  os << "\nTotal layers: " << node_infos_.size()
     << "\nTotal GFLOPs: " << total_flops / gflops
     << "\nTotal weights(MB): " << sizeof(float) * total_weights / mb
     << "\nTotal activations(MB): " << total_activations / mb << "\n";
}

} // namespace halo
