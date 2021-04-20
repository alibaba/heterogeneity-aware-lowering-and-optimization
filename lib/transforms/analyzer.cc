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

#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_map>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/data_layout.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/ir/nn_cnn_instructions.h"

namespace halo {

static DefaultDataLayout Ddl;
static DataLayout* Dl = &Ddl;

std::unordered_map<std::string, unsigned int>
    InputTensorApp; // input tensor appearance
std::unordered_map<std::string, size_t>
    IoTensorSize;                  // size of input output tensors
std::set<std::string> AliveTensor; // live tensor buffer

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

  // input shape
  auto ip_num = inst->GetNumOfOperands();
  for (size_t i = 0; i < ip_num; ++i) {
    const auto& ip_type = inst->GetOperand(i).GetType();
    if (ip_type.IsScalar()) {
      std::vector<int64_t> vec{1};
      node_info.input_shape.push_back(vec);
    } else {
      node_info.input_shape.push_back(ip_type.GetDimSizes());
    }

    if (i == 0) {
      node_info.data_type = ip_type.GetDataType();
    }

    if (IsA<Constant>(inst->GetOperand(i))) {
      node_info.weight_mem += Dl->Bytes(ip_type);
    } else {
      std::string name = inst->GetOperand(i).GetOwner()->GetName();
      if (InputTensorApp[name] > 0) {
        InputTensorApp[name]--;
      }
      AliveTensor.insert(name);
    }
  }

  // output shape
  const auto& op_type = inst->GetResultType();
  if (op_type.IsScalar()) {
    node_info.output_shape.push_back(1);
  } else {
    node_info.output_shape = op_type.GetDimSizes();
  }
  AliveTensor.insert(node_info.name);

  for (auto iter = AliveTensor.begin(); iter != AliveTensor.end();) {
    node_info.io_mem += IoTensorSize[*iter];
    if (InputTensorApp[*iter] == 0) {
      iter = AliveTensor.erase(iter);
    } else {
      iter++;
    }
  }

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
    case OpCode::CMP:
    case OpCode::ERF:
    case OpCode::SQRT:
    case OpCode::RSQRT:
    case OpCode::FLOOR:
    case OpCode::SITOFP: {
      auto& node_info = GenerateCommonInfo(inst, node_infos);
      node_info.flops = inst->GetResultType().GetTotalNumOfElements();
      break;
    }
    default: {
      HLCHECK(0 && "Unimplemented");
    }
  }
}

static void RunOnInstruction(ArgmaxInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(BatchMatMulInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);

  // batch matmul computational estimator: batch * ((2 * Cin - 1) * Cout)
  const auto& input_type = inst->GetOperand(0).GetType();
  size_t batch = 1;
  if (input_type.GetNumOfDims() >= 1) {
    batch = input_type.GetNumOfElementsInDim(input_type.GetNumOfDims() - 1);
  }

  const auto& matrixb = inst->GetOperand(1);
  const auto& matrixb_type = matrixb.GetType();
  size_t matb_size = matrixb_type.GetTotalNumOfElements();

  size_t dims = matrixb_type.GetNumOfDims();
  int64_t col = 1;
  int64_t row = 1;
  if (dims >= 2) {
    col = inst->GetTransposeB() ? matrixb_type.GetNumOfElementsInDim(dims - 2)
                                : matrixb_type.GetNumOfElementsInDim(dims - 1);
  } else if (dims == 1) {
    col = matrixb_type.GetNumOfElementsInDim(0);
  }
  const auto& matrixa_type = inst->GetOperand(0).GetType();
  dims = matrixa_type.GetNumOfDims();
  if (dims >= 2) {
    row = inst->GetTransposeA() ? matrixa_type.GetNumOfElementsInDim(dims - 1)
                                : matrixa_type.GetNumOfElementsInDim(dims - 2);
  } else if (dims == 1) {
    row = matrixa_type.GetNumOfElementsInDim(0);
  }

  node_info.flops = static_cast<float>(row * (2 * matb_size - col) * batch);
  node_info.weight_mem *= batch;
}

static void RunOnInstruction(BatchNormInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  // z = gamma * (y - mean) / sqrt(variance + epsilon) + beta
  // BatchNorm computational estimator: Cin * 4
  const auto& input_type = inst->GetOperand(0).GetType();
  node_info.flops =
      GetNumOfOperators(inst) * input_type.GetTotalNumOfElements();
}

static void RunOnInstruction(ConcatInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(Conv2DInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);

  const auto& weight_op = inst->GetOperand(1);
  const auto& weight_type = weight_op.GetType();
  size_t weight_size = weight_type.GetTotalNumOfElements();

  // TODO(unkonwn) process group and bias
  // Conv computational estimator: 2 * Kh * Kw * Cin * Hout * Wout * Cout
  const auto& out_type = inst->GetResultType();
  const size_t dims = out_type.GetNumOfDims();
  HLCHECK(dims == 4);
  node_info.flops = GetNumOfOperators(inst) * weight_size;
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
}

static void RunOnInstruction(GatherInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

static void RunOnInstruction(GemmInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);

  const auto& matrix_b = inst->GetOperand(1);
  const auto& matrix_c = inst->GetOperand(2);

  const size_t matrixb_sz = matrix_b.GetType().GetTotalNumOfElements();
  const size_t matrixc_sz = matrix_c.GetType().GetTotalNumOfElements();

  // GEMM computational estimator: out = alpha * A' * B' + beta * C
  const size_t dims = matrix_b.GetType().GetNumOfDims();
  const int64_t row_a = matrix_c.GetType().GetNumOfElementsInDim(0);
  const int64_t col_b =
      inst->GetTransposeB()
          ? matrix_b.GetType().GetNumOfElementsInDim(dims - 2)
          : matrix_b.GetType().GetNumOfElementsInDim(dims - 1);

  node_info.flops =
      static_cast<float>(row_a * (2 * matrixb_sz - col_b) + 2 * matrixc_sz);
}

static void RunOnInstruction(MatMulInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);

  const auto& matrixb = inst->GetOperand(1);
  const auto& matrixb_type = matrixb.GetType();
  size_t matb_size = matrixb_type.GetTotalNumOfElements();

  // matmul computational estimator: (2 * Cin - 1) * Cout
  size_t dims = matrixb_type.GetNumOfDims();
  int64_t col = 1;
  int64_t row = 1;
  if (dims >= 2) {
    col = inst->GetTransposeB() ? matrixb_type.GetNumOfElementsInDim(dims - 2)
                                : matrixb_type.GetNumOfElementsInDim(dims - 1);
  } else if (dims == 1) {
    col = matrixb_type.GetNumOfElementsInDim(0);
  }
  const auto& matrixa_type = inst->GetOperand(0).GetType();
  dims = matrixa_type.GetNumOfDims();
  if (dims >= 2) {
    row = inst->GetTransposeA() ? matrixa_type.GetNumOfElementsInDim(dims - 1)
                                : matrixa_type.GetNumOfElementsInDim(dims - 2);
  } else if (dims == 1) {
    row = matrixa_type.GetNumOfElementsInDim(0);
  }

  node_info.flops = static_cast<float>(row * (2 * matb_size - col));
}

static void RunOnInstruction(OneHotInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
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
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(RandomUniformInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(ReduceMaxInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(ReduceMeanInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  node_info.flops = GetNumOfOperators(inst) *
                    inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

static void RunOnInstruction(ReduceProductInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
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
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(SliceInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(SoftmaxInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  auto& node_info = GenerateCommonInfo(inst, node_infos);
  node_info.flops = GetNumOfOperators(inst) *
                    inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

static void RunOnInstruction(StackInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(TransposeInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

static void RunOnInstruction(ZExtInst* inst,
                             std::vector<Analyzer::NodeInfo>* node_infos) {
  GenerateCommonInfo(inst, node_infos);
}

void GenerateInputTensor(Instruction* inst) {
  // input
  auto ip_num = inst->GetNumOfOperands();
  for (size_t i = 0; i < ip_num; ++i) {
    auto ip_op = inst->GetOperand(i);
    if (!(IsA<Constant>(ip_op))) {
      std::string name = ip_op.GetOwner()->GetName();
      if (InputTensorApp.find(name) == InputTensorApp.end()) { // not found
        InputTensorApp[name] = 1;
        auto& ip_type = ip_op.GetType();
        IoTensorSize[name] = Dl->Bytes(ip_type);
      } else { // already exists
        InputTensorApp[name]++;
      }
    }
  }

  // output
  if (inst->GetNumOfResults() > 0) {
    const auto& op_type = inst->GetResultType();
    IoTensorSize[inst->GetName()] = Dl->Bytes(op_type);
  }
}

bool Analyzer::RunOnModule(Module* m) {
  // input tensor preprocessing
  InputTensorApp.clear();
  IoTensorSize.clear();
  AliveTensor.clear();
  for (auto& func : *m) {
    for (auto& bb : *func) {
      for (const auto& it : *bb) {
        Instruction* inst = it.get();
        GenerateInputTensor(inst);
      }
    }
  }

  // per node processing
  node_infos_.clear();
  for (auto& func : *m) {
    for (auto& bb : *func) {
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
    }
  }
  if (os_ != nullptr) {
    WriteCSVReport(*os_);
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
  float max_io = 0;
  for (const auto& it : node_infos_) {
    total_flops += it.flops;
    total_weights += it.weight_mem;

    max_io = std::max(max_io, it.io_mem);
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
     << "Total(MB), "
     << "percent(%)\n";

  for (const auto& it : node_infos_) {
    os << it.id << ", " << it.name << ", "
       << Instruction::OpCodeToString(it.type) << ", "
       << Type::DataTypeToString(it.data_type) << ", (";
    for (auto& ii : it.input_shape) {
      os << "(";
      for (const auto& i : ii) {
        os << i << " ";
      }
      os << ")";
    }
    os << "), (";
    for (const auto& ii : it.output_shape) {
      os << ii << " ";
    }
    os << "), " << it.flops / mflops << ", " << it.weight_mem / mb << ", "
       << (it.weight_mem + it.io_mem) / mb << ", "
       << ratio * it.flops / total_flops << ", "
       << "\n";
  }

  os << "\nTotal layers: " << node_infos_.size()
     << "\nTotal GFLOPs: " << total_flops / gflops
     << "\nTotal Weights(MB): " << total_weights / mb
     << "\nTotal Memory(MB): " << (total_weights + max_io) / mb << "\n";
}

} // namespace halo
