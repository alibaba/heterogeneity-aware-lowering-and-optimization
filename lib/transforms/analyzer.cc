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
#include <cmath>
#include <iostream>

namespace halo {

static DefaultDataLayout Ddl;
static DataLayout* Dl = &Ddl;

float Analyzer::GetNumOfOperators(const Instruction* inst) {
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

Analyzer::NodeInfo& Analyzer::GenerateCommonInfo(const Instruction* inst) {
  node_infos_.emplace_back();
  Analyzer::NodeInfo& node_info = node_infos_.back();
  node_info.id = node_infos_.size();
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

    size_t size = Dl->Bytes(ip_type);
    if (IsA<Constant>(inst->GetOperand(i))) {
      node_info.weight_mem += size;
    } else {
      std::string name = inst->GetOperand(i).GetOwner()->GetName();
      if (alive_tensor_.find(name) != alive_tensor_.end()) {
        alive_tensor_[name].liveness--;
      }

      node_info.io_mem += size;
    }
  }

  const size_t op_tgts = inst->GetResultsUses()[0].GetUses().size();
  // todo: only output[0] is processed here
  const auto& op_type = inst->GetResultType();
  if (op_type.IsScalar()) {
    node_info.output_shape.push_back(1);
  } else {
    node_info.output_shape = op_type.GetDimSizes();
  }
  TensorInfo tif = {op_tgts, Dl->Bytes(op_type)};
  alive_tensor_[node_info.name] = tif;

  for (auto iter = alive_tensor_.begin(); iter != alive_tensor_.end();) {
    if (iter->second.liveness == 0) {
      iter = alive_tensor_.erase(iter);
    } else {
      node_info.io_mem += iter->second.size * iter->second.liveness;
      iter++;
    }
  }

  return node_info;
}

template <class T>
void Analyzer::CalPoolingInst(const T* inst) {
  float op_num = GetNumOfOperators(inst);
  auto& node_info = GenerateCommonInfo(inst);
  auto kernel_shape = inst->GetKsize();
  auto stride_shape = inst->GetStrides();
  auto input_type = inst->GetOperand(0).GetType();
  auto input_dim = input_type.GetDimSizes();

  switch (inst->GetDataFormat()) {
    case DataFormat::NCHW: {
      node_info.flops =
          op_num * kernel_shape.back() * kernel_shape[kernel_shape.size() - 2];
      size_t rptx = input_dim[3] / stride_shape[stride_shape.size() - 2];
      size_t rpty = input_dim[2] / stride_shape[stride_shape.back()];
      node_info.flops *= static_cast<float>(rptx * rpty * input_dim[1]);
      break;
    }
    case DataFormat::NHWC: {
      node_info.flops = op_num * kernel_shape[kernel_shape.size() - 2] *
                        kernel_shape[kernel_shape.size() - 3];
      size_t rptx = input_dim[2] / stride_shape[stride_shape.size() - 2];
      size_t rpty = input_dim[1] / stride_shape[stride_shape.back()];
      node_info.flops *= static_cast<float>(rptx * rpty * input_dim[3]);
      break;
    }
    default: {
      HLCHECK(0 && "Invalid format");
    }
  }
}

template <class T>
void Analyzer::CalConvInst(const T* inst) {
  auto& node_info = GenerateCommonInfo(inst);

  const auto& weight_op = inst->GetOperand(1);
  const auto& weight_type = weight_op.GetType();
  size_t weight_size = weight_type.GetTotalNumOfElements();

  const auto& out_type = inst->GetResultType();
  size_t out_size = out_type.GetTotalNumOfElements();
  const size_t dims = out_type.GetNumOfDims();
  HLCHECK(dims == 4);
  int chn = 0;
  switch (inst->GetDataFormat()) {
    case DataFormat::NCHW: {
      chn = 1;
      break;
    }
    case DataFormat::NHWC: {
      chn = 3;
      break;
    }
    default: {
      HLCHECK(0 && "Invalid format");
    }
  }

  int grp = inst->GetGroup();

  if (grp == 1) {
    // Normal Conv computational estimator:
    // 1 kernel per pixel: (2 * Kh * Kw - 1)
    // all chanels (2 * Kh * Kw - 1) * Cin + 1 (bias)
    // all output pixels:
    // ((2 * Kh * Kw - 1) * Cin + 1) * Cout * Hout * Wout * Batch (Cout == Nk)
    node_info.flops = 2;
    node_info.flops *= static_cast<float>(weight_size * out_size);
    node_info.flops /= out_type.GetNumOfElementsInDim(chn);
    node_info.flops =
        node_info.flops -
        static_cast<float>(weight_type.GetNumOfElementsInDim(1) * out_size -
                           out_size);
  } else {
    // depthwise convolution estimator:
    // 1 kernel per pixel: (2 * Kh * Kw - 1)
    // all groups (2 * Kh * Kw - 1) * Cout + 1 (bias)
    // all output pixels:
    // ((2 * Kh * Kw - 1) * Cout + 1) * Hout * Wout * Batch
    node_info.flops = 2;
    node_info.flops *=
        static_cast<float>(weight_type.GetNumOfElementsInDim(2) *
                           weight_type.GetNumOfElementsInDim(3) * out_size);
    size_t t = out_size / out_type.GetNumOfElementsInDim(chn);
    node_info.flops = node_info.flops - static_cast<float>(out_size - t);
  }
}

void Analyzer::RunOnInstruction(Instruction* inst) {
  auto op_code = inst->GetOpCode();
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
    case OpCode::SITOFP:
    case OpCode::SIGMOID:
    case OpCode::EXP:
    case OpCode::TOPK:
    case OpCode::RANGE:
    case OpCode::RANDOMUNIFORM:
    case OpCode::RCP:
    case OpCode::TANH:
    case OpCode::FPTOSI: {
      auto& node_info = GenerateCommonInfo(inst);
      node_info.flops = inst->GetResultType().GetTotalNumOfElements();
      if (op_code == OpCode::SIGMOID) {
        node_info.flops *= 3;
      } else if (op_code == OpCode::RANGE) {
        node_info.flops *= 2;
      } else if (op_code == OpCode::TOPK) {
        node_info.flops *= std::log2f(node_info.flops);
      }
      break;
    }
    default: {
      std::cout << "Error OP: "
                << Instruction::OpCodeToString(inst->GetOpCode()) << ": "
                << static_cast<int>(inst->GetOpCode()) << "\n";
      HLCHECK(0 && "Unimplemented");
    }
  }
}

void Analyzer::RunOnInstruction(ResizeInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);

  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
  size_t dims = inst->GetOperand(0).GetType().GetNumOfDims();
  node_info.flops *= static_cast<float>((1 << dims) - 1);
  auto interp_md = inst->GetInterpolationMode();
  if (interp_md == Interpolation::LINEAR) {
    const float op_per_interp = 4.0;
    node_info.flops *= op_per_interp;
  } else if (interp_md == Interpolation::CUBIC) {
    const float op_per_interp = 7.0;
    node_info.flops *= op_per_interp;
  }
}

void Analyzer::RunOnInstruction(LSTMInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = 0;

  auto dir = inst->GetDirection();
  size_t rpt = 1;
  if (dir == Direction::BIDIRECTIONAL) {
    rpt = 2;
  }
  size_t hidden_sz = inst->GetHiddenSize();

  const auto& input_type = inst->GetOperand(0).GetType();
  HLCHECK(input_type.GetNumOfDims() >= 3);
  size_t ip_sz = input_type.GetNumOfElementsInDim(2);
  size_t b_sz = input_type.GetNumOfElementsInDim(1);
  size_t seq_sz = input_type.GetNumOfElementsInDim(0);

  // forget gate: F = sigmoid(input*W + H*W + b)
  // input gate: I = sigmoid(input*W + H*W + b), C_ = tanh(input*W + H*W + b)
  // cell state gate: C = F x C + I x C_
  // output gate: O = sigmoid(input*W + H*W + b), H = O x tanh(C)
  const int mad = 8;
  const int mul = 5;
  node_info.flops += static_cast<float>(mad * (ip_sz + hidden_sz) * hidden_sz);
  node_info.flops += static_cast<float>(mul * hidden_sz);
  node_info.flops *= static_cast<float>(rpt * b_sz * seq_sz);
}

void Analyzer::RunOnInstruction(BatchMatMulInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);

  // batch matmul computational estimator: batch * ((2 * Cin - 1) * Cout)
  size_t batch = 1;
  //   const auto& input_type = inst->GetOperand(0).GetType();
  //   if (input_type.GetNumOfDims() >= 1) {
  //     batch = input_type.GetNumOfElementsInDim(input_type.GetNumOfDims() -
  //     1);
  //   }

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

void Analyzer::RunOnInstruction(Conv2DInst* inst) {
  CalConvInst<Conv2DInst>(inst);
}

void Analyzer::RunOnInstruction(Conv2DTransposeInst* inst) {
  CalConvInst<Conv2DTransposeInst>(inst);
}

void Analyzer::RunOnInstruction(GemmInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);

  const auto& matrix_b = inst->GetOperand(1);
  const auto& matrix_c = inst->GetOperand(2);

  const size_t matrixb_sz = matrix_b.GetType().GetTotalNumOfElements();
  const size_t matrixc_sz = matrix_c.GetType().GetTotalNumOfElements();

  // GEMM computational estimator: out = alpha * A' * B' + beta * C
  const size_t dims = matrix_b.GetType().GetNumOfDims();
  const int64_t row_a = inst->GetResultType().GetNumOfElementsInDim(0);
  const int64_t col_b =
      inst->GetTransposeB()
          ? matrix_b.GetType().GetNumOfElementsInDim(dims - 2)
          : matrix_b.GetType().GetNumOfElementsInDim(dims - 1);

  node_info.flops =
      static_cast<float>(row_a * (2 * matrixb_sz - col_b) + 2 * matrixc_sz);
}

void Analyzer::RunOnInstruction(MatMulInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);

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

void Analyzer::RunOnInstruction(NonMaxSuppressionInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);

  // IOU operation: 2x min, 4x max, 4x +-, 2x */
  const int iou_op = 12;
  size_t num_box = inst->GetOperand(0).GetType().GetNumOfElementsInDim(1);
  float sort_op = num_box * std::log2f(num_box);

  node_info.flops = static_cast<float>(iou_op * num_box) + sort_op;
}

void Analyzer::RunOnInstruction(PoolingMaxInst* inst) {
  CalPoolingInst<PoolingMaxInst>(inst);
}

void Analyzer::RunOnInstruction(PoolingAvgInst* inst) {
  CalPoolingInst<PoolingAvgInst>(inst);
}

void Analyzer::RunOnInstruction(BatchNormInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  // z = gamma * (y - mean) / sqrt(variance + epsilon) + beta
  // BatchNorm computational estimator: Cin * 4
  const auto& input_type = inst->GetOperand(0).GetType();
  node_info.flops =
      GetNumOfOperators(inst) * input_type.GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ReduceMeanInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = GetNumOfOperators(inst) *
                    inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(SoftmaxInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = GetNumOfOperators(inst) *
                    inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(GatherInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ReluInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(Relu6Inst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetOperand(0).GetType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ArgmaxInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ConcatInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(OneHotInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(PadInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(RangeInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(RandomUniformInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ReduceMaxInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ReduceProductInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ReshapeInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(SetDiff1DInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(SliceInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(StackInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(TransposeInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ZExtInst* inst) {
  auto& node_info = GenerateCommonInfo(inst);
  node_info.flops = inst->GetResultType().GetTotalNumOfElements();
}

void Analyzer::RunOnInstruction(ReturnInst* inst) {
  // do nothing
}

bool Analyzer::RunOnModule(Module* m) {
  alive_tensor_.clear();
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

  GenerateRscInfo(*os_);

  return false;
}

// static int SearchBatchSize(int bsz_log2, float init_latency, float
// knl_latency,
//                            int ips) {
//   const float epsilon = 1;
//   const float ms2s = 1000;
//   float cur_qps = ms2s / (init_latency + knl_latency);
//   float qps = static_cast<float>(ips);
//   if (cur_qps > qps) {
//     return 0;
//   }
//   float bsz = static_cast<float>(1 << bsz_log2);
//   cur_qps = bsz * ms2s / (init_latency + knl_latency * bsz);
//   if (cur_qps < qps) {
//     return bsz_log2;
//   }
//   int l = 0;
//   int r = bsz_log2;

//   while (l < r) {
//     int mid = (l + r) / 2;
//     bsz = static_cast<float>(1 << mid);
//     cur_qps = bsz * ms2s / (init_latency + knl_latency * bsz);

//     if (std::abs(cur_qps - qps) < epsilon) {
//       l = mid;
//       break;
//     }
//     if (cur_qps > qps) {
//       r = mid;
//     } else {
//       l = mid + 1;
//     }
//   }

//   return l;
// }

static float NewtonSolver(const std::array<double, 4> func, int iteration,
                          float error) {
  const std::array<double, 3> func_de{func[1], func[2] * 2, func[3] * 3};
  const float init = 50;
  const float max_per = 100;
  const float min_per = 0;
  float per = init;

  for (int i = 0; i < iteration; i++) {
    if (fabs(func[0] + func[1] * per + func[2] * per * per +
             func[3] * per * per * per) < error) {
      break;
    }
    per = per - (func[0] + func[1] * per + func[2] * per * per +
                 func[3] * per * per * per) /
                    (func_de[0] + func_de[1] * per + func_de[2] * per * per);
  }
  if (per > max_per) {
    per = max_per;
  } else if (per < min_per) {
    per = min_per;
  }

  return per;
}

void Analyzer::GenerateRscInfo(std::ostream& os) {
  static constexpr float mflops = 1000000.0F;
  static constexpr float gflops = 1000 * mflops;
  static constexpr float mb = 1024 * 1024.0F;
  static constexpr int ratio = 100;

  float total_flops = 0;
  float total_weights = 0;
  float conv_flops = 0;
  float conv_act_flops = 0;
  int conv_op_num = 0;
  int conv_act_num = 0;
  bool last_is_conv = false;
  float matmul_flops = 0;
  int matmul_op_num = 0;
  int other_op_num = 0;
  size_t max_io = 0;
  for (const auto& it : node_infos_) {
    total_flops += it.flops;
    total_weights += it.weight_mem;

    max_io = std::max(max_io, it.io_mem);

    if (it.type == OpCode::MATMUL || it.type == OpCode::GEMM ||
        it.type == OpCode::BATCHMATMUL) {
      matmul_flops += it.flops;
      matmul_op_num++;
      last_is_conv = false;
    } else if (it.type == OpCode::CONV2D ||
               it.type == OpCode::CONV2DTRANSPOSE) {
      conv_flops += it.flops;
      conv_op_num++;
      last_is_conv = true;
    } else {
      other_op_num++;
      if (last_is_conv) {
        conv_act_flops += it.flops;
        conv_act_num++;
      }
      last_is_conv = false;
    }
  }

  if (opts_.print_details && os) {
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

    os << "\n";
  }

  /*-----Calculate T4 parameters-----------------*/
  total_weights /= mb;
  max_io /= mb;
  conv_flops /= gflops;
  conv_act_flops /= gflops;
  matmul_flops /= gflops;
  total_flops /= gflops;
  float init_latency =
      static_cast<float>(conv_op_num) * hw_paras_["GPU_t4"].conv_knl_init;
  init_latency +=
      static_cast<float>(matmul_op_num) * hw_paras_["GPU_t4"].mm_knl_init;
  init_latency += static_cast<float>(other_op_num - conv_act_num) *
                  hw_paras_["GPU_t4"].other_knl_init;
  float floatsrate = 0;
  float knl_latency = 0;
  adaptive_bsz_ = opts_.batch_size;

  float max_batch = (hw_paras_["GPU_t4"].max_mem - total_weights) / max_io;
  //   os << "max_batch: " << max_batch << "\n";
  //   os << "total_weights: " << total_weights << "\n";
  //   os << "max_io: " << max_io << "\n";
  //   os << "opts_.batch_size: " << opts_.batch_size << "\n";
  //   os << "conv_flops: " << conv_flops << "\n";
  //   os << "conv_act_flops: " << conv_act_flops << "\n";
  //   os << "total_flops: " << total_flops << "\n";
  HLCHECK(max_batch >= opts_.batch_size);
  const float t4_flops = hw_paras_["GPU_t4"].max_flops;
  const float step_sz = 0.05;
  float cur_qps = 0;
  const float max_step = 1.01;
  for (float step = step_sz; step <= max_step; step += step_sz) {
    std::map<std::string, HWInfo> hw_paras_step = hw_paras_;
    hw_paras_step["GPU_t4"].conv_time *= 1 / step;
    hw_paras_step["GPU_t4"].mm_time *= 1 / step;
    hw_paras_step["GPU_t4"].other_time *= 1 / step;
    const float ms2s = 1000;
    float knl_latency_temp =
        hw_paras_step["GPU_t4"].conv_time * (conv_flops + conv_act_flops);
    knl_latency_temp += hw_paras_step["GPU_t4"].mm_time * matmul_flops;
    knl_latency_temp +=
        hw_paras_step["GPU_t4"].other_time *
        (total_flops - conv_flops - conv_act_flops - matmul_flops);
    // knl_latency_temp *= static_cast<float>(adaptive_bsz_);
    cur_qps = float(opts_.batch_size) * ms2s * 4 /
              (init_latency + knl_latency_temp * float(opts_.batch_size));
    // os << "step:" << step << " ,cur_qps:" << cur_qps << "\n";
    // os << "hw_paras_step[GPU_t4].conv_time" <<
    // hw_paras_step["GPU_t4"].conv_time
    //    << "\n";
    if (cur_qps >= opts_.qps) {
      floatsrate = step * t4_flops;
      knl_latency = knl_latency_temp;
      break;
    }
  }
  // if (opts_.batch_size == 1 && opts_.qps > 0) {
  //   float max_batch =
  //       (hw_paras_step["GPU_t4"].max_mem - total_weights) / max_io;
  //   int b_log2 = static_cast<int>(std::log2f(max_batch));
  //   b_log2 = SearchBatchSize(b_log2, init_latency, knl_latency, opts_.qps);
  //   adaptive_bsz_ = 1 << b_log2;
  //   knl_latency *= static_cast<float>(adaptive_bsz_);
  //   max_io *= static_cast<float>(adaptive_bsz_);
  //   total_flops *= static_cast<float>(adaptive_bsz_);

  // } else {
  //   // error input when qps and batch_size are both given.
  //   HLCHECK(opts_.qps == 0);
  //   adaptive_bsz_ = opts_.batch_size;
  // }

  max_io *= static_cast<float>(adaptive_bsz_);
  total_flops *= static_cast<float>(adaptive_bsz_);
  if (floatsrate == 0 || knl_latency == 0) {
    os << "Can not reach required qps: " << opts_.qps
       << " with batch size: " << opts_.batch_size
       << " ,max possible qps is: " << cur_qps << " \n";
    floatsrate = t4_flops;
    knl_latency = hw_paras_["GPU_t4"].conv_time * (conv_flops + conv_act_flops);
    knl_latency += hw_paras_["GPU_t4"].mm_time * matmul_flops;
    knl_latency += hw_paras_["GPU_t4"].other_time *
                   (total_flops - conv_flops - conv_act_flops - matmul_flops);
  }
  // HLCHECK(floatsrate != 0);
  // HLCHECK(knl_latency != 0);
  float trt_mem = total_weights + max_io;
  const int trt_base = 800;
  const int adjust_bsz = 64;
  const int adjust_mem = 1;
  int trt_env = (adaptive_bsz_ > adjust_bsz)
                    ? trt_base - (adaptive_bsz_ - adjust_bsz) * adjust_mem
                    : trt_base;
  trt_env = std::max(0, trt_env);
  trt_mem += static_cast<float>(trt_env);

  float est_latency = init_latency + knl_latency;
  const float t4 = t4_flops / 100;
  const double u_sec = 1e+6;
  const int iteration = 10;
  const float error_rate = 0.001;
  const float max_percent = 100;
  if (opts_.model_type == 1) {
    // const std::array<double, 10> model{64073.283167584894, -88.91731411,
    //                                    12.78189374,        26.05789414,
    //                                    8533.30914793,      -2900.88985761};
    // const std::array<double, 4> func{
    //     model[0] +
    //         model[2] * float(opts_.batch_size) * float(opts_.batch_size) +
    //         model[4] * float(opts_.batch_size) -
    //         u_sec * float(opts_.batch_size) / opts_.qps,
    //     model[1] * float(opts_.batch_size) + model[5], model[3]};
    const int resnet_max_batch = 64;
    if (opts_.batch_size > resnet_max_batch) {
      opts_.batch_size = resnet_max_batch;
    }
    const std::array<double, 10> model{
        49902.8906207358, -3.30238451e+02, -2.17410190e+01, 9.51439925e+01,
        1.34280387e+04,   -4.18767285e+03, 2.18543166e+00,  -4.47421309e-03,
        7.32400224e-01,   -5.81271182e-01};
    const std::array<double, 4> func{
        model[0] + model[4] * float(opts_.batch_size) +
            model[8] * float(opts_.batch_size) * float(opts_.batch_size) *
                float(opts_.batch_size) +
            model[2] * float(opts_.batch_size) * float(opts_.batch_size) -
            u_sec * float(opts_.batch_size) / opts_.qps,
        model[1] * float(opts_.batch_size) + model[5] +
            model[7] * float(opts_.batch_size) * float(opts_.batch_size),
        model[3] + model[6] * float(opts_.batch_size), model[9]};
    float per =
        NewtonSolver(func, iteration,
                     error_rate * u_sec * float(opts_.batch_size) / opts_.qps);
    floatsrate = per * t4_flops / max_percent;
    os << "Model: resnet50"
       << "\n";
    os << "est latency: "
       << func[0] + func[1] * per + func[2] * per * per +
              u_sec * float(opts_.batch_size) / opts_.qps +
              func[3] * per * per * per
       << "\n";
  } else if (opts_.model_type == 2) {
    const std::array<double, 4> func{
        88324.13992776436 - u_sec * float(opts_.batch_size) / opts_.qps,
        -2.75316291e+03, 3.90359192e+01, -1.81786268e-01};
    float per =
        NewtonSolver(func, iteration,
                     error_rate * u_sec * float(opts_.batch_size) / opts_.qps);
    floatsrate = per * t4_flops / max_percent;
    os << "Model: dbnet"
       << "\n";
    os << "est latency: "
       << func[0] + func[1] * per + func[2] * per * per +
              func[3] * per * per * per +
              u_sec * float(opts_.batch_size) / opts_.qps
       << "\n";
  } else if (opts_.model_type == 3) {
    const std::array<double, 4> func{
        31525.584310580438 - u_sec * float(opts_.batch_size) / opts_.qps,
        -475.78524037, 2.58107976, 0.0};
    float per =
        NewtonSolver(func, iteration,
                     error_rate * u_sec * float(opts_.batch_size) / opts_.qps);
    floatsrate = per * t4_flops / max_percent;
    os << "Model: crnn"
       << "\n";
    os << "est latency: "
       << func[0] + func[1] * per + func[2] * per * per +
              func[3] * per * per * per +
              u_sec * float(opts_.batch_size) / opts_.qps
       << "\n";
  } else if (opts_.model_type == 4) {
    const int bert_max_batch = 128;
    if (opts_.batch_size > bert_max_batch) {
      opts_.batch_size = bert_max_batch;
    }
    const std::array<double, 6> model{438429.4914344477, -5.17070386e+02,
                                      1.96615464e+01,    2.23010546e+02,
                                      5.54527169e+04,    -2.45070285e+04};
    const std::array<double, 4> func{
        model[0] +
            model[2] * float(opts_.batch_size) * float(opts_.batch_size) +
            model[4] * float(opts_.batch_size) -
            u_sec * float(opts_.batch_size) / opts_.qps,
        model[1] * float(opts_.batch_size) + model[5], model[3]};
    float per =
        NewtonSolver(func, iteration,
                     error_rate * u_sec * float(opts_.batch_size) / opts_.qps);
    floatsrate = per * t4_flops / max_percent;
    os << "Model: bert"
       << "\n";
    os << "est latency: "
       << func[0] + func[1] * per + func[2] * per * per +
              func[3] * per * per * per +
              u_sec * float(opts_.batch_size) / opts_.qps
       << "\n";
  } else {
    os << "Model: other"
       << "\n";
  }

  os << "Device: GPU T4"
     << "\n";
  os << "batch size: " << adaptive_bsz_ << "\n";
  os << "est FLOPs: " << floatsrate << " gFlops\n";
  os << "est split: " << floatsrate / t4 << "% T4\n";
  os << "model FLOPs: " << floatsrate / t4 << "% T4\n";
  os << "est latency: " << est_latency << " ms\n";
  os << "est mem: " << trt_mem << " MB\n";
  /*-----Generated T4 parameters-----------------*/

  // fill in resource request for scheduling
  rsc_req_.clear();
  // rsc_req_.append("{");
  // rsc_req_.append("\"key:\","); // key will be inserted in vODLA IF
  rsc_req_.append("\"options\":");
  rsc_req_.append("[");
  // opt1
  rsc_req_.append("[");
  // dev1
  rsc_req_.append("{");
  rsc_req_.append("\"applyId\":\"1\",");
  rsc_req_.append("\"applyType\":\"byFlops\",");
  rsc_req_.append("\"type\":\"GPU\",");
  rsc_req_.append("\"model\":\"Tesla T4\",");
  rsc_req_.append("\"size\":1,");
  rsc_req_.append("\"flops\":\"");
  // std::string s = std::to_string(total_flops * gflops);
  std::string s = std::to_string(ceil(int(floatsrate)));
  rsc_req_.append(s.substr(0, s.find('.')));
  rsc_req_.append("\",");
  rsc_req_.append("\"precision\":\"Fp32\",");
  rsc_req_.append("\"memory\":\"");
  // s = std::to_string(trt_mem);
  s = std::to_string(ceil(trt_mem));
  rsc_req_.append(s.substr(0, s.find('.')));
  rsc_req_.append("\",");
  rsc_req_.append("\"allowSplit\":false,");
  rsc_req_.append("\"maxSplit\":2,");
  rsc_req_.append("\"minSplitSize\":5");
  rsc_req_.append("}");
  // end dev1
  rsc_req_.append("]");
  // end opt1
  rsc_req_.append("]");
  rsc_req_.append("}");
}

} // namespace halo
