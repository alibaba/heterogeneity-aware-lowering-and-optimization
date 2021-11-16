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

static int SearchBatchSize(int bsz_log2, float init_latency, float knl_latency,
                           int ips) {
  const float epsilon = 1;
  const float ms2s = 1000;
  float cur_qps = ms2s / (init_latency + knl_latency);
  float qps = static_cast<float>(ips);
  if (cur_qps > qps) {
    return 0;
  }
  float bsz = static_cast<float>(1 << bsz_log2);
  cur_qps = bsz * ms2s / (init_latency + knl_latency * bsz);
  if (cur_qps < qps) {
    return bsz_log2;
  }
  int l = 0;
  int r = bsz_log2;
  while (l < r) {
    int mid = (l + r) / 2;
    bsz = static_cast<float>(1 << mid);
    cur_qps = bsz * ms2s / (init_latency + knl_latency * bsz);
    if (std::abs(cur_qps - qps) < epsilon) {
      l = mid;
      break;
    }
    if (cur_qps > qps) {
      r = mid;
    } else {
      l = mid + 1;
    }
  }

  return l;
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
      static_cast<float>(conv_op_num) * HW_Paras["GPU_t4"].conv_knl_init;
  init_latency +=
      static_cast<float>(matmul_op_num) * HW_Paras["GPU_t4"].mm_knl_init;
  init_latency += static_cast<float>(other_op_num - conv_act_num) *
                  HW_Paras["GPU_t4"].other_knl_init;
  float knl_latency =
      HW_Paras["GPU_t4"].conv_time * (conv_flops + conv_act_flops);
  knl_latency += HW_Paras["GPU_t4"].mm_time * matmul_flops;
  knl_latency += HW_Paras["GPU_t4"].other_time *
                 (total_flops - conv_flops - conv_act_flops - matmul_flops);

  if (opts_.batch_size == 1 && opts_.qps > 0) {
    float max_batch = (HW_Paras["GPU_t4"].max_mem - total_weights) / max_io;
    int b_log2 = static_cast<int>(std::log2f(max_batch));
    b_log2 = SearchBatchSize(b_log2, init_latency, knl_latency, opts_.qps);
    adaptive_bsz = 1 << b_log2;
    knl_latency *= static_cast<float>(adaptive_bsz);
    max_io *= static_cast<float>(adaptive_bsz);
  } else {
    // error input when qps and batch_size are both given.
    HLCHECK(opts_.qps == 0);
    adaptive_bsz = opts_.batch_size;
  }

  float trt_mem = total_weights + max_io;
  const int trt_base = 800;
  const int adjust_bsz = 64;
  const int adjust_mem = 1;
  int trt_env = (adaptive_bsz > adjust_bsz)
                    ? trt_base - (adaptive_bsz - adjust_bsz) * adjust_mem
                    : trt_base;
  trt_env = std::max(0, trt_env);
  trt_mem += static_cast<float>(trt_env);

  float est_latency = init_latency + knl_latency;

  os << "Device: GPU T4"
     << "\n";
  os << "batch size: " << adaptive_bsz << "\n";
  os << "est latency: " << est_latency << "ms\n";
  os << "est mem: " << trt_mem << "MB\n";
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
  rsc_req_.append("\"applyType\":\"bySize\",");
  rsc_req_.append("\"type\":\"GPU\",");
  rsc_req_.append("\"model\":\"T4\",");
  rsc_req_.append("\"size\":1,");
  rsc_req_.append("\"flops\":\"");
  rsc_req_.append(std::to_string(total_flops * gflops));
  rsc_req_.append("\",");
  rsc_req_.append("\"precision\":\"int32\",");
  rsc_req_.append("\"memory\":\"");
  rsc_req_.append(std::to_string(trt_mem));
  rsc_req_.append("\",");
  rsc_req_.append("\"allowSplit\":false,");
  rsc_req_.append("}");
  // end dev1
  rsc_req_.append("]");
  // end opt1
  rsc_req_.append("]");
  rsc_req_.append("}");
}

} // namespace halo
