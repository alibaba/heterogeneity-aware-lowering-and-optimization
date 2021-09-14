//===- conv.cc ------------------------------------------------------------===//
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

#include "halo/lib/framework/common.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"
#include "halo/lib/transforms/type_legalizer.h"

namespace halo {

const char* StringifyDirection(Direction direction) {
  const char* str = nullptr;

  switch (direction) {
    case Direction::FORWARD:
      str = "ODLA_RNN_FORWARD";
      break;
    case Direction::REVERSE:
      str = "ODLA_RNN_REVERSE";
      break;
    case Direction::BIDIRECTIONAL:
      str = "ODLA_RNN_BIDIRECTIONAL";
      break;
    default:
      HLCHECK(false && "Invalid direction value");
  }

  return str;
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

static const std::unordered_map<RNNWeightFormat, std::string> WeightFormatNames{
    {RNNWeightFormat::LDGIO, "ODLA_RNN_LDGIO"},
    {RNNWeightFormat::LDGOI, "ODLA_RNN_LDGOI"},
    {RNNWeightFormat::INVALID, "INVALID"},
};

static const std::unordered_map<RNNGateOrder, std::string> GateOrderNames{
    {RNNGateOrder::ICOF, "ODLA_RNN_ICOF"},
    {RNNGateOrder::IFCO, "ODLA_RNN_IFCO"},
    {RNNGateOrder::IFOC, "ODLA_RNN_IFOC"},
    {RNNGateOrder::IOFC, "ODLA_RNN_IOFC"},
    {RNNGateOrder::URO, "ODLA_RNN_URO"},
    {RNNGateOrder::INVALID, "INVALID"},
};

void GenericCXXCodeGen::RunOnInstruction(LSTMInst* inst) {
  const Def& x = inst->GetOperand(LSTM_ARG_X_IDX);
  const Def& w = inst->GetOperand(LSTM_ARG_W_IDX);
  const Def& r = inst->GetOperand(LSTM_ARG_R_IDX);
  const Def& b = inst->GetOperand(LSTM_ARG_B_IDX);
  const Def& sequence_lens = inst->GetOperand(LSTM_ARG_SEQUENCE_LENGTH_IDX);

  size_t num_ops = inst->GetNumOfOperands();

  ir_mapping_[Def::GetUndefined()] = CXXValue("nullptr", CXXType("void"));

  const Def& initial_h = num_ops > LSTM_ARG_INITIAL_H_IDX
                             ? inst->GetOperand(LSTM_ARG_INITIAL_H_IDX)
                             : Def::GetUndefined();

  const Def& initial_c = num_ops > LSTM_ARG_INITIAL_C_IDX
                             ? inst->GetOperand(LSTM_ARG_INITIAL_C_IDX)
                             : Def::GetUndefined();

  const Def& p = num_ops > LSTM_ARG_P_IDX ? inst->GetOperand(LSTM_ARG_P_IDX)
                                          : Def::GetUndefined();

  CXXValue op_x = ir_mapping_[x];
  CXXValue op_w = ir_mapping_[w];
  CXXValue op_r = ir_mapping_[r];
  CXXValue op_b = ir_mapping_[b];
  CXXValue op_sequence_lens = ir_mapping_[sequence_lens];
  CXXValue op_initial_h = ir_mapping_[initial_h];
  CXXValue op_initial_c = ir_mapping_[initial_c];
  CXXValue op_p = ir_mapping_[p];

  uint32_t hidden_size = r.GetType().GetNumOfElementsInDim(2);

  std::vector<CXXValue> rets;
  rets.emplace_back(inst->GetName(),
                    TensorTypeToCXXType(inst->GetResultsTypes()[0], false));
  rets.emplace_back(inst->GetName() + "_h",
                    TensorTypeToCXXType(inst->GetResultsTypes()[1], false));
  rets.emplace_back(inst->GetName() + "_c",
                    TensorTypeToCXXType(inst->GetResultsTypes()[2], false));

  const char* str = StringifyDirection(inst->GetDirection());

  const char* outputs = "ODLA_RNN_HIDDEN_CELL_STATE";

  auto it_format = WeightFormatNames.find(inst->GetWeightFormat());
  auto it_gate = GateOrderNames.find(inst->GetGateOrder());
  HLCHECK(it_format != WeightFormatNames.end() &&
          it_gate != GateOrderNames.end());

  EmitODLACall(rets, "odla_LSTM", op_x, it_format->second, it_gate->second,
               EmitShape(w.GetType()), op_w, op_r, op_b, op_sequence_lens,
               op_initial_c, op_initial_c, op_p, hidden_size, str, outputs);

  ir_mapping_[Def(inst, 0)] = rets[0];
  ir_mapping_[Def(inst, 1)] = rets[1];
  ir_mapping_[Def(inst, 2)] = rets[2];
}

void GenericCXXCodeGen::RunOnInstruction(GRUInst* inst) {
  const Def& x = inst->GetOperand(LSTM_ARG_X_IDX);
  const Def& w = inst->GetOperand(LSTM_ARG_W_IDX);
  const Def& r = inst->GetOperand(LSTM_ARG_R_IDX);
  const Def& b = inst->GetOperand(LSTM_ARG_B_IDX);
  const Def& sequence_lens = inst->GetOperand(LSTM_ARG_SEQUENCE_LENGTH_IDX);

  size_t num_ops = inst->GetNumOfOperands();

  ir_mapping_[Def::GetUndefined()] = CXXValue("nullptr", CXXType("void"));

  const Def& initial_h = num_ops > LSTM_ARG_INITIAL_H_IDX
                             ? inst->GetOperand(LSTM_ARG_INITIAL_H_IDX)
                             : Def::GetUndefined();

  CXXValue op_x = ir_mapping_[x];
  CXXValue op_w = ir_mapping_[w];
  CXXValue op_r = ir_mapping_[r];
  CXXValue op_b = ir_mapping_[b];
  CXXValue op_sequence_lens = ir_mapping_[sequence_lens];
  CXXValue op_initial_h = ir_mapping_[initial_h];

  uint32_t hidden_size = inst->GetHiddenSize();

  std::vector<CXXValue> rets;
  rets.emplace_back(inst->GetName(),
                    TensorTypeToCXXType(inst->GetResultsTypes()[0], false));
  rets.emplace_back(inst->GetName() + "_h",
                    TensorTypeToCXXType(inst->GetResultsTypes()[1], false));

  const char* dir = StringifyDirection(inst->GetDirection());

  const char* outputs = "ODLA_RNN_HIDDEN_STATE";

  auto it_format = WeightFormatNames.find(inst->GetWeightFormat());
  auto it_gate = GateOrderNames.find(inst->GetGateOrder());
  HLCHECK(it_format != WeightFormatNames.end() &&
          it_gate != GateOrderNames.end());

  EmitODLACall(rets, "odla_GRU", op_x, it_format->second, it_gate->second,
               EmitShape(w.GetType()), op_w, op_r, op_b, op_sequence_lens,
               op_initial_h, hidden_size, dir, inst->GetLinearBeforeReset(),
               outputs);

  ir_mapping_[Def(inst, 0)] = rets[0];
  ir_mapping_[Def(inst, 1)] = rets[1];
}

} // namespace halo
