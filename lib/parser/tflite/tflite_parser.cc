//===- tflite_parser.cc ---------------------------------------------------===//
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

#include "tflite_parser.h"

#include <fstream>
#include <iostream>
#include <numeric>

#include "halo/lib/framework/common.h"
#include "halo/lib/framework/type.h"

namespace halo {

namespace {

static DataType ProcessDataType(tflite::TensorType data_type) {
  switch (data_type) {
    case tflite::TensorType_FLOAT32:
      return DataType::FLOAT32;
    case tflite::TensorType_FLOAT16:
      return DataType::FLOAT16;
    case tflite::TensorType_INT64:
      return DataType::INT64;
    case tflite::TensorType_INT32:
      return DataType::INT32;
    case tflite::TensorType_INT16:
      return DataType::INT16;
    case tflite::TensorType_INT8:
      return DataType::INT8;
    case tflite::TensorType_UINT8:
      return DataType::UINT8;
    case tflite::TensorType_BOOL:
      return DataType::BOOL;
    default:
      LOG(ERROR) << "Unsupported DataType.";
      return DataType::INVALID;
  }
}

template <typename T>
static std::vector<T> ProcessTensorShape(
    const flatbuffers::Vector<int32_t>* tensor_shape) {
  if (tensor_shape == nullptr) {
    return {};
  }
  std::vector<T> ret(tensor_shape->size());
  for (size_t i = 0; i < tensor_shape->size(); ++i) {
    ret[i] = tensor_shape->Get(i);
  }
  return ret;
}

template <typename T>
static std::vector<T> DecodeTensorContent(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers,
    const tflite::Tensor& tensor) {
  HLCHECK(tensor.buffer() != 0 && tensor.buffer() < buffers.size());
  std::vector<T> output;
  if (tensor.is_variable()) {
    return output;
  }

  auto cur_buffer = buffers[tensor.buffer()];
  if (cur_buffer != nullptr) {
    auto array = cur_buffer->data();
    if (array != nullptr) {
      size_t size = array->size();
      output.resize(size / sizeof(T));
      HLCHECK(size % sizeof(T) == 0);
      const T* p = reinterpret_cast<const T*>(array->data());
      for (size_t i = 0, n = size / sizeof(T); i < n; ++i) {
        output[i] = *p++; // NOLINT.
      }
    }
  }
  return output;
}

template <typename T>
static Tensor<T> ConvertOHWI2HWIO(Tensor<T>& input) {
  const auto& s = input.GetShape();
  HLCHECK(s.size() == 4);
  if (s[0] == 1) {
    std::vector<int64_t> new_shape(4, 0);
    static const std::vector<size_t> ohwi2hwio{1, 2, 0, 3};
    for (size_t i = 0; i < s.size(); ++i) {
      new_shape[i] = s[ohwi2hwio[i]];
    }
    input.SetShape(new_shape);
    return Tensor<T>(input);
  } else {
    std::vector<T> out(
        std::accumulate(s.begin(), s.end(), 1, std::multiplies<int64_t>()));
    const auto& in = input.GetData();
    for (int64_t c = 0, sum_c = s.back(); c < sum_c; ++c) {
      for (int64_t w = 0, sum_w = s[2]; w < sum_w; ++w) {
        for (int64_t h = 0, sum_h = s[1]; h < sum_h; ++h) {
          for (int64_t n = 0, sum_n = s[0]; n < sum_n; ++n) {
            out[n + c * sum_n + w * sum_n * sum_c + h * sum_n * sum_c * sum_w] =
                in[c + w * sum_c + h * sum_c * sum_w +
                   n * sum_c * sum_w * sum_h];
          }
        }
      }
    }

    // NHWC+HWIO
    std::vector<int64_t> new_shape(s);
    static const std::vector<size_t> ohwi2hwio{1, 2, 3, 0};
    for (size_t i = 0; i < s.size(); ++i) {
      new_shape[i] = s[ohwi2hwio[i]];
    }
    return Tensor<T>(input.GetDataType(), new_shape, out);
  }
}

auto process_padding = [](tflite::Padding padding) {
  switch (padding) {
    case tflite::Padding::Padding_SAME:
      return Padding::SAME;
    case tflite::Padding::Padding_VALID:
      return Padding::VALID;
    default:
      return Padding::INVALID;
  }
};

auto process_fused_activation_function = [](tflite::ActivationFunctionType
                                                type) {
  switch (type) {
    case tflite::ActivationFunctionType::ActivationFunctionType_RELU:
      return ActivationType::RELU;
    case tflite::ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      return ActivationType::RELU1;
    case tflite::ActivationFunctionType::ActivationFunctionType_RELU6:
      return ActivationType::RELU6;
    case tflite::ActivationFunctionType::ActivationFunctionType_TANH:
      return ActivationType::TANH;
    case tflite::ActivationFunctionType::ActivationFunctionType_SIGN_BIT:
      return ActivationType::SIGN_BIT;
    default:
      return ActivationType::NONE;
  }
};

auto process_new_shape = [](const flatbuffers::Vector<int32_t>* shape) {
  return ProcessTensorShape<int32_t>(shape);
};

auto process_squeeze_dims =
    [](const flatbuffers::Vector<int32_t>* squeeze_dims) {
      std::vector<int32_t> ret(squeeze_dims->size());
      for (size_t i = 0; i < squeeze_dims->size(); ++i) {
        ret[i] = squeeze_dims->Get(i);
      }
      return ret;
    };

} // end anonymous namespace

Status TFLITEParser::Parse(Function* function,
                           const std::vector<std::string>& file_list,
                           const armory::Opts& opts) {
  HLCHECK(file_list.size() == 1);
  std::ifstream ifs(file_list.back(), std::ios::binary);
  ifs.seekg(0, std::ios::end);
  int len = ifs.tellg();
  proto_ = new char[len];
  ifs.seekg(0, std::ios::beg);
  ifs.read(proto_, len);
  ifs.close();

  const auto& model = *tflite::GetModel(proto_);
  static const int TFLITE_SCHEMA_VERSION = 3;
  if (model.version() != TFLITE_SCHEMA_VERSION) {
    VLOG(0) << "schema version is not equal to model version!";
    return Status::ILLEGAL_PARAM;
  }

  BasicBlockBuilder bb_builder(function);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");
  return Parse(bb, model);
}

Status TFLITEParser::Parse(BasicBlock* bb, const tflite::Model& model) {
  RegisterOp();
  auto function = bb->GetParent();
  ir_builder_ = std::make_unique<IRBuilder>(bb);
  arg_builder_ = std::make_unique<ArgumentBuilder>(function);
  c_builder_ = std::make_unique<ConstantBuilder>(function);
  return ConvertToHaloIR(model);
}

Status TFLITEParser::ConvertToHaloIR(const tflite::Model& model) {
  // TODO(unknown) we assume only one subgraph in the model
  Status s = Status::SUCCESS;
  const auto subgraphs = model.subgraphs();
  HLCHECK(subgraphs->size() == 1);
  const auto& graph_def = *subgraphs->GetMutableObject(0);
  std::unordered_set<int32_t> inputs_set;
  const auto& inputs = *graph_def.inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs_set.insert(inputs[i]);
  }

  // Convert Input and Placeholder
  const auto buffers = model.buffers();
  const auto tensors = graph_def.tensors();
  for (size_t i = 0; i < tensors->size(); ++i) {
    const auto tensor = tensors->Get(i);
    if (inputs_set.count(i)) {
      s = ConvertPlaceholderNode(*tensor, i);
      if (s != Status::SUCCESS) {
        return s;
      }
    }
    ConvertConstNode(*buffers, *tensor, i);
  }

  // Convert computation node
  const auto operators = graph_def.operators();
  const auto op_codes = model.operator_codes();
  for (size_t i = 0; i < operators->size(); ++i) {
    const auto cur_node_def = operators->Get(i);
    const auto cur_op = op_codes->Get(cur_node_def->opcode_index());
    tflite::BuiltinOperator cur_op_type =
        static_cast<tflite::BuiltinOperator>(cur_op->builtin_code());
    s = ConvertOneNode(ir_builder_, *cur_node_def, cur_op_type);
    if (s != Status::SUCCESS) {
      return s;
    }
  }

  // Convert output
  std::vector<Def> outputs;
  const auto& output_ids = *graph_def.outputs();
  for (size_t i = 0; i < output_ids.size(); ++i) {
    auto it = inst_id_to_ptr_.find(output_ids[i]);
    if (it != inst_id_to_ptr_.end()) {
      auto inst = it->second;
      int idx = 0;
      HLCHECK(0 <= idx && idx <= 1024);
      outputs.emplace_back(Def{inst, idx});
    } else {
      LOG(ERROR) << "Output " << i << " not found.";
    }
  }
  if (!outputs.empty()) {
    ir_builder_->CreateReturn("output", outputs);
  }

  return Status::SUCCESS;
}

Status TFLITEParser::ConvertOneNode(
    std::unique_ptr<IRBuilder>& ir_builder, const tflite::Operator& cur_node,
    const tflite::BuiltinOperator& cur_op_type) {
  Status s = Status::SUCCESS;
  const std::string op_type(tflite::EnumNameBuiltinOperator(cur_op_type));
  auto fp = func_lists_.find(op_type);
  if (fp != func_lists_.end()) {
    s = (fp->second)(ir_builder, cur_node);
    if (s != Status::SUCCESS) {
      return s;
    }
  } else {
    LOG(ERROR)
        << "Convert function not found, Please check if it is supported: "
        << "Op: [" << cur_op_type << "], Index: [" << -1 << "]";
    return Status::ASSERTION;
  }
  return Status::SUCCESS;
}

void TFLITEParser::RegisterOp() {
#include "tflite_regist_op.h.inc"
}

template <>
const Tensor<float> TFLITEParser::ProcessTensor<float>(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers,
    const tflite::Tensor& tensor) {
  const DataType& data_type = ProcessDataType(tensor.type());
  std::vector<int64_t> shape = ProcessTensorShape<int64_t>(tensor.shape());
  auto v = DecodeTensorContent<float>(buffers, tensor);
  return Tensor<float>(data_type, shape, v);
}

template <>
const Tensor<int> TFLITEParser::ProcessTensor<int>(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers,
    const tflite::Tensor& tensor) {
  const DataType& data_type = ProcessDataType(tensor.type());
  std::vector<int64_t> shape = ProcessTensorShape<int64_t>(tensor.shape());
  auto v = DecodeTensorContent<int>(buffers, tensor);
  return Tensor<int>(data_type, shape, v);
}

template <>
const Tensor<int64_t> TFLITEParser::ProcessTensor<int64_t>(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers,
    const tflite::Tensor& tensor) {
  const DataType& data_type = ProcessDataType(tensor.type());
  std::vector<int64_t> shape = ProcessTensorShape<int64_t>(tensor.shape());
  auto v = DecodeTensorContent<int64_t>(buffers, tensor);
  return Tensor<int64_t>(data_type, shape, v);
}

template <>
const Tensor<uint8_t> TFLITEParser::ProcessTensor<uint8_t>(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers,
    const tflite::Tensor& tensor) {
  const DataType& data_type = ProcessDataType(tensor.type());
  std::vector<int64_t> shape = ProcessTensorShape<int64_t>(tensor.shape());
  auto v = DecodeTensorContent<uint8_t>(buffers, tensor);
  return Tensor<uint8_t>(data_type, shape, v);
}

IRObject* TFLITEParser::ConvertConstNode(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers,
    const tflite::Tensor& tensor, size_t id) {
  DataType data_type = ProcessDataType(tensor.type());
  IRObject* inst = nullptr;
  switch (data_type) {
    case DataType::FLOAT32: {
      Tensor<float> temp = ProcessTensor<float>(buffers, tensor);
      // Skip useless output tensor info
      if (temp.GetData().empty()) {
        return nullptr;
      }

      // TODO(unkonwn) hack here for quick validation, will replace with
      // transpose node later
      if (temp.GetShape().size() == 4) {
        const Tensor<float> trans = ConvertOHWI2HWIO<float>(temp);
        inst = c_builder_->CreateConstant(tensor.name()->str(),
                                          Type(data_type, trans.GetShape()),
                                          trans.GetData());
      } else {
        inst = c_builder_->CreateConstant(tensor.name()->str(),
                                          Type(data_type, temp.GetShape()),
                                          temp.GetData());
      }
      inst_id_to_ptr_.emplace(id, inst);
      break;
    }
    case DataType::INT32: {
      Tensor<int> temp = ProcessTensor<int>(buffers, tensor);
      // Skip useless output tensor info
      if (temp.GetData().empty()) {
        return nullptr;
      }

      // TODO(unkonwn) hack here for quick validation, will replace with
      // transpose node later
      if (temp.GetShape().size() == 4) {
        Tensor<int> trans = ConvertOHWI2HWIO<int>(temp);
        inst = c_builder_->CreateConstant(tensor.name()->str(),
                                          Type(data_type, trans.GetShape()),
                                          trans.GetData());
      } else {
        inst = c_builder_->CreateConstant(tensor.name()->str(),
                                          Type(data_type, temp.GetShape()),
                                          temp.GetData());
      }

      inst = c_builder_->CreateConstant(tensor.name()->str(),
                                        Type(data_type, temp.GetShape()),
                                        temp.GetData());
      inst_id_to_ptr_.emplace(id, inst);
      break;
    }
    case DataType::INT64: {
      Tensor<int64_t> temp = ProcessTensor<int64_t>(buffers, tensor);
      // Skip useless output tensor info
      if (temp.GetData().empty()) {
        return nullptr;
      }

      // TODO(unkonwn) hack here for quick validation, will replace with
      // transpose node later
      if (temp.GetShape().size() == 4) {
        const Tensor<int64_t> trans = ConvertOHWI2HWIO<int64_t>(temp);
        inst = c_builder_->CreateConstant(tensor.name()->str(),
                                          Type(data_type, trans.GetShape()),
                                          trans.GetData());
      } else {
        inst = c_builder_->CreateConstant(tensor.name()->str(),
                                          Type(data_type, temp.GetShape()),
                                          temp.GetData());
      }

      inst_id_to_ptr_.emplace(id, inst);
      break;
    }
    case DataType::UINT8: {
      Tensor<uint8_t> temp = ProcessTensor<uint8_t>(buffers, tensor);
      // Skip useless output tensor info
      if (temp.GetData().empty()) {
        return nullptr;
      }
      // TODO(unkonwn) hack here for quick validation, will replace with
      // transpose node later
      if (temp.GetShape().size() == 4) {
        const Tensor<uint8_t> trans = ConvertOHWI2HWIO<uint8_t>(temp);
        inst = c_builder_->CreateConstant(tensor.name()->str(),
                                          Type(data_type, trans.GetShape()),
                                          trans.GetData());
      } else {
        inst = c_builder_->CreateConstant(tensor.name()->str(),
                                          Type(data_type, temp.GetShape()),
                                          temp.GetData());
      }
      inst_id_to_ptr_.emplace(id, inst);
      break;
    }
    default:
      HLCHECK(0 && "Unsupported data type");
  }
  return inst;
}

Status TFLITEParser::ConvertPlaceholderNode(const tflite::Tensor& tensor,
                                            size_t id) {
  DataType data_type = ProcessDataType(tensor.type());
  std::vector<int64_t> shape = ProcessTensorShape<int64_t>(tensor.shape());
  auto arg = arg_builder_->CreateArgument(tensor.name()->str(),
                                          Type(data_type, shape));
  inst_id_to_ptr_.emplace(id, arg);
  return Status::SUCCESS;
}

std::vector<Def> TFLITEParser::GetInputOperands(
    const tflite::Operator& node_def) {
  std::vector<Def> operands;
  size_t operand_num = node_def.inputs()->size();
  for (size_t i = 0; i < operand_num; ++i) {
    int32_t input_node_id = node_def.inputs()->Get(i);
    std::unordered_map<int32_t, IRObject*>::iterator it =
        inst_id_to_ptr_.find(input_node_id);
    int idx = 0;
    if (it != inst_id_to_ptr_.end()) {
      auto inst = it->second;
      HLCHECK(0 <= idx && idx <= 1024);
      operands.emplace_back(Def{inst, idx});
    } else {
      LOG(ERROR) << " Cur node's " << i << "th operand:" << input_node_id
                 << " not found";
    }
  }

  return operands;
}

void TFLITEParser::InsertIDToInstMap(const tflite::Operator& node_def,
                                     IRObject* inst) {
  size_t num_outputs = node_def.outputs()->size();
  for (size_t i = 0; i < num_outputs; ++i) {
    inst_id_to_ptr_.emplace(node_def.outputs()->Get(i), inst);
  }
}

// convert to halo ir def func
#include "tflite_convert.cc.inc"

std::unique_ptr<Parser> CreateTFLITEParser() {
  return std::make_unique<TFLITEParser>();
}

} // end namespace halo
