//===- onnx_parser.cc -----------------------------------------------------===//
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

#include "onnx_parser.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>

#include "halo/lib/framework/common.h"
#include "halo/lib/framework/type.h"
#include "halo/lib/ir/extension_instructions.h"
#include "onnx.pb.h"

namespace halo {

ONNXParser::~ONNXParser() {}

Status ONNXParser::Parse(Function* function,
                         const std::vector<std::string>& file_list,
                         const armory::Opts& opts) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  HLCHECK(!file_list.empty());
  std::ifstream ifs(file_list.front());
  HLCHECK(!ifs.fail());

  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  onnx::ModelProto model_def;
  std::string data{std::istreambuf_iterator<char>{ifs},
                   std::istreambuf_iterator<char>{}};
  google::protobuf::io::ArrayInputStream input_stream(
      data.c_str(), static_cast<int>(data.size()));
  google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  coded_stream.SetTotalBytesLimit((2048LL << 20) - 1, 512LL << 20);
  if (!model_def.ParseFromCodedStream(&coded_stream)) {
    // Try to parse it as data file.
    ifs.clear();
    ifs.seekg(0);
    onnx::TensorProto tensor_def;
    if (tensor_def.ParsePartialFromIstream(&ifs)) {
      c_builder_ = std::make_unique<ConstantBuilder>(function->GetParent());
      // Use the function name as data for testing purpose.
      tensor_def.set_name(function->GetName());
      ConvertConstNode(tensor_def);
      return Status::SUCCESS;
    } else {
      LOG(ERROR) << "Encountered error(s) when parsing " << file_list.front();
      return Status::ASSERTION;
    }
  }
  if (!model_def.has_graph()) {
    LOG(ERROR) << "No graph is defined in onnx file.";
    return Status::ASSERTION;
  }
  const onnx::GraphProto& graph_def = model_def.graph();
  BasicBlockBuilder bb_builder(function);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");
  return Parse(bb, graph_def, opts);
}

Status ONNXParser::Parse(BasicBlock* bb, const onnx::GraphProto& graph_def,
                         const armory::Opts& opts) {
  RegisterOp();
  auto function = bb->GetParent();
  ir_builder_ = std::make_unique<IRBuilder>(bb);
  arg_builder_ = std::make_unique<ArgumentBuilder>(function);
  c_builder_ = std::make_unique<ConstantBuilder>(function);
  opts_ = opts;
  return ConvertToHaloIR(graph_def);
}

Status ONNXParser::ConvertToHaloIR(const onnx::GraphProto& graph_def) {
  Status s = Status::SUCCESS;
  // Convert constant.
  std::set<std::string> const_input_names;
  auto const_inputs_size = graph_def.initializer_size();
  for (int i = 0; i < const_inputs_size; ++i) {
    const_input_names.emplace(graph_def.initializer(i).name());
    ConvertConstNode(graph_def.initializer(i));
  }

  // Convert input
  auto input_infos_size = graph_def.input_size();
  // std::string info_type("input");
  for (int i = 0; i < input_infos_size; ++i) {
    // InitValueShapeInfo(input_infos[i], info_type);
    auto name = graph_def.input(i).name();
    // const node may appear in the input list
    auto it = const_input_names.find(name);
    if (it == const_input_names.end()) {
      s = ConvertPlaceholderNode(graph_def.input(i));
      if (s != Status::SUCCESS) {
        return s;
      }
    }
  }

  // Convert node
  auto node_size = graph_def.node_size();
  for (int i = 0; i < node_size; ++i) {
    // 1.Constant input not appear in graph constant inputs initializer list
    if (graph_def.node(i).op_type() == "Constant") {
      s = ConvertConstNode(graph_def.node(i));
      if (s != Status::SUCCESS) {
        return s;
      }
      continue;
    }
    if (graph_def.node(i).op_type() == "ConstantOfShape") {
      const auto& node = graph_def.node(i);
      s = ConvertOneNode(node);
      if (s != Status::SUCCESS) {
        return s;
      }
      auto it = inst_name_to_ptr_.find(node.output(0));
      HLCHECK(it != inst_name_to_ptr_.end());
      IRObject* obj = it->second.first;
      HLCHECK(IsA<ONNXExtensionInst>(obj));
      ONNXExtensionInst* inst = DynCast<ONNXExtensionInst>(obj);
      std::unique_ptr<Attribute> attr_val = Attribute::CreateFloat("value", 0);
      auto attr_dt =
          Attribute::CreateEnumDataType("data_type", DataType::FLOAT32);

      if (node.attribute_size() == 1) {
        const auto& attr = node.attribute().begin();
        HLCHECK(attr->type() == onnx::AttributeProto::TENSOR);
        HLCHECK(attr->has_t());
        const auto& tensor_def = attr->t();
        DataType data_type = ProcessDataType(tensor_def.data_type());
        attr_dt = Attribute::CreateEnumDataType("data_type", data_type);
        switch (data_type) {
          case DataType::FLOAT32: {
            const Tensor<float> temp = ProcessTensor<float>(tensor_def);
            HLCHECK(temp.GetShape().size() == 1 && temp.GetShape()[0] == 1);
            attr_val = Attribute::CreateFloat("value", temp.GetData()[0]);
            break;
          }
          case DataType::INT32: {
            const Tensor<int> temp = ProcessTensor<int>(tensor_def);
            HLCHECK(temp.GetShape().size() == 1 && temp.GetShape()[0] == 1);
            attr_val = Attribute::CreateInteger("value", temp.GetData()[0]);
            break;
          }
          case DataType::INT64: {
            const Tensor<int64_t> temp = ProcessTensor<int64_t>(tensor_def);
            HLCHECK(temp.GetShape().size() == 1 && temp.GetShape()[0] == 1);
            attr_val = Attribute::CreateInteger("value", temp.GetData()[0]);
            break;
          }
          default:
            HLCHECK(0 && "Unsupported data type");
        }
      }
      inst->AddOneAttribute(std::move(attr_dt));
      inst->AddOneAttribute(std::move(attr_val));
      continue;
    }

    s = ConvertOneNode(graph_def.node(i));
    if (s != Status::SUCCESS) {
      return s;
    }
  }

  // Convert output
  std::vector<Def> outputs;
  auto output_infos_size = graph_def.output_size();
  for (int i = 0; i < output_infos_size; ++i) {
    auto name = graph_def.output(i).name();
    auto it = inst_name_to_ptr_.find(name);
    int idx = 0;
    if (it != inst_name_to_ptr_.end()) {
      auto inst = it->second.first;
      idx = it->second.second;
      HLCHECK(0 <= idx && idx <= 1024);
      outputs.emplace_back(Def{inst, idx});
    } else {
      LOG(ERROR) << "Output " << i << " :" << name << " not found.";
    }
  }
  if (!outputs.empty()) {
    ir_builder_->CreateReturn("output", outputs);
  }

  return Status::SUCCESS;
}

Status ONNXParser::ConvertOneNode(const onnx::NodeProto& cur_node) {
  Status s = Status::SUCCESS;
  auto fp = func_lists_.find(cur_node.op_type());
  if (fp != func_lists_.end()) {
    s = (fp->second)(cur_node);
    if (s != Status::SUCCESS) {
      return s;
    }
  } else {
    if (opts_.print_diagnostic_report) {
      ONNXParser::WriteCSVReport(cur_node, std::cout);
      ConvertDummyNode(cur_node);
    } else {
      LOG(ERROR)
          << "Convert function not found, Please check if it is supported: "
          << "Name: "
          << "[" << cur_node.name() << "], Op: [" << cur_node.op_type()
          << "], Index: "
          << "[" << -1 << "]";
      return Status::ASSERTION;
    }
  }
  return Status::SUCCESS;
}

void ONNXParser::WriteCSVReport(const onnx::NodeProto& cur_node,
                                std::ostream& os) {
  os << "Name: [" << cur_node.name() << "], Op: [" << cur_node.op_type()
     << "]\n";
}

void ONNXParser::RegisterOp(){
#include "onnx_regist_op.h.inc"
}

halo::DataType ONNXParser::ProcessDataType(int data_type) {
  switch (data_type) {
    case onnx::TensorProto::FLOAT:
      return DataType::FLOAT32;
    case onnx::TensorProto::INT64:
      return DataType::INT64;
    case onnx::TensorProto::INT32:
      return DataType::INT32;
    case onnx::TensorProto::INT16:
      return DataType::INT16;
    case onnx::TensorProto::INT8:
      return DataType::INT8;
    case onnx::TensorProto::UINT8:
      return DataType::UINT8;
    case onnx::TensorProto::STRING:
      return DataType::STRING;
    case onnx::TensorProto::BOOL:
      return DataType::BOOL;
    default:
      LOG(ERROR) << "Unsupported DataType.";
      return DataType::INVALID;
  }
}

static void ProcessTensorShape(const onnx::TensorProto& tensor_def,
                               std::vector<int64_t>& shape) {
  const int dim_size = tensor_def.dims_size();
  for (int i = 0; i < dim_size; ++i) {
    shape.push_back(tensor_def.dims(i));
  }
}

static size_t GetTensorDataSize(const onnx::TensorProto& tensor_proto) {
  switch (tensor_proto.data_type()) {
    case onnx::TensorProto::FLOAT:
      return tensor_proto.float_data_size();
    case onnx::TensorProto::INT64:
      return tensor_proto.int64_data_size();
    case onnx::TensorProto::INT32:
      return tensor_proto.int32_data_size();
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::STRING:
      return tensor_proto.string_data_size();
    default:
      LOG(ERROR) << "Unsupported DataType.";
      return 0;
  }
}

template <typename T>
static void GetTensorData(const onnx::TensorProto& tensor_proto,
                          std::vector<T>& v, size_t size) {
  LOG(ERROR) << "Unsupported DataType.";
}

template <>
void GetTensorData(const onnx::TensorProto& tensor, std::vector<float>& v,
                   size_t size) {
  for (size_t i = 0; i < size; ++i) {
    v.push_back(tensor.float_data(i));
  }
}

template <>
void GetTensorData(const onnx::TensorProto& tensor, std::vector<int64_t>& v,
                   size_t size) {
  for (size_t i = 0; i < size; ++i) {
    v.push_back(tensor.int64_data(i));
  }
}

template <>
void GetTensorData(const onnx::TensorProto& tensor, std::vector<int32_t>& v,
                   size_t size) {
  for (size_t i = 0; i < size; ++i) {
    v.push_back(tensor.int32_data(i));
  }
}

template <>
void GetTensorData(const onnx::TensorProto& tensor, std::vector<int8_t>& v,
                   size_t size) {
  for (size_t i = 0; i < size; ++i) {
    v.push_back(tensor.raw_data()[i]);
  }
}

template <typename T>
Tensor<T> ONNXParser::ProcessTensor(const onnx::TensorProto& tensor_proto) {
  const DataType& data_type = ProcessDataType(tensor_proto.data_type());
  std::vector<int64_t> shape;
  ProcessTensorShape(tensor_proto, shape);

  std::vector<T> v;
  const int v_size = GetTensorDataSize(tensor_proto);
  if (v_size != 0) {
    v.reserve(v_size);
    GetTensorData(tensor_proto, v, v_size);
  } else if (!tensor_proto.raw_data().empty()) {
    v = Tensor<T>::DecodeTensorContent(tensor_proto.raw_data());
  } else {
    // TODO(unknown): handle external storage
    LOG(ERROR) << "Unsupported external data storage.";
  }

  if (shape.empty() && v.size() > 1) {
    shape.push_back(v.size());
  }
  return Tensor<T>(data_type, shape, v);
}

IRObject* ONNXParser::ConvertConstNode(const onnx::TensorProto& tensor_def) {
  DataType data_type = ProcessDataType(tensor_def.data_type());
  IRObject* inst = nullptr;
  switch (data_type) {
    case DataType::FLOAT32: {
      const Tensor<float> temp = ProcessTensor<float>(tensor_def);
      inst = c_builder_->CreateConstant(
          tensor_def.name(), Type(data_type, temp.GetShape()), temp.GetData());
      inst_name_to_ptr_.emplace(tensor_def.name(), std::make_pair(inst, 0));
      break;
    }
    case DataType::INT32: {
      const Tensor<int> temp = ProcessTensor<int>(tensor_def);
      inst = c_builder_->CreateConstant(
          tensor_def.name(), Type(data_type, temp.GetShape()), temp.GetData());
      inst_name_to_ptr_.emplace(tensor_def.name(), std::make_pair(inst, 0));
      break;
    }
    case DataType::INT64: {
      const Tensor<int64_t> temp = ProcessTensor<int64_t>(tensor_def);
      inst = c_builder_->CreateConstant(
          tensor_def.name(), Type(data_type, temp.GetShape()), temp.GetData());
      inst_name_to_ptr_.emplace(tensor_def.name(), std::make_pair(inst, 0));
      break;
    }
    case DataType::BOOL: {
      const Tensor<int8_t> temp = ProcessTensor<int8_t>(tensor_def);
      inst = c_builder_->CreateConstant(tensor_def.name(),
                                        Type(DataType::INT8, temp.GetShape()),
                                        temp.GetData());
      inst_name_to_ptr_.emplace(tensor_def.name(), std::make_pair(inst, 0));
      break;
    }
    default:
      HLCHECK(0 && "Unsupported data type");
  }
  return inst;
}

Status ONNXParser::ConvertConstNode(const onnx::NodeProto& cur_node) {
  for (const auto& attr : cur_node.attribute()) {
    HLCHECK(attr.type() == onnx::AttributeProto::TENSOR);
    HLCHECK(attr.has_t());
    auto inst = ConvertConstNode(attr.t());
    if (inst->GetName().empty()) {
      // Fix constant node name is null in generated cpp code
      inst->SetName(cur_node.name());
    }
    InsertIDToInstMap(cur_node, inst);
  }
  return Status::SUCCESS;
}

Status ONNXParser::ConvertPlaceholderNode(
    const onnx::ValueInfoProto& value_info_def) {
  HLCHECK(value_info_def.type().has_tensor_type() &&
          "Unsupported value info type.");
  auto type_def = value_info_def.type().tensor_type();
  DataType data_type = ProcessDataType(type_def.elem_type());
  auto shape_def = type_def.shape();
  const int dim_size = shape_def.dim_size();
  std::vector<int64_t> shape;
  for (int i = 0; i < dim_size; i++) {
    auto dim_def = shape_def.dim(i);
    if (dim_def.dim_value()) {
      shape.push_back(dim_def.dim_value());
    } else {
      // TODO(unknown): Handle dim_param case
      shape.push_back(-1);
    }
  }
  auto arg = arg_builder_->CreateArgument(value_info_def.name(),
                                          Type(data_type, shape));
  inst_name_to_ptr_.emplace(value_info_def.name(), std::make_pair(arg, 0));
  return Status::SUCCESS;
}

std::vector<Def> ONNXParser::GetInputOperands(const onnx::NodeProto& node_def) {
  std::vector<Def> operands;
  size_t operand_num = node_def.input_size();
  for (size_t i = 0; i < operand_num; ++i) {
    std::string input_node_name = node_def.input(i);
    std::unordered_map<std::string, std::pair<IRObject*, int>>::iterator it;
    it = inst_name_to_ptr_.find(input_node_name);
    // TODO(unknown): handle multiple outputs
    int idx = 0;
    if (it != inst_name_to_ptr_.end()) {
      auto inst = it->second.first;
      idx = it->second.second;
      HLCHECK(0 <= idx && idx <= 1024);
      operands.emplace_back(Def{inst, idx});
    } else {
      // those errors will be record in diagnostic report file
      // LOG(ERROR) << node_def.name() << " Node's" << i
      //<< "th operand:" << node_def.input(i) << " not found";
    }
  }
  return operands;
}

void ONNXParser::InsertIDToInstMap(const onnx::NodeProto& node_def,
                                   IRObject* inst) {
  size_t num_outputs = node_def.output_size();
  if (inst->GetNumOfResults() != num_outputs) {
    inst->SetNumOfResults(num_outputs);
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    inst_name_to_ptr_.emplace(node_def.output(i), std::make_pair(inst, i));
  }
}

// Process Attribute
ONNXAttrs::ONNXAttrs(const onnx::NodeProto& node_def) {
  int num_attrs = node_def.attribute_size();
  for (int i = 0; i < num_attrs; ++i) {
    const auto& attr = node_def.attribute(i);
    attr_map_.emplace(attr.name(), attr);
  }
}

template <>
bool ONNXAttrs::Process<std::string>(const std::string& key,
                                     std::string* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::STRING);
  *value = attr_map_.at(key).s();
  return true;
}

template <>
bool ONNXAttrs::Process<int64_t>(const std::string& key, int64_t* value) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::INT);
  *value = attr_map_.at(key).i();
  return true;
}

template <>
bool ONNXAttrs::Process<int>(const std::string& key, int* value) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::INT);
  *value = static_cast<int>(attr_map_.at(key).i());
  return true;
}

template <>
bool ONNXAttrs::Process<bool>(const std::string& key, bool* value) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::INT);
  *value = static_cast<bool>(attr_map_.at(key).i());
  return true;
}

template <>
bool ONNXAttrs::Process<float>(const std::string& key, float* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::FLOAT);
  *value = attr_map_.at(key).f();
  return true;
}

template <>
bool ONNXAttrs::Process<std::vector<std::string>>(
    const std::string& key, std::vector<std::string>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.type() == onnx::AttributeProto::STRINGS);
  int size = attr_value.strings_size();
  (*value).reserve(size);
  for (int i = 0; i < size; ++i) {
    (*value).emplace_back(attr_value.strings(i));
  }
  return true;
}

template <>
bool ONNXAttrs::Process<std::vector<int64_t>>(const std::string& key,
                                              std::vector<int64_t>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.type() == onnx::AttributeProto::INTS);
  int size = attr_value.ints_size();
  (*value).reserve(size);
  std::copy(attr_value.ints().begin(), attr_value.ints().end(),
            (*value).begin());
  return true;
}

template <>
bool ONNXAttrs::Process<std::vector<int>>(const std::string& key,
                                          std::vector<int>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.type() == onnx::AttributeProto::INTS);
  int size = attr_value.ints_size();
  (*value).reserve(size);
  for (int i = 0; i < size; ++i) {
    (*value).push_back(static_cast<int>(attr_value.ints(i)));
  }
  return true;
}

template <>
bool ONNXAttrs::Process<std::vector<float>>(const std::string& key,
                                            std::vector<float>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.type() == onnx::AttributeProto::FLOATS);
  (*value).reserve(attr_value.floats_size());
  std::copy(attr_value.floats().begin(), attr_value.floats().end(),
            (*value).begin());
  return true;
}

template <>
bool ONNXAttrs::Process<Padding>(const std::string& key, Padding* padding) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::STRING);
  static const std::unordered_map<std::string, Padding> enum_map{
      {"VALID", Padding::VALID},
      {"SAME_UPPER", Padding::SAME},
      {"SAME_LOWER", Padding::SAME_LOWER},
      {"NOTSET", Padding::EXPLICIT},
  };

  *padding = enum_map.count(attr_map_.at(key).s())
                 ? enum_map.at(attr_map_.at(key).s())
                 : Padding::INVALID;
  return true;
}

template <>
bool ONNXAttrs::Process<DataType>(const std::string& key, DataType* data_type) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::INT);
  *data_type = ONNXParser::ProcessDataType(attr_map_.at(key).type());
  return true;
}

template <>
bool ONNXAttrs::Process<PadMode>(const std::string& key, PadMode* pad_mode) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::STRING);
  static const std::unordered_map<std::string, PadMode> enum_map{
      {"CONSTANT", PadMode::CONSTANT},
      {"REFLECT", PadMode::REFLECT},
      {"EDGE", PadMode::EDGE},
  };

  std::string mode = attr_map_.at(key).s();
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](char c) { return std::toupper(c); });
  *pad_mode = enum_map.count(mode) ? enum_map.at(mode) : PadMode::INVALID;
  return true;
}

Status ONNXParser::ConvertDummyNode(const onnx::NodeProto& node_def) {
  std::vector<Def> operands = GetInputOperands(node_def);
  auto inst = ir_builder_->CreateDummy(
      node_def.name(), operands, node_def.output_size(), node_def.op_type());
  InsertIDToInstMap(node_def, inst);
  return Status::SUCCESS;
}

// convert to halo ir def func
#include "onnx_convert.cc.inc"

} // end namespace halo