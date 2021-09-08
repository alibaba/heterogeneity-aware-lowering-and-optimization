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
#include "halo/lib/parser/parser.h"
#include "onnx.pb.h"

namespace halo {

ONNXParser::~ONNXParser() {}

bool ONNXParser::Scope::Contains(const std::string& name) {
  return inst_name_to_ptr_.count(name) != 0;
}

Value ONNXParser::Scope::Find(const std::string& name) {
  const static Value empty_value;
  auto it = inst_name_to_ptr_.find(name);
  if (it != inst_name_to_ptr_.end()) {
    return it->second;
  }
  return (parent_ == nullptr) ? empty_value : parent_->Find(name);
}

void ONNXParser::Scope::Insert(const onnx::TensorProto& tensor,
                               const Value& def) {
  Insert(tensor.name(), def);
}

ONNXParser::Scope* ONNXParser::Scope::CreateScope() {
  sub_scopes_.push_back(std::make_unique<Scope>());
  Scope* new_scope = sub_scopes_.back().get();
  new_scope->parent_ = this;
  return new_scope;
}

void ONNXParser::Scope::Insert(const std::string& name, const Value& def) {
  if (inst_name_to_ptr_.count(name) > 0) {
    std::cerr << "Duplicated :" << std::endl;
    inst_name_to_ptr_[name].GetOwner()->Dump();
    def.GetOwner()->Dump();
    std::cerr << std::endl;
  }
  inst_name_to_ptr_[name] = def;
}

Status ONNXParser::Parse(Function* function,
                         const std::vector<const char*>& buffers,
                         const std::vector<size_t>& buffer_sizes) {
  return Status::ASSERTION;
}

Status ONNXParser::Parse(Function* function,
                         const std::vector<const void*>& model_defs) {
  if (model_defs.empty() || model_defs[0] == nullptr) {
    return Status::FILE_NOT_EXIST;
  }

  const onnx::GraphProto* graph_def =
      reinterpret_cast<const onnx::GraphProto*>(model_defs[0]);
  bb_builder_ = std::make_unique<BasicBlockBuilder>(function);
  BasicBlock* bb = bb_builder_->CreateBasicBlock("bb0");
  armory::Opts opts;
  return Parse(bb, *graph_def, opts);
}

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
  if (!model_def.ParseFromCodedStream(&coded_stream) ||
      !model_def.has_graph()) {
    // Try to parse it as data file.
    ifs.clear();
    ifs.seekg(0);
    onnx::TensorProto tensor_def;
    if (tensor_def.ParsePartialFromIstream(&ifs)) {
      c_builder_ = std::make_unique<ConstantBuilder>(function->GetParent());
      // Use the function name as data for testing purpose.
      tensor_def.set_name(function->GetName());
      ConvertConstNode(c_builder_.get(), tensor_def);
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
  bb_builder_ = std::make_unique<BasicBlockBuilder>(function);
  BasicBlock* bb = bb_builder_->CreateBasicBlock("bb0");
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
    ConvertConstNode(c_builder_.get(), graph_def.initializer(i));
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
      s = ConvertPlaceholderNode(arg_builder_.get(), graph_def.input(i));
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
      const auto& name = graph_def.node(i).output(0);
      if (curr_scope_->Contains(name)) {
        continue;
      }
      s = ConvertConstNode(c_builder_.get(), graph_def.node(i));
      if (s != Status::SUCCESS) {
        return s;
      }
      continue;
    }
    if (graph_def.node(i).op_type() == "ConstantOfShape") {
      const auto& node = graph_def.node(i);
      s = ConvertOneNode(ir_builder_.get(), node);
      if (s != Status::SUCCESS) {
        return s;
      }
      auto def = curr_scope_->Find(node.output(0));
      HLCHECK(!def.IsNull());
      IRObject* obj = def.GetOwner();
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

    s = ConvertOneNode(ir_builder_.get(), graph_def.node(i));
    if (s != Status::SUCCESS) {
      return s;
    }
  }

  // Convert output
  std::vector<Def> outputs;
  auto output_infos_size = graph_def.output_size();
  for (int i = 0; i < output_infos_size; ++i) {
    auto name = graph_def.output(i).name();
    auto val = curr_scope_->Find(name);
    if (!val.IsNull()) {
      auto inst = val.GetOwner();
      int idx = val.GetIdx();
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

Status ONNXParser::ConvertOneNode(IRBuilder* ir_builder,
                                  const onnx::NodeProto& node_def) {
  Status s = Status::SUCCESS;
  auto fp = func_lists_.find(node_def.op_type());
  if (fp != func_lists_.end()) {
    s = (fp->second)(ir_builder, node_def);
    if (s != Status::SUCCESS) {
      return s;
    }
  } else {
    if (opts_.print_diagnostic_report) {
      ONNXParser::WriteCSVReport(node_def, std::cout);
      ConvertDummyNode(ir_builder, node_def);
    } else {
      LOG(ERROR)
          << "Convert function not found, Please check if it is supported: "
          << "Name: "
          << "[" << node_def.name() << "], Op: [" << node_def.op_type()
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

void ONNXParser::RegisterOp() {
  func_lists_.emplace("Loop",
                      std::bind(&ONNXParser::ConvertLoopNode, this,
                                std::placeholders::_1, std::placeholders::_2));
#include "onnx_regist_op.h.inc"
}

halo::DataType ONNXParser::ProcessDataType(int data_type, bool allow_invalid) {
  switch (data_type) {
    case onnx::TensorProto::FLOAT:
      return DataType::FLOAT32;
    case onnx::TensorProto::FLOAT16:
      return DataType::FLOAT16;
    case onnx::TensorProto::BFLOAT16:
      return DataType::BFLOAT16;
    case onnx::TensorProto::DOUBLE:
      return DataType::FLOAT64;
    case onnx::TensorProto::INT64:
      return DataType::INT64;
    case onnx::TensorProto::UINT64:
      return DataType::UINT64;
    case onnx::TensorProto::INT32:
      return DataType::INT32;
    case onnx::TensorProto::UINT32:
      return DataType::UINT32;
    case onnx::TensorProto::INT16:
      return DataType::INT16;
    case onnx::TensorProto::UINT16:
      return DataType::UINT16;
    case onnx::TensorProto::INT8:
      return DataType::INT8;
    case onnx::TensorProto::UINT8:
      return DataType::UINT8;
    case onnx::TensorProto::STRING:
      return DataType::STRING;
    case onnx::TensorProto::BOOL:
      return DataType::BOOL;
    default:
      if (!allow_invalid) {
        LOG(ERROR) << "Unsupported DataType.";
      }
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
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::BFLOAT16:
      return tensor_proto.int32_data_size() * 2;
    case onnx::TensorProto::DOUBLE:
      return tensor_proto.double_data_size();
    case onnx::TensorProto::INT64:
      return tensor_proto.int64_data_size();
    case onnx::TensorProto::INT32:
      return tensor_proto.int32_data_size();
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::UINT16:
      return tensor_proto.int32_data_size() * 2;
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::STRING:
      return tensor_proto.string_data_size();
    case onnx::TensorProto::BOOL:
      return 0; // 0 means bool is stored in raw data
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
void GetTensorData(const onnx::TensorProto& tensor, std::vector<uint16_t>& v,
                   size_t size) {
  for (size_t i = 0; i < size / 2; ++i) {
    int32_t x = tensor.int32_data(i);
    const int16_t* v16 = reinterpret_cast<const int16_t*>(&x); // NOLINT.
    v.push_back(v16[0]);                                       // NOLINT.
    v.push_back(v16[1]);                                       // NOLINT.
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

template <>
void GetTensorData(const onnx::TensorProto& tensor, std::vector<std::string>& v,
                   size_t size) {
  for (size_t i = 0; i < size; ++i) {
    v.push_back(std::string(tensor.string_data()[i]));
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
    // TODO(unknown): handle external storage and empty tensor, e.g. roi of
    // Resize module HLCHECK(tensor_proto.data_location() ==
    //         onnx::TensorProto::DataLocation::TensorProto_DataLocation_EXTERNAL);
    // HLCHECK(0 && "Unsupported external data storage.");
    LOG(WARNING) << "Unsupported external data storage or empty tensor.";
  }
  auto elems = v.size();
  if (data_type == DataType::FLOAT16) {
    elems /= 2;
  }
  if (shape.empty() && elems > 1) {
    shape.push_back(elems);
  }
  return Tensor<T>(data_type, shape, v);
}

IRObject* ONNXParser::ConvertConstNode(ConstantBuilder* c_builder,
                                       const onnx::TensorProto& tensor_def) {
  return ConvertConstNode(c_builder, tensor_def, tensor_def.name());
}

IRObject* ONNXParser::ConvertConstNode(ConstantBuilder* c_builder,
                                       const onnx::TensorProto& tensor_def,
                                       const std::string& name) {
  DataType data_type = ProcessDataType(tensor_def.data_type());
  IRObject* inst = nullptr;
  switch (data_type) {
    case DataType::FLOAT64: {
      const Tensor<double> temp = ProcessTensor<double>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    case DataType::FLOAT32: {
      const Tensor<float> temp = ProcessTensor<float>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    case DataType::UINT16:
    case DataType::FLOAT16:
    case DataType::BFLOAT16: {
      const Tensor<uint16_t> temp = ProcessTensor<uint16_t>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData().data());
      break;
    }
    case DataType::INT16: {
      const Tensor<int16_t> temp = ProcessTensor<int16_t>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData().data());
      break;
    }
    case DataType::INT32: {
      const Tensor<int> temp = ProcessTensor<int>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    case DataType::UINT32: {
      const Tensor<uint> temp = ProcessTensor<uint>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    case DataType::INT64: {
      const Tensor<int64_t> temp = ProcessTensor<int64_t>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    case DataType::UINT64: {
      const Tensor<uint64_t> temp = ProcessTensor<uint64_t>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    case DataType::STRING: {
      const Tensor<std::string> temp = ProcessTensor<std::string>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    case DataType::UINT8: {
      const Tensor<uint8_t> temp = ProcessTensor<uint8_t>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    case DataType::INT8:
    case DataType::BOOL: {
      const Tensor<int8_t> temp = ProcessTensor<int8_t>(tensor_def);
      inst = c_builder->CreateConstant(name, Type(data_type, temp.GetShape()),
                                       temp.GetData());
      break;
    }
    default:
      HLCHECK(0 && "Unsupported data type");
  }
  if (inst != nullptr) {
    curr_scope_->Insert(name, Value(inst, 0));
  }
  return inst;
}

Status ONNXParser::ConvertConstNode(ConstantBuilder* c_builder,
                                    const onnx::NodeProto& cur_node) {
  IRObject* inst = nullptr;
  for (const auto& attr : cur_node.attribute()) {
    if (attr.name() != "value") {
      continue;
    }
    if (attr.type() == onnx::AttributeProto::TENSOR) {
      HLCHECK(attr.has_t());
      inst = ConvertConstNode(c_builder, attr.t(), cur_node.output(0));
      return Status::SUCCESS;
    }
    if (attr.type() == onnx::AttributeProto::INT) {
      int64_t val = attr.i();
      inst = c_builder->CreateConstant(cur_node.name(),
                                       Type{DataType::INT64, {1}}, &val);
    } else if (attr.type() == onnx::AttributeProto::FLOAT) {
      float val = attr.f();
      inst = c_builder->CreateConstant(cur_node.name(),
                                       Type{DataType::FLOAT32, {1}}, &val);
    }
    HLCHECK(inst && "Unhandled attribute");

    if (inst->GetName().empty()) {
      // Fix constant node name is null in generated cpp code
      inst->SetName(cur_node.name());
    }
    InsertIDToInstMap(cur_node, inst);
  }
  return Status::SUCCESS;
}

Type ONNXParser::GetType(const onnx::ValueInfoProto& value_info_def) {
  HLCHECK(value_info_def.type().has_tensor_type() &&
          "Unsupported value info type.");
  auto type_def = value_info_def.type().tensor_type();
  DataType data_type = ProcessDataType(type_def.elem_type());
  const auto& shape_def = type_def.shape();
  const int dim_size = shape_def.dim_size();
  std::vector<int64_t> shape;
  for (int i = 0; i < dim_size; ++i) {
    const auto& dim_def = shape_def.dim(i);
    if (dim_def.dim_value() != 0) {
      shape.push_back(dim_def.dim_value());
    } else {
      shape.push_back(-1);
    }
  }
  return Type(data_type, shape);
}

Status ONNXParser::ConvertSubPlaceholderNode(
    ArgumentBuilder* arg_builder, const onnx::ValueInfoProto& value_info_def) {
  HLCHECK(!loop_arg_types_.empty());
  auto& type = loop_arg_types_.top();
  if (!type.IsValid()) {
    type = GetType(value_info_def);
  }
  auto arg = arg_builder->CreateArgument(value_info_def.name(), type);
  curr_scope_->Insert(value_info_def.name(), Value{arg, 0});
  loop_arg_types_.pop();
  return Status::SUCCESS;
}

Status ONNXParser::ConvertPlaceholderNode(
    ArgumentBuilder* arg_builder, const onnx::ValueInfoProto& value_info_def) {
  auto arg = arg_builder->CreateArgument(value_info_def.name(),
                                         GetType(value_info_def));
  curr_scope_->Insert(value_info_def.name(), Value{arg, 0});
  return Status::SUCCESS;
}

std::vector<Def> ONNXParser::GetInputOperands(const onnx::NodeProto& node_def) {
  std::vector<Def> operands;
  std::unordered_map<std::string, std::pair<IRObject*, int>>::iterator it;
  for (size_t i = 0, operand_num = node_def.input_size(); i < operand_num;
       ++i) {
    const auto& input_node_name = node_def.input(i);
    Value val = curr_scope_->Find(input_node_name);
    if (!val.IsNull()) {
      auto inst = val.GetOwner();
      int idx = val.GetIdx();
      HLCHECK(0 <= idx && idx <= 1024);
      operands.emplace_back(Def{inst, idx});
    } else {
      operands.emplace_back(Def::GetUndefined());
      // those errors will be record in diagnostic report file
      LOG(ERROR) << node_def.name() << " Node's" << i
                 << "th operand:" << node_def.input(i) << " not found";
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
    int idx = i;
    curr_scope_->Insert(node_def.output(i), Value{inst, idx});
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
bool ONNXAttrs::Process<onnx::GraphProto>(const std::string& key,
                                          onnx::GraphProto* value) {
  if (attr_map_.count(key) == 0) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::GRAPH);
  *value = attr_map_.at(key).g();
  return true;
}

template <>
bool ONNXAttrs::Process<std::string>(const std::string& key,
                                     std::string* value) {
  if (attr_map_.count(key) == 0) {
    return false;
  }
  value->clear();
  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::STRING);
  *value = attr_map_.at(key).s();
  return true;
}

template <>
bool ONNXAttrs::Process<int64_t>(const std::string& key, int64_t* value) {
  if (attr_map_.count(key) == 0) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::INT);
  *value = attr_map_.at(key).i();
  return true;
}

template <>
bool ONNXAttrs::Process<int>(const std::string& key, int* value) {
  if (attr_map_.count(key) == 0) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::INT);
  *value = static_cast<int>(attr_map_.at(key).i());
  return true;
}

template <>
bool ONNXAttrs::Process<bool>(const std::string& key, bool* value) {
  if (attr_map_.count(key) == 0) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::INT);
  *value = static_cast<bool>(attr_map_.at(key).i());
  return true;
}

template <>
bool ONNXAttrs::Process<float>(const std::string& key, float* value) {
  if (attr_map_.count(key) == 0) {
    return false;
  }
  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::FLOAT);
  *value = attr_map_.at(key).f();
  return true;
}

template <>
bool ONNXAttrs::Process<std::vector<std::string>>(
    const std::string& key, std::vector<std::string>* value) {
  if (attr_map_.count(key) == 0) {
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
  if (attr_map_.count(key) == 0) {
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
  if (attr_map_.count(key) == 0) {
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
  if (attr_map_.count(key) == 0) {
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

/// Get Attr for Shape
template <>
bool ONNXAttrs::Process<std::vector<std::vector<int64_t>>>(
    const std::string& key, std::vector<std::vector<int64_t>>* value) {
  if (attr_map_.count(key) == 0) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.type() == onnx::AttributeProto::STRINGS);
  /// HgEngine only support single optput in onnx
  int size = attr_value.strings_size();
  (*value).reserve(size);
  for (int i = 0; i < size; ++i) {
    (*value).emplace_back(std::stol(attr_value.strings(i)));
  }
  return true;
}

template <>
bool ONNXAttrs::Process<Padding>(const std::string& key, Padding* padding) {
  if (attr_map_.count(key) == 0) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::STRING);
  static const std::unordered_map<std::string, Padding> enum_map{
      {"VALID", Padding::VALID},
      {"SAME_UPPER", Padding::SAME},
      {"SAME_LOWER", Padding::SAME_LOWER},
      {"NOTSET", Padding::EXPLICIT},
  };

  *padding = enum_map.count(attr_map_.at(key).s()) != 0
                 ? enum_map.at(attr_map_.at(key).s())
                 : Padding::INVALID;
  return true;
}

template <>
bool ONNXAttrs::Process<DataType>(const std::string& key, DataType* data_type) {
  if (attr_map_.count(key) == 0) {
    return false;
  }

  HLCHECK(attr_map_.at(key).type() == onnx::AttributeProto::INT);
  *data_type = ONNXParser::ProcessDataType(attr_map_.at(key).type());
  return true;
}

template <>
bool ONNXAttrs::Process<PadMode>(const std::string& key, PadMode* pad_mode) {
  if (attr_map_.count(key) == 0) {
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
  *pad_mode = enum_map.count(mode) != 0 ? enum_map.at(mode) : PadMode::INVALID;
  return true;
}

Status ONNXParser::ConvertDummyNode(IRBuilder* ir_builder,
                                    const onnx::NodeProto& node_def) {
  std::vector<Def> operands = GetInputOperands(node_def);
  auto inst = ir_builder->CreateDummy(
      node_def.name(), operands, node_def.output_size(), node_def.op_type());
  InsertIDToInstMap(node_def, inst);
  return Status::SUCCESS;
}

// convert to halo ir def func
#include "onnx_convert.cc.inc"

std::unique_ptr<Parser> CreateONNXParser() {
  return std::make_unique<ONNXParser>();
}

Status ONNXParser::ConvertLoopNode(IRBuilder* ir_builder,
                                   const onnx::NodeProto& cur_node) {
  std::string cur_node_name = cur_node.name();
  if (cur_node_name.empty()) {
    // TODO(unknown) current node name must unique
    cur_node_name = "unknown";
  }
  const auto& operands = GetInputOperands(cur_node);
  // first 2 inputs(loop_cnt/loop_cond) is optional
  for (int64_t i = operands.size() - 1; i >= 0; --i) {
    loop_arg_types_.push(operands[i].GetType());
  }

  ONNXAttrs attrs(cur_node);
  onnx::GraphProto subgraph;
  curr_scope_ = curr_scope_->CreateScope();
  attrs.Process<onnx::GraphProto>("body", &subgraph);
  auto loop_body = bb_builder_->CreateBasicBlock("bb_" + cur_node_name);
  auto loop_ir_builder = std::make_unique<IRBuilder>(loop_body);
  auto arg_builder = std::make_unique<ArgumentBuilder>(loop_body);
  auto c_builder = std::make_unique<ConstantBuilder>(loop_body);
  std::set<std::string> const_input_names;
  for (int i = 0, const_inputs_size = subgraph.initializer_size();
       i < const_inputs_size; ++i) {
    const_input_names.emplace(subgraph.initializer(i).name());
    ConvertConstNode(c_builder.get(), subgraph.initializer(i));
  }

  // Convert input
  auto input_infos_size = subgraph.input_size();
  for (int i = 0; i < input_infos_size; ++i) {
    if (const_input_names.count(subgraph.input(i).name()) == 0) {
      ConvertSubPlaceholderNode(arg_builder.get(), subgraph.input(i));
    }
  }

  for (int i = 0, node_size = subgraph.node_size(); i < node_size; ++i) {
    VLOG(1) << "sub node name: " << subgraph.node(i).name();
    if (subgraph.node(i).op_type() == "Constant") {
      ConvertConstNode(c_builder.get(), subgraph.node(i));
      continue;
    }
    ConvertOneNode(loop_ir_builder.get(), subgraph.node(i));
  }

  // Convert output. Skip the first operand as "cond" is not a real output.
  std::vector<Def> outputs;

  for (int i = 1, output_infos_size = subgraph.output_size();
       i < output_infos_size; ++i) {
    auto name = subgraph.output(i).name();
    VLOG(1) << "output node name: " << name;
    auto value = curr_scope_->Find(name);
    if (!value.IsNull()) {
      HLCHECK(0 <= value.GetIdx() && value.GetIdx() <= 1024);
      outputs.emplace_back(Def{value.GetOwner(), value.GetIdx()});
    } else {
      LOG(ERROR) << "Output " << i << " :" << name << " not found.";
    }
  }
  if (!outputs.empty()) {
    loop_ir_builder->CreateReturn("output", outputs);
  }
  curr_scope_ = curr_scope_->GetParent();
  auto loop = ir_builder->CreateLoop(cur_node.name(), operands);
  loop->SetBody(loop_body);
  loop_body->SetLoopInst(loop);
  loop->GetResultsUses().resize(cur_node.output_size());
  loop->GetResultsTypes().resize(cur_node.output_size());

  InsertIDToInstMap(cur_node, loop);
  return Status::SUCCESS;
}

} // end namespace halo
