//===- tf_parser.cc -------------------------------------------------------===//
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

#include "tf_parser.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <string_view>

#include "function.pb.h"
#include "graph.pb.h"
#include "halo/lib/framework/common.h"
#include "halo/lib/framework/type.h"
#include "halo/lib/ir/ir_builder.h"
#include "xla.pb.h"

namespace halo {

TFParser::~TFParser() {}

Status TFParser::Parse(Function* function,
                       const std::vector<std::string>& file_list,
                       const armory::Opts& opts) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  tensorflow::GraphDef graph_def;
  HLCHECK(!file_list.empty());

  std::ifstream ifs(file_list.front());
  HLCHECK(!ifs.fail());

  if (!graph_def.ParseFromIstream(&ifs)) {
    google::protobuf::io::IstreamInputStream input_stream(&ifs);
    if (!google::protobuf::TextFormat::Parse(&input_stream, &graph_def)) {
      LOG(ERROR) << "Encountered error(s) when parsing " << file_list.front();
      return Status::ASSERTION;
    }
  }

  BasicBlockBuilder bb_builder(function);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");
  return Parse(bb, graph_def, opts);
}

Status TFParser::Parse(Function* function,
                       const std::vector<const char*>& buffers,
                       const std::vector<size_t>& buffer_sizes) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  tensorflow::GraphDef graph_def;

  google::protobuf::io::ArrayInputStream ais(buffers[0], buffer_sizes[0]);

  if (!graph_def.ParseFromZeroCopyStream(&ais)) {
    if (!google::protobuf::TextFormat::Parse(&ais, &graph_def)) {
      LOG(ERROR) << "Encountered error(s) when parsing memory graph";
      return Status::ASSERTION;
    }
  }

  BasicBlockBuilder bb_builder(function);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");
  return Parse(bb, graph_def, opts_);
}

Status TFParser::Parse(Function* function,
                       const std::vector<const void*>& model_defs) {
  if (model_defs.empty() || model_defs[0] == nullptr) {
    return Status::FILE_NOT_EXIST;
  }
  BasicBlockBuilder bb_builder(function);
  BasicBlock* bb = bb_builder.CreateBasicBlock("bb0");

  const tensorflow::GraphDef* graph_def =
      reinterpret_cast<const tensorflow::GraphDef*>(model_defs[0]);

  return Parse(bb, *graph_def, opts_);
}

Status TFParser::Parse(BasicBlock* bb, const tensorflow::GraphDef& graph_def,
                       const armory::Opts& opts) {
  Init(bb, bb->GetParent(), opts);
  return ConvertToHaloIR(graph_def);
}

/// register op and create ir builder
void TFParser::Init(BasicBlock* bb, Function* function,
                    const armory::Opts& opts) {
  RegisterOp();
  ir_builder_ = std::make_unique<IRBuilder>(bb);
  arg_builder_ = std::make_unique<ArgumentBuilder>(function);
  c_builder_ = std::make_unique<ConstantBuilder>(function);
  opts_ = opts;
}

Status TFParser::ConvertToHaloIR(const tensorflow::GraphDef& graph_def) {
  int i = 0;
  Status s = Status::SUCCESS;
  std::vector<const tensorflow::NodeDef*> ret_vals;
  std::unordered_set<const tensorflow::NodeDef*> visited;
  std::vector<const tensorflow::NodeDef*> all_nodes;
  all_nodes.reserve(graph_def.node_size());
  for (const auto& node : graph_def.node()) {
    all_nodes.push_back(&node);
  }
  for (bool changed = true; !all_nodes.empty() && changed;) {
    std::vector<const tensorflow::NodeDef*> skipped;
    changed = false;
    for (const auto& node : all_nodes) {
      const auto& cur_node = *node;
      if (visited.count(node) > 0) {
        continue;
      }
      VLOG(4) << "==========layer[" << i << "]==========";
      if (cur_node.op() == "_Retval") {
        ret_vals.push_back(&cur_node);
        ++i;
        continue;
      }

      if (cur_node.input_size() >
          static_cast<signed>(GetInputOperands(cur_node).size())) {
        skipped.push_back(&cur_node);
        continue;
      }
      s = ConvertOneNode(ir_builder_.get(), cur_node, i++);
      visited.insert(&cur_node);
      changed = true;
      if (s != Status::SUCCESS) {
        return s;
      }
    }
    all_nodes.swap(skipped);
  }
  VLOG(4) << "Total convert node num: " << graph_def.node_size();
  HLCHECK(graph_def.node_size() == i);
  ConvertReturnNodes(ir_builder_.get(), ret_vals);

  // Add control dependents.
  for (const auto& kv : control_edges_) {
    auto it = inst_name_to_ptr_.find(kv.first);
    HLCHECK(it != inst_name_to_ptr_.end());
    for (const auto& dep_name : kv.second) {
      auto it_d = inst_name_to_ptr_.find(dep_name);
      HLCHECK(it_d != inst_name_to_ptr_.end());
      it->second->AddDependant(it_d->second);
    }
  }
  return Status::SUCCESS;
}

Status TFParser::ConvertReturnNodes(
    IRBuilder* ir_builder,
    const std::vector<const tensorflow::NodeDef*>& ret_vals) {
  if (ret_vals.empty()) {
    return Status::SUCCESS;
  }
  std::vector<Def> outputs;
  for (auto& op : ret_vals) {
    auto inputs = GetInputOperands(*op);
    for (auto v : inputs) {
      outputs.push_back(v);
    }
  }
  ir_builder->CreateReturn("output", outputs);
  return Status::SUCCESS;
}

Status TFParser::ConvertOneNode(IRBuilder* ir_builder,
                                const tensorflow::NodeDef& cur_node,
                                size_t index) {
  Status s = Status::SUCCESS;
  auto fp = func_lists_.find(cur_node.op());
  if (fp != func_lists_.end()) {
    s = (fp->second)(ir_builder, cur_node);
    if (s != Status::SUCCESS) {
      return s;
    }
  } else {
    if (opts_.print_diagnostic_report) {
      TFParser::WriteCSVReport(cur_node, index, std::cout);
      ConvertDummyNode(ir_builder, cur_node);
    } else {
      LOG(ERROR)
          << "Convert function not found, Please check if it is supported: "
          << "Name: "
          << "[" << cur_node.name() << "], Op: [" << cur_node.op()
          << "], Index: "
          << "[" << index << "]";
      return Status::ASSERTION;
    }
  }
  return Status::SUCCESS;
}

void TFParser::WriteCSVReport(const tensorflow::NodeDef& cur_node,
                              const size_t index, std::ostream& os) {
  os << "Name: [" << cur_node.name() << "], Op: [" << cur_node.op()
     << "], Index: [" << index << "]\n";
}

void TFParser::RegisterOp() {
  func_lists_.emplace("Const",
                      std::bind(&TFParser::ConvertConstNode, this,
                                std::placeholders::_1, std::placeholders::_2));
  func_lists_.emplace("Placeholder",
                      std::bind(&TFParser::ConvertPlaceholderNode, this,
                                std::placeholders::_1, std::placeholders::_2));
  func_lists_.emplace("_Arg",
                      std::bind(&TFParser::ConvertPlaceholderNode, this,
                                std::placeholders::_1, std::placeholders::_2));
#include "tf_regist_op.h.inc"
}

static halo::DataType ProcessDataType(const tensorflow::DataType& data_type) {
  switch (data_type) {
    case tensorflow::DT_HALF:
      return DataType::FLOAT16;
    case tensorflow::DT_FLOAT:
      return DataType::FLOAT32;
    case tensorflow::DT_INT64:
      return DataType::INT64;
    case tensorflow::DT_UINT64:
      return DataType::UINT64;
    case tensorflow::DT_INT32:
      return DataType::INT32;
    case tensorflow::DT_UINT8:
    case tensorflow::DT_QUINT8:
      return DataType::UINT8;
    case tensorflow::DT_INT16:
      return DataType::INT16;
    case tensorflow::DT_QINT8:
    case tensorflow::DT_INT8:
      return DataType::INT8;
    case tensorflow::DT_STRING:
      return DataType::STRING;
    case tensorflow::DT_BOOL:
      return DataType::BOOL;
    default:
      LOG(ERROR) << "Unsupported DataType:" << data_type;
      return DataType::INVALID;
  }
}

TFAttrs::TFAttrs(const tensorflow::NodeDef& node_def) {
  for (const auto& it : node_def.attr()) {
    attr_map_.emplace(it.first, it.second);
  }
}

/// Get Attribute for enum
template <>
bool TFAttrs::Process<CodeType>(const std::string& key, CodeType* code_type) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kS);
  static const std::unordered_map<std::string, CodeType> enum_map{
      {"CORNER", CodeType::CORNER},
      {"CENTER_SIZE", CodeType::CENTER_SIZE},
      {"CORNER_SIZE", CodeType::CORNER_SIZE},
  };

  *code_type = enum_map.count(attr_map_.at(key).s())
                   ? enum_map.at(attr_map_.at(key).s())
                   : CodeType::INVALID;
  return true;
}

template <>
bool TFAttrs::Process<DataFormat>(const std::string& key,
                                  DataFormat* data_format) {
  if (!attr_map_.count(key)) {
    return false;
  }
  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kS);

  static const std::unordered_map<std::string, DataFormat> enum_map{
      {"NHWC", DataFormat::NHWC},   {"NCHW", DataFormat::NCHW},
      {"NDHWC", DataFormat::NDHWC}, {"HWCN", DataFormat::HWCN},
      {"DNCHW", DataFormat::DNCHW},
  };

  *data_format = enum_map.count(attr_map_.at(key).s())
                     ? enum_map.at(attr_map_.at(key).s())
                     : DataFormat::INVALID;
  return true;
}

template <>
bool TFAttrs::Process<PadMode>(const std::string& key, PadMode* pad_mode) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kS);
  static const std::unordered_map<std::string, PadMode> enum_map{
      {"CONSTANT", PadMode::CONSTANT},
      {"REFLECT", PadMode::REFLECT},
      {"SYMMETRIC", PadMode::SYMMETRIC},
  };

  *pad_mode = enum_map.count(attr_map_.at(key).s())
                  ? enum_map.at(attr_map_.at(key).s())
                  : PadMode::INVALID;
  return true;
}

template <>
bool TFAttrs::Process<Padding>(const std::string& key, Padding* padding) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kS);
  static const std::unordered_map<std::string, Padding> enum_map{
      {"VALID", Padding::VALID},
      {"SAME", Padding::SAME},
      {"SAME_LOWER", Padding::SAME_LOWER},
      {"EXPLICIT", Padding::EXPLICIT},
  };

  *padding = enum_map.count(attr_map_.at(key).s())
                 ? enum_map.at(attr_map_.at(key).s())
                 : Padding::INVALID;
  return true;
}

/// Process Tensor in list scope
template <>
Tensor<std::string> TFParser::ProcessTensor<std::string>(
    const tensorflow::TensorProto& tensor_proto) {
  const DataType& data_type = ProcessDataType(tensor_proto.dtype());
  std::vector<int64_t> shape;
  if (tensor_proto.has_tensor_shape()) {
    shape = ProcessShape(tensor_proto.tensor_shape());
  }

  std::vector<std::string> v;
  // string val e.g.  s: "NHWC"
  for (const auto& it : tensor_proto.string_val()) {
    v.emplace_back(it);
  }

  // has tensor content, e.g. tensor_content: "\000\000\000\001"
  if (!tensor_proto.tensor_content().empty()) {
    HLCHECK(v.empty());
    v.emplace_back(tensor_proto.tensor_content());
    // string val and tensor content will not coexist, we use
    // `need_decode` flag to distinguish
    return Tensor<std::string>(data_type, shape, v, true);
  }
  return Tensor<std::string>(data_type, shape, v);
}

template <>
Tensor<bool> TFParser::ProcessTensor<bool>(
    const tensorflow::TensorProto& tensor_proto) {
  const DataType& data_type = ProcessDataType(tensor_proto.dtype());
  std::vector<int64_t> shape;
  if (tensor_proto.has_tensor_shape()) {
    shape = ProcessShape(tensor_proto.tensor_shape());
  }

  std::vector<bool> v;
  for (const auto& it : tensor_proto.bool_val()) {
    v.emplace_back(it);
  }
  return Tensor<bool>(data_type, shape, v);
}

template <>
Tensor<int> TFParser::ProcessTensor<int>(
    const tensorflow::TensorProto& tensor_proto) {
  const DataType& data_type = ProcessDataType(tensor_proto.dtype());
  std::vector<int64_t> shape;
  if (tensor_proto.has_tensor_shape()) {
    shape = ProcessShape(tensor_proto.tensor_shape());
  }

  std::vector<int> v;
  v.reserve(std::accumulate(shape.begin(), shape.end(), 1,
                            std::multiplies<int64_t>()));
  if (data_type == DataType::FLOAT16) {
    for (const auto& it : tensor_proto.half_val()) {
      v.emplace_back(it);
    }
  } else {
    for (const auto& it : tensor_proto.int_val()) {
      v.emplace_back(it);
    }
  }
  return Tensor<int>(data_type, shape, v);
}

template <>
Tensor<int64_t> TFParser::ProcessTensor<int64_t>(
    const tensorflow::TensorProto& tensor_proto) {
  const DataType& data_type = ProcessDataType(tensor_proto.dtype());
  std::vector<int64_t> shape;
  if (tensor_proto.has_tensor_shape()) {
    shape = ProcessShape(tensor_proto.tensor_shape());
  }

  std::vector<int64_t> v;
  for (const auto& it : tensor_proto.int64_val()) {
    v.emplace_back(it);
  }
  return Tensor<int64_t>(data_type, shape, v);
}

template <>
Tensor<float> TFParser::ProcessTensor<float>(
    const tensorflow::TensorProto& tensor_proto) {
  const DataType& data_type = ProcessDataType(tensor_proto.dtype());
  std::vector<int64_t> shape;
  if (tensor_proto.has_tensor_shape()) {
    shape = ProcessShape(tensor_proto.tensor_shape());
  }

  std::vector<float> v;
  for (const auto& it : tensor_proto.float_val()) {
    v.emplace_back(it);
  }
  return Tensor<float>(data_type, shape, v);
}

template <>
Tensor<double> TFParser::ProcessTensor<double>(
    const tensorflow::TensorProto& tensor_proto) {
  const DataType& data_type = ProcessDataType(tensor_proto.dtype());
  std::vector<int64_t> shape;
  if (tensor_proto.has_tensor_shape()) {
    shape = ProcessShape(tensor_proto.tensor_shape());
  }

  std::vector<double> v;
  for (const auto& it : tensor_proto.double_val()) {
    v.emplace_back(it);
  }
  return Tensor<double>(data_type, shape, v);
}

template <>
Tensor<char> TFParser::ProcessTensor<char>(
    const tensorflow::TensorProto& tensor_proto) {
  const DataType& data_type = ProcessDataType(tensor_proto.dtype());
  std::vector<int64_t> shape;
  if (tensor_proto.has_tensor_shape()) {
    shape = ProcessShape(tensor_proto.tensor_shape());
  }

  std::vector<char> v;
  for (const auto& it : tensor_proto.int_val()) {
    v.emplace_back(it);
  }
  return Tensor<char>(data_type, shape, v);
}

std::vector<int64_t> TFParser::ProcessShape(
    const tensorflow::TensorShapeProto& tensor_shape_proto) {
  std::vector<int64_t> shape;
  for (const auto& it : tensor_shape_proto.dim()) {
    if (0 == it.size()) {
      // dim_size = None scenario
      break;
    } else {
      shape.emplace_back(it.size());
    }
  }
  return shape;
}

// Process Attribute
template <>
bool TFAttrs::Process<std::string>(const std::string& key, std::string* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kS);
  *value = attr_map_.at(key).s();
  return true;
}

template <>
bool TFAttrs::Process<int64_t>(const std::string& key, int64_t* value) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kI);
  *value = attr_map_.at(key).i();
  return true;
}

template <>
bool TFAttrs::Process<int>(const std::string& key, int* value) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kI);
  *value = attr_map_.at(key).i();
  return true;
}

template <>
bool TFAttrs::Process<float>(const std::string& key, float* value) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kF);
  *value = attr_map_.at(key).f();
  return true;
}

template <>
bool TFAttrs::Process<bool>(const std::string& key, bool* value) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kB);
  *value = attr_map_.at(key).b();
  return true;
}

template <>
bool TFAttrs::Process<DataType>(const std::string& key, DataType* data_type) {
  if (!attr_map_.count(key)) {
    return false;
  }

  HLCHECK(attr_map_.at(key).value_case() == tensorflow::AttrValue::kType);
  *data_type = ProcessDataType(attr_map_.at(key).type());
  return true;
}

template <>
bool TFAttrs::Process<std::vector<std::string>>(
    const std::string& key, std::vector<std::string>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
  (*value).reserve(attr_value.list().s_size());
  for (const auto& it : attr_value.list().s()) {
    (*value).emplace_back(it);
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<int64_t>>(const std::string& key,
                                            std::vector<int64_t>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  if (attr_value.value_case() == tensorflow::AttrValue::kList) {
    if (auto num_of_shape = attr_value.list().shape_size()) {
      HLCHECK(num_of_shape == 1);
      *value = TFParser::ProcessShape(attr_value.list().shape(0));
    } else {
      (*value).reserve(attr_value.list().i_size());
      for (const auto& it : attr_value.list().i()) {
        (*value).push_back(it);
      }
    }
  } else if (attr_value.value_case() == tensorflow::AttrValue::kShape) {
    *value = TFParser::ProcessShape(attr_map_.at(key).shape());
  } else {
    HLCHECK(0 && "Encountered attribute type error");
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<int>>(const std::string& key,
                                        std::vector<int>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
  (*value).reserve(attr_value.list().i_size());
  for (const auto& it : attr_value.list().i()) {
    (*value).push_back(it);
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<float>>(const std::string& key,
                                          std::vector<float>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
  (*value).reserve(attr_value.list().f_size());
  for (const auto& it : attr_value.list().f()) {
    (*value).push_back(it);
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<bool>>(const std::string& key,
                                         std::vector<bool>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
  (*value).reserve(attr_value.list().b_size());
  for (const auto& it : attr_value.list().b()) {
    (*value).push_back(it);
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<DataType>>(
    const std::string& key, std::vector<DataType>* data_types) {
  if (!attr_map_.count(key)) {
    return false;
  }
  data_types->clear();
  const auto& attr_value = attr_map_.at(key);
  HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
  (*data_types).reserve(attr_value.list().type_size());
  for (const auto& it : attr_value.list().type()) {
    (*data_types)
        .emplace_back(ProcessDataType(static_cast<tensorflow::DataType>(it)));
  }
  return true;
}

/// Get Attr for Shape
template <>
bool TFAttrs::Process<std::vector<std::vector<int64_t>>>(
    const std::string& key, std::vector<std::vector<int64_t>>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  if (attr_value.value_case() == tensorflow::AttrValue::kShape) {
    *value = {TFParser::ProcessShape(attr_value.shape())};
  } else {
    HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
    (*value).reserve(attr_value.list().shape_size());
    for (const auto& it : attr_value.list().shape()) {
      (*value).emplace_back(TFParser::ProcessShape(it));
    }
  }
  return true;
}

/// Get Attr for Tensor
template <>
bool TFAttrs::Process<std::vector<Tensor<std::string>>>(
    const std::string& key, std::vector<Tensor<std::string>>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  if (attr_value.value_case() == tensorflow::AttrValue::kTensor) {
    *value = {TFParser::ProcessTensor<std::string>(attr_value.tensor())};
  } else {
    HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
    (*value).reserve(attr_value.list().tensor_size());
    for (const auto& it : attr_value.list().tensor()) {
      (*value).emplace_back(TFParser::ProcessTensor<std::string>(it));
    }
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<Tensor<bool>>>(
    const std::string& key, std::vector<Tensor<bool>>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  if (attr_value.value_case() == tensorflow::AttrValue::kTensor) {
    *value = {TFParser::ProcessTensor<bool>(attr_value.tensor())};
  } else {
    HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
    (*value).reserve(attr_value.list().tensor_size());
    for (const auto& it : attr_value.list().tensor()) {
      (*value).emplace_back(TFParser::ProcessTensor<bool>(it));
    }
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<Tensor<int>>>(
    const std::string& key, std::vector<Tensor<int>>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  if (attr_value.value_case() == tensorflow::AttrValue::kTensor) {
    (*value).reserve(1);
    (*value).emplace_back(TFParser::ProcessTensor<int>(attr_value.tensor()));
  } else {
    HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
    (*value).reserve(attr_value.list().tensor_size());
    for (const auto& it : attr_value.list().tensor()) {
      (*value).emplace_back(TFParser::ProcessTensor<int>(it));
    }
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<Tensor<int64_t>>>(
    const std::string& key, std::vector<Tensor<int64_t>>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  if (attr_value.value_case() == tensorflow::AttrValue::kTensor) {
    *value = {TFParser::ProcessTensor<int64_t>(attr_value.tensor())};
  } else {
    HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
    (*value).reserve(attr_value.list().tensor_size());
    for (const auto& it : attr_value.list().tensor()) {
      (*value).emplace_back(TFParser::ProcessTensor<int64_t>(it));
    }
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<Tensor<float>>>(
    const std::string& key, std::vector<Tensor<float>>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  if (attr_value.value_case() == tensorflow::AttrValue::kTensor) {
    *value = {TFParser::ProcessTensor<float>(attr_value.tensor())};
  } else {
    HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
    (*value).reserve(attr_value.list().tensor_size());
    for (const auto& it : attr_value.list().tensor()) {
      (*value).emplace_back(TFParser::ProcessTensor<float>(it));
    }
  }
  return true;
}

template <>
bool TFAttrs::Process<std::vector<Tensor<double>>>(
    const std::string& key, std::vector<Tensor<double>>* value) {
  if (!attr_map_.count(key)) {
    return false;
  }
  value->clear();
  const auto& attr_value = attr_map_.at(key);
  if (attr_value.value_case() == tensorflow::AttrValue::kTensor) {
    *value = {TFParser::ProcessTensor<double>(attr_value.tensor())};
  } else {
    HLCHECK(attr_value.value_case() == tensorflow::AttrValue::kList);
    (*value).reserve(attr_value.list().tensor_size());
    for (const auto& it : attr_value.list().tensor()) {
      (*value).emplace_back(TFParser::ProcessTensor<double>(it));
    }
  }
  return true;
}

std::vector<Def> TFParser::GetInputOperands(
    const tensorflow::NodeDef& node_def) {
  std::vector<Def> operands;
  size_t operand_num = node_def.input_size();
  for (size_t i = 0; i < operand_num; ++i) {
    std::string input_node_name = SkipFirstChar(node_def.input(i));
    size_t pos = input_node_name.find_last_of(':');
    std::string idx_str;
    std::unordered_map<std::string, IRObject*>::iterator it;
    if (pos != std::string::npos) {
      it = inst_name_to_ptr_.find(input_node_name.substr(0, pos));
      idx_str = input_node_name.substr(pos + 1);
      input_node_name = input_node_name.substr(0, pos);
    } else {
      it = inst_name_to_ptr_.find(input_node_name);
    }

    if (it != inst_name_to_ptr_.end()) {
      auto inst = it->second;
      int idx = (idx_str.empty()) ? 0 : std::stoi(idx_str);
      HLCHECK(0 <= idx && idx <= 1024);
      operands.emplace_back(Def{inst, idx});
    } else {
      // those errors will be record in diagnostic report file
      // LOG(ERROR) << node_def.name() << " Node's " << i
      //<< "th operand:" << node_def.input(i) << " not found";
      pos = input_node_name.find_first_of(':');
      it = inst_name_to_ptr_.find(input_node_name.substr(0, pos));
      if (it != inst_name_to_ptr_.end()) {
        auto inst = it->second;
        int idx = (idx_str.empty()) ? 0 : std::stoi(idx_str);
        HLCHECK(0 <= idx && idx <= 1024);
        operands.emplace_back(Def{inst, idx});
      }
    }
  }
  return operands;
}

void TFParser::InsertIDToInstMap(const tensorflow::NodeDef& node_def,
                                 IRObject* inst) {
  inst_name_to_ptr_.emplace(node_def.name(), inst);
}

// Define convert function
Status TFParser::ConvertPlaceholderNode(IRBuilder* ir_builder,
                                        const tensorflow::NodeDef& node_def) {
  TFAttrs attrs(node_def);
  DataType data_type = DataType::INVALID;
  // TODO(unknown): This is a temp solution to handle dynamic shape of BERT.
  // pay attention to this when need to support controlflow parser
  if (node_def.name() == "sequence_lengths") {
    Constant* inst = c_builder_->CreateConstant<int>(
        node_def.name(), Type{DataType::INT32, {1}}, {15});
    inst_name_to_ptr_.emplace(node_def.name(), inst);
  } else if (attrs.Process<DataType>(node_def.op() == "_Arg" ? "T" : "dtype",
                                     &data_type)) {
    // Add default shape if no shape info in placehold
    std::vector<int64_t> shape = {};
    attrs.Process<std::vector<int64_t>>("shape", &shape);
    if (shape.empty()) {
      attrs.Process<std::vector<int64_t>>("_output_shapes", &shape);
    }
    Argument* arg = nullptr;
    arg = arg_builder_->CreateArgument(node_def.name(), Type(data_type, shape));
    inst_name_to_ptr_.emplace(node_def.name(), arg);
    if (node_def.op() == "_Arg" && node_def.attr().contains("value")) {
      // Handle special constant arg that has "value" attr.
      ConvertConstNode(ir_builder, node_def);
    }
  }
  return Status::SUCCESS;
}

template <typename T>
Constant* TFParser::CreateConstant(TFAttrs* attrs, DataType data_type,
                                   const tensorflow::NodeDef& node_def) {
  std::vector<Tensor<std::string>> tensors;
  if (attrs->Process<std::vector<Tensor<std::string>>>("value", &tensors)) {
    HLCHECK(1 == tensors.size());
    if (tensors.back().GetNeedDecode()) {
      auto v = Tensor<T>::DecodeTensorContent(tensors.back().GetData().back());
      auto inst = c_builder_->CreateConstant(
          node_def.name(), Type(data_type, tensors.back().GetShape()), v);
      inst_name_to_ptr_[node_def.name()] =
          inst; // override existing name with constant.
      return inst;
    }
  }
  return nullptr;
}

Status TFParser::ConvertConstNode(IRBuilder* ir_builder,
                                  const tensorflow::NodeDef& node_def) {
  TFAttrs attrs(node_def);
  DataType data_type = DataType::INVALID;
  // Check for control deps
  for (int i = 0, e = node_def.input_size(); i < e; ++i) {
    const auto& dep = node_def.input(i);
    HLCHECK(!dep.empty() && dep.front() == '^');
    control_edges_[dep.substr(1)].push_back(node_def.name());
  }

  if (attrs.Process<DataType>("dtype", &data_type)) {
    switch (data_type) {
      case DataType::BOOL: {
        CreateConstant<int8_t>(&attrs, data_type, node_def);
        break;
      }
      case DataType::UINT8: {
        CreateConstant<uint8_t>(&attrs, data_type, node_def);
        break;
      }

      case DataType::INT8: {
        // definitely need decoded from tensor content
        CreateConstant<int8_t>(&attrs, data_type, node_def);
        break;
      }
      case DataType::INT32: {
        // check need decoded from tensor content
        std::vector<Tensor<std::string>> tensors;
        IRObject* inst = nullptr;
        if (CreateConstant<int>(&attrs, data_type, node_def) == nullptr) {
          std::vector<Tensor<int>> native_tensors;
          if (attrs.Process<std::vector<Tensor<int>>>("value",
                                                      &native_tensors)) {
            HLCHECK(1 == native_tensors.size());
            inst = c_builder_->CreateConstant(
                node_def.name(),
                Type(data_type, native_tensors.back().GetShape()),
                native_tensors.back().GetData());
          }
        }
        inst_name_to_ptr_.emplace(node_def.name(), inst);
        break;
      }
      case DataType::INT64: {
        // check need decoded from tensor content
        std::vector<Tensor<std::string>> tensors;
        IRObject* inst = nullptr;
        if (CreateConstant<int64_t>(&attrs, data_type, node_def) == nullptr) {
          std::vector<Tensor<int64_t>> native_tensors;
          if (attrs.Process<std::vector<Tensor<int64_t>>>("value",
                                                          &native_tensors)) {
            HLCHECK(1 == native_tensors.size());
            inst = c_builder_->CreateConstant(
                node_def.name(),
                Type(data_type, native_tensors.back().GetShape()),
                native_tensors.back().GetData());
          }
        }
        inst_name_to_ptr_.emplace(node_def.name(), inst);
        break;
      }
      case DataType::FLOAT16: {
        IRObject* inst = nullptr;
        std::vector<Tensor<std::string>> tensors;
        std::vector<uint16_t> data;
        if (attrs.Process<std::vector<Tensor<std::string>>>("value",
                                                            &tensors)) {
          HLCHECK(1 == tensors.size());
          auto& tensor = tensors.back();
          if (tensor.GetNeedDecode()) {
            data =
                Tensor<uint16_t>::DecodeTensorContent(tensor.GetData().back());
          } else {
            std::vector<Tensor<int32_t>> native_tensors;
            attrs.Process<std::vector<Tensor<int32_t>>>("value",
                                                        &native_tensors);
            HLCHECK(1 == native_tensors.size());
            const auto& orig_data = native_tensors.back().GetData();
            data.reserve(orig_data.size());
            for (auto v : orig_data) {
              data.push_back(v);
            }
          }
          Type ty(data_type, tensors.back().GetShape());
          HLCHECK(data.size() ==
                  static_cast<size_t>(ty.GetTotalNumOfElements()));
          inst = c_builder_->CreateConstant(node_def.name(), ty, data.data());
        }
        HLCHECK(inst && "Unable to handle DT_HALF");
        inst_name_to_ptr_.emplace(node_def.name(), inst);
        break;
      }
      case DataType::FLOAT32: {
        // check need decoded from tensor content
        std::vector<Tensor<std::string>> tensors;
        IRObject* inst = nullptr;
        if (attrs.Process<std::vector<Tensor<std::string>>>("value",
                                                            &tensors)) {
          HLCHECK(1 == tensors.size());
          if (tensors.back().GetNeedDecode()) {
            auto v = Tensor<float>::DecodeTensorContent(
                tensors.back().GetData().back());
            inst = c_builder_->CreateConstant(
                node_def.name(), Type(data_type, tensors.back().GetShape()), v);
          } else {
            std::vector<Tensor<float>> native_tensors;
            if (attrs.Process<std::vector<Tensor<float>>>("value",
                                                          &native_tensors)) {
              HLCHECK(1 == native_tensors.size());
              auto& data = native_tensors.back().GetData();
              Type ty{data_type, native_tensors.back().GetShape()};
              if (data.size() !=
                  static_cast<size_t>(ty.GetTotalNumOfElements())) {
                HLCHECK(data.size() == 1);
                std::vector<float> expanded_data(ty.GetTotalNumOfElements(),
                                                 data[0]);
                inst = c_builder_->CreateConstant(node_def.name(), ty,
                                                  expanded_data);
              } else {
                inst = c_builder_->CreateConstant(node_def.name(), ty, data);
              }
            }
          }
          inst_name_to_ptr_.emplace(node_def.name(), inst);
        }
        break;
      }
      case DataType::STRING: {
        // check need decoded from tensor content
        std::vector<Tensor<std::string>> tensors;
        IRObject* inst = nullptr;
        if (CreateConstant<std::string>(&attrs, data_type, node_def) ==
            nullptr) {
          std::vector<Tensor<std::string>> native_tensors;
          if (attrs.Process<std::vector<Tensor<std::string>>>(
                  "value", &native_tensors)) {
            HLCHECK(1 == native_tensors.size());
            inst = c_builder_->CreateConstant(
                node_def.name(),
                Type(data_type, native_tensors.back().GetShape()),
                native_tensors.back().GetData());
          }
        }
        inst_name_to_ptr_.emplace(node_def.name(), inst);
        break;
      }
      default:
        std::cerr << node_def.name();
        HLCHECK(0 && "Unsupported data type");
    }
  }
  return Status::SUCCESS;
}

Status TFParser::ConvertDummyNode(IRBuilder* ir_builder,
                                  const tensorflow::NodeDef& node_def) {
  std::vector<Def> operands = GetInputOperands(node_def);
  static const int max_num_outputs = 128;
  auto inst = ir_builder->CreateDummy(node_def.name(), operands,
                                      max_num_outputs, node_def.op());
  InsertIDToInstMap(node_def, inst);
  return Status::SUCCESS;
}

Status IPUParser::Parse(BasicBlock* bb, const tensorflow::GraphDef& graph_def,
                        const armory::Opts& opts) {
  return ConvertToIpuGraphDef(graph_def, opts);
}

Status IPUParser::SetAttributes(tensorflow::NodeDef* cur_node) {
  std::set<std::string> whitelists = {"Placeholder"};
  if (whitelists.count(cur_node->op())) {
    return Status::SUCCESS;
  }

  cur_node->set_device("/device:IPU:0");
  {
    tensorflow::AttrValue attr;
    attr.set_b("true");
    (*(cur_node->mutable_attr()))["_XlaCompile"] = attr;
  }
  {
    tensorflow::AttrValue attr;
    attr.set_s("jit_scope_ipu_0");
    (*(cur_node->mutable_attr()))["_XlaScope"] = attr;
  }
  {
    tensorflow::AttrValue attr;
    attr.set_b("false");
    (*(cur_node->mutable_attr()))["_XlaSeparateCompiledGradients"] = attr;
  }

  return Status::SUCCESS;
}

Status IPUParser::ManualSharding(tensorflow::NodeDef* cur_node,
                                 const armory::Opts& opts) {
  auto set_sharding_attr = [&](int64_t value) {
    ::xla::OpSharding op_sharding;
    op_sharding.set_type(::xla::OpSharding_Type::OpSharding_Type_MAXIMAL);
    op_sharding.add_tile_assignment_devices(value);
    tensorflow::AttrValue attr;
    std::string s;
    op_sharding.SerializeToString(&s);
    attr.set_s(s);
    (*(cur_node->mutable_attr()))["_XlaSharding"] = attr;
  };

  for (size_t i = 0, sharding_num = opts.split_names.size(); i < sharding_num;
       ++i) {
    bool founded = false;
    for (size_t j = 0; j < opts.split_names[i].size(); ++j) {
      if (auto it = cur_node->name().find(opts.split_names[i][j]);
          it != std::string::npos) {
        set_sharding_attr(i);
        founded = true;
      }
    }
    if (!founded) {
      set_sharding_attr(sharding_num);
    }
  }

  return Status::SUCCESS;
}

Status IPUParser::ConvertToIpuGraphDef(const tensorflow::GraphDef& graph_def,
                                       const armory::Opts& opts) {
  Status s = Status::SUCCESS;
  for (auto& cur_node : graph_def.node()) {
    s = SetAttributes(const_cast<tensorflow::NodeDef*>(&cur_node));
    if (s != Status::SUCCESS) {
      return s;
    }
    s = ManualSharding(const_cast<tensorflow::NodeDef*>(&cur_node), opts);
    if (s != Status::SUCCESS) {
      return s;
    }
  }

  // Serialization ipu format graphdef
  std::fstream ofs(opts.output_graphdef_filename,
                   std::ios::out | std::ios::binary);
  HLCHECK(!ofs.fail());
  if (!graph_def.SerializeToOstream(&ofs)) {
    LOG(ERROR) << "Serialization output pb file error";
    return Status::ASSERTION;
  }
#if 0
  {
    // Serialization to text file
    std::fstream ofs("./converted_model.pbtxt", std::ios::out);
    HLCHECK(!ofs.fail());
    google::protobuf::io::OstreamOutputStream* output =
        new google::protobuf::io::OstreamOutputStream(&ofs);
    google::protobuf::TextFormat::Print(graph_def, output);
    delete output;
  }
#endif
  return Status::SUCCESS;
}

// convert to halo ir def func
#include "tf_convert.cc.inc"

std::unique_ptr<Parser> CreateTFParser(const std::string& variant) {
  return std::make_unique<TFParser>(variant);
}

std::unique_ptr<Parser> CreateIPUParser(const std::string& variant) {
  return std::make_unique<IPUParser>(variant);
}

} // namespace halo
