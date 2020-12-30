//===- onnx_parser.h --------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_PARSER_ONNX_ONNXPARSER_H_
#define HALO_LIB_PARSER_ONNX_ONNXPARSER_H_

#include <functional>
#include <unordered_map>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/parser/parser.h"

namespace onnx {
class AttributeProto;
class GraphProto;
class NodeProto;
class TensorProto;
class ValueInfoProto;
} // namespace onnx

namespace halo {
/// Record string based attributes
class ONNXAttrs {
 public:
  template <typename T>
  using AttrMap = std::unordered_map<std::string, T>;

  ONNXAttrs() = delete;
  explicit ONNXAttrs(const onnx::NodeProto& node_def);

  template <typename T>
  bool Process(const std::string& key, T*);

  ONNXAttrs(const ONNXAttrs&) = delete;
  ONNXAttrs& operator=(const ONNXAttrs&) = delete;

 private:
  AttrMap<const onnx::AttributeProto&> attr_map_;
};

/// Parser for ONNX
class ONNXParser : public Parser {
 public:
  explicit ONNXParser(){};
  Status Parse(Function* function, const std::vector<std::string>& file_list,
               const armory::Opts& opts) override;
  Status Parse(BasicBlock* bb, const onnx::GraphProto& graph_def,
               const armory::Opts& opts);
  ~ONNXParser();

  template <typename T>
  static Tensor<T> ProcessTensor(const onnx::TensorProto& tensor_proto);

  ONNXParser(const ONNXParser&) = delete;
  ONNXParser& operator=(const ONNXParser&) = delete;

  static halo::DataType ProcessDataType(int data_type);
  static void WriteCSVReport(const onnx::NodeProto& cur_node, std::ostream& os);

 private:
  void RegisterOp();
  Status ConvertToHaloIR(const onnx::GraphProto& graph_def);
  Status ConvertOneNode(const onnx::NodeProto& node_def);
  IRObject* ConvertConstNode(const onnx::TensorProto& tensor_def);
  Status ConvertConstNode(const onnx::NodeProto& cur_node);
  Status ConvertDummyNode(const onnx::NodeProto& cur_node);
  Status ConvertPlaceholderNode(const onnx::ValueInfoProto& value_info_def);
  std::vector<Def> GetInputOperands(const onnx::NodeProto& node_def);
  void InsertIDToInstMap(const onnx::NodeProto& node_def, IRObject* inst);
/// create node function auto generatered by tablegen
#include "onnx_convert.h.inc"

 private:
  std::unique_ptr<IRBuilder> ir_builder_;
  std::unique_ptr<ArgumentBuilder> arg_builder_;
  std::unique_ptr<ConstantBuilder> c_builder_;
  armory::Opts opts_;
  std::unordered_map<std::string, std::pair<IRObject*, int>> inst_name_to_ptr_;
  std::unordered_map<std::string, std::function<Status(const onnx::NodeProto&)>>
      func_lists_;
};

} // namespace halo

#endif // HALO_LIB_PARSER_ONNX_ONNXPARSER_H_