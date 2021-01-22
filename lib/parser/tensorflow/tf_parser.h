//===- tf_parser.h ----------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_PARSER_TENSORFLOW_TFPARSER_H_
#define HALO_LIB_PARSER_TENSORFLOW_TFPARSER_H_

#include <functional>
#include <unordered_map>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/parser/parser.h"

namespace tensorflow {
class GraphDef;
class NodeDef;
class TensorProto;
class TensorShapeProto;
class AttrValue;
class AttrValue_ListValue;
} // end namespace tensorflow

namespace halo {

/// Record string based attributes
class TFAttrs {
 public:
  template <typename T>
  using AttrMap = std::unordered_map<std::string, T>;

  TFAttrs() = delete;
  explicit TFAttrs(const tensorflow::NodeDef& node_def);

  template <typename T>
  bool Process(const std::string& key, T*);

  TFAttrs(const TFAttrs&) = delete;
  TFAttrs& operator=(const TFAttrs&) = delete;

 private:
  AttrMap<const tensorflow::AttrValue&> attr_map_;
};

/// This class represents a parser for tensorflow.
class TFParser : public Parser {
 public:
  explicit TFParser(const std::string& variant) : variant_(variant) {}
  Status Parse(Function* function, const std::vector<std::string>& file_list,
               const armory::Opts& opts) override;
  virtual Status Parse(BasicBlock* bb, const tensorflow::GraphDef& graph_def,
                       const armory::Opts& opts);
  ~TFParser();

  static std::vector<int64_t> ProcessShape(
      const tensorflow::TensorShapeProto& tensor_shape_proto);

  template <typename T>
  static Tensor<T> ProcessTensor(const tensorflow::TensorProto& tensor_proto);

  TFParser(const TFParser&) = delete;
  TFParser& operator=(const TFParser&) = delete;

 private:
  void Init(BasicBlock* bb, Function* function, const armory::Opts& opts);
  void RegisterOp();
  Status ConvertToHaloIR(const tensorflow::GraphDef& graph_def);
  Status ConvertOneNode(const tensorflow::NodeDef& cur_node, size_t index);
  template <typename T>
  Constant* CreateConstant(TFAttrs* attrs, DataType data_type,
                           const tensorflow::NodeDef& node_def);

/// create node function auto generatered by tablegen
#include "tf_convert.h.inc"

  Status ConvertConstNode(const tensorflow::NodeDef& node_def);
  Status ConvertPlaceholderNode(const tensorflow::NodeDef& node_def);
  Status ConvertDummyNode(const tensorflow::NodeDef& node_def);

  std::vector<Def> GetInputOperands(const tensorflow::NodeDef& node_def);

  inline void InsertIDToInstMap(const tensorflow::NodeDef& node_def,
                                IRObject* inst);

  inline std::string SkipFirstChar(std::string input_node_name) {
    auto pos = input_node_name.find_last_of('^', 1);
    return pos == 0 ? input_node_name.substr(pos + 1) : input_node_name;
  }

  static void WriteCSVReport(const tensorflow::NodeDef& cur_node,
                             const size_t index, std::ostream& os);

 protected:
  const std::string& variant_;
  armory::Opts opts_;

 private:
  std::unique_ptr<IRBuilder> ir_builder_;
  std::unique_ptr<ArgumentBuilder> arg_builder_;
  std::unique_ptr<ConstantBuilder> c_builder_;
  std::unordered_map<std::string, IRObject*> inst_name_to_ptr_;
  std::unordered_map<std::string,
                     std::function<Status(const tensorflow::NodeDef&)>>
      func_lists_;
};

/// Convert pb tp ipu graphdef
class IPUParser : public TFParser {
 public:
  explicit IPUParser(const std::string& variant) : TFParser(variant) {}
  ~IPUParser(){};

  Status Parse(BasicBlock* bb, const tensorflow::GraphDef& graph_def,
               const armory::Opts& opts) override;

  IPUParser(const IPUParser&) = delete;
  IPUParser& operator=(const IPUParser&) = delete;

 private:
  Status SetAttributes(tensorflow::NodeDef* cur_node);
  Status ManualSharding(tensorflow::NodeDef* cur_node,
                        const armory::Opts& opts);
  Status ConvertToIpuGraphDef(const tensorflow::GraphDef& graph_def,
                              const armory::Opts& opts);
};

} // namespace halo

#endif // HALO_LIB_PARSER_TENSORFLOW_TFPARSER_H_
