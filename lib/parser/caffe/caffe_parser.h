//===- caffe_parser.h --------------------------------------------*- C++-*-===//
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

#ifndef HALO_LIB_PARSER_CAFFE_CAFFEPARSER_H_
#define HALO_LIB_PARSER_CAFFE_CAFFEPARSER_H_

#include <functional>
#include <unordered_map>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/parser/parser.h"

namespace caffe {
class NetParameter;
class LayerParameter;
class BlobShape;
} // namespace caffe

namespace google {
namespace protobuf {
class Message;
} // namespace protobuf
} // namespace google

namespace halo {

class CAFFEAttrs {
 public:
  CAFFEAttrs(const caffe::BlobShape& shape);
  std::vector<int> GetShape() { return shape_; }
  CAFFEAttrs(const CAFFEAttrs&) = delete;
  CAFFEAttrs& operator=(const CAFFEAttrs&) = delete;

 private:
  std::vector<int> shape_;
};

/// Parser for CAFFE
class CAFFEParser : public Parser {
 public:
  explicit CAFFEParser(){};
  Status Parse(Function* function, const std::vector<std::string>& file_list,
               const armory::Opts& opts) override;
  Status Parse(BasicBlock* bb, const caffe::NetParameter& layer_param,
               const caffe::NetParameter& layer_param_weight,
               const armory::Opts& opts);
  ~CAFFEParser();

  static void WriteCSVReport(const caffe::LayerParameter& layer_param,
                             std::ostream& os);

  CAFFEParser(const CAFFEParser&) = delete;
  CAFFEParser& operator=(const CAFFEParser&) = delete;

 private:
  void RegisterOp();
  Status ReadProtoFromTextFile(const std::string& file_name,
                               google::protobuf::Message* net_param);
  Status ReadWeightFromCaffeModelFile(const std::string& file_name,
                                      google::protobuf::Message* net_param);
  Status ConvertToHaloIR(const caffe::NetParameter& layer_param,
                         const caffe::NetParameter& layer_param_weight);
  Status ConvertPlaceholderNode(const caffe::NetParameter& net_param);
  Status ConvertDummyNode(const caffe::LayerParameter& layer_param,
                          const caffe::LayerParameter& layer_param_weight);
  Status ConvertOneNode(std::unique_ptr<IRBuilder>& ir_builder,
                        const caffe::LayerParameter& layer_param,
                        const caffe::LayerParameter& layer_param_weight);
  std::vector<Def> GetInputOperands(
      const caffe::LayerParameter& layer_param,
      const caffe::LayerParameter& layer_param_weight);
  const std::vector<std::string> CreateExtraOperandsOrReturn(
      const caffe::LayerParameter& layer_param,
      const caffe::LayerParameter& layer_param_weight);
  void InsertIDToInstMap(const caffe::LayerParameter& node_def, IRObject* inst);
/// create node function auto generatered by tablegen
#include "caffe_convert.h.inc"

 private:
  std::unique_ptr<IRBuilder> ir_builder_;
  std::unique_ptr<ArgumentBuilder> arg_builder_;
  std::unique_ptr<ConstantBuilder> c_builder_;
  armory::Opts opts_;
  std::unordered_map<std::string, IRObject*> inst_name_to_ptr_;
  std::unordered_map<std::string, std::string> input_to_layer_;
  using CallBack = std::function<Status(std::unique_ptr<IRBuilder>&,
                                        const caffe::LayerParameter&,
                                        const caffe::LayerParameter&)>;
  std::unordered_map<std::string, CallBack> func_lists_;
};

} // namespace halo

#endif // HALO_LIB_PARSER_ONNX_ONNXPARSER_H_
