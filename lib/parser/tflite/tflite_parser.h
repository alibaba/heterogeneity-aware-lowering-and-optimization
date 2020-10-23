//===- tflite_parser.h --------------------------------------------*- C++
//-*-===//
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

#ifndef HALO_LIB_PARSER_TFLITE_TFLITEPARSER_H
#define HALO_LIB_PARSER_TFLITE_TFLITEPARSER_H

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/parser/parser.h"
#include "tflite_v3_generated.h"

namespace halo {

/// Parser for TFLITE
class TFLITEParser : public Parser {
 public:
  explicit TFLITEParser(){};
  ~TFLITEParser() { delete[] proto_; }

  Status Parse(Function* function, const std::vector<std::string>& file_list,
               const armory::Opts& opts) override;
  Status Parse(BasicBlock* bb, const tflite::Model& model);

  template <typename T>
  static const Tensor<T> ProcessTensor(
      const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers,
      const tflite::Tensor& tensor);

  TFLITEParser(const TFLITEParser&) = delete;
  TFLITEParser& operator=(const TFLITEParser&) = delete;

 private:
  void RegisterOp();
  Status ConvertToHaloIR(const tflite::Model& model);
  Status ConvertOneNode(const tflite::Operator& cur_node,
                        const tflite::BuiltinOperator& cur_op_type);
  IRObject* ConvertConstNode(
      const flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>& buffers,
      const tflite::Tensor& tensor, size_t id);
  Status ConvertPlaceholderNode(const tflite::Tensor& tensor, size_t id);
  std::vector<Def> GetInputOperands(const tflite::Operator& node_def);
  void InsertIDToInstMap(const tflite::Operator& node_def, IRObject* inst);

/// create node function auto generatered by tablegen
#include "tflite_convert.h.inc"

 private:
  char* proto_ = nullptr;
  std::unique_ptr<IRBuilder> ir_builder_;
  std::unique_ptr<ArgumentBuilder> arg_builder_;
  std::unique_ptr<ConstantBuilder> c_builder_;
  std::unordered_map<int32_t, IRObject*> inst_id_to_ptr_;
  using CallBack = std::function<Status(const tflite::Operator&)>;
  std::unordered_map<std::string, CallBack> func_lists_;
};

} // namespace halo

#endif // HALO_LIB_PARSER_TFLITE_TFLITEPARSER_H
