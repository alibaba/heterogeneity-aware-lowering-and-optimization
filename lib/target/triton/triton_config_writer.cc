//===- codegen.cc ---------------------------------------------------------===//
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

#include "halo/lib/target/triton/triton_config_writer.h"

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <sstream>

#include "halo/api/halo_data.h"
#include "halo/lib/target/codegen.h"
#include "model_config.pb.h"

namespace halo {

static nvidia::inferenceserver::DataType GetTritonType(const Type& type) {
  const static std::unordered_map<DataType, nvidia::inferenceserver::DataType>
      triton_types{
          {DataType::INVALID, nvidia::inferenceserver::DataType::TYPE_INVALID},
          {DataType::BOOL, nvidia::inferenceserver::DataType::TYPE_BOOL},
          {DataType::UINT8, nvidia::inferenceserver::DataType::TYPE_UINT8},
          {DataType::UINT16, nvidia::inferenceserver::DataType::TYPE_UINT16},
          {DataType::UINT32, nvidia::inferenceserver::DataType::TYPE_UINT32},
          {DataType::UINT64, nvidia::inferenceserver::DataType::TYPE_UINT64},
          {DataType::INT8, nvidia::inferenceserver::DataType::TYPE_INT8},
          {DataType::INT16, nvidia::inferenceserver::DataType::TYPE_INT16},
          {DataType::INT32, nvidia::inferenceserver::DataType::TYPE_INT32},
          {DataType::INT64, nvidia::inferenceserver::DataType::TYPE_INT64},
          {DataType::FLOAT16, nvidia::inferenceserver::DataType::TYPE_FP16},
          {DataType::FLOAT32, nvidia::inferenceserver::DataType::TYPE_FP32},
          {DataType::STRING, nvidia::inferenceserver::DataType::TYPE_STRING},
      };
  auto it = triton_types.find(type.GetDataType());
  return it == triton_types.end() ? triton_types.begin()->second : it->second;
}

void TritonConfigWriter::PrintUseProtobuf(const Module& module,
                                          std::ostream* os) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  nvidia::inferenceserver::ModelConfig cfg;
  cfg.set_name(module.GetName());
  cfg.set_platform("custom");
  cfg.set_max_batch_size(max_batch_size_);
  auto instance = cfg.add_instance_group();
  instance->set_count(1);
  instance->set_kind(::nvidia::inferenceserver::ModelInstanceGroup_Kind::
                         ModelInstanceGroup_Kind_KIND_GPU);
  instance->add_gpus(0);
  const auto& func = module.Functions().front();
  for (auto& arg : func->Args()) {
    auto input = cfg.add_input();
    auto const& type = arg->GetResultType();
    input->set_name(arg->GetName());
    input->set_data_type(GetTritonType(type));
    if (max_batch_size_ > 0) {
      for (size_t i = 1; i < type.GetNumOfDims(); ++i) {
        input->add_dims(type.GetDimSizes()[i]);
      }
    } else {
      std::for_each(type.GetDimSizes().begin(), type.GetDimSizes().end(),
                    [&](int x) { input->add_dims(x); });
    }
  }
  auto return_inst = func->GetReturnInst();
  for (auto op : return_inst->GetOperands()) {
    auto output = cfg.add_output();
    const auto& type = op.GetType();
    output->set_name(op.GetDef()->GetName());
    output->set_data_type(GetTritonType(type));

    if (max_batch_size_ > 0) {
      for (size_t i = 1; i < type.GetNumOfDims(); ++i) {
        output->add_dims(type.GetDimSizes()[i]);
      }
    } else {
      std::for_each(type.GetDimSizes().begin(), type.GetDimSizes().end(),
                    [&](int x) { output->add_dims(x); });
    }
  }

  google::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(false);
  printer.SetInitialIndentLevel(0);

  google::protobuf::io::OstreamOutputStream oss(os);
  printer.Print(cfg, &oss);
}

bool TritonConfigWriter::RunOnModule(Module* module) {
  std::ostream* os = &std::cout;
  std::ofstream ofs;
  if (!filename_.empty()) {
    ofs.open(filename_);
    os = &ofs;
  }
  PrintUseProtobuf(*module, os);
  return false;
}

} // namespace halo