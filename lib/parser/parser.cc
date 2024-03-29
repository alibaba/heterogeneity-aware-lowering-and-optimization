//===- parser.cc ----------------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "halo/lib/parser/parser.h"

#include <fstream>
#include <memory>
#include <set>
#include <variant>

namespace halo {

static bool ValidateFiles(const std::vector<std::string>& file_list) {
  for (const auto& fn : file_list) {
    std::ifstream f(fn.c_str());
    if (!f) {
      std::cerr << "Unable to open file " << fn << std::endl;
      return false;
    }
  }
  return true;
}

static std::unique_ptr<Parser> GetParser(ModelFormat format,
                                         const std::string& variant,
                                         const armory::Opts& opts) {
  std::unique_ptr<Parser> parser(nullptr);
  switch (format) {
    case ModelFormat::TENSORFLOW: {
      if (opts.convert_to_ipu_graphdef) {
        parser = CreateIPUParser(variant);
      } else {
        parser = CreateTFParser(variant);
      }
      break;
    }
    case ModelFormat::ONNX: {
      parser = CreateONNXParser();
      break;
    }
    case ModelFormat::TFLITE: {
      parser = CreateTFLITEParser();
      break;
    }
    case ModelFormat::CAFFE: {
      parser = CreateCAFFEParser();
      break;
    }
    default:
      HLCHECK(0 && "Unsupported format");
  }
  return parser;
}

Status Parser::Parse(Function* function, ModelFormat format,
                     const std::string& variant,
                     const std::vector<std::string>& file_list,
                     const armory::Opts& opts) {
  if (!ValidateFiles(file_list)) {
    return Status::FILE_NOT_EXIST;
  }
  auto parser = GetParser(format, variant, opts);
  if (parser == nullptr) {
    return Status::ILLEGAL_PARAM;
  }
  return parser->Parse(function, file_list, opts);
}

Status Parser::Parse(Function* function, ModelFormat format,
                     const std::vector<std::string>& file_list,
                     const armory::Opts& opts) {
  return Parse(function, format, "", file_list, opts);
}

Status Parser::Parse(Function* function,
                     const std::vector<const char*>& buffers,
                     const std::vector<size_t>& buffer_sizes,
                     ModelFormat format) {
  armory::Opts opts;
  std::string variant;
  auto parser = GetParser(format, variant, opts);
  if (parser == nullptr) {
    return Status::ILLEGAL_PARAM;
  }
  return parser->Parse(function, buffers, buffer_sizes);
}

Status Parser::Parse(Function* function, const std::vector<const void*>& model,
                     ModelFormat format) {
  armory::Opts opts;
  std::string variant;
  auto parser = GetParser(format, variant, opts);
  if (parser == nullptr) {
    return Status::ILLEGAL_PARAM;
  }
  return parser->Parse(function, model);
}

} // namespace halo
