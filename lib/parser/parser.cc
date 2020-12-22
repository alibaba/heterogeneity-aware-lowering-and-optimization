//===- parser.cc ----------------------------------------------------------===//
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

#include "halo/lib/parser/parser.h"

#include <fstream>
#include <memory>
#include <set>
#include <variant>

#include "caffe/caffe_parser.h"
#include "onnx/onnx_parser.h"
#include "tensorflow/tf_parser.h"
#include "tflite/tflite_parser.h"

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

Status Parser::Parse(Function* function, Format format,
                     const std::string& variant,
                     const std::vector<std::string>& file_list,
                     const armory::Opts& opts) {
  if (!ValidateFiles(file_list)) {
    return Status::FILE_NOT_EXIST;
  }

  std::unique_ptr<Parser> parser(nullptr);
  switch (format) {
    case Format::TENSORFLOW: {
      if (opts.convert_to_ipu_graphdef) {
        parser = std::make_unique<IPUParser>(variant);
      } else {
        parser = std::make_unique<TFParser>(variant);
      }
      break;
    }
    case Format::ONNX: {
      parser = std::make_unique<ONNXParser>();
      break;
    }
    case Format::TFLITE: {
      parser = std::make_unique<TFLITEParser>();
      break;
    }
    case Format::CAFFE: {
      parser = std::make_unique<CAFFEParser>();
      break;
    }
    default:
      HLCHECK(0 && "Unsupported format");
  }
  return parser->Parse(function, file_list, opts);
}

Status Parser::Parse(Function* function, Format format,
                     const std::vector<std::string>& file_list,
                     const armory::Opts& opts) {
  return Parse(function, format, "", file_list, opts);
}

} // namespace halo
