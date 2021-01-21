//===- parser.h -------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_PARSER_PARSER_H_
#define HALO_LIB_PARSER_PARSER_H_

#include <string>
#include <vector>

#include "halo/lib/ir/function.h"

namespace halo {

namespace armory {

struct Opts {
  Opts(const bool& print_diagnostic)
      : print_diagnostic_report(print_diagnostic) {}
  Opts() = default;
  bool print_diagnostic_report = false;
  bool convert_to_ipu_graphdef = false;
  std::string output_graphdef_filename;
  std::vector<std::vector<std::string>> split_names;
};

} // namespace armory

/// This class represents a parser interface.
class Parser {
 public:
  enum class Format { TENSORFLOW, CAFFE, ONNX, TFLITE, MXNET, INVALID };
  virtual ~Parser() = default;

  virtual Status Parse(Function* function,
                       const std::vector<std::string>& file_list,
                       const armory::Opts& opts) = 0;

  /// Parse a file from `file_lists` based on specified format. `variant`
  /// specifies sub variants like version etc., which can be empty.
  static Status Parse(Function* function, Format format,
                      const std::string& variant,
                      const std::vector<std::string>& file_list,
                      const armory::Opts& opts);

  /// Parse a file from `file_lists` based on specified format.
  static Status Parse(Function* function, Format format,
                      const std::vector<std::string>& file_list,
                      const armory::Opts& opts);
};

template <typename T>
class Tensor {
 public:
  explicit Tensor(const DataType& data_type, const std::vector<int64_t>& shape,
                  const std::vector<T>& src_data)
      : data_type_(data_type), shape_(shape), data_(src_data) {}

  explicit Tensor(const DataType& data_type, const std::vector<int64_t>& shape,
                  const std::vector<T>& src_data, bool need_decode)
      : data_type_(data_type),
        shape_(shape),
        data_(src_data),
        need_decode_(need_decode) {}

  ~Tensor() = default;

  Tensor(const Tensor& other) = default;
  Tensor& operator=(const Tensor& other) = default;

  Tensor(Tensor&& other) = default;
  Tensor& operator=(Tensor&& other) = default;

  const DataType& GetDataType() const noexcept { return data_type_; }
  const std::vector<int64_t>& GetShape() const noexcept { return shape_; }
  void SetShape(const std::vector<int64_t>& shape) noexcept { shape_ = shape; }
  const std::vector<T>& GetData() const noexcept { return data_; }
  bool GetNeedDecode() const noexcept { return need_decode_; }
  inline static std::vector<T> DecodeTensorContent(const std::string& buf) {
    std::vector<T> output;
    output.resize(buf.size() / sizeof(T));
    HLCHECK(buf.size() % sizeof(T) == 0);
    const T* p = reinterpret_cast<const T*>(buf.c_str()); // NOLINT.
    for (size_t i = 0; i < buf.size() / sizeof(T); ++i) {
      output[i] = *p++; // NOLINT.
    }
    return output;
  }

 private:
  DataType data_type_ = DataType::INVALID;
  std::vector<int64_t> shape_;
  std::vector<T> data_;
  bool need_decode_ = false;
};

} // namespace halo

#endif // HALO_LIB_PARSER_PARSER_H_
