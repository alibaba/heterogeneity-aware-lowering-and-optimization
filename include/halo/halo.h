//===- halo.h ---------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_HALO_H_
#define HALO_HALO_H_

#include <cstddef>
#include <string>
#include <vector>

#include "version.h"
// Halo C++ interface.
namespace halo {

enum class ExecMode { Compile, Interpret };
enum class API { HALO_RT, ODLA_05 };
enum class Quantization { QUINT8, None };
enum class BF16Mode { Disable, Accuracy, Performace, Auto };
enum class Dialect {
  CXX_11,
  C99,
};

enum class ConstantDataStorage {
  DefinedAsStatic,         // internal, constant, with initializer.
  DefinedAsStaticNonConst, // internal, non-constant, no initializer.
  DeclaredAsExternal,      // external, constant, no initializer
  DefinedAsGlobal          // external, constant, with initializer.
};

enum class ModelFormat { TENSORFLOW, CAFFE, ONNX, TFLITE, MXNET, INVALID };

enum class ChannelOrder {
  None,
  ChannelFirst,
  ChannelLast,
};

struct CXXCodeGenOpts {
  CXXCodeGenOpts(const BF16Mode& mode) : bf16_mode(mode) {}
  CXXCodeGenOpts() = default;
  Dialect dialect = Dialect::CXX_11;
  bool print_mem_stats = false;
  bool emit_value_reset = false;
  bool emit_value_init = false;
  bool emit_value_id_as_int = false;
  BF16Mode bf16_mode = BF16Mode::Disable;
  ExecMode exec_mode = ExecMode::Compile;
  bool emit_inference_func_sig = false;
  bool emit_model_info_apis = false;
  bool emit_dynamic_batch = false;
  bool fp16_mode = false;
  int max_batch_size = 0;
  int min_batch_size = 0;
  int opt_batch_size = 0;
  bool enable_ipu_device = false;
  bool use_ipu_model = false;
  bool separate_constants = false;
  bool disable_broadcasting = false;
  bool remove_input_transpose = false;
  bool remove_output_transpose = false;
  bool disable_conv_bn = false;
  int64_t ipu_num = 1;
  int64_t batches_per_step = 1;
  bool check_model = false;
  API api = API::ODLA_05;
};

int CompileTFGraph(const char* pb_buf, size_t pb_buf_size,
                   const std::string& name, const std::string& temp_dir,
                   const std::vector<std::string>& input_shapes,
                   const CXXCodeGenOpts& cg_opts);

int CompileTFGraph(const void* graphdef, const std::string& name,
                   const std::string& temp_dir,
                   const std::vector<std::string>& input_shapes,
                   const CXXCodeGenOpts& cg_opts);

int Compile(ModelFormat model_format, const std::vector<const char*>& models,
            const std::vector<size_t>& model_sizes, const std::string& name,
            const std::string& temp_dir, const std::string& target, int batch,
            const std::vector<std::string>& input_shapes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const CXXCodeGenOpts& cg_opts);
} // namespace halo

#endif // HALO_HALO_H_