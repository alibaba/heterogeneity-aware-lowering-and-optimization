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
enum class Quantization { QUINT8, QUINT16, FLOAT16, None };
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

enum class ModelFormat {
  TENSORFLOW,
  CAFFE,
  ONNX,
  ODLA,
  TFLITE,
  MXNET,
  INVALID
};

enum class ChannelOrder {
  None,
  ChannelFirst,
  ChannelLast,
};

struct AnalyzerOpts {
  bool print_details = false;
  int batch_size = 1;
  int qps = 0; // image per second
};

struct CXXCodeGenOpts {
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
  bool simplify_for_preprocess = false;
  bool disable_broadcasting = true;
  bool enable_type_cast = false;
  bool remove_input_transpose = false;
  bool remove_output_transpose = false;
  bool disable_conv_bn = false;
  bool fuse_conv_bias = false;
  bool fuse_matmul_mul = false;
  bool fuse_hardswish = false;
  bool fuse_conv_relu = false;
  bool fuse_fully_connected = true;
  bool fuse_mul_to_conv = true;
  bool enable_anole_device = false;
  int64_t ipu_num = 1;
  int64_t batches_per_step = 1;
  const char* quant_tbl = nullptr;
  bool check_model = false;
  API api = API::ODLA_05;
  ChannelOrder channel_order = ChannelOrder::None;
  bool format_code = false;
  bool emit_header = false;
  bool emit_obj = false;
  bool emit_shared_lib = false;
  bool emit_pb_file = false;
  const char* linked_odla_lib = nullptr;
  const char* template_file = nullptr;
  bool save_temps = false;
  bool constant_decombine = false;
  bool convert_split_to_slice = true;
  bool convert_squared_diff = true;
};

#define HALO_MODEL_INFO_MAX_OUTPUT_NR 64
#define HALO_VODLA_MAX_OUTPUT_RSC_EST 2048
struct ModelInfo {
  size_t num_outputs;
  size_t output_buf_sizes[HALO_MODEL_INFO_MAX_OUTPUT_NR];
  int input_qps = 0;                                  // input of analyzer
  int adaptive_bsz = 0;                               // output of analyzer
  char output_rsc_est[HALO_VODLA_MAX_OUTPUT_RSC_EST]; // output of analyzer
};

int CompileTFGraph(const char* pb_buf, size_t pb_buf_size,
                   const std::vector<std::string>& input_shapes,
                   const CXXCodeGenOpts& cg_opts,
                   const std::string& main_output_file, ModelInfo* model_info);

int CompileTFGraph(const void* graphdef,
                   const std::vector<std::string>& input_shapes,
                   const CXXCodeGenOpts& cg_opts,
                   const std::string& main_output_file, ModelInfo* model_info);

int Compile(ModelFormat model_format, const std::vector<const char*>& models,
            const std::vector<size_t>& model_sizes, const std::string& target,
            int batch, const std::vector<std::string>& input_shapes,
            const std::vector<std::string>& inputs,
            const std::vector<std::string>& outputs,
            const CXXCodeGenOpts& cg_opts, const std::string& main_output_file,
            ModelInfo* model_info, bool is_compile_mode);

} // namespace halo

extern "C" {
typedef struct CXXCodeGenOps HaloCodeGenOpts;
typedef struct halo::ModelInfo HaloModelInfo;

[[deprecated]] int halo_CompileTFPbGraph(const char* pb_buf, size_t pb_buf_size,
                                         size_t num_input_shapes,
                                         const char* input_shapes[], int batch,
                                         const HaloCodeGenOpts* cg_opts,
                                         const char* main_output_file,
                                         HaloModelInfo* model_info);

int halo_Compile(halo::ModelFormat model_format, unsigned num_models,
                 const char* const models[], size_t const model_sizes[],
                 const char* target, int batch, unsigned num_input_shapes,
                 const char* const input_shapes[], unsigned num_inputs,
                 const char* const inputs[], unsigned num_outputs,
                 const char* const outputs[], const HaloCodeGenOpts* cg_opts,
                 const char* main_output_file, HaloModelInfo* model_info);

int halo_Analyze(halo::ModelFormat model_format, unsigned num_models,
                 const char* const models[], size_t const model_sizes[],
                 const char* target, int batch, unsigned num_input_shapes,
                 const char* const input_shapes[], unsigned num_inputs,
                 const char* const inputs[], unsigned num_outputs,
                 const char* const outputs[], const HaloCodeGenOpts* cg_opts,
                 const char* main_output_file, HaloModelInfo* model_info);
}

#endif // HALO_HALO_H_
