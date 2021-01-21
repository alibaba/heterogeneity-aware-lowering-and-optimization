//===- codegen.h ------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_CODEGEN_H_
#define HALO_LIB_TARGET_CODEGEN_H_

#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/pass/pass.h"

namespace halo {

// The base codegen pass. It provides the overloaded virtual RunOnInstruction()
// function for each IR class.
class CodeGen : public ModulePass {
 public:
  enum class ExecMode { Compile, Interpret };
  enum class API { HALO_RT, ODLA_05 };
  enum class Quantization { QUINT8, None };
  enum class BF16Mode { Disable, Accuracy, Performace, Auto};
  CodeGen(const std::string& pass_name)
      : ModulePass(pass_name), api_(API::ODLA_05) {}
  bool RunOnModule(Module* module) = 0;
  API GetAPI() const noexcept { return api_; }
  void SetAPI(API api) noexcept { api_ = api; }
  bool IsODLA05() const noexcept { return api_ == API::ODLA_05; }

  /// make a valid C/C++ identifier name.
  static std::string NormalizeVariableName(const std::string& name);

 protected:
  /// The entrance for all instructions. It will then forward the call to
  /// RunOnInstruction(const XXXInst&) based on op code.
  virtual void RunOnBaseInstruction(Instruction*);

#define GET_RUN_ON_INSTRUCTION_DECL
#include "halo/lib/ir/instructions_info.def"
#undef GET_RUN_ON_INSTRUCTION_DECL

  static const std::string& GetRTLibFuncName(const Instruction&);

 private:
  // TODO(unknown): we need to define the name and signature in .td file so the
  // LLVM function declaration can be emitted automatically.
  inline static const std::unordered_map<OpCode, std::string>
      RuntimeLibFuncNames{
          {OpCode::CONV2D, "_sn_rt_conv2d"},
          {OpCode::MATMUL, "_sn_rt_matmul"},
          {OpCode::GEMM, "_sn_rt_gemm"},
          {OpCode::PAD, "_sn_rt_pad"},
          {OpCode::POOLINGMAX, "_sn_rt_poolingmax"},
          {OpCode::REDUCEMEAN, "_sn_rt_reduce_mean"},
          {OpCode::SOFTMAX, "_sn_rt_softmax"},
          {OpCode::ADD, "_sn_rt_add"},
          {OpCode::BATCHNORM, "_sn_rt_bn"},
          {OpCode::RELU, "_sn_rt_relu"},
          {OpCode::SUB, "_sn_rt_sub"},
          {OpCode::MUL, "_sn_rt_mul"},
          {OpCode::DIV, "_sn_rt_div"},
          {OpCode::ERF, "_sn_rt_erf"},
          {OpCode::FLOOR, "_sn_rt_floor"},
          {OpCode::RSQRT, "_sn_rt_rsqrt"},
          {OpCode::BATCHMATMUL, "_sn_rt_batch_matmul"},
          {OpCode::ONEHOT, "_sn_rt_onehot"},
          {OpCode::GATHER, "_sn_rt_gather"},
          {OpCode::SLICE, "_sn_rt_slice"},
          {OpCode::TRANSPOSE, "_sn_rt_transpose"},
          {OpCode::SITOFP, "_sn_rt_sitofp"},
          {OpCode::SQRT, "_sn_rt_sqrt"},
          {OpCode::ARGMAX, "_sn_rt_argmax"},
      };
  API api_;
};

class CodeWriter : public ModulePass {
 public:
  explicit CodeWriter(const std::string& name, std::ostream& os)
      : ModulePass(name), os_(os) {}
  virtual ~CodeWriter() = default;

 protected:
  std::ostream& os_;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_CODEGEN_H_
