//===- generic_cxx_codegen.h ------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_GENERIC_CXX_GENERIC_CXX_CODEGEN_H_
#define HALO_LIB_TARGET_GENERIC_CXX_GENERIC_CXX_CODEGEN_H_

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/common_cast_instructions.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/instruction.h"
#include "halo/lib/ir/loss_instructions.h"
#include "halo/lib/ir/nn_activation_instructions.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/ir/nn_instructions.h"
#include "halo/lib/ir/quantization_instructions.h"
#include "halo/lib/mm/memory_analyzer.h"
#include "halo/lib/target/codegen.h"

namespace halo {

struct CXXType {
  CXXType(const std::string& name)
      : name(name), is_const(false), is_pointer(true) {}
  CXXType() = default;
  std::string Str(bool use_array_decl) const;
  std::string name{"void"};
  bool is_const = false;
  bool is_pointer = true;
};

class CXXValue {
 public:
  CXXValue() : name("undef"), str_id(""), id(-1), type("void"){};
  CXXValue(const std::string& name, const CXXType& type);
  const std::string& GetName() const { return str_id.empty() ? name : str_id; }
  static void Reset();
  std::string name;
  std::string str_id;
  size_t id;
  CXXType type;

 private:
  inline static std::unordered_map<std::string, size_t> name2id;
};

// The generic CXX compiler, which is a module pass.
class GenericCXXCodeGen : public CodeGen {
 public:
  GenericCXXCodeGen(std::ostringstream& os, std::ostringstream& header_os);
  GenericCXXCodeGen(std::ostringstream& os, std::ostringstream& header_os,
                    std::ostream& dynamic_check_os, const CXXCodeGenOpts& opts);

  virtual ~GenericCXXCodeGen();

  bool RunOnModule(Module* module) override;

 protected:
  template <typename T>
  struct ArgWithComment {
    ArgWithComment(const T& arg, const char* comment)
        : arg_(arg), comment_(comment) {}
    const T& arg_;
    const char* comment_;
  };

  virtual void RunOnFunction(Function& function);
  virtual void RunOnHostFunction(Function& function);
  virtual void RunOnConstant(Constant& constant, bool decl);
  virtual void RunOnBasicBlock(BasicBlock& bb);
  void RunOnArgument(Argument& arg);
  void PreRunOnInstruction(Instruction*);
  void PostRunOnInstruction(Instruction*);

  // TODO(unknown): The following RunOnInstruction will be generated via .td
  // file.
  virtual void RunOnInstruction(AbsInst*) override;
  virtual void RunOnInstruction(ACosInst*) override;
  virtual void RunOnInstruction(ACoshInst*) override;
  virtual void RunOnInstruction(AddInst*) override;
  virtual void RunOnInstruction(AndInst*) override;
  virtual void RunOnInstruction(ArgmaxInst*) override;
  virtual void RunOnInstruction(ArgminInst*) override;
  virtual void RunOnInstruction(ASinInst*) override;
  virtual void RunOnInstruction(ASinhInst*) override;
  virtual void RunOnInstruction(ATanInst*) override;
  virtual void RunOnInstruction(ATanhInst*) override;
  virtual void RunOnInstruction(SubInst*) override;
  virtual void RunOnInstruction(MulInst*) override;
  virtual void RunOnInstruction(CallInst*) override;
  virtual void RunOnInstruction(CeilInst*) override;
  virtual void RunOnInstruction(CmpInst*) override;
  virtual void RunOnInstruction(CumSumInst*) override;
  virtual void RunOnInstruction(CustomInst*) override;
  virtual void RunOnInstruction(ConcatInst*) override;
  virtual void RunOnInstruction(ConvertFromStringInst*) override;
  virtual void RunOnInstruction(ConvertToStringInst*) override;
  virtual void RunOnInstruction(DetInst*) override;
  virtual void RunOnInstruction(DivInst*) override;
  virtual void RunOnInstruction(ErfInst*) override;
  virtual void RunOnInstruction(ExpInst*) override;
  virtual void RunOnInstruction(ExpandDimsInst*) override;
  virtual void RunOnInstruction(FloorInst*) override;
  virtual void RunOnInstruction(RoundInst*) override;
  virtual void RunOnInstruction(RcpInst*) override;
  virtual void RunOnInstruction(FPtoSIInst*) override;
  virtual void RunOnInstruction(FPtoFPInst*) override;
  virtual void RunOnInstruction(GroupNormInst*) override;
  virtual void RunOnInstruction(HardmaxInst*) override;
  virtual void RunOnInstruction(LeakyReluInst*) override;
  virtual void RunOnInstruction(SeluInst*) override;
  virtual void RunOnInstruction(EluInst*) override;
  virtual void RunOnInstruction(CeluInst*) override;
  virtual void RunOnInstruction(ThresholdedReluInst*) override;
  virtual void RunOnInstruction(SqrtInst*) override;
  virtual void RunOnInstruction(SquaredDifferenceInst*) override;
  virtual void RunOnInstruction(RsqrtInst*) override;
  virtual void RunOnInstruction(BatchNormInst*) override;
  virtual void RunOnInstruction(InstanceNormInst*) override;
  virtual void RunOnInstruction(LpNormalizeInst*) override;
  virtual void RunOnInstruction(LayerNormInst*) override;
  virtual void RunOnInstruction(BatchMatMulInst*) override;
  virtual void RunOnInstruction(CompressInst*) override;
  virtual void RunOnInstruction(Conv2DInst*) override;
  virtual void RunOnInstruction(Conv2DTransposeInst*) override;
  virtual void RunOnInstruction(DequantizeInst*) override;
  virtual void RunOnInstruction(GatherInst*) override;
  virtual void RunOnInstruction(GatherElementsInst*) override;
  virtual void RunOnInstruction(GeluInst*) override;
  virtual void RunOnInstruction(GemmInst*) override;
  virtual void RunOnInstruction(GRUInst*) override;
  virtual void RunOnInstruction(IfInst*) override;
  virtual void RunOnInstruction(LogInst*) override;
  virtual void RunOnInstruction(LogSoftmaxInst*) override;
  virtual void RunOnInstruction(LRNInst*) override;
  virtual void RunOnInstruction(LSTMInst*) override;
  virtual void RunOnInstruction(MatMulInst*) override;
  virtual void RunOnInstruction(MaximumInst*) override;
  virtual void RunOnInstruction(MeanInst*) override;
  virtual void RunOnInstruction(MinimumInst*) override;
  virtual void RunOnInstruction(ModInst*) override;
  virtual void RunOnInstruction(MultiHeadAttentionInst*) override;
  virtual void RunOnInstruction(NegativeLogLikelihoodLossInst*) override;
  virtual void RunOnInstruction(NonMaxSuppressionInst*) override;
  virtual void RunOnInstruction(NegInst*) override;
  virtual void RunOnInstruction(NotInst*) override;
  virtual void RunOnInstruction(OneHotInst*) override;
  virtual void RunOnInstruction(OrInst*) override;
  virtual void RunOnInstruction(XorInst*) override;
  virtual void RunOnInstruction(PadInst*) override;
  virtual void RunOnInstruction(PoolingMaxInst*) override;
  virtual void RunOnInstruction(PoolingAvgInst*) override;
  virtual void RunOnInstruction(PowInst*) override;
  virtual void RunOnInstruction(PReluInst*) override;
  virtual void RunOnInstruction(QuantizeInst*) override;
  virtual void RunOnInstruction(RandomUniformInst*) override;
  virtual void RunOnInstruction(ReduceL1Inst*) override;
  virtual void RunOnInstruction(ReduceL2Inst*) override;
  virtual void RunOnInstruction(ReduceLogSumInst*) override;
  virtual void RunOnInstruction(ReduceLogSumExpInst*) override;
  virtual void RunOnInstruction(ReduceSumInst*) override;
  virtual void RunOnInstruction(ReduceSumSquareInst*) override;
  virtual void RunOnInstruction(ReduceMeanInst*) override;
  virtual void RunOnInstruction(ReduceMaxInst*) override;
  virtual void RunOnInstruction(ReduceMinInst*) override;
  virtual void RunOnInstruction(ReduceProductInst*) override;
  virtual void RunOnInstruction(ReluInst*) override;
  virtual void RunOnInstruction(Relu6Inst*) override;
  virtual void RunOnInstruction(ReshapeInst*) override;
  virtual void RunOnInstruction(ResizeInst*) override;
  virtual void RunOnInstruction(ReturnInst*) override;
  virtual void RunOnInstruction(RNNInst*) override;
  virtual void RunOnInstruction(SelectInst*) override;
  virtual void RunOnInstruction(ShapeInst*) override;
  virtual void RunOnInstruction(ShiftInst*) override;
  virtual void RunOnInstruction(ShrinkInst*) override;
  virtual void RunOnInstruction(SigmoidInst*) override;
  virtual void RunOnInstruction(SItoFPInst*) override;
  virtual void RunOnInstruction(SliceInst*) override;
  virtual void RunOnInstruction(SliceDynamicInst*) override;
  virtual void RunOnInstruction(SoftmaxInst*) override;
  virtual void RunOnInstruction(SoftplusInst*) override;
  virtual void RunOnInstruction(SoftsignInst*) override;
  virtual void RunOnInstruction(SplitInst*) override;
  virtual void RunOnInstruction(StackInst*) override;
  virtual void RunOnInstruction(SwishInst*) override;
  virtual void RunOnInstruction(HardSigmoidInst*) override;
  virtual void RunOnInstruction(HardSwishInst*) override;
  virtual void RunOnInstruction(SinInst*) override;
  virtual void RunOnInstruction(SinhInst*) override;
  virtual void RunOnInstruction(CosInst*) override;
  virtual void RunOnInstruction(CoshInst*) override;
  virtual void RunOnInstruction(TanInst*) override;
  virtual void RunOnInstruction(TanhInst*) override;
  virtual void RunOnInstruction(TopKInst*) override;
  virtual void RunOnInstruction(TransposeInst*) override;
  virtual void RunOnInstruction(TFIDFVectorizeInst*) override;
  virtual void RunOnInstruction(TileInst*) override;
  virtual void RunOnInstruction(TileDynamicInst*) override;
  virtual void RunOnInstruction(ZExtInst*) override;
  virtual void RunOnInstruction(HgEngineInst*) override;
  virtual void RunOnInstruction(HgQuantInst*) override;
  virtual void RunOnInstruction(HgDequantInst*) override;
  virtual void RunOnInstruction(IsNaNInst*) override;
  virtual void RunOnInstruction(IsInfInst*) override;
  virtual void RunOnInstruction(SignInst*) override;
  virtual void RunOnInstruction(EinsumInst*) override;

  virtual void RunOnBinaryInstruction(Instruction*);
  virtual void RunOnCastInstruction(Instruction*);
  virtual void RunOnConcatInstruction(Instruction* inst, int axis,
                                      const char* odla_func_name);
  virtual void RunOnReductionInstruction(Instruction*,
                                         const std::vector<int32_t>& axis_attr,
                                         bool keep_dims,
                                         const std::string& odla_func_name,
                                         float epsilon);
  virtual void RunOnUnaryInstruction(Instruction*);
  virtual void RunOnBranchBody(const IfInst& inst, bool is_taken);

  virtual CXXValue AllocateBuffer(const Def& def, bool on_stack);
  std::string GetFunctionDecl(const Function& func, const Instruction& ret_inst,
                              bool with_func_name, bool with_type,
                              bool public_function);
  std::string GenerateTestFunc(const Function& func,
                               const std::string& func_decl,
                               const Instruction& ret_inst);
  virtual std::string EmitShape(const halo::Type& type);
  virtual std::string EmitType(const halo::Type& type);
  virtual std::string EmitLValue(const std::string& name) const;
  virtual std::string EmitLValues(const std::string& name) const;

  void EmitODLAArgs(const std::vector<int32_t>& arg);
  void EmitODLAArgs(const std::vector<uint32_t>& arg);
  void EmitODLAArgs(const std::vector<float>& arg);
  void EmitODLAArgs(const std::vector<CXXValue>& arg);
  void EmitODLAArgs(const halo::Type& arg);
  void EmitODLAArgs(const DataType& arg);
  void EmitODLAArgs(const CXXValue& arg);
  void EmitODLAArgs(const bool& arg);
  void EmitODLAArgs(const DataFormat& arg);
  void EmitODLAArgs(const std::vector<halo::Type>& arg);
  void EmitODLAArgs(const std::vector<std::string>& arg);

  template <typename T>
  void EmitODLAArgs(const ArgWithComment<T>& arg) {
    EmitODLAArgs(arg.arg_);
    if (opts_.dialect == Dialect::CXX_11) {
      os_ << "/*" << arg.comment_ << "*/";
    } else {
      os_ << "//" << arg.comment_ << "\n";
    }
  }

  template <typename T>
  void EmitODLAArgs(const T& arg) {
    os_ << arg;
  }

  template <typename T, typename... Targs>
  void EmitODLAArgs(T arg, Targs... args) {
    EmitODLAArgs(arg);
    os_ << ", ";
    EmitODLAArgs(args...);
  }

  void EmitODLAArgs() {}

  inline void EmitODLAVauleId(const CXXValue& lhs, std::ostream& os) {
    if (opts_.emit_value_id_as_int) {
      os << lhs.id;
    } else {
      os << "\"" << (lhs.str_id.empty() ? lhs.name : lhs.str_id) << "\"";
    }
  }

  template <int indent = 2, bool is_op = true, typename... Targs>
  void EmitODLACall(const CXXValue& lhs, const char* func_name, Targs... args) {
    os_ << std::string(indent, ' ');
    if (!lhs.name.empty()) {
      os_ << EmitLValue(lhs.name) << " = ";
    }
    os_ << func_name << "(";
    EmitODLAArgs(args...);
    if (is_op) {
      os_ << ", (const odla_value_id)";
      EmitODLAVauleId(lhs, os_);
    }
    os_ << ");\n";
  }

  template <int indent = 2, typename... Targs>
  void EmitODLACustomCall(const std::vector<CXXValue>& lhs,
                          std::vector<CXXValue> inputs,
                          const std::string& op_name,
                          const std::string& func_name, Targs... args) {
    os_ << std::string(indent, ' ');
    auto ret_array = lhs[0].name + "_array";
    os_ << EmitLValues(ret_array) << " = odla_CustomOp(";
    EmitODLAArgs(inputs);
    os_ << ", ";
    EmitODLAArgs(op_name);
    os_ << ", ";

    EmitODLAArgs(func_name);
    os_ << ",  (odla_value_ids){.size = " << lhs.size() << ", .value_ids = {";
    unsigned int id = 0;
    for (auto& one : lhs) {
      os_ << "(const odla_value_id)";
      EmitODLAVauleId(one, os_);
      if (++id != lhs.size()) {
        os_ << ", ";
      }
    }
    os_ << "}}";
    constexpr std::size_t n = sizeof...(args);
    if (n > 0) {
      os_ << ",";
      EmitODLAArgs(args...);
    }

    os_ << ");\n";

    for (auto& one : lhs) {
      os_ << std::string(indent, ' ');
      os_ << EmitLValue(one.name) << " = " << ret_array << ".values[" << id++
          << "];";
    }
  }

  template <int indent = 2, bool is_op = true, typename... Targs>
  void EmitODLACall(const std::vector<CXXValue>& lhs, const char* func_name,
                    Targs... args) {
    os_ << std::string(indent, ' ');
    auto ret_array = lhs[0].name + "_array";
    os_ << EmitLValues(ret_array) << " = ";
    os_ << func_name << "(";
    constexpr std::size_t n = sizeof...(args);
    if (n > 0) {
      EmitODLAArgs(args...);
    }
    if (is_op) {
      unsigned int id = 0;
      if (n > 0) {
        os_ << ", ";
      }
      os_ << " (odla_value_ids){.size = " << lhs.size() << ", .value_ids = {";
      for (auto& one : lhs) {
        os_ << "(const odla_value_id)";
        EmitODLAVauleId(one, os_);
        if (++id != lhs.size()) {
          os_ << ", ";
        }
      }
      os_ << "}}";
    }
    os_ << ");\n";

    unsigned int id = 0;
    for (auto& one : lhs) {
      os_ << std::string(indent, ' ');
      os_ << EmitLValue(one.name) << " = " << ret_array << ".values[" << id++
          << "];";
    }
  }

  virtual const std::string& EmitNull() const noexcept;
  virtual std::string GetODLAType(halo::DataType data_type) const noexcept;

 public:
  static const std::string& EmitReturnType(bool auto_type, bool single_value);
  static CXXType SNTypeToCXXType(DataType dt);
  static CXXType TensorTypeToCXXType(const halo::Type& type, bool is_const);

  template <typename T>
  inline static std::string Join(std::vector<T> vals, char sep = ',') {
    std::ostringstream ss;
    bool is_first = true;
    for (const auto& val : vals) {
      if (!is_first) {
        ss << sep << " ";
      }
      ss << val;
      is_first = false;
    }
    return ss.str();
  }
  template <typename T>
  inline static std::string Join(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
  }
  template <typename T, typename... Targs>
  inline static std::string Join(T value, Targs... args) {
    std::ostringstream ss;
    ss << value << ", ";
    ss << Join(args...);
    return ss.str();
  }

 protected:
  bool emit_banner = true;
  std::ostringstream& os_;
  std::ostringstream& header_os_;
  std::ostream& dynamic_check_os_;
  std::unordered_map<Def, CXXValue> ir_mapping_;
  std::unique_ptr<MemoryAnalyzer> memory_analyzer_;
  std::unordered_set<const BasicBlock*> visited_;
  CXXCodeGenOpts opts_;
};

class GenericCXXConstantWriter : public CodeGen {
 public:
  virtual ~GenericCXXConstantWriter() = default;
  explicit GenericCXXConstantWriter(std::ostream& os);

  bool RunOnModule(Module* module) override;

 private:
  void static RunOnConstant(const Constant& constant, std::ostream* os);
  std::ostream& os_;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_GENERIC_CXX_GENERIC_CXX_CODEGEN_H_
