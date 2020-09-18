//===- generic_llvmir_codegen.h ---------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_GENERIC_LLVMIR_GENERIC_LLVMIR_CODEGEN_H_
#define HALO_LIB_TARGET_GENERIC_LLVMIR_GENERIC_LLVMIR_CODEGEN_H_

#include <memory>
#include <unordered_map>

#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/common_instructions.h"
#include "halo/lib/ir/common_reduction_instructions.h"
#include "halo/lib/ir/instruction.h"
#include "halo/lib/ir/nn_activation_instructions.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/target/codegen.h"

// Forward declaration here to avoid the need of LLVM header files for API
// users of this class.
namespace llvm {
class BasicBlock;
class CallInst;
class ConstantFolder;
class Function;
class FunctionCallee;
class IRBuilderDefaultInserter;
template <typename T, typename Inserter>
class IRBuilder;
class LLVMContext;
class Module;
class StringRef;
class TargetMachine;
class Type;
class Value;
template <typename T>
class ArrayRef;
} // namespace llvm

namespace halo {

// The generic LLVMIR compiler, which is a module pass.
class GenericLLVMIRCodeGen : public CodeGen {
 public:
  enum class ConstantDataStorage {
    DefinedAsStatic,         // internal, constant, with initializer.
    DefinedAsStaticNonConst, // internal, non-constant, no initializer.
    DeclaredAsExternal,      // external, constant, no initializer
    DefinedAsGlobal          // external, constant, with initializer.
  };

  GenericLLVMIRCodeGen();
  GenericLLVMIRCodeGen(ConstantDataStorage constant_data_storage);

  GenericLLVMIRCodeGen(const std::string& name,
                       ConstantDataStorage constant_data_storage);
  virtual ~GenericLLVMIRCodeGen();

  bool RunOnModule(Module* module) override;

 protected:
  virtual void RunOnFunction(Function& function);
  virtual void RunOnConstant(Constant& constant);
  virtual void RunOnBasicBlock(llvm::Function* llvm_func, BasicBlock& bb);
  virtual llvm::TargetMachine* InitTargetMachine();
  // TODO(unknown): The following RunOnInstruction will be generated via .td
  // file.
  virtual void RunOnInstruction(BatchMatMulInst*) override;
  virtual void RunOnInstruction(BatchNormInst*) override;
  virtual void RunOnInstruction(Conv2DInst*) override;
  virtual void RunOnInstruction(GatherInst*) override;
  virtual void RunOnInstruction(GemmInst*) override;
  virtual void RunOnInstruction(MatMulInst*) override;
  virtual void RunOnInstruction(PadInst*) override;
  virtual void RunOnInstruction(PoolingMaxInst*) override;
  virtual void RunOnInstruction(OneHotInst*) override;
  virtual void RunOnInstruction(ReluInst*) override;
  virtual void RunOnInstruction(ReshapeInst*) override;
  virtual void RunOnInstruction(ReturnInst*) override;
  virtual void RunOnInstruction(SItoFPInst*) override;
  virtual void RunOnInstruction(SliceInst*) override;
  virtual void RunOnInstruction(SoftmaxInst*) override;
  virtual void RunOnInstruction(TransposeInst*) override;

  virtual void RunOnInstruction(ReduceMeanInst* inst) override {
    RunOnCommonReductionInstruction(inst, inst->GetAxis());
  }
  virtual void RunOnInstruction(ArgmaxInst* inst) override {
    RunOnCommonReductionInstruction(inst, {inst->GetAxis()});
  }

  virtual void RunOnBaseInstruction(Instruction*) override;

  virtual std::string GetRuntimeLibDir() const;
  virtual std::string GetRuntimeLibPath() const;
  virtual void LinkRuntimeLib();

  /// Utility functions.
  using DefaultIRBuilder =
      llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>;

  llvm::BasicBlock* GetLLVMBasicBlock(Instruction* inst);
  virtual llvm::Value* AllocateLLVMBuffer(DefaultIRBuilder* ir_builder,
                                          const Def& def, bool on_stack);
  virtual llvm::Value* AllocateLLVMBuffer(DefaultIRBuilder*, const Def& def);
  llvm::CallInst* CreateCall(llvm::FunctionCallee* callee,
                             llvm::ArrayRef<llvm::Value*> args);

  static llvm::LLVMContext& GetLLVMContext() noexcept;
  static llvm::Type* SNTypeToLLVMType(DataType dt);
  static const std::string& SNTypeToRTLibFuncSuffix(DataType dt);
  static const std::string& DataFormatToRTLibFuncSuffix(DataFormat df);
  static llvm::Type* TensorTypeToLLVMType(const halo::Type& type,
                                          bool as_pointer);
  GlobalContext* ctx_ = nullptr;
  std::unique_ptr<llvm::Module> llvm_module_;
  llvm::TargetMachine* target_machine_ = nullptr;
  DefaultIRBuilder* current_llvm_builder_ = nullptr;
  std::unordered_map<Def, llvm::Value*> ir_mapping_;

  inline static int64_t GetMaxVectorSize() {
    // This is LLVM's limit of vector length (llvm::SDNode::getMaxNumOperands().
    size_t limit = std::numeric_limits<uint16_t>::max();
    // LLVM will round vector size to next power of 2, so we need to half it.
    limit >>= 1;
    // Large vector is legit but slows down backend codegen significantly, so
    // limit it further.
    limit = std::min(limit, size_t(2048));
    return limit;
  }
  inline static std::string RuntimeLibName = "libRT_GENERIC.a";

 private:
  static std::string GetRTLibFuncName(
      const Instruction&, DataType data_type,
      DataFormat data_format = DataFormat::INVALID);
  void RunOnMathBinaryInstruction(Instruction* inst);
  void RunOnMathUnaryInstruction(Instruction* inst);
  void RunOnCommonReductionInstruction(Instruction* inst,
                                       const std::vector<int>& axis);
  ConstantDataStorage constant_data_storage_;
};

class GenericLLVMIRWriter : public CodeWriter {
 public:
  GenericLLVMIRWriter();
  explicit GenericLLVMIRWriter(const std::string& name, std::ostream& os,
                               bool bitcode_format);
  explicit GenericLLVMIRWriter(std::ostream& os, bool bitcode_format);

  bool RunOnModule(Module* module) override;

 private:
  bool bitcode_format_; // True for Bitcode output, False for text format.
};

/// This class emits the constants to a separate LLVM module.
class GenericConstantWriter : public GenericLLVMIRCodeGen {
 public:
  GenericConstantWriter();
  explicit GenericConstantWriter(const std::string& name, std::ostream& os,
                                 bool bitcode_format);
  explicit GenericConstantWriter(std::ostream& os, bool bitcode_format);

  bool RunOnModule(Module* module) override;

 protected:
  virtual void WriteToBuf();
  Module* module_ = nullptr;
  std::ostream& os_;
  bool bitcode_format_; // True for Bitcode output, False for text format.
};

/// This class emits all global variables to a separate ELF file.
class ELFConstantWriter : public GenericConstantWriter {
 public:
  ELFConstantWriter();
  explicit ELFConstantWriter(const std::string& name, std::ostream& os);
  explicit ELFConstantWriter(std::ostream& os);

 protected:
  void WriteToBuf() override;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_GENERIC_LLVMIR_GENERIC_LLVMIR_CODEGEN_H_