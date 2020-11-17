//===- generic_llvmir_codegen.cc ------------------------------------------===//
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

#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"

#include <limits>
#include <unordered_map>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/target/codegen.h"
#include "halo/lib/target/codegen_object.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Internalize.h"

namespace halo {

GenericLLVMIRCodeGen::GenericLLVMIRCodeGen(
    const std::string& name, ConstantDataStorage constant_data_storage)
    : CodeGen(name), constant_data_storage_(constant_data_storage) {}

GenericLLVMIRCodeGen::GenericLLVMIRCodeGen()
    : GenericLLVMIRCodeGen("Generic CPU LLVM IR Compilation",
                           ConstantDataStorage::DefinedAsStatic) {}

GenericLLVMIRCodeGen::GenericLLVMIRCodeGen(
    ConstantDataStorage constant_data_storage)
    : GenericLLVMIRCodeGen("Generic CPU LLVM IR Compilation",
                           constant_data_storage) {}

GenericLLVMIRCodeGen::~GenericLLVMIRCodeGen() = default;

llvm::LLVMContext& GenericLLVMIRCodeGen::GetLLVMContext() noexcept {
  static llvm::LLVMContext ctx;
  return ctx;
}

llvm::TargetMachine* GenericLLVMIRCodeGen::InitTargetMachine() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::Triple triple(llvm::Triple::normalize(LLVM_HOST_TRIPLE));
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), error);
  HLCHECK(target);
  llvm::StringRef cpu("");
  llvm::StringRef features("");
  llvm::Reloc::Model reloc = llvm::Reloc::Static;
  llvm::CodeGenOpt::Level opt_level = llvm::CodeGenOpt::Aggressive;
  llvm::CodeModel::Model cm = llvm::CodeModel::Medium;
  llvm::TargetOptions options;
  options.UnsafeFPMath = true;
  options.NoInfsFPMath = true;
  options.NoNaNsFPMath = true;
  options.NoSignedZerosFPMath = true;

  return target->createTargetMachine(triple.getTriple(), cpu, features, options,
                                     reloc, cm, opt_level);
}

llvm::Type* GenericLLVMIRCodeGen::SNTypeToLLVMType(DataType dt) {
  switch (dt) {
    case DataType::INT8:
    case DataType::UINT8: {
      return llvm::Type::getInt8Ty(GetLLVMContext());
    }
    case DataType::INT16:
    case DataType::UINT16: {
      return llvm::Type::getInt16Ty(GetLLVMContext());
    }
    case DataType::FLOAT32: {
      return llvm::Type::getFloatTy(GetLLVMContext());
    }
    case DataType::INT32: {
      return llvm::Type::getInt32Ty(GetLLVMContext());
    }
    default: {
      HLCHECK(0 && "Unhandled Type");
    }
  }
  return nullptr;
}

const std::string& GenericLLVMIRCodeGen::SNTypeToRTLibFuncSuffix(DataType dt) {
  static const std::unordered_map<DataType, std::string> suffixes = {
      {DataType::FLOAT32, "_f32"},
      {DataType::INT32, "_i32"},
      {DataType::INVALID, "_inv"}};
  if (auto kv = suffixes.find(dt); kv != suffixes.end()) {
    return kv->second;
  }
  return suffixes.find(DataType::INVALID)->second;
}

const std::string& GenericLLVMIRCodeGen::DataFormatToRTLibFuncSuffix(
    DataFormat df) {
  static const std::unordered_map<DataFormat, std::string> suffixes = {
      {DataFormat::NHWC, "_nhwc"},
      {DataFormat::NCHW, "_nchw"},
      {DataFormat::INVALID, ""}};
  if (auto kv = suffixes.find(df); kv != suffixes.end()) {
    return kv->second;
  }
  return suffixes.find(DataFormat::INVALID)->second;
}

llvm::Type* GenericLLVMIRCodeGen::TensorTypeToLLVMType(const halo::Type& type,
                                                       bool as_pointer) {
  llvm::Type* elem_type = SNTypeToLLVMType(type.GetDataType());
  bool use_vector = type.GetTotalNumOfElements() <= GetMaxVectorSize();

  llvm::Type* ret = use_vector ? static_cast<llvm::Type*>(llvm::VectorType::get(
                                     elem_type, type.GetTotalNumOfElements()))
                               : static_cast<llvm::Type*>(llvm::ArrayType::get(
                                     elem_type, type.GetTotalNumOfElements()));

  if (as_pointer) {
    ret = ret->getPointerTo();
  }
  return ret;
}

llvm::BasicBlock* GenericLLVMIRCodeGen::GetLLVMBasicBlock(Instruction* inst) {
  llvm::Value* value = ir_mapping_[*inst->GetParent()];
  return llvm::dyn_cast<llvm::BasicBlock>(value); // NOLINT
}

llvm::Value* GenericLLVMIRCodeGen::AllocateLLVMBuffer(
    llvm::IRBuilder<>* ir_builder, const Def& def) {
  constexpr int64_t stack_threshold =
      128; // std::numeric_limits<int64_t>::max();
  bool use_stack = def.GetType().GetTotalNumOfElements() <= stack_threshold;
  return AllocateLLVMBuffer(ir_builder, def, use_stack);
}

llvm::Value* GenericLLVMIRCodeGen::AllocateLLVMBuffer(
    llvm::IRBuilder<>* ir_builder, const Def& def, bool on_stack) {
  if (on_stack) {
    llvm::Value* buf =
        ir_builder->CreateAlloca(TensorTypeToLLVMType(def.GetType(), false),
                                 nullptr, def.GetOwner()->GetName());
    return buf;
  }

  auto type = TensorTypeToLLVMType(def.GetType(), false);
  llvm::GlobalVariable* gv =
      new llvm::GlobalVariable(*llvm_module_, type, false,
                               llvm::GlobalValue::LinkageTypes::InternalLinkage,
                               nullptr, def.GetOwner()->GetName());
  gv->setInitializer(llvm::Constant::getNullValue(type));
  return gv;
}

bool GenericLLVMIRCodeGen::RunOnModule(Module* module) {
  ctx_ = &module->GetGlobalContext();
  if (target_machine_ == nullptr) {
    target_machine_ = InitTargetMachine();
    HLCHECK(target_machine_);
  }

  llvm_module_ =
      llvm::make_unique<llvm::Module>(module->GetName(), GetLLVMContext());
  llvm_module_->setDataLayout(target_machine_->createDataLayout());
  llvm_module_->setTargetTriple(target_machine_->getTargetTriple().getTriple());
  for (auto& func : *module) {
    RunOnFunction(*func);
  }
  LinkRuntimeLib();

  // Verify LLVM Module.
  std::string error;
  llvm::raw_string_ostream err_os(error);
  if (llvm::verifyModule(*llvm_module_)) {
    LOG(ERROR) << "Incorrect LLVM module" << err_os.str();
    return false;
  }

  module->GetGlobalContext().GetCodeGenObject().SetLLVMModule(
      std::move(llvm_module_));
  return false;
}

void GenericLLVMIRCodeGen::RunOnFunction(Function& function) {
  bool va_arg = false;

  std::vector<llvm::Type*> arg_types;

  // Populate input arg types.
  for (const auto& arg : function.Args()) {
    arg_types.push_back(
        TensorTypeToLLVMType(arg->GetResultType(0), true /* as_pointer */));
  }

  // Populate output arg types.
  Instruction* return_inst = function.GetReturnInst();
  HLCHECK(return_inst && "No Return Instruction found");
  for (auto& op : return_inst->GetOperands()) {
    arg_types.push_back(TensorTypeToLLVMType(op.GetType(), true));
  }

  llvm::FunctionType* func_ty = llvm::FunctionType::get(
      llvm::Type::getVoidTy(GetLLVMContext()), arg_types, va_arg);
  llvm::Function::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;
  llvm::Function* llvm_func = llvm::Function::Create(
      func_ty, linkage, function.GetName(), llvm_module_.get());
  // TODO(unknown): control function attribute via command line options or allow
  // sub-target to overide the behavior.
  llvm_func->addFnAttr(llvm::Attribute::NoUnwind);
  ir_mapping_[function] = llvm_func;

  llvm_func->setCallingConv(llvm::CallingConv::C);

  /// Setup argument attribute and name.
  size_t idx = 0;
  for (const auto& arg : function.Args()) {
    llvm::Argument* llvm_arg = llvm_func->args().begin() + idx;
    llvm_arg->setName(arg->GetName());
    llvm_arg->addAttr(llvm::Attribute::ReadOnly);
    ir_mapping_[*arg] = llvm_arg;
    ++idx;
  }

  size_t i = 0;
  for (auto& op : return_inst->GetOperands()) {
    llvm::Argument* llvm_arg = llvm_func->args().begin() + idx;
    llvm_arg->setName("out_" + op.GetDef()->GetName());
    ir_mapping_[Def(return_inst, i++)] = llvm_arg;
    ++idx;
  }

  for (auto& constant : function.Constants()) {
    RunOnConstant(*constant);
  }

  for (auto& bb : function) {
    RunOnBasicBlock(llvm_func, *bb);
  }

  // Emit return.
  llvm::BasicBlock& last_bb = llvm_func->back();
  llvm::IRBuilder<> ir_builder(&last_bb);
  ir_builder.CreateRetVoid();
} // namespace halo

llvm::CallInst* GenericLLVMIRCodeGen::CreateCall(
    llvm::FunctionCallee* callee, llvm::ArrayRef<llvm::Value*> args) {
  auto func = current_llvm_builder_->CreateCall(*callee, args);
  func->setCallingConv(llvm::CallingConv::C);
  func->setDoesNotThrow();
  return func;
}

void GenericLLVMIRCodeGen::RunOnConstant(Constant& constant) {
  const auto& sn_ty = constant.GetResultType(0);
  bool use_vector = sn_ty.GetTotalNumOfElements() <= GetMaxVectorSize();
  llvm::Constant* cv = nullptr;
  switch (sn_ty.GetDataType()) {
    case DataType::FLOAT32: {
      llvm::ArrayRef<float> data(constant.GetDataPtr<float>(),
                                 sn_ty.GetTotalNumOfElements());
      cv = use_vector
               ? llvm::ConstantDataVector::get(llvm_module_->getContext(), data)
               : llvm::ConstantDataArray::get(llvm_module_->getContext(), data);
      break;
    }
    case DataType::INT64: {
      llvm::ArrayRef<uint64_t> data(constant.GetDataPtr<uint64_t>(),
                                    sn_ty.GetTotalNumOfElements());
      cv = llvm::ConstantDataVector::get(llvm_module_->getContext(), data);
      break;
    }
    case DataType::UINT32:
    case DataType::INT32: {
      llvm::ArrayRef<uint32_t> data(constant.GetDataPtr<uint32_t>(),
                                    sn_ty.GetTotalNumOfElements());
      cv = llvm::ConstantDataArray::get(llvm_module_->getContext(), data);
      break;
    }
    case DataType::INT16:
    case DataType::UINT16: {
      llvm::ArrayRef<uint16_t> data(constant.GetDataPtr<uint16_t>(),
                                    sn_ty.GetTotalNumOfElements());
      cv = llvm::ConstantDataArray::get(llvm_module_->getContext(), data);
      break;
    }
    case DataType::INT8:
    case DataType::UINT8: {
      llvm::ArrayRef<uint8_t> data(constant.GetDataPtr<uint8_t>(),
                                   sn_ty.GetTotalNumOfElements());
      cv = llvm::ConstantDataArray::get(llvm_module_->getContext(), data);
      break;
    }
    default: {
      HLCHECK(0 && "Unsupported type");
    }
  }

  if (cv == nullptr) {
    HLCHECK(0);
    return;
  }

  auto v = llvm_module_->getOrInsertGlobal(
      NormalizeVariableName(constant.GetName()), cv->getType());
  llvm::GlobalVariable* gv = llvm::dyn_cast<llvm::GlobalVariable>(v);
  HLCHECK(gv);
  if (gv != nullptr) {
    gv->setLinkage(constant_data_storage_ ==
                           ConstantDataStorage::DefinedAsStatic
                       ? llvm::GlobalValue::LinkageTypes::InternalLinkage
                       : llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    if (constant_data_storage_ != ConstantDataStorage::DeclaredAsExternal) {
      gv->setSection("data_" + std::to_string(constant.GetId()));
      gv->setInitializer(cv);
    }
  }
  ir_mapping_[constant] = gv;
}

void GenericLLVMIRCodeGen::RunOnBasicBlock(llvm::Function* llvm_func,
                                           BasicBlock& bb) {
  llvm::BasicBlock* llvm_bb =
      llvm::BasicBlock::Create(GetLLVMContext(), bb.GetName(), llvm_func);
  ir_mapping_[bb] = llvm_bb;
  llvm::IRBuilder<> ir_builder(llvm_bb);
  // current_llvm_builder_ is only available in thihs function.
  current_llvm_builder_ = &ir_builder;

  for (auto& inst : bb) {
    RunOnBaseInstruction(inst.get());
  }
  current_llvm_builder_ = nullptr;
}

std::string GenericLLVMIRCodeGen::GetRuntimeLibDir() const {
  constexpr int n = 32;
  llvm::SmallVector<char, n> path(ctx_->GetBasePath().begin(),
                                  ctx_->GetBasePath().end());
  llvm::sys::path::append(path, "runtime", "lib");
  return std::string(path.begin(), path.end());
}

std::string GenericLLVMIRCodeGen::GetRuntimeLibPath() const {
  std::string path = GetRuntimeLibDir() +
                     llvm::sys::path::get_separator().str() + RuntimeLibName;
  HLCHECK(llvm::sys::fs::exists(path));
  return path;
}

std::string GenericLLVMIRCodeGen::GetRTLibFuncName(const Instruction& inst,
                                                   DataType data_type,
                                                   DataFormat data_format) {
  const std::string& basename = CodeGen::GetRTLibFuncName(inst);
  const std::string& suffix = SNTypeToRTLibFuncSuffix(data_type);
  const std::string& suffix2 = DataFormatToRTLibFuncSuffix(data_format);
  return basename + suffix + suffix2;
}

void GenericLLVMIRCodeGen::LinkRuntimeLib() {
  auto file_buf = llvm::MemoryBuffer::getFile(GetRuntimeLibPath(), -1, false);
  std::error_code ec = file_buf.getError();
  if (ec) {
    llvm::errs() << "error opening '" << GetRuntimeLibPath()
                 << "': " << ec.message();
  }
  llvm::Error err = llvm::Error::success();
  llvm::object::Archive archive(file_buf.get()->getMemBufferRef(), err);
  if (err) {
    auto ec = llvm::errorToErrorCode(std::move(err));
    llvm::errs() << "error opening '" << GetRuntimeLibPath()
                 << "': " << ec.message();
  }

  auto& llvm_ctx = llvm_module_->getContext();
  for (auto& c : archive.children(err)) {
    auto buf = c.getMemoryBufferRef();
    if (!buf) {
      llvm::errs() << "error reading object\n";
      HLCHECK(false);
    }
    llvm::SMDiagnostic diag_err;
    std::unique_ptr<llvm::Module> lib_module =
        llvm::parseIR(buf.get(), diag_err, llvm_ctx);
    if (lib_module == nullptr) {
      diag_err.print("Halo", llvm::errs());
      HLCHECK(false && "Failed to load runtime library");
    }

    // Force lib's data layout and triple the same as dst module and remove
    // target specific attributes.
    lib_module->setDataLayout(llvm_module_->getDataLayout());
    lib_module->setTargetTriple(llvm_module_->getTargetTriple());
    // call setFunctionAttributes(CPU, Features, M);
    for (auto& func : lib_module->functions()) {
      func.removeFnAttr("target-features");
      func.removeFnAttr("target-cpu");
    }

    // Make all gvs in library internal.
    auto callback = [](llvm::Module& m, const llvm::StringSet<>& gvs) {
      llvm::internalizeModule(m,
                              [](const llvm::GlobalValue& gv) { return true; });
    };
    llvm::Linker::linkModules(*llvm_module_, std::move(lib_module),
                              llvm::Linker::Flags::LinkOnlyNeeded, callback);
  }
  if (err) {
    llvm::errs() << "Error getting child";
  }
}

void GenericLLVMIRCodeGen::RunOnBaseInstruction(Instruction* inst) {
  switch (inst->GetOpCode()) {
    case OpCode::ADD:
    case OpCode::SUB:
    case OpCode::DIV:
    case OpCode::MUL: {
      RunOnMathBinaryInstruction(inst);
      break;
    }
    case OpCode::CEIL:
    case OpCode::FLOOR:
    case OpCode::ERF:
    case OpCode::RSQRT:
    case OpCode::SQRT: {
      RunOnMathUnaryInstruction(inst);
      break;
    }
    default: {
      CodeGen::RunOnBaseInstruction(inst);
      break;
    }
  }
}

GenericLLVMIRWriter::GenericLLVMIRWriter(const std::string& name,
                                         std::ostream& os, bool bitcode_format)
    : CodeWriter(name, os), bitcode_format_(bitcode_format) {}

GenericLLVMIRWriter::GenericLLVMIRWriter(std::ostream& os, bool bitcode_format)
    : GenericLLVMIRWriter("Generic LLVM IR Writer", os, bitcode_format) {}

GenericLLVMIRWriter::GenericLLVMIRWriter()
    : GenericLLVMIRWriter(std::cout, false) {}

bool GenericLLVMIRWriter::RunOnModule(Module* module) {
  auto llvm_module =
      module->GetGlobalContext().GetCodeGenObject().GetLLVMModule();
  llvm::raw_os_ostream llvm_os(os_);
  if (bitcode_format_) {
    llvm::WriteBitcodeToFile(*llvm_module, llvm_os);
  } else {
    llvm_module->print(llvm_os, nullptr);
  }
  return false;
}

} // namespace halo
