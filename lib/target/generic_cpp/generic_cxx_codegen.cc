//===- generic_cxx_codegen.cc ---------------------------------------------===//
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

#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

#include <cstddef>
#include <sstream>

#include "halo/api/halo_data.h"
#include "halo/lib/framework/global_context.h"
#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/ir/instruction.h"
#include "halo/lib/mm/memory_analyzer.h"
#include "halo/lib/target/codegen.h"
#include "halo/lib/target/codegen_object.h"

namespace halo {

std::string CXXType::Str(bool use_array_decl) const {
  std::string str;
  if (is_const) {
    str += "const ";
  }
  str += name;
  if (is_pointer) {
    if (use_array_decl) {
      // use array style like const float a[1*2*3].
    } else {
      str += "*";
    }
  }
  return str;
}

void CXXValue::Reset() { name2id.clear(); }

CXXValue::CXXValue(const std::string& name, const CXXType& type)
    : str_id(""), type(type) {
  this->name = CodeGen::NormalizeVariableName(name);
  if (name2id.count(name) == 0) {
    name2id[name] = name2id.size();
  }
  id = name2id[name];
}

GenericCXXCodeGen::GenericCXXCodeGen(std::ostream& os, std::ostream& header_os)
    : CodeGen("Generic CXX Compilation"),
      os_(os),
      header_os_(header_os),
      dynamic_check_os_(std::cout) {
  CXXValue::Reset();
}

GenericCXXCodeGen::GenericCXXCodeGen(std::ostream& os, std::ostream& header_os,
                                     std::ostream& dynamic_check_os,
                                     const Opts& opts)
    : CodeGen("Generic CXX Compilation"),
      os_(os),
      header_os_(header_os),
      dynamic_check_os_(dynamic_check_os),
      opts_(opts) {
  CXXValue::Reset();
}

GenericCXXCodeGen::~GenericCXXCodeGen() = default;

static const std::string& GetIncludeFile(CodeGen::API api) {
  static const std::unordered_map<CodeGen::API, std::string> headers{
      {CodeGen::API::HALO_RT, ""}, {CodeGen::API::ODLA_05, "<ODLA/odla.h>"}};
  auto it = headers.find(api);
  HLCHECK(it != headers.end());
  if (it != headers.end()) {
    return it->second;
  }
  return headers.begin()->second;
}

static void EmitBanner(std::ostream* os, std::ostream* header_os,
                       CodeGen::API api) {
  static const std::string banner(
      "//===- Halo Compiler Generated File "
      "--------------------------------===//\n\n");
  *os << banner;
  if (header_os != nullptr) {
    *header_os << banner;
  }
  *os << "#include " << GetIncludeFile(api) << "\n\n";
}

bool GenericCXXCodeGen::RunOnModule(Module* module) {
  memory_analyzer_ = std::make_unique<MemoryAnalyzer>(*module);
  Function* entry_func = nullptr;
  EmitBanner(&os_, &header_os_, GetAPI());
  for (auto& func : *module) {
    if (func->IsEntryFunction()) {
      entry_func = func.get();
    } else {
      RunOnFunction(*func);
    }
  }

  if (entry_func != nullptr) {
    if (module->Functions().size() > 1) {
      RunOnHostFunction(*entry_func);
    } else {
      RunOnFunction(*entry_func);
    }
  }

  if (opts_.print_mem_stats) {
    std::cout << "Total Read-only Memory: "
              << memory_analyzer_->GetWeightsSize() << " bytes\n";
    std::cout << "Total Writable Memory: "
              << memory_analyzer_->GetNonWeightsSize() << " bytes\n";
    std::cout << "Total Peak Memory: " << memory_analyzer_->GetPeak()
              << " bytes\n";
  }
  return false;
}

CXXValue GenericCXXCodeGen::AllocateBuffer(const Def& def, bool on_stack) {
  return CXXValue("undef", CXXType(""));
}

CXXType GenericCXXCodeGen::SNTypeToCXXType(DataType dt) {
  switch (dt) {
    case DataType::INT8:
      return (CXXType("signed char"));
    case DataType::UINT8: {
      return (CXXType("unsigned char"));
    }
    case DataType::INT16:
      return (CXXType("short"));
    case DataType::UINT16: {
      return (CXXType("unsigned short"));
    }
    case DataType::FLOAT32: {
      return (CXXType("float"));
    }
    case DataType::INT32: {
      return (CXXType("int"));
    }
    case DataType::UINT32: {
      return (CXXType("unsigned int"));
    }
    case DataType::INT64: {
      return (CXXType("int64_t"));
    }
    default: {
      HLCHECK(0 && "Unhandled Type");
    }
  }
  return CXXType();
}

CXXType GenericCXXCodeGen::TensorTypeToCXXType(const halo::Type& type,
                                               bool is_const) {
  auto ty = SNTypeToCXXType(type.GetDataType());
  ty.is_const = is_const;
  return ty;
}

std::string GenericCXXCodeGen::GetFunctionDecl(const Function& func,
                                               const Instruction& ret_inst,
                                               bool with_func_name,
                                               bool with_type,
                                               bool public_function) {
  const static std::string inference_func_decl =
      "void model_run(int num_inputs, const void* inputs[],"
      "int num_outputs, void* outputs[], int batch_size)";
  if (opts_.emit_inference_func_sig && func.IsEntryFunction() &&
      public_function) {
    return inference_func_decl;
  }

  std::ostringstream ss;
  bool is_sub = !func.IsEntryFunction();
  if (is_sub) {
    ss << "static ";
  }
  if (with_func_name) {
    ss << "void " << func.GetName();
  }
  ss << "(";
  if (is_sub) {
    ss << "odla_device device, odla_values inputs, odla_values outputs";
  }
  bool is_first = true;
  for (auto& arg : func.Args()) {
    const auto& type = arg->GetResultType();
    if (ir_mapping_.find(*arg) == ir_mapping_.end()) {
      CXXValue cv(arg->GetName(), TensorTypeToCXXType(type, true));
      ir_mapping_[*arg] = cv;
    }
    const auto& cv = ir_mapping_[*arg];
    if (!is_sub) {
      if (with_type) {
        ss << (is_first ? "" : ", ") << cv.type.Str(true) << " " << cv.name
           << "[";
        auto shape = type.GetDimSizes();
        if (type.IsDynamicBatch()) {
          shape[0] = 1;
        }
        ss << Join(shape, '*') << "]";
      } else {
        ss << (is_first ? "" : ", ") << cv.name;
      }
    }
    is_first = false;
  }
  for (auto& out : ret_inst.GetOperands()) {
    const auto& type = out.GetType();
    if (ir_mapping_.find(out) == ir_mapping_.end()) {
      CXXValue cv(out.GetDef()->GetName(), TensorTypeToCXXType(type, false));
      ir_mapping_[out] = cv;
    }
    const auto& cv = ir_mapping_[out];
    if (!is_sub) {
      if (with_type) {
        ss << (is_first ? "" : ", ")
           << TensorTypeToCXXType(type, false).Str(true) << " out_" + cv.name
           << "[";
        auto shape = type.GetDimSizes();
        if (type.IsDynamicBatch()) {
          shape[0] = 1;
        }
        ss << Join(shape, '*') << "]";
      } else {
        ss << (is_first ? "" : ", ") << " out_" + cv.name;
      }
      is_first = false;
    }
  }
  if (opts_.emit_dynamic_batch) {
    ss << ", int batch_size";
  }
  ss << ")";
  return ss.str();
}

const std::string& GenericCXXCodeGen::EmitReturnType(bool auto_type,
                                                     bool single_value) {
  const static std::string single_c_type{"odla_value"};
  const static std::string multiple_c_type{"odla_values"};
  const static std::string cxx_type{"auto"};
  return auto_type ? cxx_type : single_value ? single_c_type : multiple_c_type;
}

std::string GenericCXXCodeGen::EmitLValue(const std::string& name) const {
  return EmitReturnType(opts_.dialect == Dialect::CXX_11, true) + " " + name;
}

std::string GenericCXXCodeGen::EmitLValues(const std::string& name) const {
  return EmitReturnType(opts_.dialect == Dialect::CXX_11, false) + " " + name;
}

const std::string& GenericCXXCodeGen::EmitNull() const noexcept {
  static const std::string cxx_null{"nullptr"};
  static const std::string c_null{"NULL"};
  return opts_.dialect == Dialect::CXX_11 ? cxx_null : c_null;
}

std::string GenericCXXCodeGen::EmitShape(const halo::Type& type) {
  std::ostringstream ss;
  if (opts_.dialect == Dialect::C99) {
    ss << "(odla_value_shape)";
  }
  ss << "{.size = " << type.GetNumOfDims() << ", .dims={";
  ss << GenericCXXCodeGen::Join(type.GetDimSizes());
  ss << "}}";
  return ss.str();
}

std::string GenericCXXCodeGen::GetODLAType(DataType type) const noexcept {
  switch (type) {
    case DataType::FLOAT32: {
      return "ODLA_FLOAT32";
    }
    case DataType::INT32: {
      return "ODLA_INT32";
    }
    case DataType::UINT32: {
      return "ODLA_UINT32";
    }
    case DataType::INT64: {
      return "ODLA_INT64";
    }
    default: {
      return "INVALID";
    }
  }
}

std::string GenericCXXCodeGen::EmitType(const halo::Type& type) {
  std::string str = GetODLAType(type.GetDataType());
  str += ", " + EmitShape(type);
  if (opts_.dialect == Dialect::C99) {
    return "(odla_value_type){" + str + "}";
  }
  return "{" + str + "}";
}

static std::string DeclAsExtern(const std::string& str) {
  return "extern \"C\" {\n" + str + ";\n};\n";
}

std::string GenericCXXCodeGen::GenerateTestFunc(const Function& func,
                                                const std::string& func_decl,
                                                const Instruction& ret_inst) {
  std::ostringstream oss;
  oss << "#include \"unittests.h\"\n\n";

  auto convert_data_type = [](DataType dtype) {
    std::string data_type_str;
    switch (dtype) {
      case DataType::FLOAT32:
        data_type_str = "float";
        break;
      case DataType::INT32:
        data_type_str = "int32_t";
        break;
      case DataType::UINT32:
        data_type_str = "uint32_t";
        break;
      case DataType::INT64:
        data_type_str = "int64_t";
        break;
      default:
        HLCHECK(0);
    }
    return data_type_str;
  };

  if (func.IsEntryFunction()) {
    if (opts_.dialect == Dialect::CXX_11) {
      oss << DeclAsExtern(func_decl);
    }
    oss << "int main(int argc, char** argv) {\n";
    oss << "  std::string test_case_dir = argv[argc - 1];\n";
    oss << "  std::string device_name = argv[argc - 2];\n";
    oss << "  int data_set_id = atoi(argv[argc - 3]);\n";
    oss << "  double thre = strtod(argv[argc - 4], NULL);\n\n";
    oss << "  std::vector<const void*> inputs;\n";
    oss << "  std::vector<const void*> output_refs;\n";
    oss << "  std::vector<void*> outputs;\n\n";
    oss << "  UnitTests unittests;\n";
    int32_t i = 0;
    std::string data_type;
    // load input data
    for (auto& arg : func.Args()) {
      auto& type = arg->GetResultType();
      data_type.clear();
      data_type = convert_data_type(type.GetDataType());
      oss << "  std::vector<" << data_type << "> in_" << i
          << " = unittests.LoadInData<" << data_type << ">("
          << "test_case_dir, data_set_id, " << i << ");\n";
      oss << "  inputs.push_back(in_" << i << ".data());\n";
      i++;
    }

    i = 0;
    // load reference data and declare output
    for (auto& out : ret_inst.GetOperands()) {
      const auto& type = out.GetType();
      const auto elem_nums = type.GetTotalNumOfElements();
      data_type.clear();
      data_type = convert_data_type(type.GetDataType());
      oss << "  std::vector<" << data_type << "> out_ref_" << i
          << " = unittests.LoadOutData<" << data_type << ">("
          << "test_case_dir, data_set_id, " << i << ");\n";
      oss << "  output_refs.push_back(out_ref_" << i << ".data());\n";
      oss << "  " << data_type << " out_" << i << "[" << elem_nums
          << "] = {};\n";
      oss << "  outputs.push_back(out_" << i++ << ");\n";
    }

    // start time
    oss << "#ifdef TIME_PERF\n";
    oss << "  unittests.TimeBegin();\n";
    oss << "#endif\n";
    // execute func
    oss << "  model_run(" << func.Args().size() << ", inputs.data(), ";
    oss << ret_inst.GetOperands().size() << ", outputs.data(), 1);\n";

    oss << "  long long times = 0;\n";
    // end time
    oss << "#ifdef TIME_PERF\n";
    oss << "  unittests.TimeStop();\n";
    // elapsed time
    oss << "  times = unittests.GetDuration();\n";
    oss << "#endif\n";
    // verify output data
    oss << "  unittests.CheckResult<" << data_type << ">("
        << ret_inst.GetOperands().size() << ", outputs.data()"
        << ", output_refs.data()"
        << ", test_case_dir, device_name, times, thre);\n";
    oss << "  return 0;\n}\n";
  }
  return oss.str();
}

void GenericCXXCodeGen::RunOnHostFunction(Function& function) {
  HLCHECK(function.Constants().empty());
  Instruction* return_inst = function.GetReturnInst();
  HLCHECK(return_inst && "No Return Instruction found");

  auto func_decl = GetFunctionDecl(function, *return_inst, true, true, true);

  if (opts_.dialect == Dialect::CXX_11) {
    os_ << DeclAsExtern(func_decl);
  }

  os_ << func_decl << " {\n";
  os_ << "  static odla_device trt_dev;\n";
  os_ << "  static odla_device x86_dev;\n";
  os_ << "  static odla_device host_dev;\n";
  os_ << "  odla_AllocateDevice(" << EmitNull()
      << ", ODLA_DEVICE_DEFAULT, &host_dev);\n";
  os_ << "  odla_AllocateDevice(" << EmitNull()
      << ", ODLA_DEVICE_NVIDIA_TENSORRT, "
         "&trt_dev);\n";
  os_ << "  odla_AllocateDevice(" << EmitNull()
      << ", ODLA_DEVICE_INTEL_X86, &x86_dev);\n";
  os_ << "  odla_SetCurrentDevice(host_dev);\n";

  std::vector<std::string> created_val_names;
  int index = 0;
  for (auto& arg : function.Args()) {
    auto& type = arg->GetResultType();
    CXXValue v("in_" + arg->GetName(), TensorTypeToCXXType(type, true));

    ir_mapping_[*arg] = v;
    std::string arg_name = opts_.emit_inference_func_sig
                               ? "inputs[" + std::to_string(index++) + "]"
                               : v.name.substr(3);
    EmitODLACall(v, "odla_CreateValue", arg->GetResultType());
    os_ << "  odla_SetValueData(" << Join(v.name, arg_name) << ");\n";
    created_val_names.push_back(v.name);
  }
  for (auto& bb : function) {
    RunOnBasicBlock(*bb);
  }

  index = 0;
  for (auto& op : return_inst->GetOperands()) {
    auto& cv = ir_mapping_[op];
    std::string arg_name = opts_.emit_inference_func_sig
                               ? "outputs[" + std::to_string(index++) + "]"
                               : "out_" + cv.name;
    os_ << "  odla_GetValueData(" << Join(cv.name, arg_name) << ");\n";
  }

  for (const auto& name : created_val_names) {
    os_ << "  odla_ReleaseValue(" << name << ");\n";
  }

  os_ << "}\n";
}

void GenericCXXCodeGen::RunOnFunction(Function& function) {
  for (auto& constant : function.Constants()) {
    RunOnConstant(*constant, true);
  }

  if (function.empty() || (function.BasicBlocks().size() == 1 &&
                           function.BasicBlocks().front()->empty())) {
    return;
  }

  Instruction* return_inst = function.GetReturnInst();
  HLCHECK(return_inst && "No Return Instruction found");

  bool is_compile_mode = opts_.exec_mode == CodeGen::ExecMode::Compile;

  // Emit a separate computation builder function or not.
  bool emit_builder_func = function.GetParent()->Functions().size() == 1;

  std::string helper_func_name = is_compile_mode
                                     ? function.GetName() + "_helper"
                                     : "run_" + function.GetName();
  // Emit function for launching computation.
  auto func_decl = GetFunctionDecl(function, *return_inst, true, true, true);

  // contents in oss will be write to c file and header file.
  std::ostringstream oss;
  bool emit_triton_style =
      (function.IsEntryFunction() && opts_.emit_inference_func_sig);
  const std::string init_func_name =
      emit_triton_style ? "model_init" : function.GetName() + "_init";
  const std::string fini_func_name =
      emit_triton_style ? "model_fini" : function.GetName() + "_fini";

  if (function.IsEntryFunction()) {
    if (opts_.dialect == Dialect::CXX_11) {
      oss << "extern \"C\" {\n";
    }
    oss << "  " << func_decl << ";\n";
    oss << "void " << init_func_name << "();\n";
    oss << "void " << fini_func_name << "();\n";
    if (opts_.dialect == Dialect::CXX_11) {
      oss << "};\n";
    }
  }
  os_ << oss.str();
  header_os_ << oss.str();

  if (emit_builder_func) {
    if (is_compile_mode) {
      os_ << "static odla_computation Comp;\n";
      os_ << "static void " << helper_func_name << "() {\n";
      os_ << "  odla_CreateComputation(&Comp);\n";
      if (opts_.enable_ipu_device) {
        os_ << "bool use_ipu_model = " << opts_.use_ipu_model << ";\n";
        os_ << "int ipu_num = " << opts_.ipu_num << ";\n";
        os_ << "int batches_per_step = " << opts_.batches_per_step << ";\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_USE_SIM_MODE, "
               "(odla_item_value) &use_ipu_model);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_PROCESSOR_NUM, "
               "(odla_item_value) &ipu_num);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_BATCHES_PER_STEP, "
               "(odla_item_value) &batches_per_step);\n";
      }

      if (opts_.emit_dynamic_batch) {
        os_ << "bool is_dynamic_batch = true;\n";
        os_ << "int min_batch_size = 1;\n";
        os_ << "int max_batch_size = 8;\n";
        os_ << "int opt_batch_size = 4;\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_DYNAMIC_BATCH, "
               "(odla_item_value) &is_dynamic_batch);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_MIN_BATCH_SIZE, "
               "(odla_item_value) &min_batch_size);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_MAX_BATCH_SIZE, "
               "(odla_item_value) &max_batch_size);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_OPT_BATCH_SIZE, "
               "(odla_item_value) &opt_batch_size);\n";
      }
    } else {
      os_ << "static void " << helper_func_name
          << GetFunctionDecl(function, *return_inst, false, true, false)
          << " {\n";
    }
  } else { // emit single function
    os_ << func_decl << " {\n";
    if (!function.IsEntryFunction()) {
      os_ << "  odla_SetCurrentDevice(device);";
    }

    if (is_compile_mode) {
      os_ << "  static odla_computation Comp;\n";
      os_ << "  if (Comp == " << EmitNull() << ") {\n";
      os_ << "    odla_CreateComputation(&Comp);\n";
      if (opts_.enable_ipu_device) {
        os_ << "bool use_ipu_model = " << opts_.use_ipu_model << ";\n";
        os_ << "int ipu_num = " << opts_.ipu_num << ";\n";
        os_ << "int batches_per_step = " << opts_.batches_per_step << ";\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_USE_SIM_MODE, "
               "(odla_item_value) &use_ipu_model);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_PROCESSOR_NUM, "
               "(odla_item_value) &ipu_num);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_BATCHES_PER_STEP, "
               "(odla_item_value) &batches_per_step);\n";
      }

      if (opts_.emit_dynamic_batch) {
        os_ << "bool is_dynamic_batch = true;\n";
        os_ << "int min_batch_size = 1;\n";
        os_ << "int max_batch_size = 8;\n";
        os_ << "int opt_batch_size = 4;\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_DYNAMIC_BATCH, "
               "(odla_item_value) &is_dynamic_batch);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_MIN_BATCH_SIZE, "
               "(odla_item_value) &min_batch_size);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_MAX_BATCH_SIZE, "
               "(odla_item_value) &max_batch_size);\n";
        os_ << "odla_SetComputationItem(Comp, ODLA_OPT_BATCH_SIZE, "
               "(odla_item_value) &opt_batch_size);\n";
      }
    }
  }

  // Emit wrappers for arguments.
  for (auto& arg : function.Args()) {
    auto& type = arg->GetResultType();
    auto arg_name = NormalizeVariableName(arg->GetName());
    CXXValue v(is_compile_mode ? arg->GetName() : "in_" + arg->GetName(),
               TensorTypeToCXXType(type, true));
    if (is_compile_mode) {
      EmitODLACall<2, false>(
          v, "odla_CreateArgument", type,
          "(const odla_value_id)(\"" + arg->GetName() + "\")");
    } else {
      EmitODLACall(v, "odla_CreateValue", type);
      os_ << "  odla_SetValueData(" << v.name << ", " << v.name.substr(3)
          << ");\n";
    }
    ir_mapping_[*arg] = v;
  }
  // Emit wrappers for constants.
  for (auto& constant : function.Constants()) {
    RunOnConstant(*constant, false);
  }

  for (auto& bb : function) {
    RunOnBasicBlock(*bb);
  }

  os_ << "}\n"; // End of computation build function.

  if (opts_.check_model) {
    dynamic_check_os_ << GenerateTestFunc(function, func_decl, *return_inst);
  }

  if (emit_builder_func) {
    // Emit function for launching computation.
    if (opts_.exec_mode == CodeGen::ExecMode::Compile) {
      if (function.IsEntryFunction()) {
        os_ << "void " << fini_func_name << "(){\n";
        os_ << "  odla_DestroyComputation(Comp);\n";
        os_ << "}\n";

        os_ << "void " << init_func_name << "(){\n";
      } else {
        os_ << GetFunctionDecl(function, *return_inst, true, true, true)
            << " {\n";
      }
      os_ << "  if (Comp == " << EmitNull() << ") { " << helper_func_name
          << "(); }\n";
      os_ << "}\n";
    }
    if (function.IsEntryFunction()) {
      os_ << GetFunctionDecl(function, *return_inst, true, true, true)
          << " {\n";
      if (opts_.exec_mode == CodeGen::ExecMode::Compile) {
        os_ << "  " << init_func_name << "();\n";
      }
    }
    if (opts_.exec_mode == CodeGen::ExecMode::Interpret) {
      os_ << "  " << helper_func_name;
      if (opts_.emit_inference_func_sig) {
        os_ << "(";
        bool is_first = true;
        int i = 0;
        for (const auto& arg : function.Args()) {
          os_ << (is_first ? "" : ", ");
          CXXType ty = TensorTypeToCXXType(arg->GetResultType(), true);
          os_ << "(" << ty.Str(false) << ")inputs[" << i << "]";
          is_first = false;
          ++i;
        }
        for (int i = 0, e = return_inst->GetNumOfOperands(); i < e; ++i) {
          os_ << (is_first ? "" : ", ");
          CXXType ty =
              TensorTypeToCXXType(return_inst->GetOperand(i).GetType(), false);
          os_ << "(" << ty.Str(false) << ")outputs[" << i << "]";
          is_first = false;
        }
        os_ << ")";
      } else {
        os_ << GetFunctionDecl(function, *return_inst, false, false, false);
      }
      os_ << ";\n";
      os_ << "}\n";
      return;
    }
  }

  if (opts_.exec_mode == CodeGen::ExecMode::Compile) {
    os_ << "  static odla_context Ctx;\n";
    os_ << "  if (Ctx == " << EmitNull()
        << ") {  odla_CreateContext(&Ctx); };\n";
    if (opts_.emit_dynamic_batch) {
      os_ << "odla_SetContextItem(Ctx, ODLA_RUN_BATCH_SIZE, "
             "(odla_item_value) &batch_size);\n";
    }
  }
  int index = 0;
  bool is_sub = !function.IsEntryFunction();
  for (auto& arg : function.Args()) {
    auto& cv = ir_mapping_[*arg];
    std::string arg_name = (opts_.emit_inference_func_sig || is_sub)
                               ? (is_sub ? "inputs.values[" : "inputs[") +
                                     std::to_string(index++) + "]"
                               : cv.name;
    os_ << (is_sub ? "  odla_BindValueToArgumentById("
                   : "  odla_BindToArgumentById(")
        << Join("(const odla_value_id)\"" + arg->GetName() + "\"", arg_name,
                "Ctx")
        << ");\n";
  }
  index = 0;
  // Pre-launch binding.
  for (auto& op : return_inst->GetOperands()) {
    auto& cv = ir_mapping_[op];
    std::string arg_name = (opts_.emit_inference_func_sig || is_sub)
                               ? (is_sub ? "outputs.values[" : "outputs[") +
                                     std::to_string(index++) + "]"
                               : "out_" + cv.name;
    os_ << "  odla_Bind" << (is_sub ? "Value" : "") << "ToOutputById("
        << Join("(const odla_value_id)\"" + cv.name + "\"", arg_name, "Ctx")
        << ");\n";
  }
  if (opts_.exec_mode == CodeGen::ExecMode::Compile) {
    os_ << "  odla_ExecuteComputation(Comp, Ctx, "
           "ODLA_COMPUTE_INFERENCE, "
        << EmitNull() << ");\n";
  }
  os_ << "}\n";
}

void GenericCXXCodeGen::RunOnConstant(Constant& constant, bool decl) {
  const auto& uses = constant.GetIthResultUses(0);
  bool only_used_by_reshape = true;
  for (const auto& u : uses) {
    if (!IsA<Instruction>(u.GetUse()) ||
        DynCast<Instruction>(u.GetUse())->GetOpCode() != OpCode::RESHAPE ||
        u.GetIdx() != 1) {
      only_used_by_reshape = false;
      break;
    }
  }
  if (only_used_by_reshape) {
    return;
  }

  auto& type = constant.GetResultType();
  if (decl) {
    CXXValue value(constant.GetName(), TensorTypeToCXXType(type, true));
    if (constant.IsScalarOne()) {
      os_ << "extern const " << value.type.name << " " << value.name
          << "[1];\n";
    } else {
      os_ << "extern const " << value.type.name << " " << value.name << "["
          << Join(type.GetDimSizes(), '*') << "];\n";
    }
    ir_mapping_[constant] = value;
    return;
  }
  auto ptr_name = ir_mapping_[constant].name;
  CXXValue value(constant.GetName() + "_", TensorTypeToCXXType(type, true));

  EmitODLACall(value, "odla_CreateConstant", type, ptr_name);
  ir_mapping_[constant] = value;
}

void GenericCXXCodeGen::PreRunOnInstruction(Instruction* inst) {
  if (inst->GetOpCode() != OpCode::RETURN) {
    const auto& type = inst->GetResultType();
    CXXValue ret(inst->GetName(), TensorTypeToCXXType(type, true));
    if (opts_.emit_value_init) {
      os_ << "  odla_InitValue("
          << Join("comp", ret.name, GetODLAType(type.GetDataType()),
                  EmitShape(type))
          << " );\n";
    }
  }
}

void GenericCXXCodeGen::PostRunOnInstruction(Instruction* inst) {
  if (inst->GetOpCode() == OpCode::RETURN) {
    return;
  }
  auto deads = memory_analyzer_->Executed(inst);
  if (opts_.emit_value_reset) {
    for (auto& dead : deads) {
      const auto& type = dead.GetType();
      CXXValue val(dead.GetOwner()->GetName(), TensorTypeToCXXType(type, true));
      os_ << "  odla_ReleaseValue(" << val.name << ");\n";
    }
  }
}

void GenericCXXCodeGen::RunOnBasicBlock(BasicBlock& bb) {
  for (auto& inst : bb) {
    Instruction* i = inst.get();
    PreRunOnInstruction(i);
    RunOnBaseInstruction(i);
    PostRunOnInstruction(i);
  }
}

void GenericCXXCodeGen::EmitODLAArgs(const std::vector<int32_t>& arg) {
  os_ << "(const odla_int32[])";
  os_ << '{' << Join(arg) << '}';
}

void GenericCXXCodeGen::EmitODLAArgs(const std::vector<uint32_t>& arg) {
  os_ << "(const odla_uint32[])";
  os_ << '{' << Join(arg) << '}';
}

void GenericCXXCodeGen::EmitODLAArgs(const std::vector<CXXValue>& arg) {
  os_ << "(odla_values){.size = " << arg.size() << ", .values = {";
  for (const auto& v : arg) {
    os_ << v.name << ", ";
  }
  os_ << "}}";
}

void GenericCXXCodeGen::EmitODLAArgs(const bool& arg) { os_ << (arg ? 1 : 0); }

void GenericCXXCodeGen::EmitODLAArgs(const DataType& arg) {
  os_ << GetODLAType(arg);
}

void GenericCXXCodeGen::EmitODLAArgs(const halo::Type& arg) {
  os_ << EmitType(arg);
}

void GenericCXXCodeGen::EmitODLAArgs(const CXXValue& arg) { os_ << arg.name; }
void GenericCXXCodeGen::EmitODLAArgs(const DataFormat& arg) {
  if (opts_.dialect == Dialect::CXX_11) {
    os_ << "odla_memory_layout::";
  }
  os_ << "ODLA_";
  os_ << (arg == DataFormat::NHWC ? "CHANNELS_LAST" : "CHANNELS_FIRST");
}

} // namespace halo
