//===-modelzoo_emitter.cc =================================================//
//
// Copyright (C) 2020-2021 Alibaba Group Holding Limited.
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

#include <algorithm>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace halo {

static std::vector<llvm::Record*> GetAllModels(
    const llvm::RecordKeeper& records) {
  std::vector<llvm::Record*> models = records.getAllDerivedDefinitions("Model");
  std::sort(models.begin(), models.end(),
            [](llvm::Record* lhs, llvm::Record* rhs) {
              auto op1 = lhs->getName();
              auto op2 = rhs->getName();
              return (op1.compare(op2) <= 0);
            });
  return models;
}

static llvm::raw_ostream& EmitPreProcess();

static llvm::raw_ostream& EmitPostProcess();

static llvm::raw_ostream& EmitCopyRight(llvm::StringRef name,
                                        llvm::raw_ostream& os) {
  os << "// ==== " << name
     << "====================================================\n";
  os << "// Copyright (C) 2019-2021 Alibaba Group Holding Limited.\n";
  os << "//\n";
  os << "// Licensed under the Apache License, Version 2.0(the \"License\");\n";
  os << "// you may not use this file except in compliance with the License.\n";
  os << "// You may obtain a copy of the License at\n";
  os << "//\n";
  os << "//   http://www.apache.org/licenses/LICENSE-2.0\n";
  os << "//\n";
  os << "// Unless required by applicable law or agreed to in writing, "
        "software\n";
  os << "// distributed under the License is distributed on an \"AS IS\" "
        "BASIS,\n";
  os << "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or "
        "implied.\n";
  os << "// See the License for the specific language governing permissions "
        "and\n";
  os << "// limitations under the License\n";
  os << "// "
        "======================================================================"
        "========\n";
  return os;
}

static llvm::raw_ostream& EmitMacroDef(llvm::raw_ostream& os) {
  os << "#ifndef  TEST_ITERS\n";
  os << "#define  TEST_ITERS 10\n";
  os << "#endif "
     << "\n";
  os << "#ifndef  COMPARE_ERROR\n";
  os << "#define  COMPARE_ERROR 1-e5\n";
  os << "#endif "
     << "\n\n";
  return os;
}

static llvm::raw_ostream& EmitCallFunc(int num_inputs, int num_outputs,
                                       llvm::raw_ostream& os) {
  os.indent(4);
  os << "model_run(";
  os << num_inputs << ", ";
  os << "inputs.data(), ";
  os << num_outputs << ", ";
  os << "outputs.data(), 1);\n";
  return os;
}

static llvm::raw_ostream& EmitTimePerf(int num_inputs, int num_outputs,
                                       llvm::raw_ostream& os) {
  os.indent(2);
  os << "auto begin_time = Now();\n";
  os.indent(2);
  os << " for (int i = 0; i < TEST_ITERS; i++) {\n";
  EmitCallFunc(num_inputs, num_outputs, os);
  os.indent(2);
  os << "}\n";
  os.indent(2);
  os << "auto end_time = Now();\n";
  os.indent(2);
  os << "std::cout << \"Elapse time: \" << ";
  os << "GetDuration(begin_time, end_time) / ";
  os << "TEST_ITERS << \" seconds\\n\";\n";
  os << "return 0;\n";
  return os;
}

static llvm::raw_ostream& EmitCheckResult(int num_outputs,
                                          llvm::raw_ostream& os) {
  for (int i = 0; i < num_outputs; i++) {
    os.indent(2);
    os << "bool check_";
    os << i;
    os << " = Verify(";
    os << "output_" << i << ",";
    os << "output_ref_" << i << ",";
    os << "sizeof(output_" << i << ")";
    os << " / sizeof(output_" << i << "[0])";
    os << ", COMPARE_ERROR);\n";
  }
  os << "  if (";
  for (int i = 0; i < num_outputs - 1; i++) {
    os << "check_";
    os << i;
    os << " && ";
  }
  os << "check_" << num_outputs - 1 << ") {\n";
  os << "    std::cout << \"Result verified\" << std::endl;\n";
  os << "  } else { \n";
  os << "    std::cout << \" Failed \" << std::endl;\n";
  os << "    return 1;\n";
  os << "  };\n";
  return os;
}

static llvm::raw_ostream& EmitHeaderFunc(llvm::Record* model,
                                         llvm::raw_ostream& os) {
  auto docs = model->getValueAsDef("docs_");
  llvm::StringRef model_name = docs->getValueAsString("model_name_");
  os << "#include <vector>\n";
  os << "#include \"" << model_name << "_data.h\"\n";
  os << "#include \"test_util.h\"\n";

  EmitMacroDef(os);
  return os;
}

static llvm::raw_ostream& EmitDeclFunc(llvm::Record* model,
                                       llvm::raw_ostream& os) {
  os << "extern \"C\" {\n";
  os << "void model_run( ";
  os << "int num_inputs, ";
  os << "const void* inputs[], ";
  os << "int num_outputs, ";
  os << "void* outputs[], ";
  os << "int batch_size);\n";
  os << "};\n\n";
  return os;
}

static llvm::raw_ostream& EmitMainFunc(llvm::Record* model,
                                       llvm::raw_ostream& os) {
  int num_inputs = model->getValueAsInt("num_inputs_");
  int num_outputs = model->getValueAsInt("num_outputs_");

  os << "int main(int argc, char** argv) {\n";
  os.indent(2);
  os << "std::vector<const void*> inputs;\n";
  os.indent(2);
  os << "std::vector<const void*> output_refs;\n";
  os.indent(2);
  os << "std::vector<void*> outputs;\n\n";

  for (int i = 0; i < num_inputs; i++) {
    os.indent(2);
    os << "inputs.push_back(";
    os << "input_" << i << ");\n";
  }
  for (int i = 0; i < num_outputs; i++) {
    os.indent(2);
    os << "outputs.push_back(";
    os << "output_" << i << ");\n";
    os.indent(2);
    os << "output_refs.push_back(";
    os << "output_ref_" << i << ");\n";
  }

  EmitCallFunc(num_inputs, num_outputs, os);
  EmitCheckResult(num_outputs, os);
  EmitTimePerf(num_inputs, num_outputs, os);

  os << "}\n";
  return os;
}

static llvm::raw_ostream& EmitTestConfigs(llvm::Record* model,
                                          llvm::raw_ostream& os) {
  auto configs = model->getValueAsListOfDefs("configs");
  auto docs = model->getValueAsDef("docs_");
  llvm::StringRef model_name = docs->getValueAsString("model_name_");

  os << "// clang-format off\n";
  for (auto config : configs) {
    int batch_size = config->getValueAsInt("batch_size_");
    os << "// RUN: %halo_compiler -target cxx ";
    os << "-batch-size " << batch_size;
    os << " %halo_compile_flags ";
    os << "%model_path -o %t.cc\n";
    os << "// RUN: %cxx -c -fPIC -o %t.o %t.cc -I%odla_path/include\n";
    os << "// RUN: %cxx %flags %macro_flags %s %t.o %t.bin %incs %odla_link "
          "%device_link ";
    os << "-o %t.exe -Wno-deprecated-declarations\n";
    os << "// RUN: %t.exe | tee -a log.txt | FileCheck %s\n";
    os << "// CHECK: Result verified\n";
  }
  os << "// clang-format on\n";
  os << "#include \"";
  os << model_name << ".inc";
  os << "\"\n";
  return os;
}

static llvm::raw_ostream& EmitReportHeader(llvm::raw_ostream& os) {
  os << "### Model Zoo\n";
  os << "[models directory](../models/) contains scripts for the following "
        "models,";
  os << "which download the pretrained models, compile and deploy them using "
        "HALO on X86-CPU or NVGPU.\n";
  os << "Please refer to [Instruction.md](../models/Instruction.md) for more "
        "details about how to run the examples.\n\n";

  os << "| Model | Reference Framework | Descriptors\n";
  os << "|:----|:----|:----|\n";
  return os;
}

static llvm::raw_ostream& EmitReportDescTable(llvm::Record* model,
                                              llvm::raw_ostream& os) {
  auto docs = model->getValueAsDef("docs_");
  llvm::StringRef model_name = docs->getValueAsString("model_name_");
  llvm::StringRef ref_framework = docs->getValueAsString("ref_framework_");
  llvm::StringRef model_desc = docs->getValueAsString("model_desc_");

  os << "| " << model_name << "| " << ref_framework << "| " << model_desc
     << "|\n";

  return os;
}

void EmitReportModel(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  auto models = GetAllModels(records);
  EmitReportHeader(os);
  for (auto model : models) {
    EmitReportDescTable(model, os);
  }
}

void EmitConfigModel(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  auto models = GetAllModels(records);

  for (auto model : models) {
    auto docs = model->getValueAsDef("docs_");
    llvm::StringRef model_name = docs->getValueAsString("model_name_");
    llvm::Twine file_name("test_");
    file_name.concat(model_name);
    file_name.concat(".cc");
    EmitCopyRight(file_name.str(), os);
    EmitTestConfigs(model, os);
  }
}

void EmitTestModel(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  auto models = GetAllModels(records);
  for (auto model : models) {
    auto docs = model->getValueAsDef("docs_");
    llvm::StringRef model_name = docs->getValueAsString("model_name_");
    llvm::Twine file_name("");
    file_name.concat(model_name);
    file_name.concat(".inc");
    EmitCopyRight(file_name.str(), os);
    EmitHeaderFunc(model, os);
    EmitDeclFunc(model, os);
    EmitMainFunc(model, os);
  }
}

} // namespace halo