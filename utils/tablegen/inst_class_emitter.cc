//===- inst_class_emitter.cc ------------------------------------*- C++ -*-===//
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

#include <algorithm>

#include "halo_tblgen.h"
#include "inst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/TableGenBackend.h"

llvm::cl::OptionCategory HaloInstEmitterCat("Options for -gen-inst-class");

static llvm::cl::opt<std::string> Ifdefname(
    "ifdef", llvm::cl::desc("Name Used in #ifdef "), llvm::cl::value_desc(""),
    llvm::cl::init(""), llvm::cl::cat(HaloInstEmitterCat));

namespace halo {

namespace tablegen {

/// Instruction sub class header file emitter
class CodeEmitterGen {
 public:
  CodeEmitterGen(const llvm::RecordKeeper& records, llvm::raw_ostream& o)
      : records_(records), os_(o) {}
  /// Main entry to emit a XXX_inst.h file based on XXX_inst.td
  /// create a header file of multiple sub-instructions declarations.
  void Run();
  /// Emit instruction declarations per td
  void EmitClasses();

 private:
  const llvm::RecordKeeper& records_;
  llvm::raw_ostream& os_;
};

/// Emit #ifdef..#endif pattern
class Ifdef {
 public:
  Ifdef(const std::string& name, llvm::raw_ostream& o);
  // Destructor, auto emitting close phrase of #endif
  ~Ifdef() { os_ << "#endif // " << ifdef_name_ << "\n"; }
  // Ifdef class cannot be copied or moved. Otherwise, unintended "#endif" will
  // be emitted by the destructor.
  Ifdef(const Ifdef&) = delete;
  Ifdef(Ifdef&&) = delete;
  Ifdef operator=(const Ifdef&) = delete;
  Ifdef operator=(Ifdef&&) = delete;

 private:
  std::string ifdef_name_;
  llvm::raw_ostream& os_;
  // ifdef string
};

Ifdef::Ifdef(const std::string& name, llvm::raw_ostream& o)
    : ifdef_name_(name), os_(o) {
  os_ << "#ifndef " << ifdef_name_ << "\n";
  os_ << "#define " << ifdef_name_ << "\n\n";
}

/// Emit namespace xxx ... end namespace pattern
class Namespace {
 public:
  Namespace(const std::string& name, llvm::raw_ostream& o);
  // Destructor, anto emitting close phrase // end namespace
  ~Namespace() { os_ << "} // end namespace " << name_ << "\n"; }
  // Namespace class cannot be copied or moved. Otherwise, unintended "}" will
  // be emitted by the destructor.
  Namespace(const Namespace&) = delete;
  Namespace(Namespace&&) = delete;
  Namespace operator=(const Namespace&) = delete;
  Namespace operator=(Namespace&&) = delete;

 private:
  // namespace string
  std::string name_;
  llvm::raw_ostream& os_;
};

Namespace::Namespace(const std::string& name, llvm::raw_ostream& o)
    : name_(name), os_(o) {
  os_ << "namespace " << name_ << " {\n\n";
}

// Emit instruction classes.
void CodeEmitterGen::EmitClasses() {
  // Emit namespace
  Namespace ns_emitter("halo", os_);

  std::vector<llvm::Record*> insts = records_.getAllDerivedDefinitions("Inst");
  for (auto ic = insts.begin(), ec = insts.end(); ic != ec; ++ic) {
    Inst(*ic, os_).Run();
  }
}

// Main entry to generate a XXX_inst.h
void CodeEmitterGen::Run() {
  const std::string filename = Ifdefname + ".h";
  // Emit license
  halo::EmitSourceFileHeader(filename, os_);
  // Emit ifdef phase
  std::string ifdef_name = "HALO_LIB_IR_" + Ifdefname + "_H";
  std::transform(ifdef_name.begin(), ifdef_name.end(), ifdef_name.begin(),
                 toupper);
  Ifdef ifdef_emitter(ifdef_name, os_);
  // Emit include
  os_ << "#include \"halo/lib/ir/instruction.h\"\n\n";
  EmitClasses();
}

} // end namespace tablegen

void EmitInstClass(const llvm::RecordKeeper& records, llvm::raw_ostream& os) {
  halo::tablegen::CodeEmitterGen(records, os).Run();
}

} // end namespace halo