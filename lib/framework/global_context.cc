//===- global_context.cc --------------------------------------------------===//
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

#include "halo/lib/framework/global_context.h"

#include <experimental/filesystem>

#include "halo/halo.h"
#include "halo/lib/framework/data_layout.h"
#include "halo/lib/target/codegen_object.h"
#include "llvm/Config/llvm-config.h"

namespace halo {

class GlobalContextImpl {
 public:
  GlobalContextImpl() = default;
  ~GlobalContextImpl() = default;
  GlobalContextImpl(const GlobalContextImpl&) = delete;
  GlobalContextImpl& operator=(const GlobalContextImpl&) = delete;
  GlobalContextImpl(GlobalContextImpl&&) = delete;
  GlobalContextImpl& operator=(GlobalContextImpl&&) = delete;

  /// Return the current global counter and then increase it.
  uint64_t ReturnAndIncreaseGlobalCounter() noexcept {
    uint64_t current = global_counter_++;
    return current;
  }

  const DataLayout& GetDefaultDataLayout() const noexcept {
    return data_layout_;
  }

  const CodeGenObject& GetCodeGenObject() const noexcept {
    return code_gen_obj_;
  }

  CodeGenObject& GetCodeGenObject() noexcept { return code_gen_obj_; }

  void SetBasePath(const std::string& path) noexcept { base_path_ = path; }
  const std::string& GetBasePath() const noexcept { return base_path_; }

  void SetODLAIncludePath(const std::string& path) noexcept {
    odla_header_path_ = path;
  }
  const std::string& GetODLAIncludePath() const noexcept {
    return odla_header_path_;
  }

  void SetODLALibraryPath(const std::string& path) noexcept {
    odla_lib_path_ = path;
  }
  const std::string& GetODLALibraryPath() const noexcept {
    return odla_lib_path_;
  }

  void SetVerbosity(int verbosity) noexcept { verbosity_ = verbosity; }
  int GetVerbosity() const noexcept { return verbosity_; }

  const std::string& GetTargetTriple() const noexcept { return triple_; }
  void SetTargetTriple(const std::string& triple) noexcept { triple_ = triple; }

  const std::string& GetProcessorName() const noexcept { return processor_; }
  void SetProcessorName(const std::string& processor) noexcept {
    processor_ = processor;
  }

  ModelInfo& GetModelInfo() noexcept { return model_info_; }

  void SetPrintPass(const bool is_print_pass) noexcept {
    is_print_pass_ = is_print_pass;
  }

  bool GetPrintPass() const noexcept { return is_print_pass_; }

 private:
  // A global counter
  uint64_t global_counter_ = 0;
  DefaultDataLayout data_layout_;
  CodeGenObject code_gen_obj_;
  std::string base_path_{""};
  std::string odla_header_path_{""};
  std::string odla_lib_path_{""};
  int verbosity_ = 0;
  std::string triple_{LLVM_HOST_TRIPLE};
  std::string processor_{"native"};
  bool is_print_pass_ = false;
  ModelInfo model_info_{};
};

GlobalContext::GlobalContext() : impl_(std::make_unique<GlobalContextImpl>()) {}

GlobalContext::~GlobalContext() {}

/// Return the current global counter.
uint64_t GlobalContext::GetGlobalCounter() noexcept {
  return impl_->ReturnAndIncreaseGlobalCounter();
}

const DataLayout& GlobalContext::GetDefaultDataLayout() const noexcept {
  return impl_->GetDefaultDataLayout();
}

void GlobalContext::SetBasePath(const std::string& path) noexcept {
  impl_->SetBasePath(path);
}

const std::string& GlobalContext::GetBasePath() const noexcept {
  return impl_->GetBasePath();
}

void GlobalContext::SetVerbosity(int verbosity) noexcept {
  impl_->SetVerbosity(verbosity);
}

int GlobalContext::GetVerbosity() const noexcept {
  return impl_->GetVerbosity();
}

void GlobalContext::SetODLAIncludePath(const std::string& path) noexcept {
  impl_->SetODLAIncludePath(path);
}

const std::string& GlobalContext::GetODLAIncludePath() const noexcept {
  return impl_->GetODLAIncludePath();
}

void GlobalContext::SetODLALibraryPath(const std::string& path) noexcept {
  impl_->SetODLALibraryPath(path);
}

const std::string& GlobalContext::GetODLALibraryPath() const noexcept {
  return impl_->GetODLALibraryPath();
}

const CodeGenObject& GlobalContext::GetCodeGenObject() const noexcept {
  return impl_->GetCodeGenObject();
}

CodeGenObject& GlobalContext::GetCodeGenObject() noexcept {
  return impl_->GetCodeGenObject();
}

const std::string& GlobalContext::GetTargetTriple() const noexcept {
  return impl_->GetTargetTriple();
}

void GlobalContext::SetTargetTriple(const std::string& triple) noexcept {
  impl_->SetTargetTriple(triple);
}

const std::string& GlobalContext::GetProcessorName() const noexcept {
  return impl_->GetProcessorName();
}

void GlobalContext::SetProcessorName(const std::string& processor) noexcept {
  impl_->SetProcessorName(processor);
}

ModelInfo& GlobalContext::GetModelInfo() noexcept {
  return impl_->GetModelInfo();
}

std::ostream& GlobalContext::Dbgs() noexcept { return std::cerr; }

void GlobalContext::SetPrintPass(const bool is_print_pass) noexcept {
  impl_->SetPrintPass(is_print_pass);
}

bool GlobalContext::GetPrintPass() const noexcept {
  return impl_->GetPrintPass();
}

} // namespace halo