//===- global_context.h -----------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_FRAMEWORK_GLOBAL_CONTEXT_H_
#define HALO_LIB_FRAMEWORK_GLOBAL_CONTEXT_H_

#include <iostream>
#include <memory>

namespace halo {

class CodeGenObject;
class DataLayout;
class GlobalContextImpl;

class GlobalContext {
 public:
  GlobalContext();
  ~GlobalContext();

  // Disable copy and assignment constructors.
  GlobalContext(const GlobalContext&) = delete;
  GlobalContext& operator=(const GlobalContext&) = delete;

  /// Return the current global counter.
  uint64_t GetGlobalCounter() noexcept;

  /// Return the default data layout.
  const DataLayout& GetDefaultDataLayout() const noexcept;

  /// Return the debug output stream.
  static std::ostream& Dbgs() noexcept;

  /// Get the object used for code generation.
  const CodeGenObject& GetCodeGenObject() const noexcept;
  CodeGenObject& GetCodeGenObject() noexcept;

  /// Target triple string.
  const std::string& GetTargetTriple() const noexcept;
  void SetTargetTriple(const std::string& triple) noexcept;

  /// Target processor name.
  const std::string& GetProcessorName() const noexcept;
  void SetProcessorName(const std::string& processor) noexcept;

  /// Set the path of toolchain  so it can locate other components like runtime
  /// library. If a file path is given, it assumes the file is under
  /// `base_path`/bin/file and thus computes the `base_path`.
  void SetBasePath(const char* path) noexcept;
  const std::string& GetBasePath() const noexcept;
  void SetPrintPass(const bool is_print_pass) noexcept;
  bool GetPrintPass() const noexcept;

 private:
  const std::unique_ptr<GlobalContextImpl> impl_;
};

} // namespace halo

#endif // HALO_LIB_FRAMEWORK_GLOBAL_CONTEXT_H_