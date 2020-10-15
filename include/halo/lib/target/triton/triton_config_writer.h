//===- triton_config_writer.h -----------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_TARGET_TRITON_CONFIG_WRITER_H_
#define HALO_LIB_TARGET_TRITON_CONFIG_WRITER_H_

#include "halo/lib/pass/pass.h"
#include "halo/lib/target/codegen.h"

namespace halo {

// The class to generate config file for Triton inference service.
class TritonConfigWriter final : public CodeGen {
 public:
  TritonConfigWriter(const std::string& filename, int max_batch_size)
      : CodeGen("Triton Config Writer"),
        filename_(filename),
        max_batch_size_(max_batch_size) {}
  virtual ~TritonConfigWriter() = default;

  bool RunOnModule(Module* module) override;

 private:
  void PrintUseProtobuf(const Module& module, std::ostream* os);
  std::string filename_;
  int max_batch_size_;
};

} // end namespace halo.

#endif // HALO_LIB_TARGET_TRITON_CONFIG_WRITER_H_