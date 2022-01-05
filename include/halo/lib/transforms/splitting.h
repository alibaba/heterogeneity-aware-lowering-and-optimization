//===- splitting.h --------------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#ifndef HALO_LIB_TRANSFORMS_SPLITTING_H_
#define HALO_LIB_TRANSFORMS_SPLITTING_H_

#include "halo/lib/pass/pass.h"

namespace halo {

/// This pass splitting a function into sub functions.
class Splitting final : public FunctionPass {
 public:
  Splitting(bool group_by_device = false) : FunctionPass("Function Splitting") {
    group_by_device_ = group_by_device;
  }

  bool RunOnFunction(Function* func) override;

 private:
  bool group_by_device_;
};

} // end namespace halo.

#endif // HALO_LIB_TRANSFORMS_SPLITTING_H_
