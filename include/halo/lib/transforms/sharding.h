//===- sharding.h ---------------------------------------------------------===//
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

#ifndef HALO_LIB_TRANSFORMS_SHARDING_H_
#define HALO_LIB_TRANSFORMS_SHARDING_H_

#include "halo/lib/pass/pass.h"

namespace halo {

/// This pass splitting a function into sub functions.
class Sharding final : public FunctionPass {
 public:
  Sharding(int shards) : FunctionPass("Sharding"), shards_(shards) {}

  bool RunOnFunction(Function* func) override;

 private:
  int shards_;
};

} // end namespace halo.

#endif // HALO_LIB_TRANSFORMS_SHARDING_H_