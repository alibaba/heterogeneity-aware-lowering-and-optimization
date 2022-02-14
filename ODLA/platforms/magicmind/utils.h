//===- utils.h ------------------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
// Copyright (C) [2022] by Cambricon, Inc. All rights reserved.
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

#ifndef MAGICMIND_UTILS_H_
#define MAGICMIND_UTILS_H_

class EnvTime {
 public:
  static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

  EnvTime() = default;
  virtual ~EnvTime() = default;

  static uint64_t NowNanos() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64_t>(ts.tv_nsec));
  }
};

#endif
