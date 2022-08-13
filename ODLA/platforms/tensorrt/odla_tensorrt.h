//===- odla_tensorrt.h ----------------------------------------------------===//
//
// Copyright (C) 2020-2022 Alibaba Group Holding Limited.
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

#ifndef ODLA_TENSORRT_H
#define ODLA_TENSORRT_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <string>

#include "../include/odla_impl_common.h"
#include "common.h"

typedef struct nvmlDevice_st* nvmlDevice_t;
struct _odla_device {
  std::string name;
  std::string cuda_driver_version;
  std::string cuda_runtime_sdk_version;
  int device_count;
  int device_idx;
  std::string uuid;
  nvmlDevice_t nvml_device;
  CUdevice cu_device;
  CUcontext cu_ctx;
};

static std::string GetErrorMsg(const char* prefix, int err) {
  std::string msg(prefix);
  msg += " error: code: " + std::to_string(err);
  return msg;
}

static std::string GetErrorMsg(const char* prefix, cudaError_t err) {
  std::string msg(prefix);
  msg += std::string(" error: ") + cudaGetErrorName(err) + std::string(":") +
         cudaGetErrorString(err);
  return msg;
}

#define RETURN_ON_ERROR(lib, x, success)   \
  do {                                     \
    if (x != success) {                    \
      ODLA_LOG_ERROR(GetErrorMsg(lib, x)); \
      return ODLA_FAILURE;                 \
    }                                      \
  } while (0)

#define RETURN_ON_CUDA_ERROR(x) RETURN_ON_ERROR("cuda runtime", x, cudaSuccess)

#endif // ODLA_TENSORRT_H
