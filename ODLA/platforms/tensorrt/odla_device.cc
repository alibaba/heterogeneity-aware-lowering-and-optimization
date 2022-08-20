//===- odla_device.cc -----------------------------------------------------===//
//
// Copyright (C) 2019-2022 Alibaba Group Holding Limited.
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

#include <ODLA/odla_device.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>
#include <string>

#include "ODLA/odla_common.h"
#include "ODLA/odla_version.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "odla_tensorrt.h"

#define RETURN_ON_ERROR(x, success) \
  do {                              \
    if (x != success) {             \
      return ODLA_FAILURE;          \
    }                               \
  } while (0)

#define RETURN_ON_NVML_ERROR(x) RETURN_ON_ERROR(x, NVML_SUCCESS)
#define RETURN_ON_CUDA_ERROR(x) RETURN_ON_ERROR(x, cudaSuccess)
#define RETURN_ON_CU_ERROR(x) RETURN_ON_ERROR(x, CUDA_SUCCESS)

odla_status odla_AllocateDevice(odla_vendor vendor,
                                odla_device_name device_name,
                                odla_int32 device_idx, odla_device* device) {
  *device = nullptr;
  _odla_device dev;
  RETURN_ON_NVML_ERROR(nvmlInit_v2());
  RETURN_ON_CUDA_ERROR(cudaGetDeviceCount(&dev.device_count));
  RETURN_ON_CUDA_ERROR(cudaSetDevice(device_idx));
  RETURN_ON_CUDA_ERROR(cudaGetDevice(&dev.device_idx));
  if (device_idx != dev.device_idx) {
    assert(0);
    return ODLA_FAILURE;
  }
  RETURN_ON_NVML_ERROR(
      nvmlDeviceGetHandleByIndex_v2(dev.device_idx, &dev.nvml_device));
  cudaDeviceProp prop;
  RETURN_ON_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev.device_idx));
  dev.name = std::string(prop.name);
  dev.uuid = std::string(prop.uuid.bytes, 16);
  int ver = 0;
  RETURN_ON_CUDA_ERROR(cudaRuntimeGetVersion(&ver));
  int major = ver / 1000;
  int minor = (ver - 1000 * major) / 10;
  dev.cuda_runtime_sdk_version =
      std::to_string(major) + "." + std::to_string(minor);
  dev.cuda_driver_version.resize(80);
  RETURN_ON_NVML_ERROR(nvmlSystemGetDriverVersion(
      dev.cuda_driver_version.data(), dev.cuda_driver_version.size()));
  RETURN_ON_CU_ERROR(cuDeviceGet(&dev.cu_device, dev.device_idx));
  RETURN_ON_CU_ERROR(cuDevicePrimaryCtxRetain(&dev.cu_ctx, dev.cu_device));

  *device = new _odla_device();
  **device = dev;
  return ODLA_SUCCESS;
}

static nvmlReturn_t GetDeviceUtil(odla_device dev, bool is_gpu, float* val) {
  nvmlUtilization_t util;
  nvmlReturn_t status = nvmlDeviceGetUtilizationRates(dev->nvml_device, &util);
  *val = static_cast<float>((is_gpu ? util.gpu : util.memory) / 100.0);
  return status;
}

static void SetValue(odla_scalar_value* v, float x) {
  v->data_type = ODLA_FLOAT32;
  v->val_fp32 = x;
}

static void SetValue(odla_scalar_value* v, int x) {
  v->data_type = ODLA_INT32;
  v->val_int32 = x;
}

static void SetValue(odla_scalar_value* v, int64_t x) {
  v->data_type = ODLA_INT64;
  v->val_int64 = x;
}

static void SetValue(odla_scalar_value* v, uint64_t x) {
  v->data_type = ODLA_UINT64;
  v->val_uint64 = x;
}

static void SetValue(odla_scalar_value* v, const char* x) {
  v->data_type = ODLA_STRING;
  v->val_str = x;
}

odla_status odla_GetDeviceInfo(odla_device device, odla_device_info info_type,
                               odla_scalar_value* info_value) {
  constexpr int mw = 1000;
  switch (info_type) {
    default:
      return ODLA_FAILURE;
    case ODLA_DEVICE_INFO_ODLA_LIB_VERSION:
      SetValue(info_value, odla_GetVersionString());
      break;
    case ODLA_DEVICE_INFO_DEV_COUNT:
      SetValue(info_value, device->device_count);
      break;
    case ODLA_DEVICE_INFO_DEV_INDEX:
      SetValue(info_value, device->device_idx);
      break;
    case ODLA_DEVICE_INFO_DEV_TYPE:
      SetValue(info_value, device->name.c_str());
      break;
    case ODLA_DEVICE_INFO_DEV_UUID:
      SetValue(info_value, device->uuid.c_str());
      break;
    case ODLA_DEVICE_INFO_PROCESSOR_UTIL: {
      float val = NAN;
      RETURN_ON_NVML_ERROR(GetDeviceUtil(device, true, &val));
      SetValue(info_value, val);
      break;
    }
    case ODLA_DEVICE_INFO_MEMORY_UTIL: {
      float val = NAN;
      RETURN_ON_NVML_ERROR(GetDeviceUtil(device, false, &val));
      SetValue(info_value, val);
      break;
    }
    case ODLA_DEVICE_INFO_TOTAL_MEMORY: {
      size_t free = 0;
      size_t total = 0;
      RETURN_ON_CUDA_ERROR(cudaMemGetInfo(&free, &total));
      SetValue(info_value, total);
      break;
    }
    case ODLA_DEVICE_INFO_USED_MEMORY: {
      size_t free = 0;
      size_t total = 0;
      RETURN_ON_CUDA_ERROR(cudaMemGetInfo(&free, &total));
      SetValue(info_value, total - free);
      break;
    }
    case ODLA_DEVICE_INFO_POWER_USAGE: {
      unsigned int power = 0;
      RETURN_ON_NVML_ERROR(
          nvmlDeviceGetPowerUsage(device->nvml_device, &power));
      SetValue(info_value, static_cast<float>(static_cast<float>(power) / mw));
      break;
    }
    case ODLA_DEVICE_INFO_POWER_LIMIT: {
      unsigned int power = 0;
      RETURN_ON_NVML_ERROR(nvmlDeviceGetPowerManagementDefaultLimit(
          device->nvml_device, &power));
      SetValue(info_value, static_cast<float>(static_cast<float>(power) / mw));
      break;
    }
    case ODLA_DEVICE_INFO_DRIVER_VERSION: {
      SetValue(info_value, device->cuda_driver_version.c_str());
      break;
    }
    case ODLA_DEVICE_INFO_SDK_VERSION:
      SetValue(info_value, device->cuda_runtime_sdk_version.c_str());
      break;
  }
  return ODLA_SUCCESS;
}
