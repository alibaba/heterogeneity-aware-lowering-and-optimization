//===- odla_device.cc -----------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <popart/devicemanager.hpp>
#include <string>
#include <vector>

#include "odla_popart.h"

static constexpr const char* sDeviceType = "__device_type";
static constexpr const char* sDeviceTypeValues[] = {"ipu", "ipu_model"};

static constexpr const char* sSyncPatterm = "__sync_pattern";
static constexpr const char* sSyncPattermValues[] = {"full", "SinglePipeline",
                                                     "PingPong"};

static constexpr const char* sConnectionType = "__connection_type";
static constexpr const char* sConnectionTypeValues[] = {"Always", "OnDemand",
                                                        "Never"};

static constexpr const char* sNumIpus = "__num_ipus";
static constexpr const char* sTilesPerIpu = "__tiles_per_ipu";

struct _odla_device_config {
  std::map<std::string, std::string> items;
};

struct device_config {
  popart::DeviceType device_type = popart::DeviceType::Ipu;
  popart::SyncPattern sync_pattern = popart::SyncPattern::Full;
  popart::DeviceConnectionType connection_type =
      popart::DeviceConnectionType::Always;
  int32_t num_ipus = 1;
  int32_t tiles_per_ipu = 1216;
};

struct _odla_device {
  std::shared_ptr<popart::DeviceInfo> device;
};

namespace {
device_config GetDeviceConfig(const odla_device_config odla_dev_config) {
  device_config dev_config;
  for (const auto& item : odla_dev_config->items) {
    if (item.first.compare(sDeviceType) == 0) {
      if (item.second.compare("ipu") == 0) {
        dev_config.device_type = popart::DeviceType::Ipu;
      } else if (item.second.compare("ipu_model") == 0) {
        dev_config.device_type = popart::DeviceType::IpuModel;
      }
    } else if (item.first.compare(sNumIpus) == 0) {
      dev_config.num_ipus = std::stoi(item.second);
    }

    return dev_config;
  }
}

popart::DeviceManager& GetDeviceManager() {
  static popart::DeviceManager device_manager =
      popart::DeviceManager::createDeviceManager();
  return device_manager;
}
} // namespace

odla_status odla_CreateDeviceConfig(odla_device_config* device_config) {
  *device_config = new _odla_device_config();
  return ODLA_SUCCESS;
}

odla_status odla_DestroyDeviceConfig(odla_device_config device_config) {
  delete device_config;
  return ODLA_SUCCESS;
}

odla_status odla_SetDeviceConfigItem(
    odla_device_config device_config,
    odla_device_config_item* device_config_item, ...) {
  device_config->items.emplace(std::string(device_config_item->key),
                               std::string(device_config_item->value));
  return ODLA_SUCCESS;
}

odla_status odla_AllocateDevice(const odla_vendor vendor,
                                const odla_device_name device_name,
                                odla_device* device,
                                const odla_device_config config) {
  _odla_device* odla_dev = new _odla_device;
  auto dev_cfg = GetDeviceConfig(config);
  auto& dev_mngr = GetDeviceManager();
  std::map<std::string, std::string> options;

  switch (dev_cfg.device_type) {
    case popart::DeviceType::Ipu:
      odla_dev->device = dev_mngr.acquireAvailableDevice(
          dev_cfg.num_ipus, dev_cfg.tiles_per_ipu, dev_cfg.sync_pattern,
          dev_cfg.connection_type);
      break;
    case popart::DeviceType::IpuModel:
      options.emplace("numIPUs", std::to_string(dev_cfg.num_ipus));
      options.emplace("tilesPerIPU", std::to_string(dev_cfg.tiles_per_ipu));
      odla_dev->device = dev_mngr.createIpuModelDevice(options);
      break;
    case popart::DeviceType::Cpu:
      odla_dev->device = dev_mngr.createCpuDevice();
      break;
    default:
      std::cerr << "Non supported device type." << std::endl;
      return ODLA_UNSUPPORTED_DEVICE_TYPE;
  }
  *device = odla_dev;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyDevice(odla_device device) {
  delete device;
  return ODLA_SUCCESS;
}
