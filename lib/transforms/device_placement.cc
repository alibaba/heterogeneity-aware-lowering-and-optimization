//===- device_placement.cc ------------------------------------------------===//
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

#include "halo/lib/transforms/device_placement.h"

namespace halo {

static std::string GetDeviceInfo(Instruction* inst) {
  auto& attrs = inst->GetAttributes();
  for (auto& attr : attrs) {
    if (attr->GetKind() == Attribute::AttrKind::STRING) {
      if (attr->GetName() == "device") {
        return attr->GetValueAsString();
      }
    }
  }
  return "";
}

static void AssignDevice(Function* func) {
  constexpr static int x86_thre = 10;
  static const std::string x86 = "x86";
  static const std::string trt = "tensorrt";

  std::string device;
  for (auto& bb : *func) {
    for (auto& i : *bb) {
      Instruction* inst = i.get();
      device = GetDeviceInfo(inst);
      break;
    }
    if (!device.empty()) {
      break;
    }
  }
  if (device.empty()) {
    func->SetDeviceName(
        (*func->BasicBlocks().begin())->Instructions().size() < x86_thre ? x86
                                                                         : trt);
  } else {
    func->SetDeviceName(device);
  }
}

bool DevicePlacement::RunOnFunction(Function* func) {
  if (func->IsEntryFunction() || !func->GetDeviceName().empty()) {
    return false;
  }
  AssignDevice(func);
  return true;
}

} // end namespace halo
