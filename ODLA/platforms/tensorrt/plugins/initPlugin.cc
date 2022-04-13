//===- initPlugin.cc ------------------------------------------------------===//
//
// Copyright (C) 2020-2021 Alibaba Group Holding Limited.
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

#include "initPlugin.h"

#include <NvInferPlugin.h>

#include <memory>

#include "batchedNMSPlugin/batchedNMSPlugin.h"
#include "oneHotPlugin/oneHotPlugin.h"

extern "C" {
void initODLAPlugin(nvinfer1::ILogger* logger, const char* libNamespace) {
#if NV_TENSORRT_MAJOR < 8
  const int NUM_PLUGINS = 22;
#else
  const int NUM_PLUGINS = 32;
#endif
  initLibNvInferPlugins(static_cast<void*>(logger), libNamespace);
  REGISTER_TENSORRT_PLUGIN(BatchedNMSPluginV2Creator);
  REGISTER_TENSORRT_PLUGIN(OneHotPluginCreator);

  int num_plugins = 0;
  auto plugin_list = getPluginRegistry()->getPluginCreatorList(&num_plugins);
  if (num_plugins < NUM_PLUGINS) {
    logger->log(ILogger::Severity::kERROR, "init ODLA plugin failed.");
  }
}
}
