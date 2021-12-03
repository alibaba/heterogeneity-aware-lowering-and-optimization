/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===- batchedNMSPlugin.h ------------------------------------------------===//
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

#ifndef ODLA_TRT_BATCHED_NMS_PLUGIN_H
#define ODLA_TRT_BATCHED_NMS_PLUGIN_H

#include <string>
#include <vector>

#include "../common.h"
#include "batchedNMSInference.h"
#include "gatherNMSOutputs.h"
#include "kernel.h"
#include "nmsUtils.h"

using namespace nvinfer1::plugin;

class BatchedNMSPluginV2 : public IPluginV2Ext {
 public:
  BatchedNMSPluginV2(NMSParameters param);

  BatchedNMSPluginV2(const void* data, size_t length);

  ~BatchedNMSPluginV2() override = default;

  int getNbOutputs() const NOEXCEPT override;

  Dims getOutputDimensions(int index, const Dims* inputs,
                           int nbInputDims) NOEXCEPT override;

  int initialize() NOEXCEPT override;

  void terminate() NOEXCEPT override;

  size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override;

#if NV_TENSORRT_MAJOR >= 8
  using enqueue_output_ty = void* const*;
#else
  using enqueue_output_ty = void**;
#endif
  int enqueue(int batchSize, const void* const* inputs,
              enqueue_output_ty outputs, void* workspace,
              cudaStream_t stream) NOEXCEPT override;

  size_t getSerializationSize() const NOEXCEPT override;

  void serialize(void* buffer) const NOEXCEPT override;

  void configurePlugin(const Dims* inputDims, int nbInputs,
                       const Dims* outputDims, int nbOutputs,
                       const DataType* inputTypes, const DataType* outputTypes,
                       const bool* inputIsBroadcast,
                       const bool* outputIsBroadcast, PluginFormat floatFormat,
                       int maxBatchSize) NOEXCEPT override;

  bool supportsFormat(DataType type,
                      PluginFormat format) const NOEXCEPT override;

  const char* getPluginType() const NOEXCEPT override;

  const char* getPluginVersion() const NOEXCEPT override;

  void destroy() NOEXCEPT override;

  IPluginV2Ext* clone() const NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputType,
                                       int nbInputs) const NOEXCEPT override;

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override;

  const char* getPluginNamespace() const NOEXCEPT override;

  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                    const bool* inputIsBroadcasted,
                                    int nbInputs) const NOEXCEPT override;

  bool canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT override;

  void setClipParam(bool clip);

 private:
  NMSParameters param{};
  int boxesSize{};
  int scoresSize{};
  int numPriors{};
  std::string mNamespace;
  bool mClipBoxes{};
  const char* mPluginNamespace;
};

class BatchedNMSPluginV2Creator : public IPluginCreator {
 public:
  BatchedNMSPluginV2Creator();

  ~BatchedNMSPluginV2Creator() override = default;

  const char* getPluginName() const NOEXCEPT override;

  const char* getPluginVersion() const NOEXCEPT override;

  void setPluginNamespace(const char* libNamespace) NOEXCEPT override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const NOEXCEPT override {
    return mNamespace.c_str();
  }

  const PluginFieldCollection* getFieldNames() NOEXCEPT override;

  IPluginV2Ext* createPlugin(const char* name,
                             const PluginFieldCollection* fc) NOEXCEPT override;

  IPluginV2Ext* deserializePlugin(const char* name, const void* serialData,
                                  size_t serialLength) NOEXCEPT override;

 private:
  static PluginFieldCollection mFC;
  NMSParameters params;
  static std::vector<PluginField> mPluginAttributes;
  bool mClipBoxes;
  std::string mNamespace;
};

#endif // ODLA_TRT_BATCHED_NMS_PLUGIN_H
