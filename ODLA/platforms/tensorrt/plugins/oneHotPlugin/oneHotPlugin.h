//===- oneHotPlugin.h -----------------------------------------------------===//
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

#ifndef ODLA_TRT_ONE_HOT_PLUGIN_H
#define ODLA_TRT_ONE_HOT_PLUGIN_H

#include <string>
#include <vector>

#include "../common.h"
#include "kernel.h"
#include "plugin.h"

class OneHotPlugin : public nvinfer1::IPluginV2Ext {
 public:
  OneHotPlugin(const char* name, bool explicit_batch, int depth, int axis);

  OneHotPlugin(const char* name, const void* data, size_t length);

  int getNbOutputs() const NOEXCEPT override { return 1; };

  Dims getOutputDimensions(int index, const Dims* inputs,
                           int nbInputDims) NOEXCEPT override;

  int initialize() NOEXCEPT override { return STATUS_SUCCESS; };
  void terminate() NOEXCEPT override{};

  size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {
    return 0;
  };

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

  void setPluginNamespace(const char* ns) NOEXCEPT override {
    mNamespace = ns;
  };

  const char* getPluginNamespace() const NOEXCEPT override {
    return mNamespace.c_str();
  };

  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                    const bool* inputIsBroadcasted,
                                    int nbInputs) const NOEXCEPT override {
    return false;
  };

  bool canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT override {
    return false;
  };

 private:
  int normalizeAxis(const nvinfer1::Dims& index_dim);
  const std::string mLayerName;
  bool mExplicitBatch;
  int mDepth;
  int mAxis;
  DataType mType{nvinfer1::DataType::kFLOAT};
  int64_t mPreAxisElems;
  int64_t mPostAxisElems;
  std::string mNamespace;
};

class OneHotPluginCreator : public IPluginCreator {
 public:
  OneHotPluginCreator();

  ~OneHotPluginCreator() override = default;

  const char* getPluginName() const NOEXCEPT override;

  const char* getPluginVersion() const NOEXCEPT override;

  void setPluginNamespace(const char* ns) NOEXCEPT override { mNamespace = ns; }

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
  static std::vector<PluginField> mPluginAttributes;
  std::string mNamespace;
};

#endif // ODLA_TRT_ONE_HOT_PLUGIN_H
