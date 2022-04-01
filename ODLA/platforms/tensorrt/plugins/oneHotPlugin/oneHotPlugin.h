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

template <typename Base>
class OneHotBase : public Base {
 protected:
  struct Config {
    bool explicitBatch;
    int depth;
    int axis;
    DataType type;
    int64_t preAxisElems;
    int64_t postAxisElems;
  };

 public:
  OneHotBase(const char* name, const Config& config)
      : mNamespace(""), mLayerName(name), mConfig(config) {}
  OneHotBase(const char* name, const void* data, size_t length);
  explicit OneHotBase(const char* name)
      : OneHotBase(name, {false, 0, -1, nvinfer1::DataType::kFLOAT, 1, 1}) {}

  int getNbOutputs() const NOEXCEPT override;
  int initialize() NOEXCEPT override;
  void terminate() NOEXCEPT override;
  void destroy() NOEXCEPT override;
  void setPluginNamespace(const char* ns) NOEXCEPT override;
  const char* getPluginNamespace() const NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const NOEXCEPT override;
  const char* getPluginVersion() const NOEXCEPT override;
  const char* getPluginType() const NOEXCEPT override;

  void serialize(void* buffer) const NOEXCEPT override;
  size_t getSerializationSize() const NOEXCEPT override;
  virtual ~OneHotBase() = default;

 protected:
  int normalizeAxis(int input_rank);
  std::string mNamespace;
  const std::string mLayerName;
  struct Config mConfig;
};

class OneHotPlugin : public OneHotBase<nvinfer1::IPluginV2Ext> {
 public:
  OneHotPlugin(const OneHotPlugin& plugin);

  OneHotPlugin(const char* name, const Config& config)
      : OneHotBase<nvinfer1::IPluginV2Ext>(name, config) {}

  OneHotPlugin(const char* name, const void* data, size_t length)
      : OneHotBase<nvinfer1::IPluginV2Ext>(name, data, length) {}

  Dims getOutputDimensions(int index, const Dims* inputs,
                           int nbInputDims) NOEXCEPT override;

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

  void configurePlugin(const Dims* inputDims, int nbInputs,
                       const Dims* outputDims, int nbOutputs,
                       const DataType* inputTypes, const DataType* outputTypes,
                       const bool* inputIsBroadcast,
                       const bool* outputIsBroadcast, PluginFormat floatFormat,
                       int maxBatchSize) NOEXCEPT override;

  bool supportsFormat(DataType type,
                      PluginFormat format) const NOEXCEPT override;

  IPluginV2Ext* clone() const NOEXCEPT override;

  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                    const bool* inputIsBroadcasted,
                                    int nbInputs) const NOEXCEPT override {
    return false;
  };

  bool canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT override {
    return false;
  };
};

class OneHotPluginDynamic : public OneHotBase<nvinfer1::IPluginV2DynamicExt> {
 public:
  OneHotPluginDynamic(const char* name, const Config& config)
      : OneHotBase<nvinfer1::IPluginV2DynamicExt>(name, config) {}

  OneHotPluginDynamic(const char* name, const void* data, size_t length)
      : OneHotBase<nvinfer1::IPluginV2DynamicExt>(name, data, length) {}

  nvinfer1::IPluginV2DynamicExt* clone() const NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) NOEXCEPT override;

  bool supportsFormatCombination(int32_t pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) NOEXCEPT override;
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
