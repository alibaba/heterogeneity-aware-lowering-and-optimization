//===- oneHotPlugin.cpp ---------------------------------------------------===//
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

#include "oneHotPlugin.h"

#include <vector>

using namespace nvinfer1;

static const char* ONE_HOT_PLUGIN_VERSION{"1"};
static const char* ONE_HOT_PLUGIN_NAME{"OneHot_TRT"};

template <typename Base>
OneHotBase<Base>::OneHotBase(const char* name, const void* data, size_t length)
    : mLayerName(name) {
  const char* begin = reinterpret_cast<const char*>(data);
  const char* d = begin;
  mConfig = read<Config>(d);
  ASSERT(d == begin + length);
}

template <typename Base>
const char* OneHotBase<Base>::getPluginType() const NOEXCEPT {
  return ONE_HOT_PLUGIN_NAME;
}

template <typename Base>
const char* OneHotBase<Base>::getPluginVersion() const NOEXCEPT {
  return ONE_HOT_PLUGIN_VERSION;
}

template <typename Base>
nvinfer1::DataType OneHotBase<Base>::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes,
    int nbInputs) const NOEXCEPT {
  ASSERT(index == 0 && nbInputs > 1);
  return inputTypes[1];
}

template <typename Base>
int OneHotBase<Base>::getNbOutputs() const NOEXCEPT {
  return 1;
}

template <typename Base>
int OneHotBase<Base>::initialize() NOEXCEPT {
  return STATUS_SUCCESS;
}

template <typename Base>
void OneHotBase<Base>::terminate() NOEXCEPT {}

template <typename Base>
void OneHotBase<Base>::destroy() NOEXCEPT {
  delete this;
}

template <typename Base>
void OneHotBase<Base>::setPluginNamespace(const char* ns) NOEXCEPT {
  mNamespace = ns;
}

template <typename Base>
const char* OneHotBase<Base>::getPluginNamespace() const NOEXCEPT {
  return mNamespace.c_str();
}

template <typename Base>
size_t OneHotBase<Base>::getSerializationSize() const NOEXCEPT {
  return sizeof(mConfig);
}

template <typename Base>
void OneHotBase<Base>::serialize(void* buffer) const NOEXCEPT {
  char* begin = reinterpret_cast<char*>(buffer);
  char* d = begin;
  write(d, mConfig);
  ASSERT(d == begin + getSerializationSize());
}

template <typename Base>
int OneHotBase<Base>::normalizeAxis(int input_rank) {
  input_rank = mConfig.explicitBatch ? input_rank + 1 : input_rank;
  mConfig.axis =
      mConfig.axis < 0 ? mConfig.axis + input_rank + 1 : mConfig.axis;
  if (mConfig.explicitBatch) {
    ASSERT(mConfig.axis != 0); // dim 0 is always batch dim.
  }
  ASSERT(mConfig.axis >= 0 && mConfig.axis <= input_rank);
  return mConfig.explicitBatch ? mConfig.axis - 1 : mConfig.axis;
}

extern pluginStatus_t oneHotEncoding(cudaStream_t stream,
                                     int64_t pre_axis_elems, int depth,
                                     int64_t post_axis_elems, int axis,
                                     nvinfer1::DataType data_type,
                                     const int32_t* indices, const void* on_off,
                                     void* output);

OneHotPlugin::OneHotPlugin(const OneHotPlugin& plugin)
    : OneHotBase<nvinfer1::IPluginV2Ext>(plugin.mLayerName.c_str(),
                                         plugin.mConfig) {
  setPluginNamespace(plugin.getPluginNamespace());
}

Dims OneHotPlugin::getOutputDimensions(int index, const Dims* inputs,
                                       int nbInputDims) NOEXCEPT {
  ASSERT(nbInputDims == 2);
  ASSERT(index >= 0 && index < this->getNbOutputs());
  const auto& index_dim = inputs[0];
  if (!mConfig.explicitBatch) {
    ASSERT(inputs[1].nbDims == 1 && inputs[1].d[0] == 2);
  }
  Dims dim;
  dim.nbDims = index_dim.nbDims + 1;

  auto axis = normalizeAxis(index_dim.nbDims);
  for (int i = 0; i < dim.nbDims; ++i) {
    if (i < axis) {
      dim.d[i] = index_dim.d[i];
    } else if (i == axis) {
      dim.d[i] = mConfig.depth;
    } else {
      dim.d[i] = index_dim.d[i - 1];
    }
  }
  return dim;
}

int OneHotPlugin::enqueue(int batchSize, const void* const* inputs,
                          enqueue_output_ty outputs, void* workspace,
                          cudaStream_t stream) NOEXCEPT {
  const int32_t* indices = reinterpret_cast<const int32_t*>(inputs[0]);
  const void* on_off_vals = inputs[1];

  void* output = outputs[0];

  pluginStatus_t status = oneHotEncoding(
      stream, (mConfig.explicitBatch ? batchSize : 1) * mConfig.preAxisElems,
      mConfig.depth, mConfig.postAxisElems, mConfig.axis, mConfig.type, indices,
      on_off_vals, output);
  ASSERT(status == STATUS_SUCCESS);
  return status;
}

PluginFieldCollection OneHotPluginCreator::mFC{};
std::vector<PluginField> OneHotPluginCreator::mPluginAttributes;

OneHotPluginCreator::OneHotPluginCreator() {
  mPluginAttributes.emplace_back(
      PluginField("depth", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* OneHotPluginCreator::getPluginName() const NOEXCEPT {
  return ONE_HOT_PLUGIN_NAME;
}

const char* OneHotPluginCreator::getPluginVersion() const NOEXCEPT {
  return ONE_HOT_PLUGIN_VERSION;
}

const PluginFieldCollection* OneHotPluginCreator::getFieldNames() NOEXCEPT {
  return &mFC;
}

void OneHotPlugin::configurePlugin(
    const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes,
    const bool* inputIsBroadcast, const bool* outputIsBroadcast,
    nvinfer1::PluginFormat format, int maxBatchSize) NOEXCEPT {
  ASSERT(nbInputs == 2);
  ASSERT(nbOutputs == 1);
  ASSERT(inputTypes[0] == DataType::kINT32);
  // ASSERT(std::none_of(inputIsBroadcast, inputIsBroadcast + nbInputs,
  //                    [](bool b) { return b; }));
  ASSERT(std::none_of(outputIsBroadcast, outputIsBroadcast + nbOutputs,
                      [](bool b) { return b; }));

  mConfig.type = inputTypes[1];
  ASSERT(outputTypes[0] == mConfig.type);
  const auto& index_dim = inputDims[0];
  auto axis = normalizeAxis(index_dim.nbDims);
  mConfig.preAxisElems = 1;
  mConfig.postAxisElems = 1;
  for (int i = 0; i < index_dim.nbDims; ++i) {
    if (i < axis) {
      mConfig.preAxisElems *= index_dim.d[i];
    } else {
      mConfig.postAxisElems *= index_dim.d[i];
    }
  }
}

bool OneHotPlugin::supportsFormat(DataType type,
                                  PluginFormat format) const NOEXCEPT {
  return true;
}

IPluginV2Ext* OneHotPlugin::clone() const NOEXCEPT {
  return new OneHotPlugin(*this);
}

static unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

nvinfer1::IPluginV2DynamicExt* OneHotPluginDynamic::clone() const NOEXCEPT {
  return new OneHotPluginDynamic(*this);
}

nvinfer1::DimsExprs OneHotPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) NOEXCEPT {
  assert(outputIndex == 0);
  assert(nbInputs == 2); // indices, on/off
  DimsExprs ret;

  const auto& index_dim = inputs[0];
  ret.nbDims = index_dim.nbDims + 1;
  auto axis = normalizeAxis(index_dim.nbDims);
  for (int i = 0; i < ret.nbDims; ++i) {
    if (i < axis) {
      ret.d[i] = index_dim.d[i];
    } else if (i == axis) {
      ret.d[i] = exprBuilder.constant(mConfig.depth);
    } else {
      ret.d[i] = index_dim.d[i - 1];
    }
  }
  return {ret};
}

int OneHotPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                 const nvinfer1::PluginTensorDesc* outputDesc,
                                 const void* const* inputs,
                                 void* const* outputs, void* workspace,
                                 cudaStream_t stream) NOEXCEPT {
  const int32_t* indices = reinterpret_cast<const int32_t*>(inputs[0]);
  const void* on_off_vals = inputs[1];

  void* output = outputs[0];

  pluginStatus_t status = oneHotEncoding(
      stream, mConfig.preAxisElems, mConfig.depth, mConfig.postAxisElems,
      mConfig.axis, mConfig.type, indices, on_off_vals, output);
  ASSERT(status == STATUS_SUCCESS);
  return status;
}

bool OneHotPluginDynamic::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs,
    int32_t nbOutputs) NOEXCEPT {
  ASSERT(pos >= 0 && pos < 3);
  if (pos == 0) {
    return inOut[0].type == nvinfer1::DataType::kINT32;
  }
  if (pos == 1) {
    return (inOut[1].type == nvinfer1::DataType::kFLOAT ||
            inOut[1].type == nvinfer1::DataType::kHALF ||
            inOut[1].type == nvinfer1::DataType::kINT32);
  }
  return inOut[pos].type == inOut[1].type;
}

void OneHotPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) NOEXCEPT {
  ASSERT(nbInputs == 2);
  ASSERT(nbOutputs == 1);
  ASSERT(in[0].desc.type == DataType::kINT32);

  mConfig.type = in[1].desc.type;
  ASSERT(out[0].desc.type == mConfig.type);
  const auto& index_dim = in[0].desc.dims;
  auto axis = normalizeAxis(index_dim.nbDims);
  mConfig.preAxisElems = 1;
  mConfig.postAxisElems = 1;
  for (int i = 0; i < index_dim.nbDims; ++i) {
    if (i < axis) {
      mConfig.preAxisElems *= index_dim.d[i];
    } else {
      mConfig.postAxisElems *= index_dim.d[i];
    }
  }
}

IPluginV2Ext* OneHotPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) NOEXCEPT {
  const PluginField* fields = fc->fields;
  int depth = -1;
  int axis = -1;
  int8_t explicit_batch = 1;

  for (int i = 0; i < fc->nbFields; ++i) {
    std::string field_name(fc->fields[i].name);
    const void* data = fields[i].data;
    if (field_name == "depth") {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      depth = *(static_cast<const int32_t*>(data));
    } else if (field_name == "axis") {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      axis = *(static_cast<const int32_t*>(data));
    } else if (field_name == "explicit_batch_dimension") {
      ASSERT(fields[i].type == PluginFieldType::kINT8);
      explicit_batch = *(static_cast<const int8_t*>(data));
    }
  }
  nvinfer1::IPluginV2Ext* plugin = nullptr;

  // plugin = new OneHotPlugin(
  //    name, {explicit_batch != 0, depth, axis, DataType::kFLOAT, 1, 1});

  plugin = new OneHotPluginDynamic(
      name, {false, depth, axis, DataType::kFLOAT, 1, 1});
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

IPluginV2Ext* OneHotPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) NOEXCEPT {
  // OneHotPlugin* plugin = new OneHotPlugin(name, serialData, serialLength);
  IPluginV2Ext* plugin =
      new OneHotPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}
