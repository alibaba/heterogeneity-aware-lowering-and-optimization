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

extern pluginStatus_t oneHotEncoding(cudaStream_t stream,
                                     int64_t pre_axis_elems, int depth,
                                     int64_t post_axis_elems, int axis,
                                     nvinfer1::DataType data_type,
                                     const int32_t* indices, const void* on_off,
                                     void* output);

OneHotPlugin::OneHotPlugin(const char* name, bool explicit_batch, int depth,
                           int axis)
    : mLayerName(name),
      mExplicitBatch(explicit_batch),
      mDepth(depth),
      mAxis(axis),
      mType(DataType::kFLOAT),
      mPreAxisElems(1),
      mPostAxisElems(1),
      mNamespace("") {}

OneHotPlugin::OneHotPlugin(const char* name, const void* data, size_t length)
    : mLayerName(name) {
  const char* begin = reinterpret_cast<const char*>(data);
  const char* d = begin;
  mExplicitBatch = read<bool>(d);
  mDepth = read<int>(d);
  mAxis = read<int>(d);
  mType = read<DataType>(d);
  mPreAxisElems = read<int64_t>(d);
  mPostAxisElems = read<int64_t>(d);
  ASSERT(d == begin + length);
}

size_t OneHotPlugin::getSerializationSize() const NOEXCEPT {
  return sizeof(mExplicitBatch) + sizeof(mDepth) + sizeof(mAxis) +
         sizeof(mType) + sizeof(mPreAxisElems) + sizeof(mPostAxisElems);
}

void OneHotPlugin::serialize(void* buffer) const NOEXCEPT {
  char* begin = reinterpret_cast<char*>(buffer);
  char* d = begin;
  write(d, mExplicitBatch);
  write(d, mDepth);
  write(d, mAxis);
  write(d, mType);
  write(d, mPreAxisElems);
  write(d, mPostAxisElems);
  ASSERT(d == begin + getSerializationSize());
}

int OneHotPlugin::normalizeAxis(const Dims& index_dim) {
  auto input_rank = mExplicitBatch ? index_dim.nbDims + 1 : index_dim.nbDims;
  mAxis = mAxis < 0 ? mAxis + input_rank + 1 : mAxis;
  if (mExplicitBatch) {
    ASSERT(mAxis != 0); // dim 0 is always batch dim.
  }
  ASSERT(mAxis >= 0 && mAxis <= input_rank);
  return mExplicitBatch ? mAxis - 1 : mAxis;
}

Dims OneHotPlugin::getOutputDimensions(int index, const Dims* inputs,
                                       int nbInputDims) NOEXCEPT {
  ASSERT(nbInputDims == 2);
  ASSERT(index >= 0 && index < this->getNbOutputs());
  const auto& index_dim = inputs[0];
  if (!mExplicitBatch) {
    ASSERT(inputs[1].nbDims == 1 && inputs[1].d[0] == 2);
  }
  Dims dim;
  dim.nbDims = index_dim.nbDims + 1;

  auto axis = normalizeAxis(index_dim);
  for (int i = 0; i < dim.nbDims; ++i) {
    if (i < axis) {
      dim.d[i] = index_dim.d[i];
    } else if (i == axis) {
      dim.d[i] = mDepth;
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
      stream, (mExplicitBatch ? batchSize : 1) * mPreAxisElems, mDepth,
      mPostAxisElems, mAxis, mType, indices, on_off_vals, output);
  ASSERT(status == STATUS_SUCCESS);
  return 0;
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

  mType = inputTypes[1];
  ASSERT(outputTypes[0] == mType);
  const auto& index_dim = inputDims[0];
  auto axis = normalizeAxis(index_dim);
  mPreAxisElems = 1;
  mPostAxisElems = 1;
  for (int i = 0; i < index_dim.nbDims; ++i) {
    if (i < axis) {
      mPreAxisElems *= index_dim.d[i];
    } else {
      mPostAxisElems *= index_dim.d[i];
    }
  }
}

bool OneHotPlugin::supportsFormat(DataType type,
                                  PluginFormat format) const NOEXCEPT {
  return true;
}

const char* OneHotPlugin::getPluginType() const NOEXCEPT {
  return ONE_HOT_PLUGIN_NAME;
}

const char* OneHotPlugin::getPluginVersion() const NOEXCEPT {
  return ONE_HOT_PLUGIN_VERSION;
}

void OneHotPlugin::destroy() NOEXCEPT { delete this; }

IPluginV2Ext* OneHotPlugin::clone() const NOEXCEPT {
  auto* plugin =
      new OneHotPlugin(mLayerName.c_str(), mExplicitBatch, mDepth, mAxis);
  plugin->mType = mType;
  plugin->mPreAxisElems = mPreAxisElems;
  plugin->mPostAxisElems = mPostAxisElems;
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

nvinfer1::DataType OneHotPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes,
    int nbInputs) const NOEXCEPT {
  ASSERT(index == 0 && nbInputs > 1);
  return inputTypes[1];
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

  OneHotPlugin* plugin =
      new OneHotPlugin(name, explicit_batch != 0, depth, axis);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

IPluginV2Ext* OneHotPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) NOEXCEPT {
  OneHotPlugin* plugin = new OneHotPlugin(name, serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}
