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
//===- batchedNMSPlugin.cpp -----------------------------------------------===//
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

#include "batchedNMSPlugin.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::NMSParameters;

const char* NMS_PLUGIN_VERSION{"1"};
const char* NMS_PLUGIN_NAME{"BatchedNMS_TRT_V2"};

PluginFieldCollection BatchedNMSPluginV2Creator::mFC{};
std::vector<PluginField> BatchedNMSPluginV2Creator::mPluginAttributes;

BatchedNMSPluginV2::BatchedNMSPluginV2(NMSParameters params) : param(params) {}

BatchedNMSPluginV2::BatchedNMSPluginV2(const void* data, size_t length) {
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  param = read<NMSParameters>(d);
  boxesSize = read<int>(d);
  scoresSize = read<int>(d);
  numPriors = read<int>(d);
  mClipBoxes = read<bool>(d);
  ASSERT(d == a + length);
}

int BatchedNMSPluginV2::getNbOutputs() const { return 5; }

int BatchedNMSPluginV2::initialize() { return STATUS_SUCCESS; }

void BatchedNMSPluginV2::terminate() {}

Dims BatchedNMSPluginV2::getOutputDimensions(int index, const Dims* inputs,
                                             int nbInputDims) {
  ASSERT(nbInputDims == 2);
  ASSERT(index >= 0 && index < this->getNbOutputs());
  ASSERT(inputs[0].nbDims == 3);
  ASSERT(inputs[1].nbDims == 2);
  // boxesSize: number of box coordinates for one sample
  boxesSize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
  // scoresSize: number of scores for one sample
  scoresSize = inputs[1].d[0] * inputs[1].d[1];
  // num_detections
  if (index == 0) {
    Dims dim0{};
    dim0.nbDims = 0;
    return dim0;
  }
  // nmsed_boxes
  if (index == 1) {
    return DimsHW(param.keepTopK, 4);
  }
  // nmsed_indices
  else if (index == 4) {
    return DimsHW(param.keepTopK, 3);
  }
  // nmsed_scores or nmsed_classes
  Dims dim1{};
  dim1.nbDims = 1;
  dim1.d[0] = param.keepTopK;
  return dim1;
}

size_t BatchedNMSPluginV2::getWorkspaceSize(int maxBatchSize) const {
  return detectionInferenceWorkspaceSize(
      param.shareLocation, maxBatchSize, boxesSize, scoresSize,
      param.numClasses, numPriors, param.topK, DataType::kFLOAT,
      DataType::kFLOAT);
}

int BatchedNMSPluginV2::enqueue(int batchSize, const void* const* inputs,
                                void** outputs, void* workspace,
                                cudaStream_t stream) {
  const void* const locData = inputs[0];
  const void* const confData = inputs[1];

  void* keepCount = outputs[0];
  void* nmsedBoxes = outputs[1];
  void* nmsedScores = outputs[2];
  void* nmsedClasses = outputs[3];
  void* nmsedIndices = outputs[4];

  pluginStatus_t status = nmsInference(
      stream, batchSize, boxesSize, scoresSize, param.shareLocation,
      param.backgroundLabelId, numPriors, param.numClasses, param.topK,
      param.keepTopK, param.scoreThreshold, param.iouThreshold,
      DataType::kFLOAT, locData, DataType::kFLOAT, confData, keepCount,
      nmsedBoxes, nmsedScores, nmsedClasses, nmsedIndices, workspace,
      param.isNormalized, false, mClipBoxes);
  ASSERT(status == STATUS_SUCCESS);
  return 0;
}

size_t BatchedNMSPluginV2::getSerializationSize() const {
  // NMSParameters, boxesSize,scoresSize,numPriors
  return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool);
}

void BatchedNMSPluginV2::serialize(void* buffer) const {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write(d, param);
  write(d, boxesSize);
  write(d, scoresSize);
  write(d, numPriors);
  write(d, mClipBoxes);
  ASSERT(d == a + getSerializationSize());
}

void BatchedNMSPluginV2::configurePlugin(
    const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes,
    const bool* inputIsBroadcast, const bool* outputIsBroadcast,
    nvinfer1::PluginFormat format, int maxBatchSize) {
  ASSERT(nbInputs == 2);
  ASSERT(nbOutputs == 5);
  ASSERT(inputDims[0].nbDims == 3);
  ASSERT(inputDims[1].nbDims == 2);
  ASSERT(std::none_of(inputIsBroadcast, inputIsBroadcast + nbInputs,
                      [](bool b) { return b; }));
  ASSERT(std::none_of(outputIsBroadcast, outputIsBroadcast + nbInputs,
                      [](bool b) { return b; }));

  boxesSize = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
  scoresSize = inputDims[1].d[0] * inputDims[1].d[1];
  // num_boxes
  numPriors = inputDims[0].d[0];
  const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
  // Third dimension of boxes must be either 1 or num_classes
  ASSERT(inputDims[0].d[1] == numLocClasses);
  ASSERT(inputDims[0].d[2] == 4);
}

bool BatchedNMSPluginV2::supportsFormat(DataType type,
                                        PluginFormat format) const {
  return ((type == DataType::kFLOAT || type == DataType::kINT32) &&
          format == PluginFormat::kNCHW);
}
const char* BatchedNMSPluginV2::getPluginType() const {
  return NMS_PLUGIN_NAME;
}

const char* BatchedNMSPluginV2::getPluginVersion() const {
  return NMS_PLUGIN_VERSION;
}

void BatchedNMSPluginV2::destroy() { delete this; }

IPluginV2Ext* BatchedNMSPluginV2::clone() const {
  auto* plugin = new BatchedNMSPluginV2(param);
  plugin->boxesSize = boxesSize;
  plugin->scoresSize = scoresSize;
  plugin->numPriors = numPriors;
  plugin->setPluginNamespace(mNamespace.c_str());
  plugin->setClipParam(mClipBoxes);
  return plugin;
}

void BatchedNMSPluginV2::setPluginNamespace(const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

const char* BatchedNMSPluginV2::getPluginNamespace() const {
  return mPluginNamespace;
}

nvinfer1::DataType BatchedNMSPluginV2::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  if (index == 0 || index == 4) {
    return nvinfer1::DataType::kINT32;
  }
  return inputTypes[0];
}

void BatchedNMSPluginV2::setClipParam(bool clip) { mClipBoxes = clip; }

bool BatchedNMSPluginV2::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {
  return false;
}

bool BatchedNMSPluginV2::canBroadcastInputAcrossBatch(int inputIndex) const {
  return false;
}

BatchedNMSPluginV2Creator::BatchedNMSPluginV2Creator() : params{} {
  mPluginAttributes.emplace_back(
      PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("clipBoxes", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* BatchedNMSPluginV2Creator::getPluginName() const {
  return NMS_PLUGIN_NAME;
}

const char* BatchedNMSPluginV2Creator::getPluginVersion() const {
  return NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* BatchedNMSPluginV2Creator::getFieldNames() {
  return &mFC;
}

IPluginV2Ext* BatchedNMSPluginV2Creator::createPlugin(
    const char* name, const PluginFieldCollection* fc) {
  const PluginField* fields = fc->fields;
  mClipBoxes = true;

  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "shareLocation")) {
      params.shareLocation = *(static_cast<const bool*>(fields[i].data));
    } else if (!strcmp(attrName, "backgroundLabelId")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "numClasses")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.numClasses = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "topK")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.topK = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "keepTopK")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.keepTopK = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "scoreThreshold")) {
      ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
      params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "iouThreshold")) {
      ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
      params.iouThreshold = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "isNormalized")) {
      params.isNormalized = *(static_cast<const bool*>(fields[i].data));
    } else if (!strcmp(attrName, "clipBoxes")) {
      mClipBoxes = *(static_cast<const bool*>(fields[i].data));
    }
  }

  BatchedNMSPluginV2* plugin = new BatchedNMSPluginV2(params);
  plugin->setClipParam(mClipBoxes);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

IPluginV2Ext* BatchedNMSPluginV2Creator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call NMS::destroy()
  BatchedNMSPluginV2* plugin = new BatchedNMSPluginV2(serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}
