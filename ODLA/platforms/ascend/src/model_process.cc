#include "model_process.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>

#include "ODLA/odla.h"
#include "ge_ir_build.h"
#include "odla_ascend_acl.h"

using namespace ge;

using namespace std;
extern bool g_isDevice;

ModelProcess::ModelProcess()
    : modelId_(0),
      modelMemSize_(0),
      modelWeightSize_(0),
      modelMemPtr_(nullptr),
      modelWeightPtr_(nullptr),
      loadFlag_(false),
      modelDesc_(nullptr),
      input_(nullptr),
      output_(nullptr) {}

ModelProcess::~ModelProcess() {
  DestroyDesc();
  DestroyInput();
  DestroyOutput();
}

odla_status ModelProcess::LoadModelFromWithMem(
    ge::ModelBufferData ModelBufferData_) {
  if (loadFlag_) {
    ERROR_LOG("has already loaded a model");
    return ODLA_FAILURE;
  }
  aclError ret = aclmdlLoadFromMem(ModelBufferData_.data.get(),
                                   ModelBufferData_.length, &modelId_);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("load model from memory failed");
    return ODLA_FAILURE;
  }

  loadFlag_ = true;
  INFO_LOG("load model success");
  return ODLA_SUCCESS;
}

odla_status ModelProcess::CreateInput(void* input_ptr, size_t input_size) {
  void* inputBuffer = nullptr;
  aclError ret =
      aclrtMalloc(&inputBuffer, input_size, ACL_MEM_MALLOC_NORMAL_ONLY);
  ret = aclrtMemcpy(inputBuffer, input_size, input_ptr, input_size,
                    ACL_MEMCPY_HOST_TO_DEVICE);

  input_ = aclmdlCreateDataset();
  if (input_ == nullptr) {
    ERROR_LOG("can't create dataset, create input failed");
    return ODLA_FAILURE;
  }

  aclDataBuffer* inputData = aclCreateDataBuffer(inputBuffer, input_size);
  if (inputData == nullptr) {
    ERROR_LOG("can't create data buffer, create input failed");
    return ODLA_FAILURE;
  }

  ret = aclmdlAddDatasetBuffer(input_, inputData);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("add input dataset buffer failed");
    aclrtFree(inputBuffer);
    aclDestroyDataBuffer(inputData);
    return ODLA_FAILURE;
  }
  INFO_LOG("create model input success");

  return ODLA_SUCCESS;
}

void ModelProcess::DestroyInput() {
  if (input_ == nullptr) {
    return;
  }

  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(input_, i);
    aclDestroyDataBuffer(dataBuffer);
  }
  aclmdlDestroyDataset(input_);
  input_ = nullptr;
}

odla_status ModelProcess::CreateDesc() {
  modelDesc_ = aclmdlCreateDesc();
  if (modelDesc_ == nullptr) {
    ERROR_LOG("create model description failed");
    return ODLA_FAILURE;
  }

  aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
  if (ret != ACL_SUCCESS) {
    ERROR_LOG("get model description failed");
    return ODLA_FAILURE;
  }

  INFO_LOG("create model description success");

  return ODLA_SUCCESS;
}

void ModelProcess::DestroyDesc() {
  if (modelDesc_ != nullptr) {
    (void)aclmdlDestroyDesc(modelDesc_);
    modelDesc_ = nullptr;
  }
}

odla_status ModelProcess::CreateOutput() {
  if (modelDesc_ == nullptr) {
    ERROR_LOG("no model description, create ouput failed");
    return ODLA_FAILURE;
  }

  output_ = aclmdlCreateDataset();
  if (output_ == nullptr) {
    ERROR_LOG("can't create dataset, create output failed");
    return ODLA_FAILURE;
  }

  size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
  for (size_t i = 0; i < outputSize; ++i) {
    size_t buffer_size = aclmdlGetOutputSizeByIndex(modelDesc_, i);

    void* outputBuffer = nullptr;
    aclError ret =
        aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("can't malloc buffer, size is %zu, create output failed",
                buffer_size);
      return ODLA_FAILURE;
    }

    aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("can't create data buffer, create output failed");
      aclrtFree(outputBuffer);
      return ODLA_FAILURE;
    }

    ret = aclmdlAddDatasetBuffer(output_, outputData);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("can't add data buffer, create output failed");
      aclrtFree(outputBuffer);
      aclDestroyDataBuffer(outputData);
      return ODLA_FAILURE;
    }
  }

  INFO_LOG("create model output success");
  return ODLA_SUCCESS;
}

void ModelProcess::DumpModelOutputResult(void* output_ptr) {
  size_t outputNum = aclmdlGetDatasetNumBuffers(output_);

  for (size_t i = 0; i < outputNum; ++i) {
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
    if (dataBuffer == nullptr) {
      ERROR_LOG(
          "Get the dataset buffer from model "
          "inference output failed");
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
      ERROR_LOG(
          "Get the dataset buffer address "
          "from model inference output failed");
    }

    size_t bufferSize = aclGetDataBufferSizeV2(dataBuffer);
    if (bufferSize == 0) {
      ERROR_LOG(
          "The dataset buffer size of "
          "model inference output is 0");
    }

    aclError ret = aclrtMemcpy(output_ptr, bufferSize, dataBufferDev,
                               bufferSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("Memcpy inference result to host failed, error %d", ret);
      break;
    }
  }

  INFO_LOG("dump data success");
  return;
}

void ModelProcess::DestroyOutput() {
  if (output_ == nullptr) {
    return;
  }

  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
    void* data = aclGetDataBufferAddr(dataBuffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(dataBuffer);
  }

  (void)aclmdlDestroyDataset(output_);
  output_ = nullptr;
}

odla_status ModelProcess::Execute() {
  aclError ret = aclmdlExecute(modelId_, input_, output_);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("execute model failed, modelId is %u", modelId_);
    return ODLA_FAILURE;
  }

  INFO_LOG("model execute success");
  return ODLA_SUCCESS;
}

void ModelProcess::Unload() {
  if (!loadFlag_) {
    ERROR_LOG("no model had been loaded, unload failed");
    return;
  }

  aclError ret = aclmdlUnload(modelId_);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("unload model failed, modelId is %u", modelId_);
  }

  if (modelDesc_ != nullptr) {
    (void)aclmdlDestroyDesc(modelDesc_);
    modelDesc_ = nullptr;
  }

  loadFlag_ = false;
  INFO_LOG("unload model success, modelId is %u", modelId_);
}
