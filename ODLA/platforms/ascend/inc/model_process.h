#pragma once
#include <ODLA/odla.h>

#include <iostream>

#include "acl/acl.h"
#include "ge_ir_build.h"

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__)
using namespace ge;

class ModelProcess {
 public:
  ModelProcess();

  ~ModelProcess();

  odla_status LoadModelFromWithMem(ge::ModelBufferData ModelBufferData_);

  void Unload();

  /**
   * @brief create model desc
   * @return result
   */
  odla_status CreateDesc();

  /**
   * @brief destroy desc
   */
  void DestroyDesc();

  /**
   * @brief create model input
   * @param [in] inputDataBuffer: input buffer
   * @param [in] bufferSize: input buffer size
   * @return result
   */
  //    odla_status CreateInput(Tensor &input_tensors);
  odla_status CreateInput(void* input_ptr, size_t input_size);

  /**
   * @brief destroy input resource
   */
  void DestroyInput();

  /**
   * @brief create output buffer
   * @return result
   */
  odla_status CreateOutput();

  /**
   * @brief destroy output resource
   */
  void DestroyOutput();

  /**
   * @brief model execute
   * @return result
   */
  odla_status Execute();

  /**
   * @brief dump model output result to file
   */
  //    void DumpModelOutputResult(Tensor &input_tensors);
  void DumpModelOutputResult(void* output_ptr);

 private:
  uint32_t modelId_;
  size_t modelMemSize_;
  size_t modelWeightSize_;
  void* modelMemPtr_;
  void* modelWeightPtr_;
  bool loadFlag_; // model load flag
  aclmdlDesc* modelDesc_;
  aclmdlDataset* input_;
  aclmdlDataset* output_;
};