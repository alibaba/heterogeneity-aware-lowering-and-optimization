//===- oneHot.cu ----------------------------------------------------------===//
//
// Copyright (C) 2019-2022 Alibaba Group Holding Limited.
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

#include "../common.h"
#include "kernel.h"

template <int TPB, typename T>
__global__ void onehot_kernel(T* output, int64_t output_elems,
                              const int32_t* indices, int64_t input_elems,
                              int32_t depth, int64_t dims_post, T on, T off) {
  for (int64_t i = blockIdx.x * TPB + threadIdx.x; i < output_elems;
       i += gridDim.x * TPB) {
    output[i] = off;
  }

  int64_t extend_a = depth * dims_post;
  for (int64_t i = blockIdx.x * TPB + threadIdx.x; i < input_elems;
       i += gridDim.x * TPB) {
    int d = indices[i];
    int a = i / dims_post;
    int b = i % dims_post;
    int64_t idx = a * extend_a + d * dims_post + b;
    output[idx] = on;
  }
}

pluginStatus_t oneHotEncoding(cudaStream_t stream, int64_t pre_axis_elems,
                              int depth, int64_t post_axis_elems, int axis,
                              nvinfer1::DataType data_type,
                              const int32_t* indices, const void* on_off,
                              void* output) {
  int64_t input_elems = pre_axis_elems * depth * post_axis_elems;
  int64_t output_elems = depth * input_elems;
  constexpr int BS = 512;
  const int GS = (input_elems + BS - 1) / BS;
  if (data_type == nvinfer1::DataType::kFLOAT) {
    const float* ptr = reinterpret_cast<const float*>(on_off);
    float on = ptr[0];
    float off = ptr[1];
    onehot_kernel<BS, float>
        <<<GS, BS, 0, stream>>>((float*)output, output_elems, indices,
                                input_elems, depth, post_axis_elems, on, off);
  } else if (data_type == nvinfer1::DataType::kINT32) {
    const int32_t* ptr = reinterpret_cast<const int32_t*>(on_off);
    int32_t on = ptr[0];
    int32_t off = ptr[1];
    onehot_kernel<BS, int32_t>
        <<<GS, BS, 0, stream>>>((int32_t*)output, output_elems, indices,
                                input_elems, depth, post_axis_elems, on, off);

  } else if (data_type == nvinfer1::DataType::kHALF) {
    const half* ptr = reinterpret_cast<const half*>(on_off);
    half on = ptr[0];
    half off = ptr[1];
    onehot_kernel<BS, half><<<GS, BS, 0, stream>>>((half*)output, output_elems,
                                                   indices, input_elems, depth,
                                                   post_axis_elems, on, off);

  } else {
    ASSERT(data_type == nvinfer1::DataType::kBOOL ||
           data_type == nvinfer1::DataType::kINT8);
    const int8_t* ptr = reinterpret_cast<const int8_t*>(on_off);
    int8_t on = ptr[0];
    int8_t off = ptr[1];
    onehot_kernel<BS, int8_t>
        <<<GS, BS, 0, stream>>>((int8_t*)output, output_elems, indices,
                                input_elems, depth, post_axis_elems, on, off);
  }
  return STATUS_NOT_SUPPORTED;
}