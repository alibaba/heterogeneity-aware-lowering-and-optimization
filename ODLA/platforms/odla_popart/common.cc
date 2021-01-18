//===- odla_common.cc -----------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

#include <ODLA/odla.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "odla_popart.h"

popart::DataType GetPopartType(odla_value_type type) {
  switch (type.element_type) {
    case ODLA_FLOAT32:
      return popart::DataType::FLOAT;
    case ODLA_FLOAT16:
      return popart::DataType::FLOAT16;
    case ODLA_INT32:
      return popart::DataType::INT32;
    case ODLA_INT64:
      return popart::DataType::INT64;
    case ODLA_BOOL:
      return popart::DataType::BOOL;
    default:
      assert(false);
  }
}

odla_element_type GetOdlaType(popart::DataType type) {
  switch (type) {
    case popart::DataType::FLOAT:
      return ODLA_FLOAT32;
    case popart::DataType::FLOAT16:
      return ODLA_FLOAT16;
    case popart::DataType::INT32:
      return ODLA_INT32;
    case popart::DataType::INT64:
      return ODLA_INT64;
    case popart::DataType::BOOL:
      return ODLA_BOOL;
    default:
      assert(false);
  }
}

std::string&& GetResizeInterpolationModeName(odla_interpolation_mode mode) {
  switch (mode) {
    case ODLA_NEAREST:
      return std::move(std::string("nearest"));
    case ODLA_LINEAR:
      return std::move(std::string("linear"));
    case ODLA_CUBIC:
      return std::move(std::string("cubic"));
    default:
      assert(false);
  }
}

std::string&& GetDirectionName(odla_rnn_direction direction) {
  switch (direction) {
    case ODLA_RNN_FORWARD:
      return std::move(std::string("forward"));
    case ODLA_RNN_REVERSE:
      return std::move(std::string("reverse"));
    case ODLA_RNN_BIDIRECTIONAL:
      return std::move(std::string("bidirectional"));
    default:
      assert(false);
  }
}

popart::Shape GetPopartShape(odla_value_shape shape) {
  popart::Shape dims(shape.size);
  for (int i = 0; i < shape.size; ++i) {
    dims[i] = shape.dims[i];
  }
  return dims;
}

std::unique_ptr<popart::IArray> MakeNDArrayWrapper(const odla_void* data_ptr,
                                                   popart::DataType data_type,
                                                   std::vector<int64_t> shape) {
  std::unique_ptr<popart::IArray> pArray;
  char* ptr = const_cast<char*>(reinterpret_cast<const char*>(data_ptr));

  switch (data_type) {
    case popart::DataType::FLOAT: {
      pArray = std::unique_ptr<popart::NDArrayWrapper<float>>(
          new popart::NDArrayWrapper<float>(reinterpret_cast<float*>(ptr),
                                            shape));
      break;
    }
    case popart::DataType::UINT32: {
      pArray = std::unique_ptr<popart::NDArrayWrapper<uint32_t>>(
          new popart::NDArrayWrapper<uint32_t>(reinterpret_cast<uint32_t*>(ptr),
                                               shape));
      break;
    }
    case popart::DataType::INT32: {
      pArray = std::unique_ptr<popart::NDArrayWrapper<int32_t>>(
          new popart::NDArrayWrapper<int32_t>(reinterpret_cast<int32_t*>(ptr),
                                              shape));
      break;
    }
    case popart::DataType::INT64: {
      pArray = std::unique_ptr<popart::NDArrayWrapper<int64_t>>(
          new popart::NDArrayWrapper<int64_t>(reinterpret_cast<int64_t*>(ptr),
                                              shape));
      break;
    }
    case popart::DataType::BOOL: {
      pArray = std::unique_ptr<popart::NDArrayWrapper<bool>>(
          new popart::NDArrayWrapper<bool>(reinterpret_cast<bool*>(ptr),
                                           shape));
      break;
    }
    default:
      assert(false);
  }

  return pArray;
}

std::string&& GetTypeName(odla_element_type type_value) {
  std::string result;
  switch (type_value) {
    case ODLA_INT8:
      return std::move(std::string("INT8"));
    case ODLA_INT16:
      return std::move(std::string("INT16"));
    case ODLA_INT32:
      return std::move(std::string("INT32"));
    case ODLA_INT64:
      return std::move(std::string("INT64"));
    case ODLA_UINT8:
      return std::move(std::string("UINT8"));
    case ODLA_UINT16:
      return std::move(std::string("UINT16"));
    case ODLA_UINT32:
      return std::move(std::string("UINT32"));
    case ODLA_UINT64:
      return std::move(std::string("UINT64"));
    case ODLA_FLOAT16:
      return std::move(std::string("FLOAT16"));
    case ODLA_BFLOAT16:
      return std::move(std::string("BFLOAT16"));
    case ODLA_FLOAT32:
      return std::move(std::string("FLOAT"));
    case ODLA_FLOAT64:
      return std::move(std::string("DOUBLE"));
    case ODLA_BOOL:
      return std::move(std::string("BOOL"));
    default:
      assert(false);
  }
}
