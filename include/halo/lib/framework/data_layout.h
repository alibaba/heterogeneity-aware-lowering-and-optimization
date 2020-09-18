//===- data_layout.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
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

#ifndef HALO_LIB_FRAMEWORK_DATA_LAYOUT_H_
#define HALO_LIB_FRAMEWORK_DATA_LAYOUT_H_

#include "halo/api/halo_data.h"
#include "halo/lib/framework/type.h"

namespace halo {

/// This class definds the interface for querying about the storage information.
class DataLayout {
 public:
  /// Returns the bits of a single piece of data of type `dt` in memory.
  virtual size_t Bits(DataType dt) const noexcept = 0;
  /// Returns the bytes of a single piece of data of type `dt` used in memory.
  virtual size_t Bytes(DataType dt) const noexcept = 0;
  /// Returns the alignment in bytes of data type `dt`.
  virtual size_t Alignment(DataType dt) const noexcept = 0;
  /// Returns the total bytes of `n` pieces of data of type `dt`in memory,
  /// including both inter- and intra- element paddings.
  virtual size_t Bytes(DataType, size_t n) const noexcept = 0;
  /// Returns the padding bits of a single piece of data of type `dt` in memory.
  size_t PaddingsInBits(DataType dt) const noexcept {
    return Bytes(dt) * 8 - Bits(dt);
  }
  /// Returns the total size in bytes of `type` in memory.
  size_t Bytes(const Type& type) const noexcept {
    return Bytes(type.GetDataType(), type.GetTotalNumOfElements());
  }
};

/// This class defines the default data layout.
class DefaultDataLayout : public DataLayout {
 public:
  size_t Bits(DataType dt) const noexcept override {
    switch (dt) {
      case DataType::INT8:
      case DataType::UINT8:
      case DataType::BOOL: {
        return 8;
      }
      case DataType::INT16:
      case DataType::UINT16:
      case DataType::FLOAT16: {
        return 16;
      }
      case DataType::INT32:
      case DataType::UINT32:
      case DataType::FLOAT32: {
        return 32;
      }
      case DataType::INT64:
      case DataType::UINT64: {
        return 64;
      }
      case DataType::STRING:
      case DataType::QINT2:
      case DataType::QINT3:
      case DataType::QINT4:
      case DataType::INVALID: {
        // target dependent:
        return 0;
      }
    }
    return 0;
  }

  size_t Alignment(DataType dt) const noexcept override {
    switch (dt) {
      case DataType::INT16:
      case DataType::UINT16:
      case DataType::FLOAT16: {
        return alignof(int16_t);
      }
      case DataType::INT32:
      case DataType::UINT32:
      case DataType::FLOAT32: {
        return alignof(int32_t);
      }
      default: {
        return 1;
      }
    }
  }

  size_t Bytes(DataType dt) const noexcept override { return Bits(dt) / 8; }
  size_t Bytes(DataType dt, size_t n) const noexcept override {
    return n * Bytes(dt);
  }
};

} // namespace halo

#endif // HALO_LIB_FRAMEWORK_DATA_LAYOUT_H_