//===- odla_dnnl_debug.cc -------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "ODLA/odla_common.h"
#include "odla_dnnl.h"

template <typename T>
static void Dump(const T* data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    std::cerr << data[i] << " ";
  }
  std::cerr << "\n";
}

void odla_Dump(odla_value value) {
  size_t n = GetTotalElements(value->shape);
  auto op = [value, n]() {
    std::cerr << "Name:" << value->name << "\n";
    std::cerr << "Shape: [";
    for (int i = 0; i < value->shape.size; ++i) {
      std::cerr << value->shape.dims[i] << " ";
    }
    std::cerr << "]\n";

    const void* ptr = value->mem.get_data_handle();
    switch (value->elem_type) {
      case ODLA_FLOAT32:
        return Dump(static_cast<const float*>(ptr), n);
      case ODLA_INT8:
        return Dump(static_cast<const int8_t*>(ptr), n);
      case ODLA_UINT8:
        return Dump(static_cast<const uint8_t*>(ptr), n);
      case ODLA_STRING:
        return Dump(static_cast<const char* const*>(ptr), n);
      default:
        std::cerr << "<unimplemented>\n";
    }
    std::cerr << "]\n";
  };
  add_op(op);
  InterpretIfNeeded();
}
