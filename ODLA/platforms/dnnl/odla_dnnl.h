//===- odla_dnnl.h --------------------------------------------------------===//
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

#ifndef ODLA_DNNL_H_
#define ODLA_DNNL_H_

#include <ODLA/odla.h>

#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "ODLA/odla_common.h"
#include "dnnl.hpp"

extern thread_local odla_computation g_comp;

void add_op(dnnl::primitive prim,
            const std::unordered_map<int, dnnl::memory>& args)
    __attribute__((visibility("hidden")));

void add_op(std::function<void()> func) __attribute__((visibility("hidden")));

void InterpretIfNeeded() __attribute__((visibility("hidden")));
struct _odla_value {
  dnnl::memory mem;
  bool is_const;
  uint8_t elem_size;
  odla_element_type elem_type; // TODO: use odla_value_type
  odla_value_shape shape;
  std::string name;
  _odla_value(const dnnl::memory& m, const odla_value_shape& shape_,
              const std::string& id)
      : mem(m),
        is_const(false),
        shape(shape_),
        name(id),
        elem_type(ODLA_FLOAT32),
        elem_size(4) {
    if (shape.size == 0) {
      shape.size = 1;
      shape.dims[0] = 1;
    }
  }
};

dnnl::memory cast_odla_mem(dnnl::memory src_mem, const odla_value_shape shape,
                           dnnl::memory::data_type dt, bool is_const)
    __attribute__((visibility("hidden")));

struct operation {
  void execute(dnnl::stream& stream) {
    if (func) {
      return func();
    }
    prim.execute(stream, args);
  }
  dnnl::primitive prim;
  std::function<void()> func;
  std::unordered_map<int, dnnl::memory> args;
};

typedef struct TargetOpts {
  odla_bf16_mode bf16_mode;
} target_opts;

struct _odla_computation {
  dnnl::engine eng;
  // std::vector<dnnl::primitive> primitives;
  // std::vector<std::unordered_map<int, dnnl::memory>> args;
  std::vector<operation> ops;
  std::vector<std::unique_ptr<_odla_value>> vals;
  std::unordered_map<std::string, odla_value> inputs;
  std::unordered_map<std::string, odla_value> outputs;
  std::vector<odla_value> input_vals;
  std::vector<odla_value> output_vals;
  std::unordered_map<std::string, std::pair<odla_value, void*>> outputs_v;
  target_opts opts;
  std::vector<std::vector<char>> bufs;
  void* CreateBuffer(size_t len) {
    bufs.push_back(std::vector<char>(len));
    return bufs.back().data();
  }
  _odla_computation() : eng(dnnl::engine::kind::cpu, 0), opts({BF16_DISABLE}) {}
};

static inline dnnl::memory::dims getDims(const odla_value_shape& od) {
  auto dims = dnnl::memory::dims(od.dims, od.dims + od.size);
  return dims;
}

static inline dnnl::memory::format_tag getFormatTag(
    const odla_value_shape& od) {
  static const dnnl::memory::format_tag tags[] = {
      dnnl::memory::format_tag::undef,  dnnl::memory::format_tag::a,
      dnnl::memory::format_tag::ab,     dnnl::memory::format_tag::abc,
      dnnl::memory::format_tag::abcd,   dnnl::memory::format_tag::abcde,
      dnnl::memory::format_tag::abcdef,
  };
  return (od.size <= 0 || od.size > 6) ? tags[0] : tags[od.size];
}

static inline dnnl::memory::format_tag getFormatTag(odla_memory_layout layout,
                                                    unsigned group = 1) {
  switch (layout) {
    case ODLA_CHANNELS_FIRST:
      return dnnl::memory::format_tag::nchw;
    case ODLA_CHANNELS_LAST:
      return dnnl::memory::format_tag::nhwc;
    case ODLA_SIO:
      return (group > 1) ? dnnl::memory::format_tag::hwigo
                         : dnnl::memory::format_tag::hwio;
    case ODLA_OIS:
      return (group > 1) ? dnnl::memory::format_tag::goihw
                         : dnnl::memory::format_tag::oihw;
    case ODLA_IOS:
      return (group > 1) ? dnnl::memory::format_tag::giohw
                         : dnnl::memory::format_tag::iohw;
    default:
      assert(0);
      return dnnl::memory::format_tag::any;
  }
}

static inline int GetDataSize(dnnl::memory::data_type dt) {
  switch (dt) {
    case dnnl::memory::data_type::u8:
    case dnnl::memory::data_type::s8:
      return 1;
    case dnnl::memory::data_type::bf16:
      return 2;
    case dnnl::memory::data_type::f32:
    case dnnl::memory::data_type::s32:
      return 4;
  }
  assert(0 && "invalid data type");
  return 0;
}

static inline dnnl::memory::data_type getDataType(const odla_element_type ty) {
  dnnl::memory::data_type dt;
  switch (ty) {
    case ODLA_FLOAT64: // FIXME: This is a temporarily workaround.
    case ODLA_FLOAT32:
      dt = dnnl::memory::data_type::f32;
      break;
    case ODLA_INT8:
      dt = dnnl::memory::data_type::s8;
      break;
    case ODLA_UINT8:
      dt = dnnl::memory::data_type::u8;
      break;
    case ODLA_INT32:
      dt = dnnl::memory::data_type::s32;
      break;
    case ODLA_INT64:
      dt = dnnl::memory::data_type::s32; // FIXME:
      break;
    case ODLA_BFLOAT16:
      dt = dnnl::memory::data_type::bf16;
      break;
    case ODLA_STRING:
      dt = dnnl::memory::data_type::u8; // Actual storage is pointer but DNNL
      // has no word-sized type.
      break;
    default:
      dt = dnnl::memory::data_type::undef;
  }
  return dt;
}

static inline dnnl::memory::desc getMemoryDesc(const odla_value_shape& dims,
                                               dnnl::memory::data_type ty) {
  return dnnl::memory::desc(getDims(dims), ty, getFormatTag(dims));
}

static inline dnnl::memory::desc getMemoryDesc(const odla_value_shape& dims,
                                               odla_element_type ty) {
  return dnnl::memory::desc(getDims(dims), getDataType(ty), getFormatTag(dims));
}

static inline dnnl::memory::desc getMemoryDesc(const odla_value_type& ty) {
  return getMemoryDesc(ty.shape, ty.element_type);
}

static inline int64_t GetTotalElements(const odla_value_shape& dims) {
  return std::accumulate(dims.dims, dims.dims + dims.size, 1,
                         std::multiplies<size_t>());
}

// The strides used by DNNL's memory desc.
static inline dnnl::memory::dims getStrides(const odla_value_shape& od) {
  std::vector<int64_t> strides(od.size, 1);
  for (int i = od.size - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * od.dims[i + 1];
  auto dims = dnnl::memory::dims(strides);
  return dims;
}

// get shape total element size.
static inline int64_t GetCountFromAxis(const odla_value_shape& shape,
                                       const odla_int32 axis) {
  int64_t size = 1;
  for (int i = axis; i < shape.size; i++) {
    size = size * shape.dims[i];
  }
  return size;
}

static inline odla_value CreateValue(const dnnl::memory& mem,
                                     const odla_value_shape shape,
                                     const odla_value_id id) {
  std::string name = id == nullptr ? "" : std::string((const char*)id);
  auto v = std::make_unique<_odla_value>(mem, shape, name);
  auto ret = v.get();
  g_comp->vals.push_back(std::move(v));
  return ret;
}

#endif // ODLA_DNNL_H_