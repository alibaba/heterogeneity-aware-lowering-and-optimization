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
  odla_element_type elem_type; // TODO: use odla_value_type
  odla_value_shape shape;
  std::string name;
  _odla_value(const dnnl::memory& m, const odla_value_shape& shape_,
              const std::string& id)
      : mem(m),
        is_const(false),
        shape(shape_),
        name(id),
        elem_type(ODLA_FLOAT32) {
    if (shape.size == 0) {
      shape.size = 1;
      shape.dims[0] = 1;
    }
  }
};

dnnl::memory cast_odla_mem(dnnl::memory src_mem, const odla_value_shape& shape,
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

dnnl::memory cast_op(odla_value& input, dnnl::memory::data_type dt)
    __attribute__((visibility("hidden")));
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

#ifdef ODLA_BUILD_DNNL_GPU
  _odla_computation() : eng(dnnl::engine::kind::gpu, 0), opts({BF16_DISABLE}) {}
#else
  _odla_computation() : eng(dnnl::engine::kind::cpu, 0), opts({BF16_DISABLE}) {}
#endif
};

static inline dnnl::memory::dims getDims(const odla_value_shape& od) {
  auto dims = dnnl::memory::dims(od.dims, od.dims + od.size);
  return dims;
}

static inline dnnl::memory::format_tag getFormatTag(
    const odla_value_shape& od) {
  static const dnnl::memory::format_tag tags[] = {
      dnnl::memory::format_tag::undef,
      dnnl::memory::format_tag::a,
      dnnl::memory::format_tag::ab,
      dnnl::memory::format_tag::abc,
      dnnl::memory::format_tag::abcd,
      dnnl::memory::format_tag::abcde,
      dnnl::memory::format_tag::abcdef,
      dnnl::memory::format_tag::abcdefg,
      dnnl::memory::format_tag::abcdefgh,
      dnnl::memory::format_tag::abcdefghi,
      dnnl::memory::format_tag::abcdefghij,
      dnnl::memory::format_tag::abcdefghijk,
      dnnl::memory::format_tag::abcdefghijkl};
  return (od.size <= 0 || od.size > ODLA_MAX_DIMENSION ||
          od.size >= (sizeof(tags) / sizeof(tags[0])))
             ? tags[0]
             : tags[od.size];
}

static inline dnnl::memory::format_tag getFormatTag(odla_memory_layout layout,
                                                    unsigned group = 1,
                                                    unsigned spatial_dims = 2) {
  switch (layout) {
    case ODLA_CHANNELS_FIRST:
      assert(spatial_dims >= 1 && spatial_dims <= 3);
      if (spatial_dims == 1) {
        return dnnl::memory::format_tag::ncw;
      }
      if (spatial_dims == 2) {
        return dnnl::memory::format_tag::nchw;
      }
      return dnnl::memory::format_tag::ncdhw;
    case ODLA_CHANNELS_LAST:
      if (spatial_dims == 1) {
        return dnnl::memory::format_tag::nwc;
      } else if (spatial_dims == 2) {
        return dnnl::memory::format_tag::nhwc;
      }
      return dnnl::memory::format_tag::ndhwc;
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

static inline int64_t GetTotalElements(const odla_value_shape& dims) {
  return std::accumulate(dims.dims, dims.dims + dims.size, 1,
                         std::multiplies<size_t>());
}

static inline bool hasDNNLMemorySupport(odla_element_type ty) {
  return ty == ODLA_FLOAT32 || ty == ODLA_INT8 || ty == ODLA_UINT8 ||
         ty == ODLA_INT32 || ty == ODLA_BFLOAT16;
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
    case ODLA_BOOL:
    case ODLA_UINT8:
      dt = dnnl::memory::data_type::u8;
      break;
    case ODLA_UINT32:
    case ODLA_INT32:
      dt = dnnl::memory::data_type::s32;
      break;
    case ODLA_UINT64:
    case ODLA_INT64:
      dt = dnnl::memory::data_type::s32; // FIXME:
      break;
    case ODLA_UINT16:
    case ODLA_FLOAT16:
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

static inline size_t getElementStorageSize(odla_element_type elem_type) {
  switch (elem_type) {
    case ODLA_FLOAT64:
    case ODLA_INT64:
    case ODLA_UINT64:
    case ODLA_QINT64:
    case ODLA_QUINT64:
      return sizeof(int64_t);
    case ODLA_FLOAT32:
    case ODLA_QINT32:
    case ODLA_QUINT32:
    case ODLA_INT32:
    case ODLA_UINT32:
      return sizeof(int32_t);
    case ODLA_FLOAT16:
    case ODLA_BFLOAT16:
    case ODLA_INT16:
    case ODLA_UINT16:
    case ODLA_QINT16:
    case ODLA_QUINT16:
      return sizeof(uint16_t);
    case ODLA_INT8:
    case ODLA_UINT8:
    case ODLA_QINT8:
    case ODLA_QUINT8:
    case ODLA_BOOL:
      return sizeof(char);
    case ODLA_STRING:
      return sizeof(char*);
  }
  assert(0 && "Unhandled data type");
  return 0;
}

static inline size_t getValueStorageSize(odla_value value) {
  return GetTotalElements(value->shape) *
         getElementStorageSize(value->elem_type);
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

static inline void expand_dims(odla_value_shape& src,
                               const odla_value_shape& dst) {
  // src shape is [1,5], dst shape is [1,4,1], we expand src shape to [1,1,5]
  // src shape is [64], dst shape is [1,64,128], we expand src shape to [1,64,1]
  // src shape is [64], dst shape is [1,64,64], we expand src shape to [1,1,64]
  // src shape is [7,31,64], dst shape is [2,31,64,64], failed

  const int src_n = src.size;
  const int dst_n = dst.size;
  odla_value_shape new_shape;

  auto CompareDims = [src, dst, src_n](int loc,
                                       odla_value_shape& new_shape) -> bool {
    int src_idx = src_n - 1;
    int dst_idx = loc;
    for (int k = 0; k < src_n; k++) {
      if (dst.dims[dst_idx - k] != src.dims[src_idx - k] &&
          dst.dims[dst_idx - k] != 1 && src.dims[src_idx - k] != 1) {
        return false;
      }
      new_shape.dims[dst_idx - k] = src.dims[src_idx - k];
    }
    return true;
  };

  // slide from the last item in dst
  for (int j = dst_n - 1; j >= 0; j--) {
    if (CompareDims(j, new_shape)) {
      // the src shape cannot be expanded
      assert(j + 1 >= src_n);
      const int sub_array_start = j + 1 - src_n;
      for (int i = 0; i < sub_array_start; i++) {
        new_shape.dims[i] = 1;
      }
      break;
    } else {
      new_shape.dims[j] = 1;
    }
  }

  new_shape.size = dst_n;
  src = new_shape;
}

static inline dnnl::memory broadcast_mem(const dnnl::memory orig_mem,
                                         const odla_value_shape& orig_shape,
                                         const odla_value_shape& target_shape,
                                         bool needs_reorder) {
  assert(orig_shape.size == target_shape.size); // dims are already expanded.
  auto dt = orig_mem.get_desc().data_type();
  std::vector<int64_t> strides_v(target_shape.size, 0);
  for (int i = target_shape.size - 1, s = 1; i >= 0; --i) {
    if (orig_shape.dims[i] != target_shape.dims[i]) {
      assert(orig_shape.dims[i] == 1);
    } else {
      strides_v[i] = s;
      s *= orig_shape.dims[i];
    }
  }
  auto src_md = dnnl::memory::desc(getDims(target_shape), dt,
                                   dnnl::memory::dims(strides_v));
  if (!needs_reorder) {
    return dnnl::memory(src_md, g_comp->eng, orig_mem.get_data_handle());
  }
  auto ret_md = getMemoryDesc(target_shape, dt);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto reorder_pd =
      dnnl::reorder::primitive_desc(g_comp->eng, src_md, g_comp->eng, ret_md);
  auto reorder_prim = dnnl::reorder(reorder_pd);
  add_op(reorder_prim, {{DNNL_ARG_SRC, orig_mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();
  return ret_mem;
}

static inline std::pair<dnnl::memory, dnnl::memory> broadcast_operands(
    const odla_value& lhs, const odla_value& rhs,
    odla_value_shape* tiled_shape) {
  auto dims_lhs = lhs->shape;
  auto dims_rhs = rhs->shape;
  auto rank = std::max(dims_lhs.size, dims_rhs.size);
  if (dims_lhs.size != dims_rhs.size) {
    auto& from = dims_lhs.size > dims_rhs.size ? dims_rhs : dims_lhs;
    auto& to = dims_lhs.size > dims_rhs.size ? dims_lhs : dims_rhs;
    expand_dims(from, to);
  }
  for (int i = 0; i < rank; i++) {
    tiled_shape->dims[i] = std::max(dims_lhs.dims[i], dims_rhs.dims[i]);
  }
  tiled_shape->size = rank;
  return {
      broadcast_mem(lhs->mem, dims_lhs, *tiled_shape, true),
      broadcast_mem(rhs->mem, dims_rhs, *tiled_shape, true),
  };
}

static inline odla_value unary_eltwise_op(
    dnnl::algorithm algo, odla_value input, odla_float32 alpha,
    odla_float32 beta, const odla_value_id id,
    dnnl::primitive_attr attr = dnnl::primitive_attr()) {
  auto eltwise_d =
      dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algo,
                                  input->mem.get_desc(), alpha, beta);
  auto pd = dnnl::eltwise_forward::primitive_desc(eltwise_d, attr, g_comp->eng);

  dnnl::primitive prim = dnnl::eltwise_forward(pd);
  auto ret_mem = dnnl::memory(input->mem.get_desc(), g_comp->eng);
  odla_value v = CreateValue(ret_mem, input->shape, id);
  add_op(prim, {{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();
  return v;
}

#endif // ODLA_DNNL_H_
