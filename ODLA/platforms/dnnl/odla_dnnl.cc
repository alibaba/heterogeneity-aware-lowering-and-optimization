//===- odla_dnnl.cc -------------------------------------------------------===//
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

#include <ODLA/odla.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "ODLA/odla_compute.h"
#include "dnnl.hpp"
#include "dnnl_threadpool_iface.hpp"
#include "dnnl_utils.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

struct _odla_value {
  dnnl::memory mem;
  bool is_const;
  uint8_t elem_size;
  odla_value_shape shape;
  std::string name;
  _odla_value(const dnnl::memory& m, const odla_value_shape& shape_,
              const std::string& id)
      : mem(m), is_const(false), shape(shape_), name(id), elem_size(4) {
    if (shape.size == 0) {
      shape.size = 1;
      shape.dims[0] = 1;
    }
  }
};

typedef struct TargetOpts {
  odla_bf16_mode bf16_mode;
} target_opts;

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

struct _odla_computation {
  dnnl::engine eng;
  // std::vector<dnnl::primitive> primitives;
  // std::vector<std::unordered_map<int, dnnl::memory>> args;
  std::vector<operation> ops;
  std::vector<std::unique_ptr<_odla_value>> vals;
  std::unordered_map<std::string, odla_value> inputs;
  std::unordered_map<std::string, odla_value> outputs;
  std::unordered_map<std::string, std::pair<odla_value, void*>> outputs_v;
  target_opts opts;

  _odla_computation() : eng(dnnl::engine::kind::cpu, 0), opts({BF16_DISABLE}) {}
};

struct _odla_context {
  odla_computation comp;
  std::unique_ptr<dnnl::stream> stream;
};

static dnnl::memory::format_tag getFormatTag(const odla_value_shape& od) {
  static const dnnl::memory::format_tag tags[] = {
      dnnl::memory::format_tag::undef,  dnnl::memory::format_tag::a,
      dnnl::memory::format_tag::ab,     dnnl::memory::format_tag::abc,
      dnnl::memory::format_tag::abcd,   dnnl::memory::format_tag::abcde,
      dnnl::memory::format_tag::abcdef,
  };
  return (od.size <= 0 || od.size > 6) ? tags[0] : tags[od.size];
}

static dnnl::memory::format_tag getFormatTag(odla_memory_layout layout,
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

static int64_t GetTotalElements(const odla_value_shape& dims) {
  return std::accumulate(dims.dims, dims.dims + dims.size, 1,
                         std::multiplies<size_t>());
}

// The strides used by DNNL's memory desc.
static dnnl::memory::dims getStrides(const odla_value_shape& od) {
  std::vector<int64_t> strides(od.size, 1);
  for (int i = od.size - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * od.dims[i + 1];
  auto dims = dnnl::memory::dims(strides);
  return dims;
}

static dnnl::memory::data_type getDataType(const odla_element_type ty) {
  dnnl::memory::data_type dt;
  switch (ty) {
    case ODLA_FLOAT32:
      dt = dnnl::memory::data_type::f32;
      break;
    case ODLA_INT8:
      dt = dnnl::memory::data_type::s8;
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
    default:
      dt = dnnl::memory::data_type::undef;
  }
  return dt;
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

template <typename Tin, typename Tout>
static void ArgMax(const Tin* input, Tout* output, odla_int32 axis,
                   const odla_value_shape& shape) {
  int64_t dim, axis_dist;
  axis = axis < 0 ? shape.size + axis : axis;
  dim = shape.dims[axis];
  // Distance between values of axis in blob
  axis_dist = GetCountFromAxis(shape, axis) / dim;
  int num = GetCountFromAxis(shape, 0) / dim;

  std::vector<std::pair<Tin, Tout>> output_data_vector(dim);
  for (int i = 0; i < num; ++i) {
    for (Tout j = 0; j < dim; ++j) {
      output_data_vector[j] = std::make_pair(
          input[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    std::partial_sort(output_data_vector.begin(),
                      output_data_vector.begin() + 1, output_data_vector.end(),
                      std::greater<std::pair<Tin, Tout>>());
    // Produces max_ind per axis
    output[(i / axis_dist) * axis_dist + i % axis_dist] =
        output_data_vector[0].second;
  }
}

static dnnl::memory::dims getDims(const odla_value_shape& od) {
  auto dims = dnnl::memory::dims(od.dims, od.dims + od.size);
  return dims;
}

static dnnl::memory::desc getMemoryDesc(const odla_value_shape& dims,
                                        dnnl::memory::data_type ty) {
  return dnnl::memory::desc(getDims(dims), ty, getFormatTag(dims));
}

static dnnl::memory::desc getMemoryDesc(const odla_value_shape& dims,
                                        odla_element_type ty) {
  return dnnl::memory::desc(getDims(dims), getDataType(ty), getFormatTag(dims));
}

static dnnl::memory::desc getMemoryDesc(const odla_value_type& ty) {
  return getMemoryDesc(ty.shape, ty.element_type);
}

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;
thread_local bool g_interpret_mode = false;

static void add_op(dnnl::primitive prim,
                   const std::unordered_map<int, dnnl::memory>& args) {
  operation op;
  op.prim = prim;
  op.args = args;
  g_comp->ops.emplace_back(op);
}

static void add_op(std::function<void()> func) {
  operation op;
  op.func = func;
  g_comp->ops.emplace_back(op);
}

#ifdef ODLA_DNNL_BUILD_AS_INTERPRETER
struct Initializer {
  Initializer() {
    odla_CreateComputation(nullptr);
    g_interpret_mode = true;
  }
};
static Initializer interpreter_initializer;
#endif

static odla_value CreateValue(const dnnl::memory& mem,
                              const odla_value_shape shape,
                              const odla_value_id id) {
  std::string name = id == nullptr ? "" : std::string((const char*)id);
  auto v = std::make_unique<_odla_value>(mem, shape, name);
  auto ret = v.get();
  g_comp->vals.push_back(std::move(v));
  return ret;
}

static dnnl::memory cast_odla_mem(dnnl::memory src_mem,
                                  const odla_value_shape shape,
                                  const dnnl::memory::data_type dt,
                                  const bool is_const) {
  auto dst_md = dnnl::memory::desc(getDims(shape), dt, getFormatTag(shape));
  auto dst_mem = dnnl::memory(dst_md, g_comp->eng);
  auto r = dnnl::reorder(src_mem, dst_mem);
  if (is_const) {
    r.execute(dnnl::stream(g_comp->eng),
              {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, dst_mem}});
  } else {
    add_op(r, {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, dst_mem}});
  }
  return dst_mem;
}

static dnnl::memory cast_op(odla_value& input, dnnl::memory::data_type dt) {
  auto src_md = dnnl::memory::desc(getDims(input->shape),
                                   input->mem.get_desc().data_type(),
                                   getFormatTag(input->shape));
  auto src_mem =
      dnnl::memory(src_md, g_comp->eng, input->mem.get_data_handle());
  return cast_odla_mem(src_mem, input->shape, dt, input->is_const);
}

extern "C" {

odla_status odla_SetComputationItem(odla_computation comp, odla_item_type type,
                                    odla_item_value value) {
  switch (type) {
    case ODLA_BF16_MODE:
      comp->opts.bf16_mode = *(reinterpret_cast<odla_bf16_mode*>(value));
      break;
    default:
      std::cerr << "Unsupported property type: " << type << std::endl;
      return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

odla_status odla_CreateComputation(odla_computation* computation) {
  g_comps.push_back(std::make_unique<_odla_computation>());
  g_comp = g_comps.back().get();
  if (computation != nullptr) {
    *computation = g_comp;
  }
  return ODLA_SUCCESS;
}

odla_status odla_SetActiveComputation(odla_computation computation) {
  g_comp = computation;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation computation) {
  // TODO:
  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* ctx) {
  *ctx = new _odla_context();
  (*ctx)->comp = g_comp;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context ctx) {
  delete (ctx);
  return ODLA_SUCCESS;
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  if (context->stream == nullptr) {
    context->stream = std::make_unique<dnnl::stream>(comp->eng);
  }
  for (auto& op : comp->ops) {
    op.execute(*context->stream);
  }
  context->stream->wait();
  // copy to outputs
  auto outputs_v = context->comp->outputs_v;
  for (auto& output_pair : outputs_v) {
    auto& src_val = output_pair.second.first;
    auto& dst_ptr = output_pair.second.second;
    memcpy(dst_ptr, src_val->mem.get_data_handle(),
           src_val->mem.get_desc().get_size() * (src_val->elem_size / 4));
  }
  return ODLA_SUCCESS;
}

static void InterpretIfNeeded() {
#if ODLA_DNNL_BUILD_AS_INTERPRETER
  if (!g_interpret_mode) {
    return;
  }
  static odla_context context;
  if (!context) {
    odla_CreateContext(&context);
  }
  if (context->stream == nullptr) {
    context->stream = std::make_unique<dnnl::stream>(g_comp->eng);
  }
  for (auto& op : g_comp->ops) {
    op.execute(*context->stream);
  }
  context->stream->wait();
  g_comp->ops.clear();
#endif
}

void rewrite_scalar_type(odla_value_type& type) {
  if (type.shape.size == 0) {
    type.shape.size = 1;
    type.shape.dims[0] = 1;
  }
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  const char* name = (const char*)id;
  rewrite_scalar_type(type);
  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE &&
      type.element_type == ODLA_FLOAT32) {
    type.element_type = ODLA_BFLOAT16;
  }
  dnnl::memory::desc md = getMemoryDesc(type);
  dnnl::memory mem = dnnl::memory(md, g_comp->eng);
  odla_value v = CreateValue(mem, type.shape, id);
  if (type.element_type == ODLA_INT64) {
    v->elem_size = 8;
  }
  g_comp->inputs[name] = v;
  return v;
}

odla_value odla_CreateValue(odla_value_type type, const odla_value_id id) {
  assert(g_interpret_mode);
  rewrite_scalar_type(type);
  auto v = odla_CreateArgument(type, id);
  return v;
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  value_type->element_type = ODLA_FLOAT32;
  value_type->shape = value->shape;
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE &&
      value->mem.get_desc().data_type() == dnnl::memory::data_type::bf16) {
    auto src_md = dnnl::memory::desc(value->mem.get_desc().dims(),
                                     getDataType(ODLA_FLOAT32),
                                     getFormatTag(value->shape));
    auto src_mem =
        dnnl::memory(src_md, g_comp->eng, const_cast<void*>(data_ptr));
    auto r = dnnl::reorder(src_mem, value->mem);
    r.execute(dnnl::stream(g_comp->eng),
              {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, value->mem}});
  } else {
    value->mem.set_data_handle(const_cast<void*>(data_ptr));
  }
  return ODLA_SUCCESS;
}

odla_status odla_SetValueData(odla_value value, const void* ptr) {
  assert(g_interpret_mode);
  value->mem.set_data_handle(const_cast<void*>(ptr));
  return ODLA_SUCCESS;
}

odla_status odla_GetValueData(const odla_value value, odla_void* data_ptr) {
  assert(g_interpret_mode == true);
  memcpy(data_ptr, value->mem.get_data_handle(),
         value->mem.get_desc().get_size());
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  std::string name((const char*)value_id);
  odla_value value = context->comp->inputs[name];
  return odla_BindToArgument(value, data_ptr, context);
}

odla_value odla_CreateConstant(odla_value_type type, const void* ptr,
                               const odla_value_id id) {
  // TODO:
  // dnnl::memory::desc md(getDims(dims), getDataType(type),
  //                      dnnl::memory::format_tag::hwio);
  rewrite_scalar_type(type);
  dnnl::memory::desc md = getMemoryDesc(type);
  dnnl::memory mem = dnnl::memory(md, g_comp->eng, const_cast<void*>(ptr));

  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE &&
      type.element_type == ODLA_FLOAT32) {
    type.element_type = ODLA_BFLOAT16;
    mem = cast_odla_mem(mem, type.shape, getDataType(type.element_type), true);
  }

  odla_value v = CreateValue(mem, type.shape, id);
  v->is_const = true;
  if (type.element_type == ODLA_INT64) {
    v->elem_size = 8;
  }

  return v;
}

odla_status odla_SetValueAsOutput(const odla_value val) {
  g_comp->outputs[val->name] = val;
  // convert output to float32
  if (g_comp->opts.bf16_mode != BF16_DISABLE &&
      val->mem.get_desc().data_type() == dnnl::memory::data_type::bf16) {
    val->mem =
        cast_odla_mem(val->mem, val->shape, getDataType(ODLA_FLOAT32), false);
  }
  return ODLA_SUCCESS;
}
odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  // Handle the case of output is constant due to compile-time optimization.
  if (value->is_const) {
    size_t len = value->mem.get_desc().get_size();
    if (value->elem_size == 8) {
      len *= 2;
    }
    memcpy(data_ptr, value->mem.get_data_handle(), len);
  } else {
    value->mem.set_data_handle(data_ptr);
  }
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  std::string name((const char*)value_id);
  auto& outputs_v = context->comp->outputs_v;
  auto val = context->comp->outputs[name];
  if (outputs_v.find(name) != outputs_v.end()) {
    outputs_v[name].second = data_ptr;
  } else {
    outputs_v[name] = {val, data_ptr};
  }
  return ODLA_SUCCESS;
}

odla_value odla_Floor(odla_value input, const odla_value_id id) {
  int64_t total_elems = GetTotalElements(input->shape);
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  std::function<void()> op;
  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE) {
    op = [total_elems, input, ret_mem]() {
      dnnl_utils::floorbf_func(total_elems,
                               (int16_t*)input->mem.get_data_handle(),
                               (float*)ret_mem.get_data_handle());
    };
  } else {
    op = [total_elems, input, ret_mem]() {
      dnnl_utils::floorf_func(total_elems, (float*)input->mem.get_data_handle(),
                              (float*)ret_mem.get_data_handle());
    };
  }
  add_op(op);
  InterpretIfNeeded();
  return CreateValue(ret_mem, input->shape, id);
}

odla_value odla_Rsqrt(odla_value input, const odla_value_id id) {
  int64_t total_elems = GetTotalElements(input->shape);
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  std::function<void()> op;
  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE) {
    op = [total_elems, input, ret_mem]() {
      dnnl_utils::rsqrtbf_func(total_elems,
                               (int16_t*)input->mem.get_data_handle(),
                               (float*)ret_mem.get_data_handle());
    };

  } else {
    op = [total_elems, input, ret_mem]() {
      dnnl_utils::rsqrtf_func(total_elems, (float*)input->mem.get_data_handle(),
                              (float*)ret_mem.get_data_handle());
    };
  }
  add_op(op);
  InterpretIfNeeded();
  return CreateValue(ret_mem, input->shape, id);
}

odla_value odla_Gather(odla_value params, const odla_value indices,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  std::function<void()> op;
  axis = axis < 0 ? params->shape.size + axis : axis;
  size_t batch_size;
  size_t idx_size;
  auto dt = params->mem.get_desc().data_type();
  if (indices->shape.size > 1) {
    batch_size = indices->shape.dims[0];
    idx_size = indices->shape.dims[1];
  } else {
    batch_size = 1;
    for (int i = 0; i < axis; ++i) {
      batch_size *= params->shape.dims[i];
    }
    idx_size = indices->shape.dims[0];
  }
  size_t inner_size = 1;
  for (int i = axis + 1; i < params->shape.size; ++i) {
    inner_size *= params->shape.dims[i];
  }
  auto ret_md =
      dnnl::memory::desc(getDims(output_dims), dt, getStrides(output_dims));
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  int byte_size =
      params->mem.get_desc().get_size() / GetTotalElements(params->shape);

  op = [params, indices, batch_size, idx_size, inner_size, byte_size, ret_mem,
        axis]() {
    int32_t* indices_ptr = (int32_t*)indices->mem.get_data_handle();
    std::vector<int> indices_i32;
    if (indices->elem_size == 8) {
      int64_t* src = (int64_t*)indices_ptr;
      indices_i32.insert(indices_i32.begin(), src, src + idx_size);
      indices_ptr = indices_i32.data();
    }
    dnnl_utils::gather_func((char*)params->mem.get_data_handle(), indices_ptr,
                            batch_size, idx_size, params->shape.dims[axis],
                            inner_size, byte_size,
                            (char*)ret_mem.get_data_handle());
  };
  add_op(op);
  InterpretIfNeeded();
  return CreateValue(ret_mem, output_dims, id);
}

odla_value odla_Cast(odla_value input, odla_element_type target_type,
                     const odla_value_id id) {
  auto dst_mem = cast_op(input, getDataType(target_type));
  InterpretIfNeeded();
  return CreateValue(dst_mem, input->shape, id);
}

static odla_value unary_eltwise_op(
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

static odla_value binary_eltwise_s32(dnnl::algorithm alg, dnnl::memory lhs_mem,
                                     dnnl::memory rhs_mem,
                                     odla_value_shape shape,
                                     const odla_value_id id) {
  std::function<void()> elem_op;
  int ln = GetTotalElements(shape);
  elem_op = [lhs_mem, rhs_mem, ln, alg]() {
    int32_t* rhs_ptr = nullptr;
    rhs_ptr = (int32_t*)rhs_mem.get_data_handle();
    dnnl_utils::binary_s32_func(alg, (int32_t*)lhs_mem.get_data_handle(),
                                rhs_ptr, rhs_ptr, ln);
  };
  add_op(elem_op);
  odla_value v = CreateValue(rhs_mem, shape, id);
  InterpretIfNeeded();
  return v;
}

static void expand_dims(odla_value& src, odla_value& dst) {
  // src shape is [1,5], dst shape is [1,4,1], we expand src shape to [1,1,5]
  // src shape is [64], dst shape is [1,64,128], we expand src shape to [1,64,1]
  // src shape is [64], dst shape is [1,64,64], we expand src shape to [1,1,64]
  // src shape is [7,31,64], dst shape is [2,31,64,64], failed

  const int src_n = src->shape.size;
  const int dst_n = dst->shape.size;
  odla_value_shape new_shape;

  auto CompareDims = [src, dst, src_n](int loc,
                                       odla_value_shape& new_shape) -> bool {
    int src_idx = src_n - 1;
    int dst_idx = loc;
    for (int k = 0; k < src_n; k++) {
      if (dst->shape.dims[dst_idx - k] != src->shape.dims[src_idx - k] &&
          dst->shape.dims[dst_idx - k] != 1 &&
          src->shape.dims[src_idx - k] != 1) {
        return false;
      }
      new_shape.dims[dst_idx - k] = src->shape.dims[src_idx - k];
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
  src->shape = new_shape;
}

static odla_value broadcast_func(odla_value& input, odla_value_shape shape) {
  bool skip_broadcast = true;
  for (int i = 0; i < shape.size; i++) {
    skip_broadcast &= (input->shape.dims[i] == shape.dims[i]);
  }
  if (skip_broadcast) return input;
  std::vector<int64_t> strides_v(input->shape.size, 0);
  std::function<void()> op;
  auto ln = GetTotalElements(shape);
  auto rn = GetTotalElements(input->shape);
  if (rn != 1) { // if there is only one elemets, it's strides = 0
    for (int i = shape.size - 1, s = 1; i >= 0; --i) {
      if (input->shape.dims[i] != shape.dims[i]) {
        assert(input->shape.dims[i] == 1);
      } else {
        strides_v[i] = s;
        s *= input->shape.dims[i];
      }
    }
  }
  auto src_md =
      dnnl::memory::desc(getDims(shape), input->mem.get_desc().data_type(),
                         dnnl::memory::dims(strides_v));
  auto ret_md = dnnl::memory::desc(
      getDims(shape), input->mem.get_desc().data_type(), getFormatTag(shape));
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto reorder_pd =
      dnnl::reorder::primitive_desc(g_comp->eng, src_md, g_comp->eng, ret_md);
  auto reorder_prim = dnnl::reorder(reorder_pd);
  add_op(reorder_prim, {{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  return CreateValue(ret_mem, shape, nullptr);
}

std::pair<odla_value, odla_value> broadcast_op(odla_value lhs, odla_value rhs) {
  auto dims_lhs = lhs->shape;
  auto dims_rhs = rhs->shape;
  if (dims_lhs.size != dims_rhs.size) {
    auto& from = dims_lhs.size > dims_rhs.size ? rhs : lhs;
    auto& to = dims_lhs.size > dims_rhs.size ? lhs : rhs;
    expand_dims(from, to);
  }
  uint32_t repeat_lhs[10];
  uint32_t repeat_rhs[10];
  odla_value_shape tiled_shape;
  for (int i = 0; i < lhs->shape.size; i++) {
    auto curr_output_dim = lhs->shape.dims[i] >= rhs->shape.dims[i]
                               ? lhs->shape.dims[i]
                               : rhs->shape.dims[i];
    repeat_lhs[i] = curr_output_dim / lhs->shape.dims[i];
    repeat_rhs[i] = curr_output_dim / rhs->shape.dims[i];
    tiled_shape.dims[i] = curr_output_dim;
  }
  tiled_shape.size = lhs->shape.size;
  auto new_lhs = broadcast_func(lhs, tiled_shape);
  auto new_rhs = broadcast_func(rhs, tiled_shape);
  lhs->shape = dims_lhs;
  rhs->shape = dims_rhs;
  return std::pair<odla_value, odla_value>(new_lhs, new_rhs);
}

static odla_value binary_eltwise(dnnl::algorithm algo, odla_value lhs,
                                 odla_value rhs, const odla_value_id id) {
  if (lhs->mem.get_data_handle() != rhs->mem.get_data_handle()) {
    auto new_inputs = broadcast_op(lhs, rhs);
    lhs = new_inputs.first;
    rhs = new_inputs.second;
  }
  auto type = lhs->mem.get_desc().data_type();
  if (type == dnnl::memory::data_type::s32) {
    return binary_eltwise_s32(algo, lhs->mem, rhs->mem, lhs->shape, id);
  }
  const auto& dims_lhs = lhs->shape;
  const auto& dims_rhs = rhs->shape;
  auto lhs_md = dnnl::memory::desc(
      getDims(dims_lhs), lhs->mem.get_desc().data_type(), getStrides(dims_lhs));
  auto rhs_md = lhs_md;
  auto ret_md = lhs_md;
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  dnnl::binary::desc bd(algo, lhs_md, rhs_md, ret_md);
  dnnl::binary::primitive_desc pd(bd, g_comp->eng);
  dnnl::primitive prim = dnnl::binary(pd);

  add_op(prim, {{DNNL_ARG_SRC_0, lhs->mem},
                {DNNL_ARG_SRC_1, rhs->mem},
                {DNNL_ARG_DST, ret_mem}});

  odla_value v = CreateValue(ret_mem, lhs->shape, id);
  InterpretIfNeeded();
  return v;
}

odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_eltwise(dnnl::algorithm::binary_add, lhs, rhs, id);
}

odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  if (lhs == rhs) {
    return unary_eltwise_op(dnnl::algorithm::eltwise_square, lhs, 1.f, 0.f, id);
  }
  return binary_eltwise(dnnl::algorithm::binary_mul, lhs, rhs, id);
}

odla_value odla_Sub(odla_value lhs, odla_value rhs, const odla_value_id id) {
#if (DNNL_VERSION_MINOR != 8)
  auto v = unary_eltwise_op(dnnl::algorithm::eltwise_linear, rhs, -1.f, 0.f,
                            nullptr);
  return binary_eltwise(dnnl::algorithm::binary_add, lhs, v, id);
#else
  return binary_eltwise(dnnl::algorithm::binary_sub, lhs, rhs, id);
#endif
}

odla_value odla_Div(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_eltwise(dnnl::algorithm::binary_div, lhs, rhs, id);
}

odla_value odla_Round(odla_value input, const odla_value_id id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_round, input, 0.f, 0.f, id);
}

odla_value odla_Exp(odla_value input, const odla_value_id value_id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_exp, input, 0.f, 0.f,
                          value_id);
}

odla_value odla_Sqrt(odla_value input, const odla_value_id value_id) {
  auto v = unary_eltwise_op(dnnl::algorithm::eltwise_sqrt, input, 0.f, 0.f,
                            value_id);
  return v;
}

odla_value odla_Sigmoid(odla_value input, const odla_value_id id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_logistic, input, 0.f, 0.f,
                          id);
}

odla_value odla_LeakyRelu(odla_value input, odla_float32 alpha,
                          const odla_value_id id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_relu, input, alpha, 0.f, id);
}

odla_value odla_Relu(odla_value input, const odla_value_id value_id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_relu, input, 0.f, 0.f,
                          value_id);
}

odla_value odla_PRelu(odla_value input, odla_value slope,
                      const odla_value_id value_id) {
  // current dnnl is not support prelu primitive, so we use below equation to
  // get prelu():
  // prelu(input, slope) = (input < 0 ? input * slope : input)
  //                     = relu(input) - relu(-input) * slope
  //                     = relu(input) - mul(relu(mul(input, -1)),  slope)
  //                     = relu(input) + mul(-slope,relu(mul(input,-1)))

  auto relu_v = odla_Relu(input, nullptr);

  auto neg_input = unary_eltwise_op(dnnl::algorithm::eltwise_linear, input,
                                    -1.f, 0.f, nullptr);
  auto neg_relu_v = odla_Relu(neg_input, nullptr);
  auto neg_slope = unary_eltwise_op(dnnl::algorithm::eltwise_linear, slope,
                                    -1.f, 0.f, nullptr);
  auto neg_relu_v_mul = odla_Mul(neg_relu_v, neg_slope, nullptr);
  auto v = odla_Add(neg_relu_v_mul, relu_v, value_id);
  InterpretIfNeeded();
  return v;
}

odla_value odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
                      const odla_value_id id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_clip, input, lo, hi, id);
}

static odla_value_shape getNCHWDims(const odla_value_shape& src_dims) {
  assert(src_dims.size == 4);
  return {
      src_dims.size,
      {src_dims.dims[0], src_dims.dims[3], src_dims.dims[1], src_dims.dims[2]}};
}

static odla_value_shape getOIHWDims(const odla_value_shape& src_dims) {
  assert(src_dims.size == 4);
  return {
      src_dims.size,
      {src_dims.dims[3], src_dims.dims[2], src_dims.dims[0], src_dims.dims[1]}};
}

static odla_value_shape getGOIHWDims(const odla_value_shape& src_dims,
                                     unsigned groups, unsigned data_in_ch,
                                     odla_memory_layout layout) {
  assert(src_dims.size == 4);
  assert(layout == ODLA_OIS);
  auto group_in_ch = data_in_ch / groups;
  auto group_out_ch =
      src_dims.dims[0] * src_dims.dims[1] / (groups * group_in_ch);
  return {
      src_dims.size + 1,
      {groups, group_out_ch, group_in_ch, src_dims.dims[2], src_dims.dims[3]}};
}

odla_value odla_Transpose(odla_value input, odla_value_shape permutations,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  const auto& input_dims = input->shape;
  auto strides = getStrides(input_dims);
  auto new_strides = strides;
  for (int i = 0; i < permutations.size; ++i)
    new_strides[i] = strides[permutations.dims[i]];
  auto type = input->mem.get_desc().data_type();
  dnnl::memory::desc src_md(getDims(output_dims), type, new_strides);
  dnnl::memory::desc dst_md(getDims(output_dims), type,
                            getStrides(output_dims));
  auto src_mem = dnnl::memory(src_md, g_comp->eng, nullptr);
  auto dst_mem = dnnl::memory(dst_md, g_comp->eng);
  auto prim = dnnl::reorder(src_mem, dst_mem);

  add_op(prim, {{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, dst_mem}});
  auto v = CreateValue(dst_mem, output_dims, id);

  InterpretIfNeeded();

  return v;
}

odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id) {
  return CreateValue(input->mem, output_dims, id);
}

odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  auto input_dims = input->shape;
  auto kernel_dims = kernel->shape;
  auto dt = input->mem.get_desc().data_type();
  auto dt_dst = (g_comp->opts.bf16_mode == BF16_ACCURACY_MODE)
                    ? getDataType(ODLA_BFLOAT16)
                    : dt;
  dnnl::memory::dims stride_dims{strides[0], strides[1]};
  dnnl::memory::dims paddings_before{paddings_front[0], paddings_front[1]};
  dnnl::memory::dims paddings_after{paddings_back[0], paddings_back[1]};
  odla_value_shape orig_output_dims = output_dims;

  if (input_layout == ODLA_CHANNELS_LAST) {
    input_dims = getNCHWDims(input_dims);
    output_dims = getNCHWDims(output_dims);
  }

  if (kernel_layout == ODLA_SIO) {
    kernel_dims = getOIHWDims(kernel_dims);
  }
  if (group > 1) {
    if (kernel_layout == ODLA_SIO) {
      if (kernel_dims.dims[0] * group == kernel_dims.dims[1])
        std::swap(kernel_dims.dims[0], kernel_dims.dims[1]);
    }
    kernel_dims =
        getGOIHWDims(kernel_dims, group, input_dims.dims[1], ODLA_OIS);
  }

  auto ret_md_any = dnnl::memory::desc(getDims(output_dims), dt_dst,
                                       dnnl::memory::format_tag::any);
  auto input_md_any = dnnl::memory::desc(getDims(input_dims), dt_dst,
                                         dnnl::memory::format_tag::any);
  auto input_md_src =
      dnnl::memory::desc(getDims(input_dims), dt, getFormatTag(input_layout));

  auto kernel_md_any = dnnl::memory::desc(getDims(kernel_dims), dt_dst,
                                          dnnl::memory::format_tag::any);
  auto kernel_md_src = dnnl::memory::desc(getDims(kernel_dims), dt,
                                          getFormatTag(kernel_layout, group));

  auto kernel_mem =
      dnnl::memory(kernel_md_src, g_comp->eng, kernel->mem.get_data_handle());

  dnnl::memory::desc bias_md;
  if (bias != nullptr) {
    odla_value_shape scalar{.size = 1, .dims = {GetTotalElements(bias->shape)}};
    bias_md = dnnl::memory::desc(getDims(scalar), dt_dst,
                                 dnnl::memory::format_tag::a);
  }
  assert(dilations[0] == 1 && dilations[1] == 1);
  auto conv_desc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
      input_md_any, kernel_md_any, bias_md, ret_md_any, stride_dims,
      paddings_before, paddings_after);
  auto pd = dnnl::convolution_forward::primitive_desc(conv_desc, g_comp->eng);
  auto ret_md_exp =
      dnnl::memory::desc(getDims(output_dims), dt, getFormatTag(input_layout));

  bool need_reorder_src = pd.src_desc() != input_md_src;
  bool need_reorder_kernel = pd.weights_desc() != kernel_mem.get_desc();
  bool need_reorder_dst = pd.dst_desc() != ret_md_exp;

  dnnl::memory orig_mem;
  if (need_reorder_src) {
    auto conv_src_mem = dnnl::memory(pd.src_desc(), g_comp->eng);
    auto r = dnnl::reorder(
        dnnl::memory(input_md_src, g_comp->eng, input->mem.get_data_handle()),
        conv_src_mem);
    orig_mem = input->mem;
    add_op(r, {{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, conv_src_mem}});
    input->mem = conv_src_mem;
  }
  if (need_reorder_kernel) {
    auto conv_kernel_mem = dnnl::memory(pd.weights_desc(), g_comp->eng);
    auto r = dnnl::reorder(
        dnnl::memory(kernel_md_src, g_comp->eng, kernel->mem.get_data_handle()),
        conv_kernel_mem);
    r.execute(dnnl::stream(g_comp->eng),
              {{DNNL_ARG_FROM, kernel->mem}, {DNNL_ARG_TO, conv_kernel_mem}});
    kernel->mem = conv_kernel_mem;
  }

  auto prim = dnnl::convolution_forward(pd);
  auto conv_ret_mem = dnnl::memory(pd.dst_desc(), g_comp->eng);
  odla_value v = CreateValue(conv_ret_mem, orig_output_dims, id);

  add_op(prim, {{DNNL_ARG_SRC, input->mem},
                {DNNL_ARG_WEIGHTS, kernel->mem},
                {DNNL_ARG_DST, conv_ret_mem}});

  if (bias != nullptr) {
    g_comp->ops.back().args[DNNL_ARG_BIAS] = bias->mem;
  }

  if (need_reorder_src) {
    input->mem = orig_mem;
  }

  if (need_reorder_dst) {
    auto ret_mem = dnnl::memory(ret_md_exp, g_comp->eng);
    auto r = dnnl::reorder(conv_ret_mem, ret_mem);
    add_op(r, {{DNNL_ARG_FROM, conv_ret_mem}, {DNNL_ARG_TO, ret_mem}});
    v->mem = ret_mem;
  }
  InterpretIfNeeded();
  return v;
}

odla_value odla_DeConv(odla_value input, odla_memory_layout input_layout,
                       odla_uint32 group, odla_value kernel,
                       odla_memory_layout kernel_layout,
                       const odla_uint32* strides, const odla_uint32* dilations,
                       const odla_uint32* paddings_front,
                       const odla_uint32* paddings_back, odla_value bias,
                       odla_value_shape output_dims, const odla_value_id id) {
  auto input_dims = input->shape;
  auto kernel_dims = kernel->shape;

  dnnl::memory::dims stride_dims{strides[0], strides[1]};
  dnnl::memory::dims paddings_before{paddings_front[0], paddings_front[1]};
  dnnl::memory::dims paddings_after{paddings_back[0], paddings_back[1]};
  auto dt = input->mem.get_desc().data_type();
  auto dt_dst = (g_comp->opts.bf16_mode == BF16_ACCURACY_MODE)
                    ? getDataType(ODLA_BFLOAT16)
                    : dt;

  auto orig_output_dims = output_dims;
  if (input_layout == ODLA_CHANNELS_LAST) {
    input_dims = getNCHWDims(input_dims);
    output_dims = getNCHWDims(output_dims);
  }

  // change kernel layout to NCHW
  if (kernel_layout == odla_memory_layout::ODLA_SIO) {
    kernel_dims = getOIHWDims(kernel_dims);
  } else if (kernel_layout == odla_memory_layout::ODLA_IOS) {
    std::swap(kernel_dims.dims[0], kernel_dims.dims[1]);
  }

  if (group > 1) {
    kernel_dims =
        getGOIHWDims(kernel_dims, group, input_dims.dims[1], ODLA_OIS);
  }
  dnnl::memory::desc ret_md;

  auto ret_md_any = dnnl::memory::desc(getDims(output_dims), dt_dst,
                                       dnnl::memory::format_tag::any);
  auto input_md_any = dnnl::memory::desc(getDims(input_dims), dt_dst,
                                         dnnl::memory::format_tag::any);
  auto input_md_src =
      dnnl::memory::desc(getDims(input_dims), dt, getFormatTag(input_layout));

  auto kernel_md_any = dnnl::memory::desc(getDims(kernel_dims), dt_dst,
                                          dnnl::memory::format_tag::any);
  auto kernel_md_src = dnnl::memory::desc(
      getDims(kernel_dims), dt,
      /*dnnl::memory::format_tag::iohw */ getFormatTag(kernel_layout, group));
  auto kernel_mem =
      dnnl::memory(kernel_md_src, g_comp->eng, kernel->mem.get_data_handle());

  assert(dilations[0] == 1 && dilations[1] == 1);
  auto conv_desc = dnnl::deconvolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
      input_md_any, kernel_md_any, ret_md_any, stride_dims, paddings_before,
      paddings_after);
  auto pd = dnnl::deconvolution_forward::primitive_desc(conv_desc, g_comp->eng);

  auto ret_mem = dnnl::memory(pd.dst_desc(), g_comp->eng);
  bool needs_reorder_input = pd.src_desc() != input_md_src;
  if (pd.weights_desc() != kernel_mem.get_desc()) {
    auto reordered_w = dnnl::memory(pd.weights_desc(), g_comp->eng);
    auto rec = dnnl::reorder(kernel_mem, reordered_w);
    add_op(rec, {{DNNL_ARG_FROM, kernel->mem}, {DNNL_ARG_TO, reordered_w}});
    kernel->mem = reordered_w;
  }

  dnnl::memory orig_mem;
  if (needs_reorder_input) {
    orig_mem = input->mem;
    auto reordered_mem = dnnl::memory(pd.src_desc(), g_comp->eng);
    auto r = dnnl::reorder(
        dnnl::memory(input_md_src, g_comp->eng, input->mem.get_data_handle()),
        reordered_mem);

    add_op(r, {{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, reordered_mem}});

    input->mem = reordered_mem;
  }
  auto prim = dnnl::deconvolution_forward(pd);

  odla_value v =
      CreateValue(ret_mem, orig_output_dims, bias != nullptr ? nullptr : id);
  add_op(prim, {{DNNL_ARG_SRC, input->mem},
                {DNNL_ARG_WEIGHTS, kernel->mem},
                {DNNL_ARG_DST, ret_mem}});
  if (needs_reorder_input) {
    input->mem = orig_mem;
  }
  auto ret_md_exp =
      dnnl::memory::desc(getDims(output_dims), dt, getFormatTag(input_layout));
  if (pd.dst_desc() != ret_md_exp) {
    auto reordered_mem = dnnl::memory(ret_md_exp, g_comp->eng);
    auto r = dnnl::reorder(ret_mem, reordered_mem);
    add_op(r, {{DNNL_ARG_FROM, ret_mem}, {DNNL_ARG_TO, reordered_mem}});

    v->mem = reordered_mem;
  }
  InterpretIfNeeded();
  return bias ? odla_Add(v, bias, id) : v; // TODO: add bias into primitive
}

odla_value odla_Concat(odla_values inputs, odla_int32 axis,
                       odla_value_shape output_dims, const odla_value_id id) {
  auto num = inputs.size;
  auto type = inputs.values[0]->mem.get_desc().data_type();
  auto ret_md = getMemoryDesc(output_dims, type);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  std::vector<dnnl::memory::desc> src_mds;
  std::vector<dnnl::memory> src_mems;
  for (int i = 0; i < num; ++i) {
    src_mds.push_back(getMemoryDesc(inputs.values[i]->shape, type));
    src_mems.push_back(inputs.values[i]->mem);
  }
  if (axis < 0) {
    axis = inputs.values[0]->shape.size + axis;
  }
  auto concat_pd =
      dnnl::concat::primitive_desc(ret_md, axis, src_mds, g_comp->eng);
  auto prim = dnnl::concat(concat_pd);
  odla_value v = CreateValue(ret_mem, output_dims, id);
  std::unordered_map<int, dnnl::memory> concat_args;
  for (int i = 0; i < num; ++i) {
    concat_args.emplace(DNNL_ARG_MULTIPLE_SRC + i, src_mems[i]);
  }
  concat_args.emplace(DNNL_ARG_DST, ret_mem);
  add_op(prim, concat_args);
  InterpretIfNeeded();
  return v;
}

static odla_value BasePool(odla_value input, odla_memory_layout input_layout,
                           const odla_uint32* window_dims,
                           const odla_uint32* strides,
                           const odla_uint32* paddings_front,
                           const odla_uint32* paddings_back,
                           odla_value_shape output_dims,
                           const odla_value_id value_id,
                           dnnl::algorithm algorithm) {
  dnnl::memory::dims stride_dims{strides[0], strides[1]};
  dnnl::memory::dims kernel_dims{window_dims[0], window_dims[1]};
  dnnl::memory::dims paddings_before{paddings_front[0], paddings_front[1]};
  dnnl::memory::dims paddings_after{paddings_back[0], paddings_back[1]};
  auto dt = input->mem.get_desc().data_type();

  auto input_dims = input->shape;
  auto orig_output_dims = output_dims;
  if (input_layout == ODLA_CHANNELS_LAST) {
    input_dims = getNCHWDims(input_dims);
    output_dims = getNCHWDims(output_dims);
  }
  auto ret_md =
      dnnl::memory::desc(getDims(output_dims), dt, getFormatTag(input_layout));
  auto input_md =
      dnnl::memory::desc(getDims(input_dims), dt, getFormatTag(input_layout));

  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  auto pool_desc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, algorithm, input_md, ret_md,
      stride_dims, kernel_dims, paddings_before, paddings_after);
  auto pd = dnnl::pooling_forward::primitive_desc(pool_desc, g_comp->eng);
  auto prim = dnnl::pooling_forward(pd);

  add_op(prim, {{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  odla_value v = CreateValue(ret_mem, orig_output_dims, value_id);
  InterpretIfNeeded();
  return v;
}

odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id) {
  odla_uint32 new_paddings_back[2];
  int start_id = 2;
  int end_id = input->shape.size;
  if (input_layout == ODLA_CHANNELS_LAST) {
    start_id -= 1;
    end_id -= 1;
  }
  for (int i = start_id; i < end_id; ++i) {
    const int src = input->shape.dims[i];
    const int dst = output_dims.dims[i];
    const int ker = window_dims[i - start_id];
    const int pad_l = paddings_front[i - start_id];
    const int pad_r = paddings_back[i - start_id];
    const int str = strides[i - start_id];
    new_paddings_back[i - start_id] = pad_r;
    if ((src - ker + pad_l + pad_r) / str + 1 != dst) {
      new_paddings_back[i - start_id] += ker - 1;
    }
  }
  return BasePool(input, input_layout, window_dims, strides, paddings_front,
                  new_paddings_back, output_dims, value_id,
                  dnnl::algorithm::pooling_max);
}

odla_value odla_AveragePool(odla_value input, odla_memory_layout input_layout,
                            const odla_uint32* window_dims,
                            const odla_uint32* strides,
                            const odla_uint32* paddings_front,
                            const odla_uint32* paddings_back,
                            odla_value_shape output_dims,
                            const odla_value_id value_id) {
  return BasePool(input, input_layout, window_dims, strides, paddings_front,
                  paddings_back, output_dims, value_id,
                  dnnl::algorithm::pooling_avg);
}

odla_value odla_BatchNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id value_id) {
  dnnl::memory oring_mem;
  dnnl::memory::data_type dtype = input->mem.get_desc().data_type();
  // black list op should convert to fp32
  bool bf16_mode = (dtype == dnnl::memory::data_type::bf16 ||
                    g_comp->opts.bf16_mode != BF16_DISABLE)
                       ? true
                       : false;
  if (bf16_mode) {
    auto f32_input_mem = cast_op(input, dnnl::memory::data_type::f32);
    mean->mem = cast_op(mean, dnnl::memory::data_type::f32);
    var->mem = cast_op(var, dnnl::memory::data_type::f32);
    if (scale != nullptr && offset != nullptr) {
      scale->mem = cast_op(scale, dnnl::memory::data_type::f32);
      offset->mem = cast_op(offset, dnnl::memory::data_type::f32);
    }
    oring_mem = input->mem;
    input->mem = f32_input_mem;
  }

  dnnl::normalization_flags flags = dnnl::normalization_flags::use_global_stats;
  auto input_md = input->mem.get_desc();
  auto input_dims = input->shape;
  const auto& type = input_md.data_type();
  auto orig_dims = input_dims;
  if (input_layout == ODLA_CHANNELS_LAST) {
    input_dims = getNCHWDims(input_dims);
    input_md = dnnl::memory::desc(getDims(input_dims), type,
                                  getFormatTag(input_layout));
  }

  unsigned channels = input_dims.dims[1];
  dnnl::memory::desc weight_md(dnnl::memory::dims{2, channels}, type,
                               dnnl::memory::format_tag::nc);
  dnnl::memory weight_mem = dnnl::memory(weight_md, g_comp->eng);

  if (scale != nullptr && offset != nullptr) {
    flags |= dnnl::normalization_flags::use_scale_shift;
    auto scale_md =
        dnnl::memory::desc({1, channels}, type, dnnl::memory::format_tag::nc);
    auto scale_mem =
        dnnl::memory(scale_md, g_comp->eng, scale->mem.get_data_handle());
    auto offset_md =
        dnnl::memory::desc({1, channels}, type, dnnl::memory::format_tag::nc);
    auto c_pd = dnnl::concat::primitive_desc(
        weight_md, 0, {scale_md, offset_md}, g_comp->eng);
    auto c = dnnl::concat(c_pd);
    c.execute(dnnl::stream(g_comp->eng),
              {{DNNL_ARG_MULTIPLE_SRC, scale->mem},
               {DNNL_ARG_MULTIPLE_SRC + 1, offset->mem},
               {DNNL_ARG_DST, weight_mem}});
  }
  auto op_desc = dnnl::batch_normalization_forward::desc(
      dnnl::prop_kind::forward, input_md, epsilon, flags);
  auto pd =
      dnnl::batch_normalization_forward::primitive_desc(op_desc, g_comp->eng);
  auto prim = dnnl::batch_normalization_forward(pd);
  auto ret_mem = dnnl::memory(input_md, g_comp->eng);

  odla_value v = CreateValue(ret_mem, orig_dims, value_id);
  add_op(prim, {{DNNL_ARG_SRC, input->mem},
                {DNNL_ARG_MEAN, mean->mem},
                {DNNL_ARG_VARIANCE, var->mem},
                {DNNL_ARG_SCALE_SHIFT, weight_mem},
                {DNNL_ARG_DST, ret_mem}});
  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE) {
    v->mem = cast_op(v, dnnl::memory::data_type::bf16);
    input->mem = oring_mem;
  }

  InterpretIfNeeded();

  return v;
}

odla_value odla_Resize(odla_value input, odla_interpolation_mode interpolation,
                       odla_resize_coordinate_mode mode, odla_uint32 axes_mask,
                       odla_value_shape output_dims,
                       const odla_value_id value_id) {
  auto algo = interpolation == ODLA_NEAREST
                  ? dnnl::algorithm::resampling_nearest
                  : dnnl::algorithm::resampling_linear;
  auto input_md = input->mem.get_desc();
  auto dt = input->mem.get_desc().data_type();

  auto ret_md = dnnl::memory::desc(getDims(output_dims), dt,
                                   dnnl::memory::format_tag::nchw);

  auto op_desc = dnnl::resampling_forward::desc(
      dnnl::prop_kind::forward_inference, algo, input_md, ret_md);
  auto pd = dnnl::resampling_forward::primitive_desc(op_desc, g_comp->eng);
  auto prim = dnnl::resampling_forward(pd);

  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  odla_value v = CreateValue(ret_mem, output_dims, value_id);
  add_op(prim, {{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});

  InterpretIfNeeded();
  return v;
}

odla_value odla_LRN(odla_value input, odla_memory_layout input_layout,
                    odla_int32 window_size, odla_float32 alpha,
                    odla_float32 beta, odla_float32 bias,
                    const odla_value_id value_id) {
  assert(window_size & 1);
  auto input_md = input->mem.get_desc();
  auto input_dims = input->shape;
  const auto& type = input_md.data_type();
  auto orig_dims = input_dims;
  if (input_layout == ODLA_CHANNELS_LAST) {
    input_dims = getNCHWDims(input_dims);
    input_md = dnnl::memory::desc(getDims(input_dims), type,
                                  getFormatTag(input_layout));
  }

  auto op_desc = dnnl::lrn_forward::desc(
      dnnl::prop_kind::forward, dnnl::algorithm::lrn_across_channels, input_md,
      (window_size - 1) / 2, alpha, beta, bias);
  auto pd = dnnl::lrn_forward::primitive_desc(op_desc, g_comp->eng);
  auto prim = dnnl::lrn_forward(pd);
  auto ret_mem = dnnl::memory(input_md, g_comp->eng);

  odla_value v = CreateValue(ret_mem, orig_dims, value_id);
  add_op(prim, {{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});

  InterpretIfNeeded();

  return v;
}

odla_value odla_Softmax(odla_value input, odla_int32 axis,
                        const odla_value_id id) {
  const auto& dims = input->shape;
  auto type = input->mem.get_desc().data_type();
  axis = axis < 0 ? dims.size - 1 : axis;
  dnnl::memory::desc input_md = getMemoryDesc(dims, type);
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  auto sm_desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference,
                                             input_md, axis);

  auto pd = dnnl::softmax_forward::primitive_desc(sm_desc, g_comp->eng);
  auto prim = dnnl::softmax_forward(pd);

  odla_value v = CreateValue(ret_mem, input->shape, id);
  add_op(prim, {{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();

  return v;
}

static odla_value reduce_op(dnnl::algorithm alg, odla_value input,
                            odla_size_t num_of_axes, const odla_uint32* axes,
                            odla_bool keep_dims, odla_value_shape output_dims,
                            const odla_value_id id) {
  std::function<void()> op;
  auto dnnl_out_dims = getDims(input->shape);

  for (int i = 0; i < num_of_axes; i++) {
    dnnl_out_dims[axes[i]] = 1;
  }

  auto type = input->mem.get_desc().data_type();
  auto output_md =
      dnnl::memory::desc(dnnl_out_dims, input->mem.get_desc().data_type(),
                         getFormatTag(input->shape));
  auto input_md = dnnl::memory::desc(getDims(input->shape),
                                     input->mem.get_desc().data_type(),
                                     getFormatTag(input->shape));
  auto ret_mem = dnnl::memory(output_md, g_comp->eng);
  auto reduction_desc =
      dnnl::reduction::desc(alg, input_md, output_md, 0.f, 0.f);
  auto pd = dnnl::reduction::primitive_desc(reduction_desc, g_comp->eng);
  auto prim = dnnl::reduction(pd);
  add_op(prim, {{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();
  odla_value v = CreateValue(ret_mem, output_dims, id);
  return v;
}

odla_value odla_ReduceMax(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_max, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_ReduceMin(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_min, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_ReduceSum(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_sum, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_mean, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  auto getGemmDims = [](odla_value_shape shape,
                        odla_bool transpose) -> dnnl::memory::dims {
    auto dnnl_dims = getDims(shape);
    int ndims = dnnl_dims.size();
    int64_t group_size = 1;

    for (int i = 0; i < ndims - 2; i++) {
      group_size *= dnnl_dims[i];
    }
    if (transpose) {
      std::swap(dnnl_dims[ndims - 2], dnnl_dims[ndims - 1]);
    }

    if (ndims == 2) {
      return {dnnl_dims[ndims - 2], dnnl_dims[ndims - 1]};
    } else {
      return {group_size, dnnl_dims[ndims - 2], dnnl_dims[ndims - 1]};
    }
  };

  auto getGemmStrides = [](dnnl::memory::dims dims,
                           odla_bool transpose) -> dnnl::memory::dims {
    if (dims.size() == 2) {
      return transpose ? dnnl::memory::dims{1, dims[0]}
                       : dnnl::memory::dims{dims[1], 1};
    }

    return transpose ? dnnl::memory::dims{dims[1] * dims[2], 1, dims[1]}
                     : dnnl::memory::dims{dims[1] * dims[2], dims[2], 1};
  };

  auto lhs_dims = getGemmDims(lhs->shape, transpose_lhs);
  auto rhs_dims = getGemmDims(rhs->shape, transpose_rhs);
  // check the dim
  assert(lhs_dims.size() == rhs_dims.size());
  if (lhs_dims.size() == 2) {
    assert(lhs_dims[1] == rhs_dims[0]);
  } else {
    assert(lhs_dims[0] == rhs_dims[0]);
    assert(lhs_dims[2] == rhs_dims[1]);
  }

  auto lhs_strides = getGemmStrides(lhs_dims, transpose_lhs);
  auto rhs_strides = getGemmStrides(rhs_dims, transpose_rhs);
  auto ret_dims = getGemmDims(output_dims, false);
  auto ret_strides = getGemmStrides(ret_dims, false);
  assert(lhs->mem.get_desc().data_type() == rhs->mem.get_desc().data_type());
  auto dt = lhs->mem.get_desc().data_type();
  auto dt_dst = (g_comp->opts.bf16_mode == BF16_ACCURACY_MODE)
                    ? getDataType(ODLA_BFLOAT16)
                    : dt;

  // keep original memory layout
  auto orig_lhs_mem = lhs->mem;
  auto orig_rhs_mem = rhs->mem;

  // add reorder process
  if (dt != dt_dst) {
    lhs->mem = cast_op(lhs, dnnl::memory::data_type::bf16);
    rhs->mem = cast_op(rhs, dnnl::memory::data_type::bf16);
    if (bias) {
      bias->mem = cast_op(bias, dnnl::memory::data_type::bf16);
    }
  }

  dnnl::memory::desc lhs_md(lhs_dims, dt_dst, lhs_strides);
  dnnl::memory::desc rhs_md(rhs_dims, dt_dst, rhs_strides);
  dnnl::memory::desc ret_md(ret_dims, dt, ret_strides);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  bool is_elements_add = false;
  int64_t N = ret_dims[output_dims.size - 1];
  if (bias) {
    auto bias_elements = GetTotalElements(bias->shape);
    if (bias_elements == N) {
      dnnl::memory::desc bias_md({1, N}, dt_dst, dnnl::memory::format_tag::ab);
      auto bias_mem =
          dnnl::memory(bias_md, g_comp->eng, bias->mem.get_data_handle());
      dnnl::matmul::desc md(lhs_md, rhs_md, bias_md, ret_md);
      dnnl::matmul::primitive_desc pd(md, g_comp->eng);
      dnnl::primitive prim = dnnl::matmul(pd);
      add_op(prim, {{DNNL_ARG_SRC, lhs->mem},
                    {DNNL_ARG_WEIGHTS, rhs->mem},
                    {DNNL_ARG_BIAS, bias_mem},
                    {DNNL_ARG_DST, ret_mem}});
    } else if (bias_elements == 1) {
      dnnl::post_ops ops;
      dnnl::primitive_attr gemm_attr;
      float beta = ((float*)bias->mem.get_data_handle())[0];
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_linear, 0.f, beta);
      gemm_attr.set_post_ops(ops);
      dnnl::matmul::desc md(lhs_md, rhs_md, ret_md);
      dnnl::matmul::primitive_desc pd(md, gemm_attr, g_comp->eng);
      dnnl::primitive prim = dnnl::matmul(pd);
      add_op(prim, {{DNNL_ARG_SRC, lhs->mem},
                    {DNNL_ARG_WEIGHTS, rhs->mem},
                    {DNNL_ARG_DST, ret_mem}});
    } else
      is_elements_add = true;
  } else {
    dnnl::matmul::desc md(lhs_md, rhs_md, ret_md);
    dnnl::matmul::primitive_desc pd(md, g_comp->eng);
    dnnl::primitive prim = dnnl::matmul(pd);
    add_op(prim, {{DNNL_ARG_SRC, lhs->mem},
                  {DNNL_ARG_WEIGHTS, rhs->mem},
                  {DNNL_ARG_DST, ret_mem}});
  }
  odla_value v =
      CreateValue(ret_mem, output_dims, is_elements_add ? nullptr : id);

  InterpretIfNeeded();
  lhs->mem = orig_lhs_mem;
  rhs->mem = orig_rhs_mem;
  return is_elements_add ? odla_Add(v, bias, id) : v;
}

odla_value odla_Erf(odla_value input, const odla_value_id value_id) {
  std::function<void()> op;
  const auto& input_shape = input->shape;
  dnnl::memory::data_type type = input->mem.get_desc().data_type();

  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  size_t total_size = GetTotalElements(input->shape);
  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE) {
    op = [total_size, input, ret_mem]() {
      dnnl_utils::erf_bf16_func((int16_t*)input->mem.get_data_handle(),
                                (float*)ret_mem.get_data_handle(), total_size);
    };
  } else {
    op = [total_size, input, ret_mem]() {
      dnnl_utils::erf_func((float*)input->mem.get_data_handle(),
                           (float*)ret_mem.get_data_handle(), total_size);
    };
  }
  add_op(op);
  odla_value v = CreateValue(ret_mem, input->shape, value_id);

  InterpretIfNeeded();

  return v;
}

odla_value odla_Slice(odla_value input, const odla_uint32* start,
                      const odla_uint32* end, const odla_uint32* strides,
                      odla_value_shape output_dims, const odla_value_id id) {
  const auto& input_dims = input->shape;
  int dims = input_dims.size;
  auto offsets = dnnl::memory::dims(start, start + dims);
  // only support stride of 1
  for (int i = 0; i < dims; ++i) {
    assert(strides[i] == 1);
  }
  dnnl::memory::data_type type = input->mem.get_desc().data_type();
  dnnl::memory::desc input_md = getMemoryDesc(input_dims, type);
  dnnl::memory::desc src_sub_md =
      input_md.submemory_desc(getDims(output_dims), offsets);
  dnnl::memory::desc dst_md = getMemoryDesc(output_dims, type);

  // dummy reorder
  auto src_mem = dnnl::memory(src_sub_md, g_comp->eng, nullptr);
  auto dst_mem = dnnl::memory(dst_md, g_comp->eng);
  auto prim = dnnl::reorder(src_mem, dst_mem);
  add_op(prim, {{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, dst_mem}});
  InterpretIfNeeded();
  return CreateValue(dst_mem, output_dims, id);
}

odla_values odla_TopK(odla_value input, odla_uint32 K, odla_bool largest,
                      odla_bool sorted, odla_uint32 axis,
                      odla_value_type output_value_type,
                      odla_value_type output_value_index_type,
                      const odla_value_ids value_ids) {
  dnnl::memory::data_type type = input->mem.get_desc().data_type();
  // black list op should convert to fp32
  bool bf16_mode = (type == dnnl::memory::data_type::bf16 ||
                    g_comp->opts.bf16_mode != BF16_DISABLE)
                       ? true
                       : false;
  if (bf16_mode) {
    auto f32_mem = cast_op(input, dnnl::memory::data_type::f32);
    input->mem = f32_mem;
  }
  auto input_ptr = (float*)input->mem.get_data_handle();
  auto output_dims = output_value_type.shape;
  dnnl::memory::desc dst_md =
      getMemoryDesc(output_dims, dnnl::memory::data_type::f32);
  auto dst_mem = dnnl::memory(dst_md, g_comp->eng);
  auto dst_idx_mem = dnnl::memory(dst_md, g_comp->eng);
  std::vector<int32_t> input_shape;
  for (int i = 0; i < input->shape.size; i++) {
    input_shape.push_back(input->shape.dims[i]);
  }
  auto op = [=]() {
    dnnl_utils::topk_func(input_ptr, (float*)dst_mem.get_data_handle(),
                          (int*)dst_idx_mem.get_data_handle(), input_shape, K,
                          largest, sorted, axis);
  };
  add_op(op);
  auto dst_elements = CreateValue(dst_mem, output_dims, value_ids.value_ids[0]);
  auto dst_idxs = CreateValue(dst_idx_mem, output_dims, value_ids.value_ids[1]);

  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE) {
    dst_elements->mem = cast_op(dst_elements, dnnl::memory::data_type::bf16);
  }
  return {.size = 2, .values = {dst_elements, dst_idxs}};
}

odla_value odla_NMS(odla_value input_boxes, odla_value input_scores,
                    odla_uint32 max_num_outputs, odla_float32 iou_threshold,
                    odla_float32 score_threshold,
                    odla_value_type output_value_type,
                    const odla_value_id value_id) {
  auto type = input_boxes->mem.get_desc().data_type();
  // black list op should convert to fp32
  bool bf16_mode = (type == dnnl::memory::data_type::bf16 ||
                    g_comp->opts.bf16_mode != BF16_DISABLE)
                       ? true
                       : false;
  if (bf16_mode) {
    auto f32_boxes_mem = cast_op(input_boxes, dnnl::memory::data_type::f32);
    auto f32_scores_mem = cast_op(input_scores, dnnl::memory::data_type::f32);
    input_boxes->mem = f32_boxes_mem;
    input_scores->mem = f32_boxes_mem;
  }
  std::function<void()> op;
  int32_t batch_size =
      input_boxes->shape.size == 2 ? 1 : input_boxes->shape.dims[0];
  int32_t boxes_size = input_boxes->shape.size == 2
                           ? input_boxes->shape.dims[0]
                           : input_boxes->shape.dims[1];
  int32_t input_shape[3] = {batch_size, boxes_size, 4};
  size_t class_num =
      input_scores->shape.size == 1 ? 1 : input_scores->shape.dims[1];
  odla_value_shape boxes_shape = {.size = 2, .dims = {boxes_size, 4}};
  size_t sample_size = GetTotalElements(boxes_shape);

  auto ret_md = getMemoryDesc(output_value_type.shape,
                              getDataType(output_value_type.element_type));
  assert(output_value_type.shape.dims[0] == (batch_size * max_num_outputs));
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  dnnl::memory::data_type boxes_type = input_boxes->mem.get_desc().data_type();
  dnnl::memory::data_type scores_type =
      input_scores->mem.get_desc().data_type();

  assert(boxes_type == dnnl_f32 && scores_type == dnnl_f32);
  assert(output_value_type.element_type == ODLA_INT32);

  size_t output_num = output_value_type.shape.dims[0];

  op = [input_boxes, input_scores, batch_size, class_num, sample_size,
        boxes_size, max_num_outputs, iou_threshold, score_threshold,
        ret_mem]() {
#pragma omp parallel
    for (int i = 0; i < batch_size; i++) {
      auto boxes_ptr =
          (float*)input_boxes->mem.get_data_handle() + i * sample_size;
      auto scores_ptr =
          (float*)input_scores->mem.get_data_handle() + i * boxes_size;
      auto output_ptr =
          (int32_t*)ret_mem.get_data_handle() + i * 3 * max_num_outputs;
      dnnl_utils::nms_func(boxes_ptr, scores_ptr, i, class_num, boxes_size,
                           max_num_outputs, score_threshold, iou_threshold,
                           output_ptr);
    }
  };

  add_op(op);
  InterpretIfNeeded();
  odla_value v = CreateValue(ret_mem, input_scores->shape, value_id);

  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE) {
    v->mem = cast_op(v, dnnl::memory::data_type::bf16);
  }

  return v;
}

odla_value odla_Tile(odla_value input, const odla_uint32* repeat,
                     odla_value_shape output_dims,
                     const odla_value_id value_id) {
  bool skip_tile = true;
  for (int i = 0; i < output_dims.size; i++) {
    skip_tile &= (repeat[i] == 1);
  }
  if (skip_tile) {
    return input;
  }
  std::function<void()> rewrite_input_ptr_op;
  rewrite_input_ptr_op = [input]() {
    auto op = g_comp->ops[1];
    // if tile is the first op then we rewrite the src_mem data handle to bind
    // the input to it. condition: the first op has primitive && primitive is
    // dnnl_concat && all data handle of srcs is same.
    if (!op.prim) return;
    auto kind = op.prim.get_kind();
    if (kind == dnnl::primitive::kind::concat) {
      auto iterator0 = op.args.find(DNNL_ARG_MULTIPLE_SRC);
      auto iterator1 = op.args.find(DNNL_ARG_MULTIPLE_SRC + 1);
      if (iterator0->second.get_data_handle() ==
          iterator1->second.get_data_handle()) {
        for (int i = 0; i < op.args.size() - 1; i++) {
          op.args[DNNL_ARG_MULTIPLE_SRC + i].set_data_handle(
              input->mem.get_data_handle());
        }
      }
    }
  };
  add_op(rewrite_input_ptr_op);

  auto dim_size = output_dims.size;
  auto ret_md = dnnl::memory::desc(getDims(output_dims),
                                   input->mem.get_desc().data_type(),
                                   getFormatTag(output_dims));

  std::vector<int64_t> input_shape;
  for (int i = 0; i < dim_size; i++) {
    input_shape.push_back(input->shape.dims[i]);
  }
  auto input_ptr = input->mem.get_data_handle();
  for (int i = 0; i < dim_size; i++) {
    std::vector<dnnl::memory::desc> src_mds;
    std::unordered_map<int, dnnl::memory> concat_args;
    auto curr_dim = dnnl::memory::dims(input_shape);
    if (repeat[i] == 1) continue;
    auto src_md =
        dnnl::memory::desc(curr_dim, input->mem.get_desc().data_type(),
                           getFormatTag(input->shape));
    auto src_mem = dnnl::memory(src_md, g_comp->eng, input_ptr);
    for (int j = 0; j < repeat[i]; j++) {
      src_mds.push_back(src_md);
      concat_args.insert({DNNL_ARG_MULTIPLE_SRC + j, src_mem});
    }
    auto concat_pd =
        dnnl::concat::primitive_desc(ret_md, i, src_mds, g_comp->eng);
    auto dst_mem = dnnl::memory(concat_pd.dst_desc(), g_comp->eng);
    concat_args.insert({DNNL_ARG_DST, dst_mem});
    auto concat_prim = dnnl::concat(concat_pd);
    auto kd = concat_prim.get_kind();
    add_op(concat_prim, concat_args);
    input_ptr = dst_mem.get_data_handle();
    input_shape[i] = input_shape[i] * repeat[i];
  }
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng, input_ptr);
  InterpretIfNeeded();
  odla_value v = CreateValue(ret_mem, output_dims, value_id);
  v->is_const = true;
  return v;
}

odla_value odla_ArgMax(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id value_id) {
  dnnl::memory::desc md = getMemoryDesc(output_value_type);
  dnnl::memory dst_mem = dnnl::memory(md, g_comp->eng);
  std::function<void()> op;
  if (input->mem.get_desc().data_type() != dnnl::memory::data_type::f32) {
    input->mem = cast_odla_mem(input->mem, input->shape,
                               dnnl::memory::data_type::f32, false);
  }

  if (output_value_type.element_type == ODLA_INT64) {
    op = [input, dst_mem, axis]() {
      ArgMax<float, int64_t>(static_cast<float*>(input->mem.get_data_handle()),
                             static_cast<int64_t*>(dst_mem.get_data_handle()),
                             axis, input->shape);
    };
  } else if (output_value_type.element_type == ODLA_INT32) {
    op = [input, dst_mem, axis]() {
      ArgMax<float, int32_t>(static_cast<float*>(input->mem.get_data_handle()),
                             static_cast<int32_t*>(dst_mem.get_data_handle()),
                             axis, input->shape);
    };
  } else {
    assert(0);
  }

  add_op(op);
  InterpretIfNeeded();
  return CreateValue(dst_mem, output_value_type.shape, value_id);
}

} // C extern
