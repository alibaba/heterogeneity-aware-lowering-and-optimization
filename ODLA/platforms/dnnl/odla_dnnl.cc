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

#include "odla_dnnl.h"

#include <stdio.h>

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "ODLA/odla_common.h"
#include "ODLA/odla_value.h"
#include "ODLA/ops/odla_ops_math.h"
#include "ODLA/ops/odla_ops_nn.h"
#include "dnnl_threadpool_iface.hpp"
#include "dnnl_utils.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

enum class alg_unary_eltwise {
  isnan,
  isinf,
  abs,
  acos,
  asin,
  atan,
  ceil,
  cos,
  cosh,
  sin,
  sinh,
  log,
  tan,
  tanh,
  sqrt,
  neg,
  acosh,
  asinh,
  atanh,
  reciprocal,
  sign,
};

struct _odla_context {
  odla_computation comp;
  std::unique_ptr<dnnl::stream> stream;
};

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;
thread_local bool g_interpret_mode = false;

void add_op(dnnl::primitive prim,
            const std::unordered_map<int, dnnl::memory>& args) {
  operation op;
  op.prim = prim;
  op.args = args;
  g_comp->ops.emplace_back(op);
}

void add_op(std::function<void()> func) {
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
    auto len = getValueStorageSize(src_val);
    memcpy(dst_ptr, src_val->mem.get_data_handle(), len);
  }
  return ODLA_SUCCESS;
}

void InterpretIfNeeded() {
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
  v->elem_type = type.element_type;
  g_comp->inputs[name] = v;
  g_comp->input_vals.push_back(v);
  return v;
}

odla_status odla_GetNumOfArgsFromComputation(const odla_computation computation,
                                             odla_uint32* num_args) {
  *num_args = computation->input_vals.size();
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromComputationByIdx(const odla_computation computation,
                                            const odla_uint32 arg_idx,
                                            odla_value* arg_value) {
  *arg_value = nullptr;
  if (arg_idx >= computation->input_vals.size()) {
    return ODLA_FAILURE;
  }
  *arg_value = computation->input_vals[arg_idx];
  return ODLA_SUCCESS;
}

odla_value odla_CreateValue(odla_value_type type, const odla_value_id id) {
  assert(g_interpret_mode);
  rewrite_scalar_type(type);
  auto v = odla_CreateArgument(type, id);
  return v;
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  value_type->shape = value->shape;
  value_type->element_type = value->elem_type;
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  if (context->comp->opts.bf16_mode == BF16_PERFORMACE_MODE &&
      value->mem.get_desc().data_type() == dnnl::memory::data_type::bf16) {
    auto src_md = dnnl::memory::desc(value->mem.get_desc().dims(),
                                     getDataType(ODLA_FLOAT32),
                                     getFormatTag(value->shape));
    auto src_mem =
        dnnl::memory(src_md, context->comp->eng, const_cast<void*>(data_ptr));
    auto r = dnnl::reorder(src_mem, value->mem);
    r.execute(dnnl::stream(context->comp->eng),
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
  v->elem_type = type.element_type;
  return v;
}

template <typename T>
static odla_value CreateConstantFromScalar(odla_element_type dt, const T& v,
                                           int rank = 1,
                                           odla_value_id id = nullptr) {
  void* buf = g_comp->CreateBuffer(sizeof(v));
  memcpy(buf, &v, sizeof(v));
  odla_value_shape shape;
  shape.size = rank;
  for (int i = 0; i < rank; ++i) {
    shape.dims[i] = 1;
  }
  return odla_CreateConstant({dt, shape}, buf, id);
}

static odla_value CreateConstantFromScalar(float v, int rank = 1,
                                           odla_value_id id = nullptr) {
  return CreateConstantFromScalar(ODLA_FLOAT32, v, rank, id);
}

odla_status odla_SetValueAsOutput(const odla_value val) {
  g_comp->outputs[val->name] = val;
  // convert output to float32
  if (g_comp->opts.bf16_mode != BF16_DISABLE &&
      val->mem.get_desc().data_type() == dnnl::memory::data_type::bf16) {
    val->mem =
        cast_odla_mem(val->mem, val->shape, getDataType(ODLA_FLOAT32), false);
  }
  g_comp->output_vals.push_back(val);
  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfOutputsFromComputation(
    const odla_computation computation, odla_uint32* num_outputs) {
  *num_outputs = computation->output_vals.size();
  return ODLA_SUCCESS;
}

odla_status odla_GetOutputFromComputationByIdx(
    const odla_computation computation, const odla_uint32 output_idx,
    odla_value* output_value) {
  *output_value = nullptr;
  if (output_idx >= computation->output_vals.size()) {
    return ODLA_FAILURE;
  }
  *output_value = computation->output_vals[output_idx];
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  // Handle the case of output is constant due to compile-time optimization.
  if (value->is_const) {
    size_t len = getValueStorageSize(value);
    memcpy(data_ptr, value->mem.get_data_handle(), len);
  } else {
    auto name = value->name;
    auto& outputs_v = context->comp->outputs_v;
    auto val = context->comp->outputs[name];
    outputs_v[name] = {val, data_ptr};
  }
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  std::string name((const char*)value_id);
  auto& outputs_v = context->comp->outputs_v;
  auto val = context->comp->outputs[name];
  outputs_v[name] = {val, data_ptr};
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
    idx_size = indices->shape.dims[0];
  }
  size_t inner_size = 1;
  size_t outer_loop = 1;
  for (int i = axis + 1; i < params->shape.size; ++i) {
    inner_size *= params->shape.dims[i];
  }
  size_t outer_size = inner_size * params->shape.dims[axis];
  for (int i = 0; i < axis; ++i) {
    outer_loop *= params->shape.dims[i];
  }
  auto ret_md =
      dnnl::memory::desc(getDims(output_dims), dt, getFormatTag(output_dims));
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  int byte_size =
      params->mem.get_desc().get_size() / GetTotalElements(params->shape);

  auto one_batch_byte_size =
      GetTotalElements(output_dims) / batch_size * byte_size;
  op = [params, indices, batch_size, idx_size, inner_size, outer_loop,
        outer_size, byte_size, ret_mem, one_batch_byte_size, axis]() {
    int32_t* indices_ptr = (int32_t*)indices->mem.get_data_handle();
    std::vector<int> indices_i32;
    if (getElementStorageSize(indices->elem_type) == 8) {
      int64_t* src = (int64_t*)indices_ptr;
      indices_i32.insert(indices_i32.begin(), src, src + idx_size);
      indices_ptr = indices_i32.data();
    }
    char* ret_ptr = (char*)ret_mem.get_data_handle();
#pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
      dnnl_utils::gather_func((char*)params->mem.get_data_handle(),
                              indices_ptr + i * idx_size, idx_size, inner_size,
                              outer_loop, outer_size, byte_size,
                              ret_ptr + i * one_batch_byte_size);
    }
  };
  add_op(op);
  InterpretIfNeeded();
  return CreateValue(ret_mem, output_dims, id);
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

static void expand_dims(odla_value_shape& src, odla_value_shape& dst) {
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

static dnnl::memory broadcast_mem(const dnnl::memory orig_mem,
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

static std::pair<dnnl::memory, dnnl::memory> broadcast_operands(
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

static odla_value binary_eltwise(dnnl::algorithm algo, odla_value lhs,
                                 odla_value rhs, const odla_value_id id) {
  odla_value_shape ret_shape;
  auto new_mems = broadcast_operands(lhs, rhs, &ret_shape);
  auto lhs_m = new_mems.first;
  auto rhs_m = new_mems.second;

  auto type = lhs->mem.get_desc().data_type();
  if (type == dnnl::memory::data_type::s32) {
    return binary_eltwise_s32(algo, lhs_m, rhs_m, lhs->shape, id);
  }
  auto ret_md = getMemoryDesc(ret_shape, type);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  dnnl::binary::desc bd(algo, lhs_m.get_desc(), rhs_m.get_desc(), ret_md);
  dnnl::binary::primitive_desc pd(bd, g_comp->eng);
  dnnl::primitive prim = dnnl::binary(pd);

  add_op(prim, {{DNNL_ARG_SRC_0, lhs_m},
                {DNNL_ARG_SRC_1, rhs_m},
                {DNNL_ARG_DST, ret_mem}});

  odla_value v = CreateValue(ret_mem, ret_shape, id);
  v->elem_type = lhs->elem_type;
  InterpretIfNeeded();
  return v;
}

odla_value odla_Abs(odla_value input, const odla_value_id value_id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_abs, input, 0.f, 0.f,
                          value_id);
}

odla_value odla_Tanh(odla_value input, const odla_value_id value_id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_tanh, input, 0.f, 0.f,
                          value_id);
}

odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_eltwise(dnnl::algorithm::binary_add, lhs, rhs, id);
}

odla_value odla_Max(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_eltwise(dnnl::algorithm::binary_max, lhs, rhs, id);
}

odla_value odla_Min(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_eltwise(dnnl::algorithm::binary_min, lhs, rhs, id);
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

odla_value odla_Log(odla_value input, const odla_value_id value_id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_log, input, 0.f, 0.f,
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

odla_value odla_HardSigmoid(odla_value input, odla_float32 alpha,
                            odla_float32 beta, const odla_value_id value_id) {
  auto linear_input = unary_eltwise_op(dnnl::algorithm::eltwise_linear, input,
                                       alpha, beta, nullptr);
  int n = GetTotalElements(input->shape);
  auto elem_type = input->elem_type;
  // Prepare dest memory.
  dnnl::memory::desc dst_md = getMemoryDesc({elem_type, input->shape});
  dnnl::memory dst_mem = dnnl::memory(dst_md, g_comp->eng);
  auto v = CreateValue(dst_mem, input->shape, value_id);
  v->elem_type = elem_type;

  auto op = [linear_input, v, n] {
    void* dst = v->mem.get_data_handle();
    const void* data = linear_input->mem.get_data_handle();
    const float* input_t = static_cast<const float*>(data);
    Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>> in(input_t, n);
    float* dst_t = static_cast<float*>(dst);
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> out(dst_t, n);
    auto t = (0 < in).select(in, 0.0f);
    out = (1 > t).select(t, 1.0f);
  };

  add_op(op);
  InterpretIfNeeded();
  return v;
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

odla_value odla_Elu(odla_value input, odla_float32 alpha,
                    const odla_value_id id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_elu, input, alpha, 0.f, id);
}

odla_value odla_Selu(odla_value input, odla_float32 alpha, odla_float32 lambda,
                     const odla_value_id id) {
  auto elu_v = odla_Elu(input, alpha, id);
  auto rank = input->shape.size;
  auto lambda_v = CreateConstantFromScalar(lambda, rank);
  auto v = odla_Mul(elu_v, lambda_v, id);
  return v;
}

odla_value odla_Celu(odla_value input, odla_float32 alpha,
                     const odla_value_id id) {
  auto relu_v = odla_Relu(input, nullptr);
  auto rank = input->shape.size;
  auto alpha_v = CreateConstantFromScalar(alpha, rank);
  auto input_divalpha = odla_Div(input, alpha_v, nullptr);
  auto v_divalpha = odla_Elu(input_divalpha, alpha, nullptr);
  auto neg_input = unary_eltwise_op(dnnl::algorithm::eltwise_linear, v_divalpha,
                                    -1.f, 0.f, nullptr);
  auto neg_relu_v = odla_Relu(neg_input, nullptr);
  auto input_negpart = unary_eltwise_op(dnnl::algorithm::eltwise_linear,
                                        neg_relu_v, -1.f, 0.f, nullptr);
  auto v = odla_Add(input_negpart, relu_v, id);
  return v;
}

odla_value odla_ThresholdedRelu(odla_value input, odla_float32 alpha,
                                const odla_value_id id) {
  int n = GetTotalElements(input->shape);
  auto elem_type = input->elem_type;
  // Prepare dest memory.
  dnnl::memory::desc dst_md = getMemoryDesc({elem_type, input->shape});
  dnnl::memory dst_mem = dnnl::memory(dst_md, g_comp->eng);
  auto v = CreateValue(dst_mem, input->shape, id);
  v->elem_type = elem_type;

  auto op = [input, alpha, dst_mem, n] {
    void* dst = dst_mem.get_data_handle();
    const void* data = input->mem.get_data_handle();
    const float* input_t = static_cast<const float*>(data);
    Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>> in(input_t, n);
    float* dst_t = static_cast<float*>(dst);
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>> out(dst_t, n);
    out = (alpha < in).select(in, 0.0f);
  };

  add_op(op);
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

template <typename T>
static void left_shift(void* dst, const void* input, const void* shift_amt,
                       size_t n) {
  T* dst_t = static_cast<T*>(dst);
  const T* input_t = static_cast<const T*>(input);
  const T* shift_amt_t = static_cast<const T*>(shift_amt);
  for (size_t i = 0; i < n; ++i) {
    dst_t[i] = input_t[i] << shift_amt_t[i];
  }
}

template <typename T>
static void right_shift(void* dst, const void* input, const void* shift_amt,
                        size_t n) {
  T* dst_t = static_cast<T*>(dst);
  const T* input_t = static_cast<const T*>(input);
  const T* shift_amt_t = static_cast<const T*>(shift_amt);
  for (size_t i = 0; i < n; ++i) {
    dst_t[i] = input_t[i] >> shift_amt_t[i];
  }
}

odla_value odla_Shift(odla_value input, odla_value shift_amount,
                      odla_bool is_left_shift, const odla_value_id id) {
  auto elem_type = input->elem_type;
  bool is_left = is_left_shift != 0;
  assert(elem_type != ODLA_FLOAT32 && elem_type != ODLA_FLOAT64 &&
         elem_type != ODLA_BFLOAT16 && elem_type != ODLA_BFLOAT16);
  assert(elem_type = shift_amount->elem_type);
  int n = GetTotalElements(input->shape);
  // Prepare dest memory.
  dnnl::memory dst_mem;
  dnnl::memory::desc dst_md = getMemoryDesc({elem_type, input->shape});
  if (elem_type == ODLA_INT64 || elem_type == ODLA_UINT64) {
    auto buf = g_comp->CreateBuffer(getElementStorageSize(elem_type) * n);
    dst_mem = dnnl::memory(dst_md, g_comp->eng, buf);
  } else {
    dst_mem = dnnl::memory(dst_md, g_comp->eng);
  }
  auto v = CreateValue(dst_mem, input->shape, id);
  v->elem_type = elem_type;
  auto op = [input, shift_amount, dst_mem, is_left, n] {
    void* dst = dst_mem.get_data_handle();
    const void* data = input->mem.get_data_handle();
    const void* shifts = shift_amount->mem.get_data_handle();
    if (input->elem_type == ODLA_UINT8) {
      is_left ? left_shift<uint8_t>(dst, data, shifts, n)
              : right_shift<uint8_t>(dst, data, shifts, n);
    } else if (input->elem_type == ODLA_UINT16) {
      is_left ? left_shift<uint16_t>(dst, data, shifts, n)
              : right_shift<uint16_t>(dst, data, shifts, n);
    } else if (input->elem_type == ODLA_UINT32) {
      is_left ? left_shift<uint32_t>(dst, data, shifts, n)
              : right_shift<uint32_t>(dst, data, shifts, n);
    } else if (input->elem_type == ODLA_UINT64) {
      is_left ? left_shift<uint64_t>(dst, data, shifts, n)
              : right_shift<uint64_t>(dst, data, shifts, n);
    } else {
      assert(0);
    }
  };
  add_op(op);

  InterpretIfNeeded();
  return v;
}

template <typename T>
static void unary_eltwise_T(alg_unary_eltwise alg, void* dst, const void* input,
                            int n) {
  const T* input_t = static_cast<const T*>(input);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> in(input_t, n);
  T* dst_t = static_cast<T*>(dst);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> out(dst_t, n);
  switch (alg) {
    case alg_unary_eltwise::abs:
      out = in.abs();
      break;
    case alg_unary_eltwise::neg:
      out = -in;
      break;
    case alg_unary_eltwise::sign:
      out = (0 < in).select(1, in);
      out = (0 > out).select(-1, out);
      break;
    case alg_unary_eltwise::ceil:
      out = in.ceil();
      break;
    case alg_unary_eltwise::log:
      out = in.log();
      break;
    case alg_unary_eltwise::sqrt:
      out = in.sqrt();
      break;
    case alg_unary_eltwise::reciprocal:
      out = in.pow(-1);
      break;
    case alg_unary_eltwise::sin:
      out = in.sin();
      break;
    case alg_unary_eltwise::cos:
      out = in.cos();
      break;
    case alg_unary_eltwise::tan:
      out = in.tan();
      break;
    case alg_unary_eltwise::acos:
      out = in.acos();
      break;
    case alg_unary_eltwise::asin:
      out = in.asin();
      break;
    case alg_unary_eltwise::asinh:
      out = in.asinh();
      break;
    case alg_unary_eltwise::atan:
      out = in.atan();
      break;
    case alg_unary_eltwise::atanh:
      out = in.atanh();
      break;
    case alg_unary_eltwise::sinh:
      out = in.sinh();
      break;
    case alg_unary_eltwise::tanh:
      out = in.tanh();
      break;
    case alg_unary_eltwise::cosh:
      out = in.cosh();
      break;
    case alg_unary_eltwise::acosh:
      out = in.acosh();
      break;
    default:
      assert(0);
  }
}

template <typename T>
static void unary_eltwise_bool(alg_unary_eltwise alg, void* dst,
                               const void* input, int n) {
  const T* input_t = static_cast<const T*>(input);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> in(input_t, n);
  bool* dst_t = static_cast<bool*>(dst);
  Eigen::Map<Eigen::Array<bool, Eigen::Dynamic, 1>> out(dst_t, n);
  switch (alg) {
    case alg_unary_eltwise::isnan:
      out = in.isNaN();
      break;
    case alg_unary_eltwise::isinf:
      out = in.isInf();
      break;
    default:
      assert(0);
  }
}

static odla_value odla_unary_eltwise(alg_unary_eltwise alg, odla_value input,
                                     const odla_value_id value_id) {
  // Extract type and size
  auto elem_type = input->elem_type;
  bool ret_bool =
      (alg == alg_unary_eltwise::isnan || alg == alg_unary_eltwise::isinf);
  if (ret_bool) {
    elem_type = ODLA_BOOL;
  }
  int n = GetTotalElements(input->shape);
  // Prepare destination memory
  dnnl::memory dst_mem;
  dnnl::memory::desc dst_md = getMemoryDesc({elem_type, input->shape});
  dst_mem = dnnl::memory(dst_md, g_comp->eng);
  auto v = CreateValue(dst_mem, input->shape, value_id);
  v->elem_type = elem_type;
  // Create lambda operation
  auto op = [alg, ret_bool, input, dst_mem, n] {
    void* dst = dst_mem.get_data_handle();
    const void* data = input->mem.get_data_handle();
    if (input->elem_type == ODLA_FLOAT32) {
      ret_bool ? unary_eltwise_bool<float>(alg, dst, data, n)
               : unary_eltwise_T<float>(alg, dst, data, n);
    } else if (input->elem_type == ODLA_FLOAT64) {
      ret_bool ? unary_eltwise_bool<double>(alg, dst, data, n)
               : unary_eltwise_T<double>(alg, dst, data, n);
    } else if (input->elem_type == ODLA_UINT8) {
      ret_bool ? unary_eltwise_bool<uint8_t>(alg, dst, data, n)
               : unary_eltwise_T<uint8_t>(alg, dst, data, n);
    } else if (input->elem_type == ODLA_UINT16) {
      ret_bool ? unary_eltwise_bool<uint16_t>(alg, dst, data, n)
               : unary_eltwise_T<uint16_t>(alg, dst, data, n);
    } else if (input->elem_type == ODLA_UINT32) {
      ret_bool ? unary_eltwise_bool<uint32_t>(alg, dst, data, n)
               : unary_eltwise_T<uint32_t>(alg, dst, data, n);
    } else if (input->elem_type == ODLA_UINT64) {
      ret_bool ? unary_eltwise_bool<uint64_t>(alg, dst, data, n)
               : unary_eltwise_T<uint64_t>(alg, dst, data, n);
    } else {
      assert(0);
    }
  };
  // Postprocess
  add_op(op);
  InterpretIfNeeded();
  return v;
}

odla_value odla_IsNaN(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::isnan, input, value_id);
}

odla_value odla_IsInf(odla_value input, odla_bool detect_negative,
                      odla_bool detect_positive, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::isinf, input, value_id);
}

odla_value odla_Cos(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::cos, input, value_id);
}

odla_value odla_Sin(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::sin, input, value_id);
}

odla_value odla_Tan(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::tan, input, value_id);
}

odla_value odla_ACos(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::acos, input, value_id);
}

odla_value odla_ACosh(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::acosh, input, value_id);
}

odla_value odla_ASin(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::asin, input, value_id);
}

odla_value odla_ASinh(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::asinh, input, value_id);
}

odla_value odla_ATan(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::atan, input, value_id);
}

odla_value odla_ATanh(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::atanh, input, value_id);
}

odla_value odla_Sinh(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::sinh, input, value_id);
}

odla_value odla_Cosh(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::cosh, input, value_id);
}

odla_value odla_Ceil(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::ceil, input, value_id);
}

odla_value odla_Neg(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::neg, input, value_id);
}

odla_value odla_Reciprocal(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::reciprocal, input, value_id);
}

odla_value odla_Sign(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::sign, input, value_id);
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
  v->elem_type = input->elem_type;
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
  dnnl::memory origin_mem;
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
    origin_mem = input->mem;
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
  dnnl::memory scale_offset_mem = dnnl::memory();
  if (scale != nullptr || offset != nullptr || scalar_offset != 0.0F ||
      scalar_scale != 1.0F) {
    // make a tensor [scale, bias].
    auto get_value = [channels](odla_value x, float scalar) {
      if (x == nullptr) {
        x = CreateConstantFromScalar(scalar, 2);
      }
      return odla_Reshape(x, {2, {1, channels}}, nullptr);
    };
    odla_value s = get_value(scale, scalar_scale);
    odla_value b = get_value(offset, scalar_offset);
    flags |= dnnl::normalization_flags::use_scale_shift;
    auto scale_offset =
        odla_Concat({2, {s, b}}, 0, {2, {2, channels}}, nullptr);
    scale_offset_mem = scale_offset->mem;
  }
  auto op_desc = dnnl::batch_normalization_forward::desc(
      dnnl::prop_kind::forward, input_md, epsilon, flags);
  auto pd =
      dnnl::batch_normalization_forward::primitive_desc(op_desc, g_comp->eng);
  auto prim = dnnl::batch_normalization_forward(pd);
  auto ret_mem = dnnl::memory(input_md, g_comp->eng);

  add_op(prim, {{DNNL_ARG_SRC, input->mem},
                {DNNL_ARG_MEAN, mean->mem},
                {DNNL_ARG_VARIANCE, var->mem},
                {DNNL_ARG_SCALE_SHIFT, scale_offset_mem},
                {DNNL_ARG_DST, ret_mem}});
  odla_value v = CreateValue(ret_mem, orig_dims, value_id);
  if (g_comp->opts.bf16_mode == BF16_PERFORMACE_MODE) {
    v->mem = cast_op(v, dnnl::memory::data_type::bf16);
    input->mem = origin_mem;
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

odla_value odla_InstanceNormalization(
    odla_value input, odla_memory_layout input_layout, odla_float32 epsilon,
    odla_value scale, odla_value offset, odla_float32 scalar_scale,
    odla_float32 scalar_offset, const odla_value_id value_id) {
  bool ch_first = input_layout == odla_memory_layout::ODLA_CHANNELS_FIRST;
  auto mean_shape = input->shape;
  int rank = input->shape.size;
  std::vector<odla_uint32> axes;
  int m = 1;
  axes.reserve(rank - 2);

  if (ch_first) {
    for (int i = 2; i < rank; ++i) {
      axes.push_back(i);
      mean_shape.dims[i] = 1;
      m *= input->shape.dims[i];
    }
  } else {
    for (int i = 1; i < rank - 1; ++i) {
      axes.push_back(i);
      mean_shape.dims[i] = 1;
      m *= input->shape.dims[i];
    }
  }

  auto scale_shape = mean_shape;
  scale_shape.dims[0] = 1;
  if (ch_first) {
    scale = odla_Reshape(scale, scale_shape, nullptr);
    offset = odla_Reshape(offset, scale_shape, nullptr);
  }

  auto elems = CreateConstantFromScalar(static_cast<float>(m));
  auto eps = CreateConstantFromScalar(epsilon);
  auto mean = odla_ReduceMean(input, rank - 2, axes.data(), 1 /* keep_dims */,
                              mean_shape, nullptr);
  auto x_minus_mean = odla_Sub(input, mean, nullptr);
  auto var = odla_Div(odla_ReduceSumSquare(x_minus_mean, rank - 2, axes.data(),
                                           1, mean_shape, nullptr),
                      elems, nullptr);
  auto norm = odla_Div(
      x_minus_mean, odla_Sqrt(odla_Add(var, eps, nullptr), nullptr), nullptr);
  return odla_Add(odla_Mul(norm, scale, nullptr), offset, value_id);
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

template <typename T>
static void DoHardmax(const int* max_val_indices, T* output_ptr, int axis,
                      const odla_value_shape& shape) {
  auto dim = shape.dims[axis];
  auto elems_from_axis = GetCountFromAxis(shape, axis);
  auto extents_on_axis = elems_from_axis / dim;
  auto elems_before_axis = GetTotalElements(shape) / elems_from_axis;
  for (int64_t i = 0; i < elems_before_axis; ++i) {
    for (int64_t j = 0; j < extents_on_axis; ++j) {
      for (int idx = 0; idx < dim; ++idx) {
        auto offset_dst = i * elems_from_axis + idx * extents_on_axis + j;
        auto offset_src = i * extents_on_axis + j;
        output_ptr[offset_dst] = (idx == max_val_indices[offset_src]) ? 1 : 0;
      }
    }
  }
}

odla_value odla_Hardmax(odla_value input, odla_int32 axis,
                        const odla_value_id id) {
  auto arg_max = odla_ArgMax(input, axis, 1 /* keep dims */, 0 /* no reverse */,
                             {ODLA_INT32, input->shape}, nullptr);
  const auto& shape = input->shape;
  auto type = input->mem.get_desc().data_type();
  assert(type == dnnl::memory::data_type::f32);
  axis = axis < 0 ? shape.size - 1 : axis;
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  const void* input_ptr = arg_max->mem.get_data_handle();
  void* output_ptr = ret_mem.get_data_handle();
  std::function<void()> op = [input_ptr, output_ptr, axis, shape]() {
    DoHardmax<float>(static_cast<const int32_t*>(input_ptr),
                     static_cast<float*>(output_ptr), axis, shape);
  };

  add_op(op);

  InterpretIfNeeded();
  return CreateValue(ret_mem, input->shape, id);
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

static void strided_slice(const void* src, int elem_size,
                          const odla_value_shape& input_dims,
                          const odla_uint32* start, const odla_uint32* end,
                          const odla_uint32* strides, void* dst,
                          const odla_value_shape& output_dims) {
  int64_t dst_elems = GetTotalElements(output_dims);
  int dims = input_dims.size;
  std::vector<int> dst_dims(dims);
  std::vector<size_t> src_strides(dims, 1);
  const char* src_ptr = reinterpret_cast<const char*>(src);
  char* dst_ptr = reinterpret_cast<char*>(dst);

  for (int k = dims - 2; k >= 0; --k) {
    src_strides[k] = src_strides[k + 1] * input_dims.dims[k + 1];
  }
  for (int64_t i = 0; i < dst_elems; ++i) {
    // map dst position to src position.
    std::vector<int> src_dims(dims);
    for (int k = 0; k < dims; ++k) {
      src_dims[k] = start[k] + strides[k] * dst_dims[k];
    }
    size_t src_offset =
        elem_size * std::inner_product(src_dims.begin(), src_dims.end(),
                                       src_strides.begin(), 0);
    memcpy(dst_ptr, src_ptr + src_offset, elem_size);

    // Increase the dims of dst;
    for (int k = dims - 1; k >= 0; --k) {
      ++dst_dims[k];
      if (dst_dims[k] == output_dims.dims[k]) {
        dst_dims[k] = 0;
      } else {
        break;
      }
    }
    dst_ptr += elem_size;
  }
}

odla_value odla_Slice(odla_value input, const odla_uint32* start,
                      const odla_uint32* end, const odla_uint32* strides,
                      odla_value_shape output_dims, const odla_value_id id) {
  const auto& input_dims = input->shape;
  int dims = input_dims.size;
  auto offsets = dnnl::memory::dims(start, start + dims);
  // only support stride of 1
  bool simple_stride = true;
  for (int i = 0; i < dims; ++i) {
    if (strides[i] != 1) {
      simple_stride = false;
      break;
    }
  }
  dnnl::memory::data_type type = input->mem.get_desc().data_type();
  dnnl::memory::desc dst_md = getMemoryDesc(output_dims, type);
  auto dst_mem = dnnl::memory(dst_md, g_comp->eng);
  if (simple_stride) {
    dnnl::memory::desc input_md = getMemoryDesc(input_dims, type);
    dnnl::memory::desc src_sub_md =
        input_md.submemory_desc(getDims(output_dims), offsets);

    // dummy reorder
    auto src_mem = dnnl::memory(src_sub_md, g_comp->eng, nullptr);
    auto prim = dnnl::reorder(src_mem, dst_mem);
    add_op(prim, {{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, dst_mem}});
  } else {
    int elem_size = getElementStorageSize(input->elem_type);
    auto op = [=]() {
      strided_slice(input->mem.get_data_handle(), elem_size, input_dims, start,
                    end, strides, dst_mem.get_data_handle(), output_dims);
    };
    add_op(op);
  }
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

odla_values odla_LSTM(odla_value input, odla_rnn_weight_format weight_format,
                      odla_rnn_gate_order gate_order,
                      odla_value_shape weight_dims, odla_value W, odla_value R,
                      odla_value B, odla_value sequence_lens,
                      odla_value initial_h, odla_value initial_c, odla_value P,
                      odla_int32 hidden_size, odla_rnn_direction direction,
                      odla_rnn_outputs outputs,
                      const odla_value_ids value_ids) {
  dnnl::rnn_direction dir;
  int d = 1;
  switch (direction) {
    case ODLA_RNN_FORWARD:
      dir = dnnl::rnn_direction::unidirectional_left2right;
      break;
    case ODLA_RNN_REVERSE:
      dir = dnnl::rnn_direction::unidirectional_right2left;
      break;
    default:
      d = 2;
      dir = dnnl::rnn_direction::bidirectional_concat;
  }

  assert(weight_format == ODLA_RNN_LDGOI);
  assert(gate_order == ODLA_RNN_IOFC);

  // DNNL assumes the following layout:
  // src : { T (Seq), N (Batch), SLC(Input)}  tag::tnc
  // src_iter: {L, D, N, SIC} tag::ldnc
  // src_iter_c: {L, D, N, DIC} tag:ldnc
  // W: {L, D, SLC, 4, DIC}  tag:ldigo
  // R(W_it): {L, D, SIC, 4, DIC}  tag:ldigo
  // dst : {T, N, DIC} tag:tnc
  // dst_it: {L, D, N, DIC} tag:ldnc
  // dst_it_c: {L, D, N, DIC} tag:ldnc
  //
  constexpr int64_t l = 1;
  auto t = input->shape.dims[0]; // sequence.
  const auto n = input->shape.dims[1];
  const auto slc = input->shape.dims[2];
  constexpr int num_gates = 4;
  assert(d == W->shape.dims[0]);
  assert(W->shape.dims[1] == num_gates * hidden_size);
  assert(W->shape.dims[2] == slc);
  assert(R->shape.dims[1] == num_gates * hidden_size);
  assert(R->shape.dims[2] == hidden_size);

  if (B != nullptr) {
    assert(B->shape.dims[0] == d);
    assert(B->shape.dims[1] == num_gates * hidden_size ||
           B->shape.dims[1] == 2 * num_gates * hidden_size);
  }

  auto dt = getDataType(input->elem_type);
  dnnl::memory::desc nil;

  dnnl::memory::desc src_md({t, n, slc}, dt, dnnl::memory::format_tag::tnc);

  // Either both are empty or both are valid.
  assert((initial_h == nullptr && initial_c == nullptr) ||
         (initial_c != nullptr && initial_h != nullptr));
  auto src_iter_desc = initial_h != nullptr
                           ? dnnl::memory::desc({l, d, n, hidden_size}, dt,
                                                dnnl::memory::format_tag::ldnc)
                           : nil;
  auto src_iter_c_desc =
      initial_c != nullptr ? dnnl::memory::desc({l, d, n, hidden_size}, dt,
                                                dnnl::memory::format_tag::ldnc)
                           : nil;

  dnnl::memory::desc w_desc({l, d, slc, num_gates, hidden_size}, dt,
                            dnnl::memory::format_tag::ldigo);
  dnnl::memory::desc w_it_desc({l, d, hidden_size, num_gates, hidden_size}, dt,
                               dnnl::memory::format_tag::ldigo);

  dnnl::memory::desc ret_md({t, n, d * hidden_size}, dt,
                            dnnl::memory::format_tag::tnc);
  dnnl::memory::desc ret_it_md({l, d, n, hidden_size}, dt,
                               dnnl::memory::format_tag::ldnc);

  odla_value_shape ret_shape{4, {t, d, n, hidden_size}};
  odla_value_shape ret_iter_shape{3, {d, n, hidden_size}};

  const auto& w_peephole_desc =
      P != nullptr ? dnnl::memory::desc({l, d, 3, hidden_size}, dt,
                                        dnnl::memory::format_tag::ldnc)
                   : nil;
  auto bias_desc = B != nullptr
                       ? dnnl::memory::desc({l, d, num_gates, hidden_size}, dt,
                                            dnnl::memory::format_tag::ldgo)
                       : nil;

  dnnl::lstm_forward::desc desc(dnnl::prop_kind::forward_inference, dir, src_md,
                                src_iter_desc, src_iter_c_desc, w_desc,
                                w_it_desc, w_peephole_desc, bias_desc, ret_md,
                                ret_it_md, ret_it_md);
  auto pd = dnnl::lstm_forward::primitive_desc(desc, g_comp->eng);
  auto prim = dnnl::lstm_forward(pd);

  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto ret_h_mem = dnnl::memory(ret_it_md, g_comp->eng);
  auto ret_c_mem = dnnl::memory(ret_it_md, g_comp->eng);

  dnnl::memory zero;

  auto w_mem = W->mem;
  auto r_mem = R->mem;
  auto b_mem = (B == nullptr) ? zero : B->mem;
  if (W->is_const && R->is_const && (B == nullptr || B->is_const)) {
    // W/R/B are in [iofc] but DNNL assumes [ifco] for weights and [ifo] for
    // Peephole. See https://oneapi-src.github.io/oneDNN/dev_guide_rnn.html
    // For W/R, we also need to transpose from ldgoi to ldigo.
    w_mem = dnnl::memory(w_desc, g_comp->eng);
    r_mem = dnnl::memory(w_it_desc, g_comp->eng);
    b_mem = dnnl::memory(bias_desc, g_comp->eng);
    auto shuffle = [num_gates](float* dst, const float* src, int dir,
                               int output_chs, int input_chs) {
      const int gate_mapping[4] = {0, 3, 1, 2}; // iofc -> ifco
      for (int d = 0; d < dir; ++d) {
        size_t offset_base = d * num_gates * output_chs * input_chs;
        for (int src_gate = 0; src_gate < num_gates; ++src_gate) {
          auto dst_gate = gate_mapping[src_gate];
          for (int o_ch = 0; o_ch < output_chs; ++o_ch) {
            for (int i_ch = 0; i_ch < input_chs; ++i_ch) {
              size_t dst_offset = offset_base + i_ch * num_gates * output_chs +
                                  dst_gate * output_chs + o_ch;
              dst[dst_offset] = *src;
              ++src;
            }
          }
        }
      }
    };
    shuffle(static_cast<float*>(w_mem.get_data_handle()),
            static_cast<const float*>(W->mem.get_data_handle()), d, hidden_size,
            slc);
    shuffle(static_cast<float*>(r_mem.get_data_handle()),
            static_cast<const float*>(R->mem.get_data_handle()), d, hidden_size,
            hidden_size);
    if (B != nullptr) {
      auto shuffle_bias = [num_gates](float* dst, const float* src, int dir,
                                      int chs) {
        const int gate_mapping[4] = {0, 3, 1, 2}; // iofc -> ifco
        for (int d = 0, s = num_gates * chs; d < dir; ++d) {
          for (int src_gate = 0; src_gate < num_gates; ++src_gate) {
            auto dst_gate = gate_mapping[src_gate];
            memcpy(&dst[d * s + dst_gate * chs], &src[d * s + src_gate * chs],
                   sizeof(dst[0]) * chs);
          }
        }
      };

      shuffle_bias(static_cast<float*>(b_mem.get_data_handle()),
                   static_cast<const float*>(B->mem.get_data_handle()), d,
                   hidden_size);
    }
  } else {
  }

  // Primitive arguments
  std::unordered_map<int, dnnl::memory> args;
  args[DNNL_ARG_SRC_LAYER] = input->mem;
  args[DNNL_ARG_SRC_ITER] = (initial_h != nullptr) ? initial_h->mem : zero;
  args[DNNL_ARG_SRC_ITER_C] = (initial_h != nullptr) ? initial_c->mem : zero;
  args[DNNL_ARG_WEIGHTS_LAYER] = w_mem;
  args[DNNL_ARG_WEIGHTS_ITER] = r_mem;
  args[DNNL_ARG_WEIGHTS_PEEPHOLE] = (P != nullptr) ? P->mem : zero;
  args[DNNL_ARG_BIAS] = b_mem;
  args[DNNL_ARG_DST_LAYER] = ret_mem;
  args[DNNL_ARG_DST_ITER] = ret_h_mem;
  args[DNNL_ARG_DST_ITER_C] = ret_c_mem;

  add_op(prim, args);
  InterpretIfNeeded();

  auto ret = CreateValue(ret_mem, ret_shape, value_ids.value_ids[0]);
  auto ret_h = CreateValue(ret_h_mem, ret_iter_shape, value_ids.value_ids[1]);
  auto ret_c = CreateValue(ret_c_mem, ret_iter_shape, value_ids.value_ids[2]);

  return {.size = 3, .values = {ret, ret_h, ret_c}};
}

odla_values odla_GRU(odla_value input, odla_rnn_weight_format weight_format,
                     odla_rnn_gate_order gate_order,
                     odla_value_shape weight_dims, odla_value W, odla_value R,
                     odla_value B, odla_value sequence_lens,
                     odla_value initial_h, odla_int32 hidden_size,
                     odla_rnn_direction direction,
                     odla_bool linear_before_reset, odla_rnn_outputs outputs,
                     const odla_value_ids value_ids) {
  dnnl::rnn_direction dir;
  int d = 1;
  switch (direction) {
    case ODLA_RNN_FORWARD:
      dir = dnnl::rnn_direction::unidirectional_left2right;
      break;
    case ODLA_RNN_REVERSE:
      dir = dnnl::rnn_direction::unidirectional_right2left;
      break;
    default:
      d = 2;
      dir = dnnl::rnn_direction::bidirectional_concat;
  }

  assert(gate_order == ODLA_RNN_URO);

  // DNNL assumes the following layout:
  // src : { T (Seq), N (Batch), SLC(Input)}  tag::tnc
  // src_iter: {L, D, N, SIC} tag::ldnc
  // src_iter_c: {L, D, N, DIC} tag:ldnc
  // W: {L, D, SLC, 4, DIC}  tag:ldigo
  // R(W_it): {L, D, SIC, 4, DIC}  tag:ldigo
  // dst : {T, N, DIC} tag:tnc
  // dst_it: {L, D, N, DIC} tag:ldnc
  // dst_it_c: {L, D, N, DIC} tag:ldnc
  //
  constexpr int64_t l = 1;
  auto t = input->shape.dims[0]; // sequence.
  const auto n = input->shape.dims[1];
  const auto slc = input->shape.dims[2];
  constexpr int num_gates = 3;
  assert(d == W->shape.dims[0]);
  assert(W->shape.dims[1] == num_gates * hidden_size);
  assert(W->shape.dims[2] == slc);
  assert(R->shape.dims[1] == num_gates * hidden_size);
  assert(R->shape.dims[2] == hidden_size);

  if (B != nullptr) {
    assert(B->shape.dims[0] == d);
    assert(B->shape.dims[1] == num_gates * hidden_size ||
           B->shape.dims[1] == 2 * num_gates * hidden_size);
  }

  auto dt = getDataType(input->elem_type);
  dnnl::memory::desc nil;

  dnnl::memory::desc src_md({t, n, slc}, dt, dnnl::memory::format_tag::tnc);

  auto src_iter_desc = initial_h != nullptr
                           ? dnnl::memory::desc({l, d, n, hidden_size}, dt,
                                                dnnl::memory::format_tag::ldnc)
                           : nil;
  dnnl::memory::desc w_desc({l, d, slc, num_gates, hidden_size}, dt,
                            dnnl::memory::format_tag::ldigo);
  dnnl::memory::desc w_it_desc({l, d, hidden_size, num_gates, hidden_size}, dt,
                               dnnl::memory::format_tag::ldigo);

  dnnl::memory::desc ret_md({t, n, d * hidden_size}, dt,
                            dnnl::memory::format_tag::tnc);
  dnnl::memory::desc ret_it_md({l, d, n, hidden_size}, dt,
                               dnnl::memory::format_tag::ldnc);

  odla_value_shape ret_shape{4, {t, d, n, hidden_size}};
  odla_value_shape ret_iter_shape{3, {d, n, hidden_size}};

  auto bias_desc = B != nullptr
                       ? dnnl::memory::desc({l, d, num_gates, hidden_size}, dt,
                                            dnnl::memory::format_tag::ldgo)
                       : nil;
  dnnl::primitive prim;
  if (linear_before_reset != 0) {
    dnnl::lbr_gru_forward::desc desc(dnnl::prop_kind::forward_inference, dir,
                                     src_md, src_iter_desc, w_desc, w_it_desc,
                                     bias_desc, ret_md, ret_it_md);
    auto pd = dnnl::lbr_gru_forward::primitive_desc(desc, g_comp->eng);
    prim = dnnl::lbr_gru_forward(pd);
  } else {
    dnnl::gru_forward::desc desc(dnnl::prop_kind::forward_inference, dir,
                                 src_md, src_iter_desc, w_desc, w_it_desc,
                                 bias_desc, ret_md, ret_it_md);
    auto pd = dnnl::gru_forward::primitive_desc(desc, g_comp->eng);
    prim = dnnl::gru_forward(pd);
  }

  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto ret_h_mem = dnnl::memory(ret_it_md, g_comp->eng);

  dnnl::memory zero;

  auto w_mem = W->mem;
  auto r_mem = R->mem;
  auto b_mem = (B == nullptr) ? zero : B->mem;
  auto initial_h_mem = (initial_h == nullptr) ? zero : initial_h->mem;

  if (weight_format == ODLA_RNN_LDGOI) {
    if (W->is_const && R->is_const) {
      // W/R/B are in [URO].
      // For W/R, we also need to transpose from ldgoi to ldigo.
      w_mem = dnnl::memory(w_desc, g_comp->eng);
      r_mem = dnnl::memory(w_it_desc, g_comp->eng);
      b_mem = dnnl::memory(bias_desc, g_comp->eng);
      auto reorder = [num_gates](float* dst, const float* src, int dir,
                                 int output_chs, int input_chs) {
        for (int d = 0; d < dir; ++d) {
          size_t offset_base = d * num_gates * output_chs * input_chs;
          for (int gate = 0; gate < num_gates; ++gate) {
            for (int o_ch = 0; o_ch < output_chs; ++o_ch) {
              for (int i_ch = 0; i_ch < input_chs; ++i_ch) {
                size_t dst_offset = offset_base +
                                    i_ch * num_gates * output_chs +
                                    gate * output_chs + o_ch;
                dst[dst_offset] = *src;
                ++src;
              }
            }
          }
        }
      };
      reorder(static_cast<float*>(w_mem.get_data_handle()),
              static_cast<const float*>(W->mem.get_data_handle()), d,
              hidden_size, slc);
      reorder(static_cast<float*>(r_mem.get_data_handle()),
              static_cast<const float*>(R->mem.get_data_handle()), d,
              hidden_size, hidden_size);
    } else { // transpose from ldgoi to ldigo.
      odla_value_shape perm{5, {0, 1, 4, 2, 3}};
      auto wt = odla_Transpose(
          odla_Reshape(W,
                       odla_value_shape{5, {l, d, num_gates, hidden_size, slc}},
                       nullptr),
          perm, odla_value_shape{5, {l, d, slc, num_gates, hidden_size}},
          nullptr);
      auto rt = odla_Transpose(
          odla_Reshape(
              R,
              odla_value_shape{5, {l, d, hidden_size, num_gates, hidden_size}},
              nullptr),
          perm,
          odla_value_shape{5, {l, d, hidden_size, num_gates, hidden_size}},
          nullptr);
      w_mem = wt->mem;
      r_mem = rt->mem;
    }
  }

  // Primitive arguments
  std::unordered_map<int, dnnl::memory> args;
  args[DNNL_ARG_SRC_LAYER] = input->mem;
  args[DNNL_ARG_SRC_ITER] = initial_h_mem;
  args[DNNL_ARG_WEIGHTS_LAYER] = w_mem;
  args[DNNL_ARG_WEIGHTS_ITER] = r_mem;
  args[DNNL_ARG_BIAS] = b_mem;
  args[DNNL_ARG_DST_LAYER] = ret_mem;
  args[DNNL_ARG_DST_ITER] = ret_h_mem;
  add_op(prim, args);
  InterpretIfNeeded();

  auto ret = CreateValue(ret_mem, ret_shape, value_ids.value_ids[0]);
  auto ret_h = CreateValue(ret_h_mem, ret_iter_shape, value_ids.value_ids[1]);

  return {.size = 3, .values = {ret, ret_h}};
}
