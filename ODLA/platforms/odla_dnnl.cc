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
#include <immintrin.h>

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

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

struct _odla_value {
  dnnl::memory mem;
  bool is_const;
  odla_value_shape shape;
  std::string name;
  _odla_value(const dnnl::memory& m, const odla_value_shape& shape,
              const std::string& id)
      : mem(m), is_const(false), shape(shape), name(id) {}
};

typedef struct TargetOpts {
  bool enable_bf16;
} target_opts;

struct _odla_computation {
  dnnl::engine eng;
  std::vector<dnnl::primitive> primitives;
  std::vector<std::unordered_map<int, dnnl::memory>> args;
  std::vector<std::unique_ptr<_odla_value>> vals;
  std::unordered_map<std::string, odla_value> inputs;
  std::unordered_map<std::string, odla_value> outputs;
  target_opts opts;

  _odla_computation() : eng(dnnl::engine::kind::cpu, 0), opts({false}) {}
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
extern "C" {

void odla_ConfigTargetOptions(odla_computation comp, target_opts opts) {
  comp->opts.enable_bf16 = opts.enable_bf16;
}

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;
thread_local bool g_interpret_mode = false;

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
  for (size_t i = 0, e = comp->primitives.size(); i < e; ++i) {
    comp->primitives[i].execute(*context->stream, comp->args[i]);
  }
  context->stream->wait();
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
  for (size_t i = 0, e = g_comp->primitives.size(); i < e; ++i) {
    g_comp->primitives[i].execute(*context->stream, g_comp->args[i]);
  }
  context->stream->wait();
  g_comp->primitives.clear();
  g_comp->args.clear();
#endif
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  const char* name = (const char*)id;
  dnnl::memory::desc md = getMemoryDesc(type.shape, ODLA_FLOAT32);
  dnnl::memory mem = dnnl::memory(md, g_comp->eng);
  odla_value v = CreateValue(mem, type.shape, id);
  g_comp->inputs[name] = v;
  return v;
}

odla_value odla_CreateValue(odla_value_type type, const odla_value_id id) {
  assert(g_interpret_mode);
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
  value->mem.set_data_handle(const_cast<void*>(data_ptr));
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
  dnnl::memory::desc md = getMemoryDesc(type);

  dnnl::memory mem = dnnl::memory(md, g_comp->eng, const_cast<void*>(ptr));
  odla_value v = CreateValue(mem, type.shape, id);
  v->is_const = true;
  return v;
}

odla_status odla_SetValueAsOutput(const odla_value val) {
  g_comp->outputs[val->name] = val;
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  // Handle the case of output is constant due to compile-time optimization.
  if (value->is_const) {
    memcpy(data_ptr, value->mem.get_data_handle(),
           value->mem.get_desc().get_size());
  } else {
    value->mem.set_data_handle(data_ptr);
  }
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  std::string name((const char*)value_id);
  auto val = context->comp->outputs[name];
  return odla_BindToOutput(val, data_ptr, context);
}

static int calculat_offset(int len, int vec_size) {
  /*
  calculate the offset when using intrinsics.
  example:
    when len is 108 vec_size is 32 when using bf16
    the result is 108 % 32 = 12
    so we need to set the mask to 0b00000000000000000000111111111111
  */
  int offset = len;
  int expo = 0;
  int dst = 0;
  while (offset - vec_size > 0) {
    offset -= vec_size;
  }
  while (offset > 0) {
    dst += pow(2, expo);
    offset -= 1;
    expo += 1;
  }
  return dst;
}

#if defined(__AVX512F__)
inline __m512 _mm512_cvtbf16f32_load(__mmask16 mask, void* mem_addr) {
  auto dst = _mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask, mem_addr)), 0x10);
  return _mm512_castsi512_ps(dst);
}
#endif

#if defined(__GNUC__) && (__GNUC__ > 9)
inline void floorbf_func(int len, float* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 16;
  __mmask16 mask16 = 0xFFFF;
  auto alpha_vec = _mm512_set1_ps(0.0);
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto a1 = _mm512_cvtbf16f32_load(mask16, src + i + 16);

    auto out0 = _mm512_floor_ps(a0);
    auto out1 = _mm512_floor_ps(a1);

    auto C_bf16 = _mm512_cvtne2ps_pbh(out1, out0);
    _mm512_mask_storeu_ps(dst + i, mask16, _mm512_castsi512_ps(C_bf16));
  }
  if ((len - i) > 16) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = _mm512_floor_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(C_bf16));
    i += vec_size;
  }
  if (len - i) {
    __mmask16 tail_mask = calculat_offset(i, vec_size);
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = _mm512_floor_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
}
#elif defined(__GNUC__) && (__GNUC__ > 8)
inline void floorbf_func(int len, float* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;
  auto alpha_vec = _mm512_set1_ps(0.0);
  auto tail_mask = calculat_offset(len, vec_size);
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = _mm512_floor_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(C_bf16));
  }
  if (len - i) {
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = _mm512_floor_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
}
#else
inline void floorbf_func(int len, float* src, float* dst) {}
#endif

#if defined(__AVX512F__)
inline void floorf_func(int len, float* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;

  for (; i <= len - vec_size; i += vec_size) {
    auto a1 = _mm512_loadu_ps(src + i);
    auto out1 = _mm512_floor_ps(a1);
    _mm512_mask_storeu_ps(dst + i, mask16, out1);
  }
  if (len - i) {
    auto tail_mask = calculat_offset(len - i, vec_size);
    auto a1 = _mm512_maskz_loadu_ps(tail_mask, src + i);
    auto out1 = _mm512_floor_ps(a1);
    _mm512_mask_storeu_ps(dst + i, tail_mask, out1);
  }
}

odla_value odla_Floor(odla_value input, const odla_value_id id) {
  int64_t total_elems = GetTotalElements(input->shape);
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  if (g_comp->opts.enable_bf16) {
    floorbf_func(total_elems, (float*)input->mem.get_data_handle(),
                 (float*)ret_mem.get_data_handle());
  } else {
    floorf_func(total_elems, (float*)input->mem.get_data_handle(),
                (float*)ret_mem.get_data_handle());
  }
  InterpretIfNeeded();
  return CreateValue(ret_mem, input->shape, id);
}
#endif

#if defined(__AVX512F__)
inline void rsqrtf_func(int len, float* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;

  for (; i <= len - vec_size; i += vec_size) {
    auto a1 = _mm512_loadu_ps(src + i);
    auto out1 = _mm512_rsqrt14_ps(a1);
    _mm512_mask_storeu_ps(dst + i, mask16, out1);
  }
  if (len - i) {
    auto tail_mask = calculat_offset(len - i, vec_size);
    auto a1 = _mm512_maskz_loadu_ps(tail_mask, src + i);
    auto out1 = _mm512_rsqrt14_ps(a1);
    _mm512_mask_storeu_ps(dst + i, tail_mask, out1);
  }
}
#endif

#if defined(__GNUC__) && (__GNUC__ > 9)
inline void rsqrtbf_func(int len, float* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 16;
  __mmask16 mask16 = 0xFFFF;
  auto alpha_vec = _mm512_set1_ps(0.0);
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto a1 = _mm512_cvtbf16f32_load(mask16, src + i + 16);

    auto out0 = _mm512_rsqrt14_ps(a0);
    auto out1 = _mm512_rsqrt14_ps(a1);

    auto C_bf16 = _mm512_cvtne2ps_pbh(out1, out0);
    _mm512_mask_storeu_ps(dst + i, mask16, _mm512_castsi512_ps(C_bf16));
  }
  if ((len - i) > 16) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = _mm512_rsqrt14_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(C_bf16));
    i += vec_size;
  }
  if (len - i) {
    auto tail_mask = calculat_offset(len - i, vec_size);
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = _mm512_rsqrt14_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
}
#elif defined(__GNUC__) && (__GNUC__ > 8)
inline void rsqrtbf_func(int len, float* src, float* dst) {
  int i = 0;
  int vec_size = 512 / 32;
  __mmask16 mask16 = 0xFFFF;
  auto alpha_vec = _mm512_set1_ps(0.0);
  auto tail_mask = calculat_offset(len, vec_size);
  for (; i <= len - vec_size; i += vec_size) {
    auto a0 = _mm512_cvtbf16f32_load(mask16, src + i);
    auto out0 = _mm512_rsqrt14_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(C_bf16));
  }
  if (len - i) {
    auto a0 = _mm512_cvtbf16f32_load(tail_mask, src + i);
    auto out0 = _mm512_rsqrt14_ps(a0);
    auto C_bf16 = _mm512_cvtneps_pbh(out0);
    _mm256_mask_storeu_ps(dst + i, tail_mask, _mm256_castsi256_ps(C_bf16));
  }
}
#else
inline void rsqrtbf_func(int len, float* src, float* dst) {}
#endif

#if defined(__AVX512F__)
odla_value odla_Rsqrt(odla_value input, const odla_value_id id) {
  int64_t total_elems = GetTotalElements(input->shape);
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  if (g_comp->opts.enable_bf16) {
    rsqrtbf_func(total_elems, (float*)input->mem.get_data_handle(),
                 (float*)ret_mem.get_data_handle());
  } else {
    rsqrtf_func(total_elems, (float*)input->mem.get_data_handle(),
                (float*)ret_mem.get_data_handle());
  }
  InterpretIfNeeded();
  return CreateValue(ret_mem, input->shape, id);
}
#endif

odla_value odla_Gather(odla_value params, const odla_value indices,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  axis = axis < 0 ? params->shape.size + axis : axis;
  int64_t batch_size;
  int64_t idx_size;
  if (indices->shape.size > 1) {
    batch_size = indices->shape.dims[0];
    idx_size = indices->shape.dims[1];
  } else {
    batch_size = 1;
    idx_size = indices->shape.dims[0];
  }
  int64_t inner_size = 1;
  for (int i = axis + 1; i < params->shape.size; ++i) {
    inner_size *= params->shape.dims[i];
  }
  auto buffer_size = g_comp->opts.enable_bf16 ? sizeof(int16_t) : sizeof(float);
  auto ret_md = dnnl::memory::desc(getDims(output_dims),
                                   params->mem.get_desc().data_type(),
                                   getStrides(output_dims));
  auto output_buffer = malloc(GetTotalElements(output_dims) * buffer_size);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng, output_buffer);

  if (g_comp->opts.enable_bf16) {
    int16_t* output_ptr = (int16_t*)ret_mem.get_data_handle();
    int16_t* params_base = (int16_t*)params->mem.get_data_handle();
    int32_t* idx_base = (int32_t*)indices->mem.get_data_handle();
    size_t bytes = ret_md.get_size();
    const size_t slice_bytes = inner_size * buffer_size;
    // copy the memory:
#pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < idx_size; j++) {
        int32_t curr_idx = idx_base[i * batch_size + j];
        memcpy(output_ptr + (i * idx_size + j) * inner_size,
               params_base + (curr_idx)*inner_size, slice_bytes);
      }
    }
  } else {
    int32_t* output_ptr = (int32_t*)ret_mem.get_data_handle();
    int32_t* params_base = (int32_t*)params->mem.get_data_handle();
    int32_t* idx_base = (int32_t*)indices->mem.get_data_handle();
    size_t bytes = ret_md.get_size();
    const size_t slice_bytes = inner_size * buffer_size;
    // copy the memory:
#pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < idx_size; j++) {
        int32_t curr_idx = idx_base[i * batch_size + j];
        memcpy(output_ptr + (i * idx_size + j) * inner_size,
               params_base + (curr_idx)*inner_size, slice_bytes);
      }
    }
  }
#if 0 // below code is aligned with tensorflow

  for (int i = 0; i < axis; ++i) {
    result_shape.push_back(params->shape.dims[i]);
    outer_size *= params->shape.dims[i];
  }
  for (int i = 0; i < indices->shape.size; ++i) {
    result_shape.push_back(indices->shape.dims[i]);
  }
  for (int i = axis + 1; i < params->shape.size; ++i) {
    result_shape.push_back(params->shape.dims[i]);
    inner_size *= params->shape.dims[i];
  }
  int64_t gather_dim_size = params->shape.dims(axis);

#endif

  InterpretIfNeeded();
  return CreateValue(ret_mem, output_dims, id);
}

static odla_value binary_eltwise(dnnl::algorithm algo, odla_value lhs,
                                 odla_value rhs, const odla_value_id id) {
  const auto& dims_lhs = lhs->shape;
  const auto& dims_rhs = rhs->shape;
  auto lhs_md =
      dnnl::memory::desc(getDims(dims_lhs), lhs->mem.get_desc().data_type(),
                         getFormatTag(dims_lhs));
  auto ret_md = lhs_md;
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  auto rhs_md = rhs->mem.get_desc();

  auto ln = GetTotalElements(dims_lhs);
  auto rn = GetTotalElements(dims_rhs);

  if (ln != rn) {
    std::vector<int64_t> strides_v(dims_lhs.size, 0);
    assert(ln >= rn && ln % rn == 0);
    if (dims_rhs.size == 1) {
      for (int i = dims_lhs.size - 1; i >= 0; --i) {
        if (dims_lhs.dims[i] == rn) {
          strides_v[i] = 1;
          break;
        }
      }
    } else {
      for (int i = dims_lhs.size - 1, j = dims_rhs.size - 1, s = 1;
           i >= 0 && j >= 0; --i, --j) {
        if (dims_lhs.dims[i] != dims_rhs.dims[j]) {
          assert(dims_rhs.dims[j] == 1);
        } else {
          strides_v[i] = s;
          s *= dims_rhs.dims[j];
        }
      }
    }
    for (int i = 0; i < dims_lhs.size; ++i)
      rhs_md.data.format_desc.blocking.strides[i] = strides_v[i];
  }
  dnnl::binary::desc bd(algo, lhs_md, rhs_md, ret_md);
  dnnl::binary::primitive_desc pd(bd, g_comp->eng);
  dnnl::primitive prim = dnnl::binary(pd);

  g_comp->primitives.push_back(prim);

  odla_value v = CreateValue(ret_mem, lhs->shape, id);
  g_comp->args.push_back({{DNNL_ARG_SRC_0, lhs->mem},
                          {DNNL_ARG_SRC_1, rhs->mem},
                          {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();
  return v;
}

odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_eltwise(dnnl::algorithm::binary_add, lhs, rhs, id);
}

odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_eltwise(dnnl::algorithm::binary_mul, lhs, rhs, id);
}

odla_value odla_Round(odla_value input, const odla_value_id id) {
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                          dnnl::algorithm::eltwise_round,
                                          input->mem.get_desc());
  auto pd = dnnl::eltwise_forward::primitive_desc(desc, g_comp->eng);
  auto prim = dnnl::eltwise_forward(pd);

  g_comp->primitives.push_back(prim);

  odla_value v = CreateValue(ret_mem, input->shape, id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();

  return v;
}

odla_value odla_Sigmoid(odla_value input, const odla_value_id id) {
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                          dnnl::algorithm::eltwise_logistic,
                                          input->mem.get_desc());
  auto pd = dnnl::eltwise_forward::primitive_desc(desc, g_comp->eng);
  auto prim = dnnl::eltwise_forward(pd);

  g_comp->primitives.push_back(prim);

  odla_value v = CreateValue(ret_mem, input->shape, id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();

  return v;
}

odla_value odla_LeakyRelu(odla_value input, odla_float32 alpha,
                          const odla_value_id id) {
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  // MKL uses leaky relu: f(x) = x >= 0 ? x : x * negative_slope
  float negative_slope = alpha;
  auto relu_desc = dnnl::eltwise_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_relu,
      input->mem.get_desc(), negative_slope);
  auto pd = dnnl::eltwise_forward::primitive_desc(relu_desc, g_comp->eng);
  auto prim = dnnl::eltwise_forward(pd);

  g_comp->primitives.push_back(prim);

  odla_value v = CreateValue(ret_mem, input->shape, id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();

  return v;
}

odla_value odla_Relu(odla_value input, const odla_value_id value_id) {
  return odla_LeakyRelu(input, 0, value_id);
}

odla_value odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
                      const odla_value_id id) {
  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  float negative_slope = -0.0;
  auto relu_desc = dnnl::eltwise_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_clip,
      input->mem.get_desc(), lo, hi);
  auto pd = dnnl::eltwise_forward::primitive_desc(relu_desc, g_comp->eng);
  auto prim = dnnl::eltwise_forward(pd);

  g_comp->primitives.push_back(prim);

  odla_value v = CreateValue(ret_mem, input->shape, id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();

  return v;
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
                                     unsigned groups,
                                     odla_memory_layout layout) {
  assert(src_dims.size == 4);
  assert(layout == ODLA_OIS);
  return {src_dims.size + 1,
          {groups, src_dims.dims[0] / groups, src_dims.dims[1],
           src_dims.dims[2], src_dims.dims[3]}};
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

  g_comp->primitives.push_back(prim);
  g_comp->args.push_back({{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, dst_mem}});
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
  dnnl::memory::dims stride_dims{strides[0], strides[1]};
  dnnl::memory::dims paddings_before{paddings_front[0], paddings_front[1]};
  dnnl::memory::dims paddings_after{paddings_back[0], paddings_back[1]};
  auto dt_dst = g_comp->opts.enable_bf16 ? getDataType(ODLA_BFLOAT16) : dt;

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
    kernel_dims = getGOIHWDims(kernel_dims, group, ODLA_OIS);
  }

  dnnl::memory::desc ret_md;
  if (g_comp->opts.enable_bf16) {
    ret_md = dnnl::memory::desc(getDims(output_dims), dt_dst,
                                dnnl::memory::format_tag::any);
  } else {
    ret_md = dnnl::memory::desc(getDims(output_dims), dt,
                                getFormatTag(input_layout));
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
  dnnl::memory::desc bias_md;
  if (bias != nullptr) {
    odla_value_shape scalar{.size = 1, .dims = {GetTotalElements(bias->shape)}};
    bias_md =
        dnnl::memory::desc(getDims(scalar), dt, dnnl::memory::format_tag::a);
  }
  assert(dilations[0] == 1 && dilations[1] == 1);
  auto conv_desc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward, dnnl::algorithm::convolution_direct,
      input_md_any, kernel_md_any, bias_md, ret_md_any, stride_dims,
      paddings_before, paddings_after);
  auto pd = dnnl::convolution_forward::primitive_desc(conv_desc, g_comp->eng);

  auto ret_mem = dnnl::memory(pd.dst_desc(), g_comp->eng);

  if (pd.weights_desc() != kernel_md_src) {
    auto reordered_w = dnnl::memory(pd.weights_desc(), g_comp->eng);
    auto rec = dnnl::reorder(
        dnnl::memory(kernel_md_src, g_comp->eng, kernel->mem.get_data_handle()),
        reordered_w);
    dnnl::stream s(g_comp->eng);
    rec.execute(s, {{DNNL_ARG_FROM, kernel->mem}, {DNNL_ARG_TO, reordered_w}});
    s.wait();
    kernel->mem = reordered_w;
  }

  dnnl::memory orig_mem;
  bool needs_reorder_input = pd.src_desc() != input_md_src;
  if (needs_reorder_input) {
    orig_mem = input->mem;
    auto reordered_mem = dnnl::memory(pd.src_desc(), g_comp->eng);
    auto r = dnnl::reorder(
        dnnl::memory(input_md_src, g_comp->eng, input->mem.get_data_handle()),
        reordered_mem);
    g_comp->primitives.push_back(r);
    g_comp->args.push_back(
        {{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, reordered_mem}});

    input->mem = reordered_mem;
  }

  auto prim = dnnl::convolution_forward(pd);
  g_comp->primitives.push_back(prim);
  odla_value v = CreateValue(ret_mem, orig_output_dims, id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem},
                          {DNNL_ARG_WEIGHTS, kernel->mem},
                          {DNNL_ARG_DST, ret_mem}});
  if (bias != nullptr) {
    g_comp->args.back()[DNNL_ARG_BIAS] = bias->mem;
  }
  if (needs_reorder_input) {
    input->mem = orig_mem;
  }
  auto ret_md_exp =
      dnnl::memory::desc(getDims(output_dims), dt, getFormatTag(input_layout));
  if (pd.dst_desc() != ret_md_exp) {
    auto reordered_mem = dnnl::memory(ret_md_exp, g_comp->eng);
    auto r = dnnl::reorder(ret_mem, reordered_mem);
    g_comp->primitives.push_back(r);
    g_comp->args.push_back(
        {{DNNL_ARG_FROM, ret_mem}, {DNNL_ARG_TO, reordered_mem}});
    v->mem = reordered_mem;
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
  auto dt_dst = g_comp->opts.enable_bf16 ? getDataType(ODLA_BFLOAT16) : dt;

  auto orig_output_dims = output_dims;
  if (input_layout == ODLA_CHANNELS_LAST) {
    input_dims = getNCHWDims(input_dims);
    output_dims = getNCHWDims(output_dims);
  }

  if (kernel_layout == ODLA_SIO) {
    kernel_dims = getOIHWDims(kernel_dims);
  } else if (kernel_layout == odla_memory_layout::ODLA_IOS) {
    std::swap(kernel_dims.dims[0], kernel_dims.dims[1]);
  }

  if (group > 1) {
    kernel_dims = getGOIHWDims(kernel_dims, group, ODLA_OIS);
  }
  dnnl::memory::desc ret_md;
  if (g_comp->opts.enable_bf16) {
    ret_md = dnnl::memory::desc(getDims(output_dims), dt_dst,
                                dnnl::memory::format_tag::any);
  } else {
    ret_md = dnnl::memory::desc(getDims(output_dims), dt,
                                getFormatTag(input_layout));
  }
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

  assert(dilations[0] == 1 && dilations[1] == 1);
  auto conv_desc = dnnl::deconvolution_forward::desc(
      dnnl::prop_kind::forward, dnnl::algorithm::deconvolution_direct,
      input_md_any, kernel_md_any, ret_md_any, stride_dims, paddings_before,
      paddings_after);
  auto pd = dnnl::deconvolution_forward::primitive_desc(conv_desc, g_comp->eng);

  auto ret_mem = dnnl::memory(pd.dst_desc(), g_comp->eng);
  bool needs_reorder_input = pd.src_desc() != input_md_src;
  if (pd.weights_desc() != kernel_md_src) {
    auto reordered_w = dnnl::memory(pd.weights_desc(), g_comp->eng);
    auto rec = dnnl::reorder(
        dnnl::memory(kernel_md_src, g_comp->eng, kernel->mem.get_data_handle()),
        reordered_w);
    dnnl::stream s(g_comp->eng);
    rec.execute(s, {{DNNL_ARG_FROM, kernel->mem}, {DNNL_ARG_TO, reordered_w}});
    s.wait();
    kernel->mem = reordered_w;
  }

  dnnl::memory orig_mem;
  if (needs_reorder_input) {
    orig_mem = input->mem;
    auto reordered_mem = dnnl::memory(pd.src_desc(), g_comp->eng);
    auto r = dnnl::reorder(
        dnnl::memory(input_md_src, g_comp->eng, input->mem.get_data_handle()),
        reordered_mem);
    g_comp->primitives.push_back(r);
    g_comp->args.push_back(
        {{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, reordered_mem}});

    input->mem = reordered_mem;
  }
  auto prim = dnnl::deconvolution_forward(pd);

  g_comp->primitives.push_back(prim);

  odla_value v =
      CreateValue(ret_mem, orig_output_dims, bias != nullptr ? nullptr : id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem},
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
    g_comp->primitives.push_back(r);
    g_comp->args.push_back(
        {{DNNL_ARG_FROM, ret_mem}, {DNNL_ARG_TO, reordered_mem}});

    v->mem = reordered_mem;
  }
  InterpretIfNeeded();
  return bias ? odla_Add(v, bias, id) : v;
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
  g_comp->primitives.push_back(prim);
  odla_value v = CreateValue(ret_mem, output_dims, id);
  std::unordered_map<int, dnnl::memory> concat_args;
  for (int i = 0; i < num; ++i) {
    concat_args.emplace(DNNL_ARG_MULTIPLE_SRC + i, src_mems[i]);
  }
  concat_args.emplace(DNNL_ARG_DST, ret_mem);
  g_comp->args.push_back(concat_args);
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

  g_comp->primitives.push_back(prim);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
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
  return BasePool(input, input_layout, window_dims, strides, paddings_front,
                  paddings_back, output_dims, value_id,
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

    size_t bytes = weight_md.get_size() / 2;
    assert(bytes == scale->mem.get_desc().get_size() &&
           bytes == offset->mem.get_desc().get_size());
    char* weight_data = static_cast<char*>(weight_mem.get_data_handle());
    std::memcpy(weight_data, scale->mem.get_data_handle(), bytes);
    std::memcpy(weight_data + bytes, offset->mem.get_data_handle(), bytes);
  }
  auto op_desc = dnnl::batch_normalization_forward::desc(
      dnnl::prop_kind::forward, input_md, epsilon, flags);
  auto pd =
      dnnl::batch_normalization_forward::primitive_desc(op_desc, g_comp->eng);
  auto prim = dnnl::batch_normalization_forward(pd);
  auto ret_mem = dnnl::memory(input_md, g_comp->eng);

  g_comp->primitives.push_back(prim);
  odla_value v = CreateValue(ret_mem, orig_dims, value_id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem},
                          {DNNL_ARG_MEAN, mean->mem},
                          {DNNL_ARG_VARIANCE, var->mem},
                          {DNNL_ARG_SCALE_SHIFT, weight_mem},
                          {DNNL_ARG_DST, ret_mem}});
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

  auto op_desc = dnnl::resampling_forward::desc(dnnl::prop_kind::forward, algo,
                                                input_md, ret_md);
  auto pd = dnnl::resampling_forward::primitive_desc(op_desc, g_comp->eng);
  auto prim = dnnl::resampling_forward(pd);

  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  g_comp->primitives.push_back(prim);
  odla_value v = CreateValue(ret_mem, output_dims, value_id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});

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

  g_comp->primitives.push_back(prim);
  odla_value v = CreateValue(ret_mem, orig_dims, value_id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});

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

  auto sm_desc =
      dnnl::softmax_forward::desc(dnnl::prop_kind::forward, input_md, axis);

  auto pd = dnnl::softmax_forward::primitive_desc(sm_desc, g_comp->eng);
  auto prim = dnnl::softmax_forward(pd);

  g_comp->primitives.push_back(prim);

  odla_value v = CreateValue(ret_mem, input->shape, id);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();

  return v;
}

odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  const auto& dims = input->shape;
  assert(num_of_axes == 2 &&
         dims.size == 4); // TODO: handle more generic cases.
  // axes must be contiguous.
  for (int i = 1; i < num_of_axes; ++i) {
    assert(axes[i] == axes[i - 1] + 1);
  }
  // Use avg pooling:
  // batch: all dims before the first reduction axis
  // channel: all dims after the last reduction axis.
  // spatial: all reduction axes.
  odla_int64 batch = 1;
  for (int i = 0; i < axes[0]; ++i) batch *= dims.dims[i];
  odla_int64 channels = 1;
  for (int i = axes[num_of_axes - 1] + 1; i < dims.size; ++i)
    channels *= dims.dims[i];
  odla_int64 hw = 1;
  for (int i = 0; i < num_of_axes; ++i) hw *= dims.dims[axes[i]];
  dnnl::memory::dims stride_dims{1, static_cast<unsigned>(hw)};
  dnnl::memory::dims paddings{0, 0};

  auto dt = input->mem.get_desc().data_type();

  odla_value_shape input_dims =
      odla_value_shape{.size = 4, .dims = {batch, channels, 1, hw}};
  auto orig_output_dims = output_dims;
  output_dims = odla_value_shape{.size = 4, .dims = {batch, channels, 1, 1}};

  auto ret_md = dnnl::memory::desc(getDims(output_dims), dt,
                                   dnnl::memory::format_tag::nhwc);
  auto input_md = dnnl::memory::desc(getDims(input_dims), dt,
                                     dnnl::memory::format_tag::nhwc);

  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);

  auto pool_desc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_avg,
      input_md, ret_md, stride_dims, stride_dims, paddings, paddings);
  auto pd = dnnl::pooling_forward::primitive_desc(pool_desc, g_comp->eng);
  auto prim = dnnl::pooling_forward(pd);

  g_comp->primitives.push_back(prim);
  g_comp->args.push_back({{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();

  return CreateValue(ret_mem, orig_output_dims, id);
}

odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  const auto& lhs_dims = lhs->shape;
  const auto& rhs_dims = rhs->shape;
  auto dt = lhs->mem.get_desc().data_type();
  assert(lhs_dims.size == 2 && rhs_dims.size == 2);
  long M = output_dims.dims[0], N = output_dims.dims[1],
       K = !transpose_rhs ? rhs_dims.dims[0] : rhs_dims.dims[1];
  long lda = transpose_lhs ? M : K;
  long ldb = transpose_rhs ? K : N;
  long ldc = N;
  dnnl::memory::desc lhs_md(
      {M, K}, dt,
      transpose_lhs ? dnnl::memory::dims{1, lda} : dnnl::memory::dims{lda, 1});

  dnnl::memory::desc rhs_md(
      {K, N}, dt,
      transpose_rhs ? dnnl::memory::dims{1, ldb} : dnnl::memory::dims{ldb, 1});

  dnnl::memory::desc ret_md({M, N}, dt, {ldc, 1});
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto lhs_mem = dnnl::memory(lhs_md, g_comp->eng, lhs->mem.get_data_handle());
  auto rhs_mem = dnnl::memory(rhs_md, g_comp->eng, rhs->mem.get_data_handle());

  dnnl::matmul::desc md(lhs_md, rhs_md, ret_md);
  dnnl::matmul::primitive_desc pd(md, g_comp->eng);
  dnnl::primitive prim = dnnl::matmul(pd);

  g_comp->primitives.push_back(prim);
  g_comp->args.push_back({{DNNL_ARG_SRC, lhs->mem},
                          {DNNL_ARG_WEIGHTS, rhs->mem},
                          {DNNL_ARG_DST, ret_mem}});

  odla_value v = CreateValue(ret_mem, output_dims, bias ? nullptr : id);

  InterpretIfNeeded();

  return bias ? odla_Add(v, bias, id) : v;
}

#if defined(__GNUC__) && (__GNUC__ > 9)
static inline __m512 pexp(const __m512& _x) {
  __m512 p16f_1 = _mm512_set1_ps(1.0f);
  __m512 p16f_half = _mm512_set1_ps(0.5f);
  __m512 p16f_127 = _mm512_set1_ps(127.f);
  __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
  __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

  __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

  __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
  __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
  __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
  __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
  __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
  __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

  // Clamp x.
  __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

  // Express exp(x) as exp(m*ln(2) + r), start by extracting
  // m = floor(x/ln(2) + 0.5).
  __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x, p16f_cephes_LOG2EF, p16f_half));

  // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
  // subtracted out in two parts, m*C1+m*C2 = m*ln(2), to avoid accumulating
  // truncation errors. Note that we don't use the "pmadd" function here to
  // ensure that a precision-preserving FMA instruction is used.
  __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
  __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);

  __m512 r2 = _mm512_mul_ps(r, r);

  // TODO(gonnet): Split into odd/even polynomials and try to exploit
  //               instruction-level parallelism.
  __m512 y = p16f_cephes_exp_p0;
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
  y = _mm512_fmadd_ps(y, r2, r);
  y = _mm512_add_ps(y, p16f_1);

  // Build emm0 = 2^m.
  __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
  emm0 = _mm512_slli_epi32(emm0, 23);

  // Return 2^m * exp(r).
  return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
};

static inline __m512 erf_avx512(const __m512& src512) {
  const __m512 coeff0 = _mm512_set1_ps(+7.853861353153693E-5);
  const __m512 coeff1 = _mm512_set1_ps(-8.010193625184903E-4);
  const __m512 coeff2 = _mm512_set1_ps(+5.188327685732524E-3);
  const __m512 coeff3 = _mm512_set1_ps(-2.685381193529856E-2);
  const __m512 coeff4 = _mm512_set1_ps(+1.128358514861418E-1);
  const __m512 coeff5 = _mm512_set1_ps(-3.761262582423300E-1);
  const __m512 coeff6 = _mm512_set1_ps(+1.128379165726710E+0);

  __m512 dst512;
  __m512 base512 = _mm512_mul_ps(src512, src512);
  dst512 = _mm512_fmadd_ps(coeff0, base512, coeff1);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff2);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff3);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff4);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff5);
  dst512 = _mm512_fmadd_ps(dst512, base512, coeff6);
  dst512 = _mm512_mul_ps(dst512, src512);

  return dst512;
}

static inline __m512 erfc_avx512(const __m512& src512) {
  const __m512 Pcoeff0 = _mm512_set1_ps(+2.326819970068386E-2);
  const __m512 Pcoeff1 = _mm512_set1_ps(-1.387039388740657E-1);
  const __m512 Pcoeff2 = _mm512_set1_ps(+3.687424674597105E-1);
  const __m512 Pcoeff3 = _mm512_set1_ps(-5.824733027278666E-1);
  const __m512 Pcoeff4 = _mm512_set1_ps(+6.210004621745983E-1);
  const __m512 Pcoeff5 = _mm512_set1_ps(-4.944515323274145E-1);
  const __m512 Pcoeff6 = _mm512_set1_ps(+3.404879937665872E-1);
  const __m512 Pcoeff7 = _mm512_set1_ps(-2.741127028184656E-1);
  const __m512 Pcoeff8 = _mm512_set1_ps(+5.638259427386472E-1);

  const __m512 Rcoeff0 = _mm512_set1_ps(-1.047766399936249E+1);
  const __m512 Rcoeff1 = _mm512_set1_ps(+1.297719955372516E+1);
  const __m512 Rcoeff2 = _mm512_set1_ps(-7.495518717768503E+0);
  const __m512 Rcoeff3 = _mm512_set1_ps(+2.921019019210786E+0);
  const __m512 Rcoeff4 = _mm512_set1_ps(-1.015265279202700E+0);
  const __m512 Rcoeff5 = _mm512_set1_ps(+4.218463358204948E-1);
  const __m512 Rcoeff6 = _mm512_set1_ps(-2.820767439740514E-1);
  const __m512 Rcoeff7 = _mm512_set1_ps(+5.641895067754075E-1);

  const __m512 one = _mm512_set1_ps(1.0);
  const __m512 two = _mm512_set1_ps(2.0);
  const __m512 zero = _mm512_set1_ps(0.0);
  const __m512 MinorMaxlog = _mm512_set1_ps(-88.72283905206835);

  __m512 abssrc = _mm512_abs_ps(src512);
  __m512 nabssrc = _mm512_sub_ps(zero, abssrc);
  __m512 v = _mm512_mul_ps(abssrc, nabssrc);
  __m512 z = pexp(v);
  __m512 q = _mm512_div_ps(one, abssrc);
  __m512 y = _mm512_mul_ps(q, q);

  __mmask16 PCoeff_mask = _mm512_cmplt_ps_mask(abssrc, two); // < 2
  __mmask16 RCoeff_mask = ~PCoeff_mask;

  __m512 pP;
  __m512 pR;
  if (PCoeff_mask) {
    pP = _mm512_fmadd_ps(Pcoeff0, y, Pcoeff1);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff2);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff3);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff4);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff5);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff6);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff7);
    pP = _mm512_fmadd_ps(pP, y, Pcoeff8);
  }

  if (RCoeff_mask) {
    pR = _mm512_fmadd_ps(Rcoeff0, y, Rcoeff1);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff2);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff3);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff4);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff5);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff6);
    pR = _mm512_fmadd_ps(pR, y, Rcoeff7);
  }

  pP = _mm512_mask_mov_ps(pP, RCoeff_mask, pR);
  //  y = z * q * p;
  //  float y_clamp = z < -kMaxlog ? 0 : y;

  //  return x < 0 ? 2 - y_clamp : y_clamp;
  y = _mm512_mul_ps(z, q);
  y = _mm512_mul_ps(y, pP);
  __mmask16 y_clamp_mask = _mm512_cmplt_ps_mask(z, MinorMaxlog);
  __m512 y_clamp = _mm512_mask_mov_ps(y, y_clamp_mask, zero);
  __mmask16 x_mask = _mm512_cmplt_ps_mask(src512, zero);
  __m512 y_clamp2 = _mm512_sub_ps(two, y_clamp);
  y = _mm512_mask_mov_ps(y_clamp, x_mask, y_clamp2);
  y = _mm512_sub_ps(one, y);

  return y;
}

odla_value odla_Erf(odla_value input, const odla_value_id value_id) {
  auto erf_p = [](float* src, float* dst, size_t len) {
    int i;
    for (i = 0; i + 16 <= len; i += 16) {
      __m512 src512 = _mm512_loadu_ps(src + i);
      __m512 abssrc = _mm512_abs_ps(src512);
      __mmask16 erf_mask =
          _mm512_cmplt_ps_mask(abssrc, _mm512_set1_ps(1.0)); // < 1
      __mmask16 erfc_mask = ~erf_mask;
      // printf("erf_mask:%x, erfc_mask=%x\n", erf_mask, erfc_mask);

      if (erf_mask) { // call erf
        __m512 dst512 = erf_avx512(src512);
        _mm512_mask_storeu_ps(dst + i, erf_mask, dst512);
      }
      if (erfc_mask) { // call erfc
        __m512 dst512 = erfc_avx512(src512);
        _mm512_mask_storeu_ps(dst + i, erfc_mask, dst512);
      }
      // printf("erf_p main...\n");
    }

    int remain = len - i;
    if (remain) {
      __mmask16 mask = 0xffff;
      mask = mask >> (16 - remain);
      __m512 src512 = _mm512_maskz_loadu_ps(mask, src + i);
      __mmask16 erf_mask =
          _mm512_cmplt_ps_mask(src512, _mm512_set1_ps(1.0)); // < 1
      __mmask16 erfc_mask = ~erf_mask;

      if (erf_mask) {
        __m512 dst512 = erf_avx512(src512);
        _mm512_mask_store_ps(dst + i, mask, dst512);
      }
      if (erfc_mask) {
        __m512 dst512 = erfc_avx512(src512);
        _mm512_mask_storeu_ps(dst + i, mask, dst512);
      }
      // printf("erf_p remain...\n");
    }

    return;
  };

  const auto& input_shape = input->shape;
  dnnl::memory::data_type type = input->mem.get_desc().data_type();
  assert(type == dnnl_f32);

  float* src_ptr = (float*)input->mem.get_data_handle();

  auto ret_md = input->mem.get_desc();
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  float* dst_ptr = (float*)ret_mem.get_data_handle();

  size_t total_size = GetTotalElements(input->shape);

  erf_p(src_ptr, dst_ptr, total_size);

  odla_value v = CreateValue(ret_mem, input->shape, value_id);

  InterpretIfNeeded();

  return v;
}

#endif

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
  g_comp->primitives.push_back(prim);
  g_comp->args.push_back({{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, dst_mem}});
  InterpretIfNeeded();
  return CreateValue(dst_mem, output_dims, id);
}

} // C extern
