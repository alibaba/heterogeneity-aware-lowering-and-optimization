//===- odla_xnnpack.cc ----------------------------------------------------===//
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
#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthreadpool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xnnpack.h>

#include "ODLA/odla_common.h"

#define MAX_BINDINGS 16

typedef enum xnn_status xnn_status_t;

struct _odla_computation {
  xnn_subgraph_t graph;
  int num_bindings;
  odla_value external_vals[MAX_BINDINGS];
};

struct _odla_computation g_comp;

static int64_t GetTotalElements(const odla_value_shape* shape) {
  int64_t ret = 1;
  for (int i = 0; i < shape->size; ++i) ret *= shape->dims[i];
  return ret;
}

#ifdef USE_SUBGRAPH
struct _odla_value {
  uint32_t id;
  odla_value_id vid;
};

struct _odla_context {
  xnn_runtime_t rt;
  int bindings_nr;
  struct xnn_external_value bindings[MAX_BINDINGS];
};

typedef struct {
  odla_value* data;
  size_t capacity;
  size_t size; // Number of vectors in it at present
} value_vec;

static odla_value CreateValue(const odla_value_id vid) {
  odla_value buf = (odla_value)malloc(sizeof(struct _odla_value));
  buf->vid = vid;
  return buf;
}

odla_status odla_CreateComputation(odla_computation* computation) {
  g_comp.graph = NULL;
  g_comp.num_bindings = 0;
  // g_comp.num_inputs = 0;
  // g_comp.num_outputs = 0;

  xnn_status_t s = xnn_initialize(NULL);
  assert(s == xnn_status_success);
  s = xnn_create_subgraph(MAX_BINDINGS, 0, &g_comp.graph);
  assert(s == xnn_status_success);
  *computation = &g_comp;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation comp) {
  xnn_delete_subgraph(comp->graph);
  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
  *context = (odla_context)calloc(1, sizeof(struct _odla_context));
  xnn_status_t s =
      xnn_create_runtime_v2(g_comp.graph, NULL, 0, &(*context)->rt);
  assert(s == xnn_status_success);
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context context) {
  xnn_delete_runtime(context->rt);
  free(context);
  return ODLA_SUCCESS;
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  assert(type.element_type == ODLA_FLOAT32);
  odla_value val = CreateValue(id);
  xnn_status_t s = xnn_define_tensor_value(
      g_comp.graph, xnn_datatype_fp32, type.shape.size, type.shape.dims, NULL,
      g_comp.num_bindings, XNN_VALUE_FLAG_EXTERNAL_INPUT, &val->id);
  assert(s == xnn_status_success);
  assert(val->id == g_comp.num_bindings);
  g_comp.external_vals[g_comp.num_bindings++] = val;
  return val;
}

odla_status odla_SetValueAsOutput(const odla_value val) {
  g_comp.external_vals[g_comp.num_bindings++] = val;
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  context->bindings[context->bindings_nr].id = value->id;
  context->bindings[context->bindings_nr++].data = (odla_void*)data_ptr;
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  odla_value val = NULL;
  for (int i = 0; i < g_comp.num_bindings && val == NULL; ++i) {
    if (strcmp(g_comp.external_vals[i]->vid, value_id) == 0) {
      val = g_comp.external_vals[i];
    }
  }
  assert(val);
  return odla_BindToArgument(val, data_ptr, context);
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  return odla_BindToArgument(value, data_ptr, context);
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  return odla_BindToArgumentById(value_id, data_ptr, context);
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  assert(comp->num_bindings == context->bindings_nr);
  xnn_status_t s =
      xnn_setup_runtime(context->rt, context->bindings_nr, context->bindings);
  assert(s == xnn_status_success);

  s = xnn_invoke_runtime(context->rt);
  assert(s == xnn_status_success);
  context->bindings_nr = 0;
}

#else
static void init() __attribute__((constructor));
static void deinit() __attribute__((destructor));

struct _odla_value {
  odla_value_shape shape;
  float* data;
  xnn_operator_t op;
  xnn_operator_t op2;
  float* buf0;
  float* buf1;
  int is_extern_data;
  int needs_setup;
};

#define MAX_VALUE 1000
struct _odla_context {
  int inited;
  pthreadpool_t threadpool;
  int value_cnt;
  odla_value_id keys[MAX_VALUE];
  odla_value vals[MAX_VALUE];
};

static struct _odla_context g_ctx;

static void init() {
  if (g_ctx.inited) return;
  xnn_status_t s = xnn_initialize(NULL);
  assert(s == xnn_status_success);
  // g_ctx.threadpool = pthreadpool_create(0);
  g_ctx.inited = 1;
}

static void deinit() {
  if (g_ctx.inited == 0) return;
  for (int i = 0; i < MAX_VALUE; ++i) {
    odla_value v = g_ctx.vals[i];
    if (v == NULL) continue;
    if (!v->is_extern_data) {
      free(v->data);
    }
    if (v->buf0) free(v->buf0);
    if (v->buf1) free(v->buf1);
    if (v->op) xnn_delete_operator(v->op);
    if (v->op2) xnn_delete_operator(v->op2);
    free(v);
  }
  pthreadpool_destroy(g_ctx.threadpool);
  xnn_deinitialize();
  memset(&g_ctx, 0, sizeof(g_ctx));
}

static odla_value GetOrCreateValue(const odla_value_shape* shape,
                                   const odla_value_id id) {
#ifdef VALUE_ID_AS_PTR
  for (int i = 0; i < g_ctx.value_cnt; ++i) {
    if (g_ctx.keys[i] == id) return g_ctx.vals[i];
  }
#else
  size_t idx = (size_t)id;
  assert(idx <= MAX_VALUE);
  if (g_ctx.vals[idx] != NULL) return g_ctx.vals[idx];
#endif

  odla_value val = (odla_value)calloc(1, sizeof(struct _odla_value));
  val->shape = *shape;
  val->needs_setup = 1;

#ifdef VALUE_ID_AS_PTR
  g_ctx.vals[g_ctx.value_cnt] = val;
  g_ctx.keys[g_ctx.value_cnt] = id;
  assert(g_ctx.value_cnt <= MAX_VALUE);
#else
  g_ctx.vals[idx] = val;
#endif
  ++g_ctx.value_cnt;
  return val;
}

odla_status odla_ReleaseValue(odla_value v) {
  if (!v->is_extern_data) {
    free(v->data);
    v->data = NULL;
    v->needs_setup = 1;
  }
}

static odla_value GetValue(const odla_value_shape* shape,
                           const odla_value_id id) {
  odla_value val = GetOrCreateValue(shape, id);
  if (val->data == NULL) {
    val->data = malloc(sizeof(float) * GetTotalElements(shape));
  }
  return val;
}

odla_status odla_CreateContext(odla_context* context) {
  if (context != NULL) {
    *context = &g_ctx;
  }
  if (g_ctx.inited == 0) {
    init();
    g_ctx.inited = 1;
  }
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context ctx) {
  if (ctx != NULL) {
    assert(ctx == &g_ctx);
  }
  deinit();
  return ODLA_SUCCESS;
}

odla_value odla_CreateValue(odla_value_type type, const odla_value_id id) {
  assert(type.element_type == ODLA_FLOAT32);
  odla_value v = GetValue(&type.shape, id);
  return v;
}

odla_status odla_SetValueData(odla_value val, const void* ptr) {
  val->data = (void*)ptr;
  val->is_extern_data = 1;
  val->needs_setup = 1;
  return ODLA_SUCCESS;
}

odla_status odla_GetValueData(const odla_value value, odla_void* data_ptr) {
  memcpy(data_ptr, value->data,
         sizeof(float) * GetTotalElements(&value->shape));
}
#endif

odla_value odla_CreateConstant(odla_value_type type, const void* ptr,
                               const odla_value_id id) {
  assert(type.element_type == ODLA_FLOAT32);
  odla_value val = GetOrCreateValue(&type.shape, id);
  if (val->data == NULL) {
    val->data = (void*)ptr;
    val->is_extern_data = 1;
    val->needs_setup = 0;
  }
#ifdef USE_SUBGRAPH
  xnn_status_t s = xnn_define_tensor_value(
      g_comp.graph, xnn_datatype_fp32, type.shape.size, type.shape.dims, ptr,
      XNN_INVALID_VALUE_ID, 0, &val->id);
  assert(s == xnn_status_success);
#endif
  return val;
}

void odla_Dump(odla_value v) {
  int t = 1; // v->shape.dims[v->shape.size - 1];
  float* data = v->data;
  bool use_nchw = false;
  if (!use_nchw) {
    for (int i = 0, e = GetTotalElements(&v->shape) / t; i < e; ++i) {
      for (int j = 0; j < t; ++j) printf("%.3f", *data++);
      printf("\n");
    }
    return;
  }

  if (v->shape.size == 4) {
    int ch = v->shape.dims[3];
    int s = v->shape.dims[1] * v->shape.dims[2];
    for (int c = 0; c < ch; ++c)
      for (int i = 0; i < s; ++i) printf("%.5f\n", data[i * ch + c]);
  } else {
    for (int i = 0, e = GetTotalElements(&v->shape); i < e; ++i)
      printf("%.5f\n", data[i]);
  }
}

odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  xnn_status_t s;

  assert(input_layout == ODLA_CHANNELS_LAST);
  odla_value val = GetValue(&output_dims, id);
  if (val->op == NULL) {
    int out_ch = kernel->shape.dims[3], in_ch = kernel->shape.dims[2],
        ss = kernel->shape.dims[0] * kernel->shape.dims[1];
    if (out_ch * group == in_ch) {
      int t = out_ch;
      out_ch = in_ch;
      in_ch = t;
    }
    // kernel->shape.dims[0] = kernel->shape.dims[2];
    // kernel->shape.dims[1] = kernel->shape.dims[3];
    // kernel->shape.dims[2] = in_ch;
    // kernel->shape.dims[3] = out_ch;
    float* w_t = malloc(sizeof(float) * out_ch * in_ch * ss);

    for (int i = 0; i < in_ch; ++i)
      for (int o = 0; o < out_ch; ++o)
        for (int s = 0; s < ss; ++s)
          w_t[o * in_ch * ss + s * in_ch + i] =
              kernel->data[s * in_ch * out_ch + i * out_ch + o];
    kernel->data = w_t;
    kernel->is_extern_data = 0;

    // output_dims.dims[3] = output_dims.dims[1];
    // output_dims.dims[1] = output_dims.dims[2];

    val->shape = output_dims;

    xnn_status_t s = xnn_create_convolution2d_nhwc_f32(
        paddings_front[0] /*input_padding_top*/,
        paddings_back[1] /*input_padding_right*/,
        paddings_back[0] /*input_padding_bottom*/,
        paddings_front[1] /*input_padding_left*/,
        kernel->shape.dims[0] /* kernel_height */,
        kernel->shape.dims[1] /*kernel_width*/,
        strides[0] /*subsampling_height*/, strides[1] /*subsampling_width*/,
        dilations[0] /*dilation_height*/, dilations[1] /*dilation_width*/,
        group, input->shape.dims[3] / group /*group_input_channels*/,
        output_dims.dims[3] / group /*group_output_channels*/,
        input->shape.dims[3] /* input_pixel_stride */,
        output_dims.dims[3] /*output_pixel_stride*/, kernel->data,
        (bias == NULL) ? NULL : bias->data, -FLT_MAX /*output_min*/,
        FLT_MAX /*output_max*/, 0 /*flags*/, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || input->needs_setup) {
    s = xnn_setup_convolution2d_nhwc_f32(
        val->op, input->shape.dims[0], input->shape.dims[1] /*input_height*/,
        input->shape.dims[2] /*input_width*/, input->data, val->data,
        g_ctx.threadpool);
    val->needs_setup = 0;
  }

  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_DeConv(odla_value input, odla_memory_layout input_layout,
                       odla_uint32 group, odla_value kernel,
                       odla_memory_layout kernel_layout,
                       const odla_uint32* strides, const odla_uint32* dilations,
                       const odla_uint32* paddings_front,
                       const odla_uint32* paddings_back, odla_value bias,
                       odla_value_shape output_dims, const odla_value_id id) {
  xnn_status_t s;

  assert(input_layout == ODLA_CHANNELS_LAST);
  odla_value val = GetValue(&output_dims, id);
  if (val->op == NULL) {
    int out_ch = kernel->shape.dims[3], in_ch = kernel->shape.dims[2],
        ss = kernel->shape.dims[0] * kernel->shape.dims[1];
    float* w_t = malloc(sizeof(float) * out_ch * in_ch * ss);

    for (int i = 0; i < in_ch; ++i)
      for (int o = 0; o < out_ch; ++o)
        for (int s = 0; s < ss; ++s)
          w_t[o * in_ch * ss + s * in_ch + i] =
              kernel->data[s * in_ch * out_ch + i * out_ch + o];
    kernel->data = w_t;
    kernel->is_extern_data = 0;

    val->shape = output_dims;

    xnn_status_t s = xnn_create_deconvolution2d_nhwc_f32(
        paddings_front[0] /*input_padding_top*/,
        paddings_back[1] /*input_padding_right*/,
        paddings_back[0] /*input_padding_bottom*/,
        paddings_front[1] /*input_padding_left*/,
        kernel->shape.dims[0] /* kernel_height */,
        kernel->shape.dims[1] /*kernel_width*/,
        strides[0] /*subsampling_height*/, strides[1] /*subsampling_width*/,
        dilations[0] /*dilation_height*/, dilations[1] /*dilation_width*/,
        group, input->shape.dims[3] / group /*group_input_channels*/,
        output_dims.dims[3] / group /*group_output_channels*/,
        input->shape.dims[3] /* input_pixel_stride */,
        output_dims.dims[3] /*output_pixel_stride*/, kernel->data,
        (bias == NULL) ? NULL : bias->data, -FLT_MAX /*output_min*/,
        FLT_MAX /*output_max*/, 0 /*flags*/, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || input->needs_setup) {
    s = xnn_setup_deconvolution2d_nhwc_f32(
        val->op, input->shape.dims[0], input->shape.dims[1] /*input_height*/,
        input->shape.dims[2] /*input_width*/, 0, 0 /*adjustment?*/, input->data,
        val->data, g_ctx.threadpool);
    val->needs_setup = 0;
  }

  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_Concat(odla_values inputs, odla_int32 axis,
                       odla_value_shape output_dims, const odla_value_id id) {
  odla_value val = GetValue(&output_dims, id);
  assert(axis == 3 && output_dims.size == 4);
  int s = 1;
  for (int i = 0; i < inputs.values[0]->shape.size - 1; ++i) {
    s *= inputs.values[0]->shape.dims[i];
  }
  float* dst = val->data;
  for (int i = 0; i < s; ++i) {
    for (int j = 0; j < inputs.size; ++j) {
      int ch = inputs.values[j]->shape.dims[3];
      memcpy(dst, inputs.values[j]->data + i * ch, sizeof(float) * ch);
      dst += ch;
    }
  }
  return val;
}

odla_value odla_BatchNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id value_id) {
  odla_value val = GetValue(&input->shape, value_id);
  xnn_status_t s;
  int C = mean->shape.dims[0];
  if (val->op == NULL) {
    val->buf0 = (float*)malloc(sizeof(float) * C);
    val->buf1 = (float*)malloc(sizeof(float) * C);

    for (int64_t i = 0; i < C; ++i) {
      float s = (scale != NULL) ? scale->data[i] : scalar_scale;
      val->buf0[i] = s / sqrtf(var->data[i] + epsilon);
      float b = (offset != NULL) ? offset->data[i] : scalar_offset;
      val->buf1[i] = b - mean->data[i] * val->buf0[i];
    }
  }
  int batchs = GetTotalElements(&input->shape) / C;
  if (val->needs_setup || input->needs_setup) {
    xnn_operator_t op_mul = NULL, op_add = NULL;
    s = xnn_create_multiply_nd_f32(-FLT_MAX, FLT_MAX, 0, &op_mul);
    assert(s == xnn_status_success);
    s = xnn_setup_multiply_nd_f32(op_mul, 2, (size_t[]){batchs, C}, 1,
                                  (size_t[]){C}, input->data, val->buf0,
                                  val->data, g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->op = op_mul;
    s = xnn_create_add_nd_f32(-FLT_MAX, FLT_MAX, 0, &op_add);
    assert(s == xnn_status_success);

    s = xnn_setup_add_nd_f32(op_add, 2, (size_t[]){batchs, C}, 1, (size_t[]){C},
                             val->data, val->buf1, val->data, g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->op2 = op_add;
  }
  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  s = xnn_run_operator(val->op2, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_Relu(odla_value input, const odla_value_id id) {
  return odla_Clamp(input, 0, FLT_MAX, id);
}

odla_value odla_LeakyRelu(odla_value input, odla_float32 alpha,
                          const odla_value_id id) {
  odla_value val = GetValue(&input->shape, id);
  size_t elem_cnt = GetTotalElements(&input->shape);
  for (size_t i = 0; i < elem_cnt; ++i) {
    val->data[i] =
        (input->data[i]) >= 0 ? input->data[i] : input->data[i] * alpha;
  }
  return val;
}

odla_value odla_Resize(odla_value input, odla_interpolation_mode interpolation,
                       odla_resize_coordinate_mode mode, odla_uint32 axes_mask,
                       odla_value_shape output_dims,
                       const odla_value_id value_id) {
  odla_value val = GetValue(&output_dims, value_id);
  assert(interpolation == ODLA_NEAREST);
  assert(input->shape.size == 4 && axes_mask == -1);
  int out_h = output_dims.dims[1];
  int out_w = output_dims.dims[2];
  int ch = output_dims.dims[3];
  int in_h = input->shape.dims[1];
  int in_w = input->shape.dims[2];

  float* dst_ptr = val->data;
  assert(ch == input->shape.dims[3]);
  size_t copy_size = sizeof(float) * ch;
  const float* src_ptr = input->data;
  for (int n = 0; n < output_dims.dims[0]; ++n) {
    for (int h = 0; h < out_h; ++h) {
      int src_h = in_h * h / out_h;
      for (int w = 0; w < out_w; ++w) {
        int src_w = in_w * w / out_w;
        memcpy(dst_ptr, src_ptr + src_h * in_w * ch + src_w * ch, copy_size);
        dst_ptr += ch;
      }
    }
    src_ptr += in_h * in_w * ch;
  }
  return val;
}

odla_value odla_Sigmoid(odla_value input, const odla_value_id id) {
  xnn_status_t s;
  odla_value val = GetValue(&input->shape, id);
  if (val->op == NULL) {
    int ch = input->shape.dims[3];
    s = xnn_create_sigmoid_nc_f32(ch, ch, ch, 0, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || input->needs_setup) {
    size_t ss =
        input->shape.dims[0] * input->shape.dims[1] * input->shape.dims[2];
    s = xnn_setup_sigmoid_nc_f32(val->op, ss, input->data, val->data,
                                 g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }

  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_Softmax(odla_value input, odla_int32 axis,
                        const odla_value_id id) {
  xnn_status_t s;
  odla_value val = GetValue(&input->shape, id);
  if (val->op == NULL) {
    s = xnn_create_softmax_nc_f32(input->shape.dims[1], input->shape.dims[1],
                                  input->shape.dims[1], 0, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || input->needs_setup) {
    s = xnn_setup_softmax_nc_f32(val->op, input->shape.dims[0], input->data,
                                 input->data, g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }

  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);

  return input; // TODO: check if in-place is OK.
}

odla_value odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
                      const odla_value_id id) {
  // FIXME: fuse
  xnn_status_t s;
  odla_value val = GetValue(&input->shape, id);
  if (val->op == NULL) {
    s = xnn_create_clamp_nc_f32(input->shape.dims[1], input->shape.dims[1],
                                input->shape.dims[1], lo, hi, 0, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || input->needs_setup) {
    size_t ss =
        input->shape.dims[0] * input->shape.dims[2] * input->shape.dims[3];
    s = xnn_setup_clamp_nc_f32(val->op, ss, input->data, val->data,
                               g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }

  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
  // return input; // TODO: check if in-place is OK.
}

odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  odla_value val = GetValue(&lhs->shape, id);
  xnn_status_t s;
  size_t lhs_dims[ODLA_MAX_DIMENSION];
  size_t rhs_dims[ODLA_MAX_DIMENSION];
  for (int i = 0; i < ODLA_MAX_DIMENSION; ++i) {
    lhs_dims[i] = (size_t)lhs->shape.dims[i];
    rhs_dims[i] = (size_t)rhs->shape.dims[i];
  }
  if (val->op == NULL) {
    s = xnn_create_add_nd_f32(-FLT_MAX, FLT_MAX, 0, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || lhs->needs_setup || rhs->needs_setup) {
    s = xnn_setup_add_nd_f32(val->op, lhs->shape.size, lhs_dims,
                             rhs->shape.size, rhs_dims, lhs->data, rhs->data,
                             val->data, g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }
  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  odla_value val = GetValue(&lhs->shape, id);
  xnn_status_t s;
  size_t lhs_dims[ODLA_MAX_DIMENSION];
  size_t rhs_dims[ODLA_MAX_DIMENSION];
  for (int i = 0; i < ODLA_MAX_DIMENSION; ++i) {
    lhs_dims[i] = (size_t)lhs->shape.dims[i];
    rhs_dims[i] = (size_t)rhs->shape.dims[i];
  }
  if (val->op == NULL) {
    s = xnn_create_multiply_nd_f32(-FLT_MAX, FLT_MAX, 0, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || lhs->needs_setup || rhs->needs_setup) {
    s = xnn_setup_multiply_nd_f32(val->op, lhs->shape.size, lhs_dims,
                                  rhs->shape.size, rhs_dims, lhs->data,
                                  rhs->data, val->data, g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }
  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_AveragePool(odla_value input, odla_memory_layout input_layout,
                            const odla_uint32* window_dims,
                            const odla_uint32* strides,
                            const odla_uint32* paddings_front,
                            const odla_uint32* paddings_back,
                            odla_value_shape output_dims,
                            const odla_value_id value_id) {
  if (window_dims[0] == 1 && window_dims[1] == 1) {
    return odla_MaxPool(input, input_layout, window_dims, strides,
                        paddings_front, paddings_back, output_dims, value_id);
  }

  odla_value val = GetValue(&output_dims, value_id);
  int ch = output_dims.dims[3];
  xnn_status_t s;
  if (val->op == NULL) {
    s = xnn_create_average_pooling2d_nhwc_f32(
        paddings_front[0] /*input_padding_top*/,
        paddings_back[1] /*input_padding_right*/,
        paddings_back[0] /*input_padding_bottom*/,
        paddings_front[1] /*input_padding_left*/,
        window_dims[0] /* kernel_height */, window_dims[1] /*kernel_width*/,
        strides[0], strides[1], ch /* channels */,
        input->shape.dims[3] /* input_pixel_stride */,
        output_dims.dims[3] /*output_pixel_stride*/, -FLT_MAX /*output_min*/,
        FLT_MAX /*output_max*/, 0 /*flags*/, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || input->needs_setup) {
    s = xnn_setup_average_pooling2d_nhwc_f32(
        val->op, input->shape.dims[0] /* batch */, input->shape.dims[1],
        input->shape.dims[2] /*width*/, input->data, val->data,
        g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }
  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id) {
  odla_value val = GetValue(&output_dims, value_id);
  int ch = output_dims.dims[3];
  if (window_dims[0] == 1 && window_dims[1] == 1) {
    if (paddings_front[0] == 0 && paddings_front[1] == 0 &&
        paddings_back[0] == 0 && paddings_back[1] == 0) {
      float* out = val->data;
      const float* in = input->data;
      for (int b = 0; b < output_dims.dims[0]; ++b) {
        const float* in_row = in;
        for (int h = 0; h < output_dims.dims[1]; ++h) {
          const float* in_col = in_row;
          for (int w = 0; w < output_dims.dims[2]; ++w) {
            memcpy(out, in_col, sizeof(float) * ch);
            out += ch;
            in_col += ch * strides[1];
          }
          in_row += strides[1] * ch * input->shape.dims[2];
        }
      }
      in += ch * input->shape.dims[1] * input->shape.dims[2];
    }
    return val;
  }
  xnn_status_t s;
  if (val->op == NULL) {
    s = xnn_create_max_pooling2d_nhwc_f32(
        paddings_front[0] /*input_padding_top*/,
        paddings_back[1] /*input_padding_right*/,
        paddings_back[0] /*input_padding_bottom*/,
        paddings_front[1] /*input_padding_left*/,
        window_dims[0] /* kernel_height */, window_dims[1] /*kernel_width*/,
        strides[0], strides[1], 1, 1, output_dims.dims[3] /* channels */,
        input->shape.dims[3] /* input_pixel_stride */,
        output_dims.dims[3] /*output_pixel_stride*/, -FLT_MAX /*output_min*/,
        FLT_MAX /*output_max*/, 0 /*flags*/, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || input->needs_setup) {
    s = xnn_setup_max_pooling2d_nhwc_f32(
        val->op, input->shape.dims[0] /* batch */, input->shape.dims[1],
        input->shape.dims[2] /*width*/, input->data, val->data,
        g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }
  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id) {
  odla_value val = GetOrCreateValue(&output_dims, id);
  val->data = input->data;
  val->is_extern_data = 1;
  return val;
}

odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  odla_value val = GetValue(&output_dims, id);
  xnn_status_t s;
  if (val->op == NULL) {
    s = xnn_create_global_average_pooling_nwc_f32(
        input->shape.dims[3] /*channels*/,
        input->shape.dims[3] /*input_stride*/,
        input->shape.dims[3] /*output_stride*/, -FLT_MAX, FLT_MAX, 0 /*flags*/,
        &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || input->needs_setup) {
    s = xnn_setup_global_average_pooling_nwc_f32(
        val->op, input->shape.dims[0] /* batch */,
        input->shape.dims[1] * input->shape.dims[2] /*width*/, input->data,
        val->data, g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }
  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_value odla_Transpose(odla_value input, odla_value_shape permutations,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  odla_value val = GetValue(&output_dims, id);
  int dims = input->shape.size;

  odla_value_shape orig_strides;
  orig_strides.size = dims;
  orig_strides.dims[dims - 1] = 1;
  for (int i = dims - 2; i >= 0; --i) {
    orig_strides.dims[i] = orig_strides.dims[i + 1] * input->shape.dims[i + 1];
  }
  odla_value_shape perm_strides;
  perm_strides.size = dims;
  for (int i = 0; i < dims; ++i) {
    perm_strides.dims[i] = orig_strides.dims[permutations.dims[i]];
  }

  odla_value_shape pos; // tracks the position of dst tensor.
  for (int j = 0; j < dims; ++j) pos.dims[j] = 0;

  size_t elem_size = sizeof(float);
  size_t elem_cnt = GetTotalElements(&output_dims);
  for (size_t i = 0; i < elem_cnt; ++i) {
    size_t offset = 0;
    for (int j = 0; j < dims; ++j) {
      offset += pos.dims[j] * perm_strides.dims[j];
    }
    val->data[i] = input->data[offset];
    int c = 1;
    for (int j = dims - 1; j >= 0 && c == 1; --j) {
      pos.dims[j] += c;
      if (pos.dims[j] >= output_dims.dims[j]) {
        pos.dims[j] = 0;
      } else {
        c = 0;
      }
    }
  }
  return val;
}

odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  odla_value val = GetValue(&output_dims, id);
  xnn_status_t s;
  if (val->op == NULL) {
    s = xnn_create_fully_connected_nc_f32(
        rhs->shape.dims[1] /* input_channels*/,
        rhs->shape.dims[0] /* output_channels*/,
        rhs->shape.dims[1] /*input_stride*/,
        rhs->shape.dims[0] /*output_stride*/, rhs->data,
        bias ? bias->data : NULL, -FLT_MAX, FLT_MAX, 0 /*flags*/, &val->op);
    assert(s == xnn_status_success);
  }
  if (val->needs_setup || lhs->needs_setup || rhs->needs_setup) {
    s = xnn_setup_fully_connected_nc_f32(
        val->op, lhs->shape.dims[0] /*batch_size*/, lhs->data, val->data,
        g_ctx.threadpool);
    assert(s == xnn_status_success);
    val->needs_setup = 0;
  }
  s = xnn_run_operator(val->op, g_ctx.threadpool);
  assert(s == xnn_status_success);
  return val;
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  value_type->element_type = ODLA_FLOAT32;
  value_type->shape = value->shape;
  return ODLA_SUCCESS;
}