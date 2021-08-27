//===- odla_dnnl_reduction.cc ---------------------------------------------===//
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

static odla_value reduce_op(dnnl::algorithm alg, odla_value input,
                            odla_size_t num_of_axes, const odla_uint32* axes,
                            odla_bool keep_dims, odla_value_shape output_dims,
                            const odla_value_id id, float p = 0,
                            float eps = 0) {
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
  auto reduction_desc = dnnl::reduction::desc(alg, input_md, output_md, p, eps);
  auto pd = dnnl::reduction::primitive_desc(reduction_desc, g_comp->eng);
  auto prim = dnnl::reduction(pd);
  add_op(prim, {{DNNL_ARG_SRC, input->mem}, {DNNL_ARG_DST, ret_mem}});
  InterpretIfNeeded();
  odla_value v = CreateValue(ret_mem, output_dims, id);
  return v;
}

odla_value odla_ReduceL1(odla_value input, odla_size_t num_of_axes,
                         const odla_uint32* axes, odla_bool keep_dims,
                         odla_value_shape output_dims, const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_norm_lp_sum, input, num_of_axes,
                   axes, keep_dims, output_dims, id, 1 /* P */);
}

odla_value odla_ReduceL2(odla_value input, odla_size_t num_of_axes,
                         const odla_uint32* axes, odla_bool keep_dims,
                         odla_value_shape output_dims, const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_norm_lp_sum, input, num_of_axes,
                   axes, keep_dims, output_dims, id, 2 /* P */);
}

odla_value odla_ReduceLogSum(odla_value input, odla_size_t num_of_axes,
                             const odla_uint32* axes, odla_bool keep_dims,
                             odla_value_shape output_dims,
                             const odla_value_id id) {
  const auto& name = std::string(reinterpret_cast<const char*>(id)) + "_extra";
  return odla_Log(
      odla_ReduceSum(input, num_of_axes, axes, keep_dims, output_dims,
                     (const odla_value_id)name.c_str()),
      id);
}

odla_value odla_ReduceLogSumExp(odla_value input, odla_size_t num_of_axes,
                                const odla_uint32* axes, odla_bool keep_dims,
                                odla_value_shape output_dims,
                                const odla_value_id id) {
  const auto& name = std::string(reinterpret_cast<const char*>(id));
  odla_value_shape keep_dim_shape = input->shape;
  for (int i = 0; i < num_of_axes; ++i) {
    keep_dim_shape.dims[axes[i]] = 1;
  }
  std::string name_max = name + "_max";
  std::string name_reshape = name + "_reshape";
  std::string name_exp = name + "_exp";
  std::string name_logsum = name + "_logsum";
  std::string name_sub = name + "_sub";
  auto reduce_max =
      odla_ReduceMax(input, num_of_axes, axes, keep_dims, output_dims,
                     (const odla_value_id)name_max.c_str());
  auto reduce_max_keep_dim = odla_Reshape(
      reduce_max, keep_dim_shape, (const odla_value_id)name_reshape.c_str());
  auto exp_delta = odla_Exp(odla_Sub(input, reduce_max_keep_dim,
                                     (const odla_value_id)name_sub.c_str()),
                            (const odla_value_id)name_exp.c_str());
  return odla_Add(
      odla_ReduceLogSum(exp_delta, num_of_axes, axes, keep_dims, output_dims,
                        (const odla_value_id)name_logsum.c_str()),
      reduce_max, id);
}

odla_value odla_ReduceMax(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_max, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_mean, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_ReduceMin(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_min, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_ReduceProd(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_mul, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_ReduceSum(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce_op(dnnl::algorithm::reduction_sum, input, num_of_axes, axes,
                   keep_dims, output_dims, id);
}

odla_value odla_ReduceSumSquare(odla_value input, odla_size_t num_of_axes,
                                const odla_uint32* axes, odla_bool keep_dims,
                                odla_value_shape output_dims,
                                const odla_value_id id) {
  auto sq = odla_Mul(input, input, (odla_value_id) "square");
  return odla_ReduceSum(sq, num_of_axes, axes, keep_dims, output_dims, id);
}

template <typename T>
void DoCumSum(const T* input_ptr, T* output_ptr, int axis,
              const odla_value_shape& shape, bool exclusion, bool reverse) {
  if (axis < 0) {
    axis += shape.size;
  }

  auto dim = shape.dims[axis];
  auto elems_from_axis = GetCountFromAxis(shape, axis);
  auto extents_on_axis = elems_from_axis / dim;
  auto elems_before_axis = GetTotalElements(shape) / elems_from_axis;
  for (int64_t i = 0; i < elems_before_axis; ++i) {
    for (int64_t j = 0; j < extents_on_axis; ++j) {
      auto idx = (reverse == 0) ? 0 : dim - 1;
      T acc = 0;
      int step = (reverse == 0) ? 1 : -1;
      for (int k = 0; k < dim; ++k) {
        auto offset = i * elems_from_axis + idx * extents_on_axis + j;
        T curr = input_ptr[offset];
        output_ptr[offset] = (exclusion == 0) ? acc + curr : acc;
        acc += curr;
        idx += step;
      }
    }
  }
}

odla_value odla_CumSum(odla_value input, odla_value axis, odla_bool exclusion,
                       odla_bool reverse, const odla_value_id id) {
  const auto& shape = input->shape;
  auto elem_n = GetTotalElements(input->shape);
  auto dt = input->mem.get_desc().data_type();
  auto ret_md = getMemoryDesc(shape, dt);
  assert(dt == dnnl::memory::data_type::f32);
  bool is_double = input->elem_type == ODLA_FLOAT64;
  auto ret_mem =
      is_double ? dnnl::memory(ret_md, g_comp->eng,
                               g_comp->CreateBuffer(elem_n * sizeof(double)))
                : dnnl::memory(ret_md, g_comp->eng);

  void* output_ptr = ret_mem.get_data_handle();

  std::function<void()> op = [shape, input, axis, exclusion, reverse,
                              output_ptr, is_double]() {
    const void* input_ptr = input->mem.get_data_handle();
    int ax = ((const int*)axis->mem.get_data_handle())[0];
    if (is_double) {
      DoCumSum<double>(static_cast<const double*>(input_ptr),
                       static_cast<double*>(output_ptr), ax, shape,
                       exclusion != 0, reverse != 0);
    } else {
      DoCumSum<float>(static_cast<const float*>(input_ptr),
                      static_cast<float*>(output_ptr), ax, shape,
                      exclusion != 0, reverse != 0);
    }
  };
  add_op(op);
  InterpretIfNeeded();
  odla_value v = CreateValue(ret_mem, input->shape, id);
  v->elem_type = input->elem_type;
  return v;
}

template <typename Tin, typename Tout>
static void ArgMax(const Tin* input, Tout* output, odla_int32 axis,
                   const odla_value_shape& shape, bool return_last_index,
                   bool is_arg_max) {
  axis = axis < 0 ? shape.size + axis : axis;
  auto dim = shape.dims[axis];
  // Distance between values of axis in blob
  auto axis_elems = GetCountFromAxis(shape, axis);
  auto axis_dist = axis_elems / dim;
  auto num = GetCountFromAxis(shape, 0) / axis_elems;

  auto cmp = [return_last_index, is_arg_max](Tin curr, Tin val) {
    return (is_arg_max && curr > val) || (!is_arg_max && curr < val) ||
           (curr == val && return_last_index);
  };
  for (int64_t i = 0; i < num; ++i) {
    for (int64_t j = 0; j < axis_dist; ++j) {
      Tin val = input[i * axis_elems + j];
      int64_t idx = 0;
      for (int k = 1; k < dim; ++k) {
        const Tin& curr = input[i * axis_elems + k * axis_dist + j];
        if (cmp(curr, val)) {
          val = curr;
          idx = k;
        }
      }
      output[i * axis_dist + j] = idx;
    }
  }
}

static odla_value arg_min_max(bool is_arg_max, odla_value input,
                              odla_int32 axis, odla_bool keep_dims,
                              odla_bool return_last_index,
                              const odla_value_type& output_value_type,
                              const odla_value_id value_id) {
  size_t elem_n = GetTotalElements(output_value_type.shape);
  dnnl::memory::desc md = getMemoryDesc(output_value_type);
  dnnl::memory dst_mem =
      (output_value_type.element_type == ODLA_INT64)
          ? dnnl::memory(md, g_comp->eng,
                         g_comp->CreateBuffer(elem_n * sizeof(int64_t)))
          : dnnl::memory(md, g_comp->eng);
  std::function<void()> op;
  if (input->mem.get_desc().data_type() != dnnl::memory::data_type::f32) {
    input->mem = cast_odla_mem(input->mem, input->shape,
                               dnnl::memory::data_type::f32, false);
  }

  if (output_value_type.element_type == ODLA_INT64) {
    op = [input, dst_mem, axis, return_last_index, is_arg_max]() {
      ArgMax<float, int64_t>(static_cast<float*>(input->mem.get_data_handle()),
                             static_cast<int64_t*>(dst_mem.get_data_handle()),
                             axis, input->shape, return_last_index != 0,
                             is_arg_max);
    };
  } else if (output_value_type.element_type == ODLA_INT32) {
    op = [input, dst_mem, axis, return_last_index, is_arg_max]() {
      ArgMax<float, int32_t>(static_cast<float*>(input->mem.get_data_handle()),
                             static_cast<int32_t*>(dst_mem.get_data_handle()),
                             axis, input->shape, return_last_index != 0,
                             is_arg_max);
    };
  } else {
    assert(0);
  }

  add_op(op);
  InterpretIfNeeded();
  auto v = CreateValue(dst_mem, output_value_type.shape, value_id);
  v->elem_type = output_value_type.element_type;
  return v;
}

odla_value odla_ArgMax(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id value_id) {
  return arg_min_max(true, input, axis, keep_dims, return_last_index,
                     output_value_type, value_id);
}

odla_value odla_ArgMin(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id value_id) {
  return arg_min_max(false, input, axis, keep_dims, return_last_index,
                     output_value_type, value_id);
}
