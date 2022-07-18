#include "odla_ascend_acl.h"

template <class Op>
odla_value binary_helper(odla_value lhs, odla_value rhs,
                         const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x1_tensor = lhs->op;
  ge::Operator x2_tensor = rhs->op;

  auto add = Op(name)
                 .set_input_x1_by_name(x1_tensor, lhs->outputname)
                 .set_input_x2_by_name(x2_tensor, rhs->outputname);

  ge::DataType ascend_type = GetAscendType(lhs->type);
  TensorDesc add_input_desc_x1(ge::Shape(), FORMAT_ND, ascend_type);
  TensorDesc add_input_desc_x2(ge::Shape(), FORMAT_ND, ascend_type);
  TensorDesc add_output_desc_y(ge::Shape(), FORMAT_ND, ascend_type);

  add.update_input_desc_x1(add_input_desc_x1);
  add.update_input_desc_x2(add_input_desc_x2);
  add.update_output_desc_y(add_output_desc_y);

  return CreateValue(add, lhs->type, id);
}

odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_helper<op::Mul>(lhs, rhs, id);
};

odla_value odla_Sub(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_helper<op::Sub>(lhs, rhs, id);
};

odla_value odla_Div(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_helper<op::Div>(lhs, rhs, id);
};

odla_value odla_Equal(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_helper<op::Equal>(lhs, rhs, id);
};

odla_value odla_Greater(odla_value lhs, odla_value rhs,
                        const odla_value_id id) {
  return binary_helper<op::Greater>(lhs, rhs, id);
};

odla_value odla_Less(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_helper<op::Less>(lhs, rhs, id);
};

odla_value odla_Max(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return binary_helper<op::Maximum>(lhs, rhs, id);
};

odla_value odla_Pow(odla_value base, odla_value exponent,
                    const odla_value_id id) {
  return binary_helper<op::Pow>(base, exponent, id);
};

odla_value odla_Sigmoid(odla_value input, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x_tensor = input->op;

  auto sigmoid = op::Sigmoid(name).set_input_x(x_tensor);

  ge::DataType ascend_type = GetAscendType(input->type);
  TensorDesc add_input_desc_x(ge::Shape(), FORMAT_ND, ascend_type);
  TensorDesc add_output_desc_y(ge::Shape(), FORMAT_ND, ascend_type);

  sigmoid.update_input_desc_x(add_input_desc_x);
  sigmoid.update_output_desc_y(add_output_desc_y);

  return CreateValue(sigmoid, input->type, id);
};

odla_value odla_Cast(odla_value input, odla_element_type target_type,
                     const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x_tensor = input->op;
  ge::DataType output_type = GetAscendType(target_type);

  auto cast = op::Cast(name).set_input_x(x_tensor).set_attr_dst_type(
      static_cast<int64_t>(output_type));

  ge::DataType input_type = GetAscendType(input->type);
  TensorDesc add_input_desc_x(ge::Shape(), FORMAT_ND, input_type);
  TensorDesc add_output_desc_y(ge::Shape(), FORMAT_ND, output_type);

  cast.update_input_desc_x(add_input_desc_x);
  cast.update_output_desc_y(add_output_desc_y);

  return CreateValue(cast, input->type, id);
};

// op::Slice
odla_value odla_Slice(odla_value input, const odla_int32* start,
                      const odla_int32* end, const odla_int32* stride,
                      odla_value_shape output_dims, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x_op = input->op;

  const auto& input_shape = input->type.shape;
  int32_t dims = input_shape.size;

  TensorDesc begin_desc(ge::Shape({1, dims}), FORMAT_ND, DataType::DT_INT32);
  Tensor begin_tensor(begin_desc, (uint8_t*)start, dims * sizeof(odla_int32));
  auto begin_op = op::Constant((string(name) + "_start").c_str())
                      .set_attr_value(begin_tensor);

  TensorDesc end_desc(ge::Shape({1, dims}), FORMAT_ND, DataType::DT_INT32);
  Tensor end_tensor(end_desc, (uint8_t*)end, dims * sizeof(odla_int32));
  auto end_op =
      op::Constant((string(name) + "_end").c_str()).set_attr_value(end_tensor);

  TensorDesc stride_desc(ge::Shape({1, dims}), FORMAT_ND, DataType::DT_INT32);
  Tensor stride_tensor(stride_desc, (uint8_t*)stride,
                       dims * sizeof(odla_int32));
  auto stride_op = op::Constant((string(name) + "_stride").c_str())
                       .set_attr_value(stride_tensor);

  auto slice = op::StridedSlice(name)
                   .set_input_x(x_op)
                   .set_input_begin(begin_op)
                   .set_input_end(end_op)
                   .set_input_strides(stride_op);

  ge::DataType input_type = GetAscendType(input->type);
  TensorDesc add_input_desc_x(ge::Shape(), FORMAT_ND, input_type);
  TensorDesc add_output_desc_y(ge::Shape(), FORMAT_ND, input_type);

  slice.update_input_desc_x(add_input_desc_x);
  slice.update_output_desc_y(add_output_desc_y);

  return CreateValue(
      slice, odla_value_type{input->type.element_type, output_dims}, id);
};

odla_value odla_Transpose(odla_value input, odla_value_shape permutations,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);

  ge::Operator x_tensor = input->op;

  int shape_list[permutations.size];
  for (auto i = 0; i < permutations.size; i++) {
    shape_list[i] = permutations.dims[i];
  }

  TensorDesc shape_desc(ge::Shape({permutations.size}), FORMAT_ND, DT_INT32);
  Tensor shape_tensor(shape_desc, (uint8_t*)shape_list,
                      permutations.size * sizeof(int));

  auto shape =
      op::Const((string(name) + "_shape").c_str()).set_attr_value(shape_tensor);

  auto transpose =
      op::Transpose(name).set_input_x(x_tensor).set_input_perm(shape);

  TensorDesc transpose_input_desc_x(ge::Shape(), FORMAT_ND,
                                    GetAscendType(input->type));
  TensorDesc transpose_input_desc_perm(ge::Shape(), FORMAT_ND,
                                       GetAscendType(input->type));
  TensorDesc transpose_output_desc_y(ge::Shape(), FORMAT_ND,
                                     GetAscendType(input->type));

  transpose.update_input_desc_x(transpose_input_desc_x);
  transpose.update_input_desc_perm(transpose_input_desc_perm);
  transpose.update_output_desc_y(transpose_output_desc_y);

  return CreateValue(
      transpose, odla_value_type{input->type.element_type, output_dims}, id);
};

odla_value odla_Concat(odla_values inputs, odla_int32 axis,
                       odla_value_shape output_dims, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);

  vector<ge::Operator> inputs_vector;
  for (uint64_t i = 0; i < inputs.size; i++) {
    inputs_vector.push_back(inputs.values[i]->op);
  }

  TensorDesc axis_desc(ge::Shape({1}), FORMAT_ND, DT_INT32);
  Tensor axis_tensor(axis_desc, (uint8_t*)&axis, sizeof(odla_int32));

  auto concat_dim = op::Const((string(name) + "_concat_dim").c_str());
  concat_dim.set_attr_value(axis_tensor);
  concat_dim.update_output_desc_y(
      TensorDesc(ge::Shape({1}), FORMAT_ND, DT_INT32));

  auto concat = op::Concat(name).create_dynamic_input_x(inputs.size);

  for (uint64_t i = 0; i < inputs.size; i++) {
    concat.set_dynamic_input_x(i, inputs_vector[i]);
  }

  concat.set_input_concat_dim(concat_dim).set_attr_N(inputs.size);

  TensorDesc concat_input_desc_concat_dim(ge::Shape({1}), FORMAT_ND, DT_INT32);
  DataType dtype = GetAscendType(inputs.values[0]->type);
  TensorDesc concat_output_desc_y(ge::Shape(), FORMAT_ND, dtype);

  concat.update_input_desc_concat_dim(concat_input_desc_concat_dim);
  concat.update_output_desc_y(concat_output_desc_y);

  return CreateValue(
      concat, odla_value_type{inputs.values[0]->type.element_type, output_dims},
      id);
};

odla_value odla_Resize(odla_value input, odla_interpolation_mode interpolation,
                       odla_resize_coordinate_mode mode, odla_uint32 axes_mask,
                       odla_value_shape output_dims,
                       const odla_value_id value_id) {
  const char* name = reinterpret_cast<const char*>(value_id);

  ge::Operator x_tensor = input->op;

  std::string algo = interpolation == ODLA_NEAREST ? "nearest" : "linear";

  bool is_align_corners = false;
  bool is_half_pixel_centers = false;
  std::string coord_trans_mode;
  if (mode == ODLA_ASSYMMETRIC) {
    coord_trans_mode = "asymmetric";
  } else if (mode == ODLA_HALF_PIXEL_TF) {
    coord_trans_mode = "tf_half_pixel_for_nn";
  } else if (mode == ODLA_ALIGN_CORNERS) {
    coord_trans_mode = "align_corners";
    is_align_corners = true;
  } else {
    coord_trans_mode = "half_pixel";
    is_half_pixel_centers = true;
  }

  std::vector<int64_t> sizes{output_dims.dims[2], output_dims.dims[3]};

  TensorDesc sizes_desc(ge::Shape({1, 2}), FORMAT_ND, DT_INT64);
  Tensor sizes_tensor(sizes_desc, (uint8_t*)sizes.data(), 2 * sizeof(int64_t));
  auto sizes_op =
      op::Const((string(name) + "_sizes").c_str()).set_attr_value(sizes_tensor);

  auto resize = op::ResizeNearestNeighborV2(name)
                    .set_input_x(x_tensor)
                    .set_input_size(sizes_op)
                    .set_attr_align_corners(is_align_corners)
                    .set_attr_half_pixel_centers(is_half_pixel_centers);

  ge::DataType input_type = GetAscendType(input->type);
  TensorDesc resize_input_desc_x(ge::Shape(), FORMAT_NCHW, input_type);
  resize.update_input_desc_x(resize_input_desc_x);

  return CreateValue(
      resize, odla_value_type{input->type.element_type, output_dims}, value_id);
};

odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id) {
  const char* name = reinterpret_cast<const char*>(value_id);
  const uint32_t* window_dims_ = reinterpret_cast<const uint32_t*>(window_dims);
  const uint32_t* strides_ = reinterpret_cast<const uint32_t*>(strides);

  const uint32_t ksize[4] = {G_UINT32_1, G_UINT32_1, window_dims_[0],
                             window_dims_[1]};
  const uint32_t stride[4] = {G_UINT32_1, G_UINT32_1, strides_[0], strides_[1]};

  int64_t rank = 4;
  ge::Operator x_tensor = input->op;
  std::vector<int64_t> stride_vetor;
  std::vector<int64_t> ksize_vetor;
  std::vector<int64_t> padding = {paddings_front[0], paddings_front[1],
                                  paddings_back[0], paddings_back[1]};

  for (auto i = 0; i < rank; i++) {
    stride_vetor.emplace_back(stride[i]);
    ksize_vetor.emplace_back(ksize[i]);
  }

  auto maxpool = op::MaxPoolV3(name)
                     .set_input_x(x_tensor)
                     .set_attr_ksize(ksize_vetor)
                     .set_attr_strides(stride_vetor)
                     .set_attr_pads(padding);

  TensorDesc maxpool_input_desc_x(ge::Shape(), FORMAT_ND,
                                  GetAscendType(input->type));
  TensorDesc maxpool_output_desc_y(ge::Shape(), FORMAT_ND,
                                   GetAscendType(input->type));

  maxpool.update_input_desc_x(maxpool_input_desc_x);
  maxpool.update_output_desc_y(maxpool_output_desc_y);

  return CreateValue(maxpool,
                     odla_value_type{input->type.element_type, output_dims},
                     value_id);
};

odla_value odla_DeConv(odla_value input, odla_memory_layout input_layout,
                       odla_uint32 group, odla_value kernel,
                       odla_memory_layout kernel_layout,
                       const odla_uint32* strides, const odla_uint32* dilations,
                       const odla_uint32* paddings_front,
                       const odla_uint32* paddings_back, odla_value bias,
                       odla_value_shape output_dims, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);

  ge::Operator x_tensor = input->op;
  ge::Operator filter_tensor = kernel->op;

  vector<int64_t> stride_vector({strides[0], strides[1]});
  vector<int64_t> pad_vector({paddings_front[0], paddings_front[1],
                              paddings_back[0], paddings_back[1]});
  vector<int64_t> dilation_vector(
      {G_UINT32_1, dilations[0], dilations[1], G_UINT32_1});

  auto deconv = op::Deconvolution(name)
                    .set_input_x(x_tensor)
                    .set_input_filter(filter_tensor)
                    .set_attr_strides(stride_vector)
                    .set_attr_pads(pad_vector)
                    .set_attr_dilations(dilation_vector);

  if (group > 1) {
    deconv.set_attr_groups((uint32_t)group);
  }
  if (bias != nullptr) {
    ge::Operator bias_tensor = bias->op;
    deconv.set_input_bias(bias_tensor);
  }

  TensorDesc deconvolution_input_desc_x(
      ge::Shape(), GetAscendFormat(input_layout), GetAscendType(input->type));
  TensorDesc deconvolution_input_desc_filter(ge::Shape(), FORMAT_NCHW,
                                             GetAscendType(kernel->type));
  TensorDesc deconvolution_output_desc_y(
      ge::Shape(), GetAscendFormat(input_layout), GetAscendType(input->type));

  deconv.update_input_desc_x(deconvolution_input_desc_x);
  deconv.update_input_desc_filter(deconvolution_input_desc_filter);
  deconv.update_output_desc_y(deconvolution_output_desc_y);

  return CreateValue(deconv, {input->type.element_type, output_dims}, id);
};

odla_values odla_TopK(odla_value input, odla_uint32 K, odla_bool largest,
                      odla_bool sorted, odla_uint32 axis,
                      odla_value_type output_value_type,
                      odla_value_type output_value_index_type,
                      const odla_value_ids value_ids) {
  const char* name = reinterpret_cast<const char*>(value_ids.value_ids[0]);
  ge::Operator x_op = input->op;

  int32_t k_value = K;

  TensorDesc K_desc(ge::Shape({1}), FORMAT_ND, DT_INT32);
  Tensor K_tensor(K_desc, (uint8_t*)&k_value, sizeof(int32_t));

  auto K_op = op::Const((string(name) + "_K").c_str()).set_attr_value(K_tensor);

  auto topk = op::TopK(name)
                  .set_input_x(x_op)
                  .set_input_k(K_op)
                  .set_attr_sorted(sorted)
                  .set_attr_largest(largest)
                  .set_attr_dim(axis);

  TensorDesc topk_input_desc_x(ge::Shape(), FORMAT_ND,
                               GetAscendType(input->type));
  TensorDesc topk_output_desc_values(ge::Shape(), FORMAT_ND,
                                     GetAscendType(input->type));
  TensorDesc topk_output_desc_indices(ge::Shape(), FORMAT_ND, DT_INT32);

  topk.update_input_desc_x(topk_input_desc_x);
  topk.update_output_desc_values(topk_output_desc_values);
  topk.update_output_desc_indices(topk_output_desc_indices);

  auto values =
      CreateValue(topk, output_value_type, value_ids.value_ids[0], "values");

  auto cast = op::Cast((string(name) + "_cast").c_str())
                  .set_input_x_by_name(topk, "indices")
                  .set_attr_dst_type(static_cast<int64_t>(
                      GetAscendType(output_value_index_type)));

  auto indices =
      CreateValue(cast, output_value_index_type, value_ids.value_ids[1]);

  return odla_values({.size = 2, .values = {values, indices}});
};

odla_value odla_Gather(odla_value input, const odla_value indices,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);

  ge::Operator x_op = input->op;
  ge::Operator indices_op = indices->op;

  TensorDesc axis_desc(ge::Shape({1}), FORMAT_ND, DT_INT32);
  Tensor axis_tensor(axis_desc, (uint8_t*)&axis, sizeof(int32_t));

  auto axis_op =
      op::Const((string(name) + "_axis").c_str()).set_attr_value(axis_tensor);

  auto gather = op::GatherV2(name)
                    .set_input_x(x_op)
                    .set_input_indices(indices_op)
                    .set_input_axis(axis_op);

  TensorDesc gather_input_desc_x(ge::Shape(), FORMAT_ND,
                                 GetAscendType(input->type));
  TensorDesc gather_input_desc_indices(ge::Shape(), FORMAT_ND,
                                       GetAscendType(indices->type));
  TensorDesc gather_output_desc_y(ge::Shape(), FORMAT_ND,
                                  GetAscendType(input->type));

  gather.update_input_desc_x(gather_input_desc_x);
  gather.update_input_desc_indices(gather_input_desc_indices);
  gather.update_output_desc_y(gather_output_desc_y);

  return CreateValue(
      gather, odla_value_type{input->type.element_type, output_dims}, id);
};

odla_value odla_ExpandDims(odla_value input, odla_value_shape output_dims,
                           const odla_value_id value_id) {
  const char* name = reinterpret_cast<const char*>(value_id);
  ge::Operator x_tensor = input->op;

  vector<int64_t> shape_value =
      vector<int64_t>(output_dims.dims, output_dims.dims + output_dims.size);
  TensorDesc shape_desc(ge::Shape({output_dims.size}), FORMAT_ND, DT_INT64);
  Tensor shape_tensor(shape_desc, (uint8_t*)shape_value.data(),
                      output_dims.size * sizeof(int64_t));

  auto shape_op =
      op::Const((string(name) + "_shape").c_str()).set_attr_value(shape_tensor);

  auto expand_dims =
      op::Expand(name).set_input_x(x_tensor).set_input_shape(shape_op);

  TensorDesc expand_dims_input_desc_x(ge::Shape(), FORMAT_NCHW,
                                      GetAscendType(input->type));
  TensorDesc expand_dims_output_desc_y(ge::Shape(), FORMAT_NCHW,
                                       GetAscendType(input->type));

  expand_dims.update_input_desc_x(expand_dims_input_desc_x);
  expand_dims.update_output_desc_y(expand_dims_output_desc_y);

  return CreateValue(expand_dims,
                     odla_value_type{input->type.element_type, output_dims},
                     value_id);
};

odla_value odla_Tile(odla_value input, const odla_uint32* repeat,
                     odla_value_shape output_dims,
                     const odla_value_id value_id) {
  const char* name = reinterpret_cast<const char*>(value_id);
  ge::Operator x_tensor = input->op;

  const auto& input_shape = input->type.shape;
  int32_t dims = input_shape.size;
  vector<int64_t> multiplies_value;
  multiplies_value.reserve(dims);
  for (int32_t i = 0; i < dims; i++) {
    multiplies_value.push_back(repeat[i]);
  }

  TensorDesc multiples_desc(ge::Shape({dims}), FORMAT_ND, DT_INT64);
  Tensor multiples_tensor(multiples_desc, (uint8_t*)multiplies_value.data(),
                          dims * sizeof(int64_t));

  auto multiples_op = op::Const((string(name) + "_multiples").c_str())
                          .set_attr_value(multiples_tensor);

  auto tile =
      op::Tile(name).set_input_x(x_tensor).set_input_multiples(multiples_op);

  TensorDesc tile_input_desc_x(ge::Shape(), FORMAT_NCHW,
                               GetAscendType(input->type));
  TensorDesc tile_output_desc_y(ge::Shape(), FORMAT_NCHW,
                                GetAscendType(input->type));

  tile.update_input_desc_x(tile_input_desc_x);
  tile.update_output_desc_y(tile_output_desc_y);

  return CreateValue(
      tile, odla_value_type{input->type.element_type, output_dims}, value_id);
};

template <class Op>
odla_value reduce_helper(odla_value input, odla_size_t num_of_axes,
                         const odla_uint32* axes, odla_bool keep_dims,
                         odla_value_shape output_dims, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x_tensor = input->op;

  vector<int64_t> axes_value;
  axes_value.reserve(num_of_axes);
  for (uint64_t i = 0; i < num_of_axes; i++) {
    axes_value.push_back(axes[i]);
  }

  TensorDesc axes_desc(ge::Shape({(long int)num_of_axes}), FORMAT_ND, DT_INT64);
  Tensor axes_tensor(axes_desc, (uint8_t*)axes_value.data(),
                     num_of_axes * sizeof(int64_t));

  auto axes_op =
      op::Const((string(name) + "_axes").c_str()).set_attr_value(axes_tensor);

  auto reducemin =
      Op(name).set_input_x(x_tensor).set_input_axes(axes_op).set_attr_keep_dims(
          keep_dims);

  TensorDesc reducemin_input_desc_x(ge::Shape(), FORMAT_NCHW,
                                    GetAscendType(input->type));
  TensorDesc reducemin_output_desc_y(ge::Shape(), FORMAT_NCHW,
                                     GetAscendType(input->type));

  reducemin.update_input_desc_x(reducemin_input_desc_x);
  reducemin.update_output_desc_y(reducemin_output_desc_y);

  return CreateValue(reducemin, {input->type.element_type, output_dims}, id);
}

odla_value odla_ReduceSum(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce_helper<op::ReduceSum>(input, num_of_axes, axes, keep_dims,
                                      output_dims, id);
};

odla_value odla_ReduceMin(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return reduce_helper<op::ReduceMin>(input, num_of_axes, axes, keep_dims,
                                      output_dims, id);
};

odla_value odla_ArgMin(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x_tensor = input->op;
  DataType dtype = GetAscendType(output_value_type);

  TensorDesc dim_desc(ge::Shape({}), FORMAT_ND,
                      DT_INT32); // must be a scalar, shape is empty
  Tensor dim_tensor(dim_desc, (uint8_t*)&axis, sizeof(odla_int32));

  auto dim_op =
      op::Const((string(name) + "_dim").c_str()).set_attr_value(dim_tensor);

  auto reducemin = op::ArgMin(name)
                       .set_input_x(x_tensor)
                       .set_input_dimension(dim_op)
                       .set_attr_dtype(dtype);

  TensorDesc reducemin_input_desc_x(ge::Shape(), FORMAT_NCHW,
                                    GetAscendType(input->type));
  TensorDesc reducemin_output_desc_y(ge::Shape(), FORMAT_NCHW,
                                     GetAscendType(input->type));

  reducemin.update_input_desc_x(reducemin_input_desc_x);
  reducemin.update_output_desc_y(reducemin_output_desc_y);

  return CreateValue(reducemin, output_value_type, id);
};
