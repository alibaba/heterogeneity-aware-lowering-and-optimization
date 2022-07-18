#include "odla_ascend_acl.h"

const uint32_t G_UINT32_1 = 1;
thread_local odla_computation g_comp;
std::vector<std::unique_ptr<_odla_computation>> g_comps;
ModelProcess model;

// ascend::DataType
ge::DataType GetAscendType(odla_element_type element_type) {
  switch (element_type) {
    case ODLA_FLOAT32:
      return ge::DataType::DT_FLOAT;
    case ODLA_FLOAT16:
      return ge::DataType::DT_FLOAT16;
    case ODLA_INT32:
      return ge::DataType::DT_INT32;
    case ODLA_INT64:
      return ge::DataType::DT_INT64;
    case ODLA_BOOL:
      return ge::DataType::DT_BOOL;
    default:
      return ge::DataType::DT_FLOAT;
  }
};

ge::DataType GetAscendType(odla_value_type type) {
  return GetAscendType(type.element_type);
};

size_t GetElementSize(odla_value_type type) {
  switch (type.element_type) {
    case ODLA_FLOAT32:
      return sizeof(float);
    case ODLA_FLOAT16:
      return sizeof(int16_t);
    case ODLA_INT32:
      return sizeof(int32_t);
    case ODLA_INT64:
      return sizeof(int64_t);
    case ODLA_INT8:
      return sizeof(int8_t);
    case ODLA_BOOL:
      return sizeof(bool);
    default:
      return sizeof(float);
  }
}

ge::Format GetAscendFormat(odla_memory_layout input_layout) {
  switch (input_layout) {
    case ODLA_CHANNELS_FIRST:
      return FORMAT_NCHW;
    case ODLA_CHANNELS_LAST:
      return FORMAT_NHWC;
    case ODLA_SIO:
      return FORMAT_HWCN;
    case ODLA_OIS:
    case ODLA_IOS:
      return FORMAT_ND;
    default:
      return FORMAT_ND;
  }
}

ge::Format GetKernelFormat(odla_memory_layout input_layout) {
  switch (input_layout) {
    case ODLA_CHANNELS_FIRST:
    case ODLA_OIS:
      return FORMAT_NCHW;
    case ODLA_CHANNELS_LAST:
    case ODLA_SOI:
      return FORMAT_HWCN;
    case ODLA_IOS:
    case ODLA_SIO:
    default:
      ERROR_LOG("unsupported format");
      return FORMAT_ND;
  }
}

std::string GetAscendFmtString(odla_memory_layout input_layout) {
  switch (input_layout) {
    case ODLA_CHANNELS_FIRST:
      return "NCHW";
    case ODLA_CHANNELS_LAST:
      return "NHWC";
    case ODLA_SIO:
      return "HWCN";
    default:
      return "unsupport";
  }
}

int GetElementNums(const odla_value_shape shape) {
  int count = 1;
  for (int i = 0; i < shape.size; i++) {
    count *= shape.dims[i];
  }

  return count;
}

odla_value odla_CreateConstant(odla_value_type type, const odla_void* data_ptr,
                               const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  vector<int64_t> output_shape;
  for (auto i = 0; i < type.shape.size; i++) {
    output_shape.emplace_back(type.shape.dims[i]);
  }

  TensorDesc weight_desc(ge::Shape(output_shape), FORMAT_ND,
                         GetAscendType(type));
  int element_nums = GetElementNums(type.shape);
  Tensor weight_tensor(weight_desc, (uint8_t*)data_ptr,
                       element_nums * GetElementSize(type));
  auto weight = op::Const(name).set_attr_value(weight_tensor);

  return CreateValue(weight, type, id);
}

odla_status odla_SetComputationItem(odla_computation computation,
                                    odla_item_type type,
                                    odla_item_value value) {
  switch (type) {
    case ODLA_BF16_MODE:
      break;
    default:
      return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

const string ConvPad(odla_value input, odla_value kernel,
                     const uint32_t* strides, const uint32_t* dilations,
                     const uint32_t* paddings_front,
                     const uint32_t* paddings_back) {
  int dilation_kernel[2];
  dilation_kernel[0] = input->type.shape.dims[2] +
                       (input->type.shape.dims[2] - 1) * (dilations[0] - 1);
  dilation_kernel[1] = input->type.shape.dims[3] +
                       (input->type.shape.dims[3] - 1) * (dilations[1] - 1);

  uint32_t pad_result_cols = (input->type.shape.dims[1] + paddings_front[0] +
                              paddings_back[0] - dilation_kernel[0]) /
                                 strides[0] +
                             1;
  uint32_t pad_result_rows = (input->type.shape.dims[2] + paddings_front[1] +
                              paddings_back[1] - dilation_kernel[1]) /
                                 strides[1] +
                             1;

  if ((pad_result_cols == input->type.shape.dims[1]) &&
      (pad_result_rows == input->type.shape.dims[2])) {
    return "SAME";
  } else {
    return "VALID";
  }
}

const string MaxPoolPad(odla_value input, const uint32_t* window_dims,
                        const uint32_t* strides, const uint32_t* paddings_front,
                        const uint32_t* paddings_back) {
  uint32_t pad_result_cols = (input->type.shape.dims[1] + paddings_front[0] +
                              paddings_back[0] - window_dims[0]) /
                                 strides[0] +
                             1;
  uint32_t pad_result_rows = (input->type.shape.dims[2] + paddings_front[1] +
                              paddings_back[1] - window_dims[1]) /
                                 strides[1] +
                             1;

  if ((pad_result_cols == input->type.shape.dims[1]) &&
      (pad_result_rows == input->type.shape.dims[2])) {
    return "SAME";
  } else {
    return "VALID";
  }
}

size_t GetBiasChannels(const odla_value_shape dims) {
  return std::accumulate(dims.dims, dims.dims + dims.size, 1,
                         std::multiplies<size_t>());
}

size_t GetChannels(odla_value input, odla_memory_layout input_layout) {
  if (GetAscendFormat(input_layout) == FORMAT_NCHW) {
    return input->type.shape.dims[1];
  } else if (GetAscendFormat(input_layout) == FORMAT_NHWC) {
    return input->type.shape.dims[3];
  } else
    return input->type.shape.dims[1];
}

bool ChannelCompare(const odla_value_shape dims, odla_value input,
                    odla_memory_layout input_layout) {
  return (GetBiasChannels(dims) == GetChannels(input, input_layout));
}

// op:Add
odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x1_tensor = lhs->op;
  ge::Operator x2_tensor = rhs->op;

  auto add = op::Add(name).set_input_x1(x1_tensor).set_input_x2_by_name(
      x2_tensor, rhs->outputname);

  ge::DataType ascend_type = GetAscendType(lhs->type);
  TensorDesc add_input_desc_x1(ge::Shape(), FORMAT_ND, ascend_type);
  TensorDesc add_input_desc_x2(ge::Shape(), FORMAT_ND, ascend_type);
  TensorDesc add_output_desc_y(ge::Shape(), FORMAT_ND, ascend_type);

  add.update_input_desc_x1(add_input_desc_x1);
  add.update_input_desc_x2(add_input_desc_x2);
  add.update_output_desc_y(add_output_desc_y);

  return CreateValue(add, lhs->type, id);
}

// op:Conv
odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  const uint32_t* strides_ = reinterpret_cast<const uint32_t*>(strides);
  const uint32_t* dilations_ = reinterpret_cast<const uint32_t*>(dilations);
  const uint32_t* paddings_front_ =
      reinterpret_cast<const uint32_t*>(paddings_front);
  const uint32_t* paddings_back_ =
      reinterpret_cast<const uint32_t*>(paddings_back);
  uint32_t stride[4] = {1, 1, 1, 1};

  if (GetAscendFormat(input_layout) == FORMAT_NCHW) {
    stride[2] = strides_[0];
    stride[3] = strides_[1];
  } else if (GetAscendFormat(input_layout) == FORMAT_NHWC) {
    stride[1] = strides_[0];
    stride[2] = strides_[1];
  }
  const uint32_t dilation[4] = {G_UINT32_1, dilations_[0], dilations_[1],
                                G_UINT32_1};
  const uint32_t pad[4] = {paddings_front_[0], paddings_front_[1],
                           paddings_back_[0], paddings_back_[1]};

  ge::Operator x_tensor = input->op;
  ge::Operator fliter_tensor = kernel->op;

  vector<int64_t> stride_vetor;
  vector<int64_t> pad_vetor;
  vector<int64_t> dilation_vetor;
  for (auto i = 0; i < 4; i++) {
    stride_vetor.emplace_back(stride[i]);
    pad_vetor.emplace_back(pad[i]);
    dilation_vetor.emplace_back(dilation[i]);
  }

  auto conv2d = op::Conv2D(name)
                    .set_input_x_by_name(x_tensor, input->outputname)
                    .set_input_filter(fliter_tensor)
                    .set_attr_strides(stride_vetor)
                    .set_attr_pads(pad_vetor)
                    .set_attr_dilations(dilation_vetor);

  if (group > 1) {
    conv2d.set_attr_groups((uint32_t)group);
  }
  if (bias != nullptr) {
    ge::Operator bias_tensor = bias->op;
    conv2d.set_input_bias(bias_tensor);
  }

  TensorDesc conv2d_input_desc_x(ge::Shape(), GetAscendFormat(input_layout),
                                 GetAscendType(input->type));
  TensorDesc conv2d_input_desc_filter(
      ge::Shape(), GetKernelFormat(kernel_layout), GetAscendType(kernel->type));
  TensorDesc conv2d_output_desc_y(ge::Shape(), GetAscendFormat(input_layout),
                                  GetAscendType(input->type));

  conv2d.update_input_desc_x(conv2d_input_desc_x);
  conv2d.update_input_desc_filter(conv2d_input_desc_filter);
  conv2d.update_output_desc_y(conv2d_output_desc_y);

  return CreateValue(conv2d, {input->type.element_type, output_dims}, id);
}

// op:BatchNorm
odla_value odla_BatchNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id value_id) {
  const char* name = reinterpret_cast<const char*>(value_id);

  ge::Operator x_tensor = input->op;
  ge::Operator scale_tensor = scale->op;
  ge::Operator offset_tensor = offset->op;
  ge::Operator mean_tensor = mean->op;
  ge::Operator variance_tensor = var->op;

  auto batchnorm = op::BatchNorm(name)
                       .set_input_x_by_name(x_tensor, input->outputname)
                       .set_input_scale(scale_tensor)
                       .set_input_offset(offset_tensor)
                       .set_input_mean(mean_tensor)
                       .set_input_variance(variance_tensor)
                       .set_attr_epsilon(epsilon)
                       .set_attr_data_format(GetAscendFmtString(input_layout))
                       .set_attr_is_training(false);

  TensorDesc batchnorm_input_desc_x(ge::Shape(), GetAscendFormat(input_layout),
                                    GetAscendType(input->type));
  TensorDesc batchnorm_input_desc_scale(
      ge::Shape(), GetAscendFormat(input_layout), GetAscendType(scale->type));
  TensorDesc batchnorm_input_desc_offset(
      ge::Shape(), GetAscendFormat(input_layout), GetAscendType(offset->type));
  TensorDesc batchnorm_input_desc_mean(
      ge::Shape(), GetAscendFormat(input_layout), GetAscendType(mean->type));
  TensorDesc batchnorm_input_desc_variance(
      ge::Shape(), GetAscendFormat(input_layout), GetAscendType(var->type));
  TensorDesc batchnorm_output_desc_y(ge::Shape(), GetAscendFormat(input_layout),
                                     GetAscendType(input->type));

  batchnorm.update_input_desc_x(batchnorm_input_desc_x);
  batchnorm.update_input_desc_scale(batchnorm_input_desc_scale);
  batchnorm.update_input_desc_offset(batchnorm_input_desc_offset);
  batchnorm.update_input_desc_mean(batchnorm_input_desc_mean);
  batchnorm.update_input_desc_variance(batchnorm_input_desc_variance);
  batchnorm.update_output_desc_y(batchnorm_output_desc_y);

  return CreateValue(batchnorm, input->type, value_id);
}

// op:Relu
odla_value odla_Relu(odla_value input, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);

  ge::Operator x_tensor = input->op;
  auto relu = op::Relu(name).set_input_x_by_name(x_tensor, input->outputname);

  TensorDesc relu_input_desc_x(ge::Shape(), FORMAT_ND,
                               GetAscendType(input->type));
  TensorDesc relu_output_desc_y(ge::Shape(), FORMAT_ND,
                                GetAscendType(input->type));

  relu.update_input_desc_x(relu_input_desc_x);
  relu.update_output_desc_y(relu_output_desc_y);

  return CreateValue(relu, input->type, id);
}

// op:ReduceMean
odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x_tensor = input->op;

  TensorDesc axes_desc(ge::Shape({(long int)num_of_axes}), FORMAT_ND, DT_INT32);
  Tensor axes_tensor(axes_desc, (uint8_t*)axes, num_of_axes * sizeof(int));

  auto axes_op =
      op::Const((string(name) + "_axes").c_str()).set_attr_value(axes_tensor);

  auto reducemean = op::ReduceMean(name)
                        .set_input_x(x_tensor)
                        .set_input_axes(axes_op)
                        .set_attr_keep_dims(keep_dims);

  TensorDesc reducemean_input_desc_x(ge::Shape(), FORMAT_NCHW,
                                     GetAscendType(input->type));
  TensorDesc reducemean_input_desc_axes(ge::Shape(), FORMAT_ND, DT_UINT32);
  TensorDesc reducemean_output_desc_y(ge::Shape(), FORMAT_NCHW,
                                      GetAscendType(input->type));

  reducemean.update_input_desc_x(reducemean_input_desc_x);
  reducemean.update_input_desc_axes(reducemean_input_desc_axes);
  reducemean.update_output_desc_y(reducemean_output_desc_y);

  return CreateValue(reducemean, input->type, id);
}

// op:Reshape
odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x_tensor = input->op;

  TensorDesc shape_desc(ge::Shape({output_dims.size}), FORMAT_ND, DT_INT64);
  Tensor shape_tensor(shape_desc, (uint8_t*)output_dims.dims,
                      output_dims.size * sizeof(int64_t));

  auto shape =
      op::Const((string(name) + "_shape").c_str()).set_attr_value(shape_tensor);

  auto reshape = op::Reshape(name)
                     .set_input_x_by_name(x_tensor, input->outputname)
                     .set_input_shape(shape);

  ge::DataType input_type = GetAscendType(input->type);
  TensorDesc reshape_input_desc_x(ge::Shape(), FORMAT_ND, input_type);
  TensorDesc reshape_input_desc_shape(ge::Shape(), FORMAT_ND, input_type);
  TensorDesc reshape_output_desc_y(ge::Shape(), FORMAT_ND, input_type);

  reshape.update_input_desc_x(reshape_input_desc_x);
  reshape.update_input_desc_shape(reshape_input_desc_shape);
  reshape.update_output_desc_y(reshape_output_desc_y);

  return CreateValue(reshape, {input->type.element_type, output_dims}, id);
}

// op:MatMulV2
odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  const char* name = reinterpret_cast<const char*>(id);
  ge::Operator x1_tensor = lhs->op;
  ge::Operator x2_tensor = rhs->op;
  ge::Operator bias_tensor = bias->op;

  auto matmul = op::MatMulV2(name)
                    .set_input_x1(x1_tensor)
                    .set_input_x2(x2_tensor)
                    .set_input_bias(bias_tensor)
                    .set_attr_transpose_x1(transpose_lhs)
                    .set_attr_transpose_x2(transpose_rhs);

  TensorDesc matmul_input_desc_x1(ge::Shape(), FORMAT_ND,
                                  GetAscendType(lhs->type));
  TensorDesc matmul_input_desc_x2(ge::Shape(), FORMAT_ND,
                                  GetAscendType(rhs->type));
  TensorDesc matmul_input_desc_bias(ge::Shape(), FORMAT_ND,
                                    GetAscendType(bias->type));
  TensorDesc matmul_output_desc_y(ge::Shape(), FORMAT_ND,
                                  GetAscendType(lhs->type));

  matmul.update_input_desc_x1(matmul_input_desc_x1);
  matmul.update_input_desc_x2(matmul_input_desc_x2);
  matmul.update_input_desc_bias(matmul_input_desc_bias);
  matmul.update_output_desc_y(matmul_output_desc_y);

  return CreateValue(matmul,
                     odla_value_type{lhs->type.element_type, output_dims}, id);
}

odla_status odla_DestroyComputation(odla_computation comp) {
  static bool isDestroyed = false;
  if (isDestroyed) return ODLA_FAILURE;

  isDestroyed = true;

  model.Unload();
  model.DestroyDesc();

  aclError ret;

  if (g_comp->acl_ctx != nullptr) {
    ret = aclrtDestroyContext(g_comp->acl_ctx);
    if (ret != ACL_ERROR_NONE) {
      ERROR_LOG("Destroy acl context failed");
    }
    g_comp->acl_ctx = nullptr;
  }
  INFO_LOG("Destroy acl context success");

  ret = aclrtResetDevice(0);
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("reset device failed");
  }
  INFO_LOG("Reset device success");

  ret = aclFinalize();
  if (ret != ACL_ERROR_NONE) {
    ERROR_LOG("finalize acl failed");
  }
  INFO_LOG("Finalize acl success");

  for (auto& c : g_comps) {
    if (c.get() == comp) {
      c.reset();
      return ODLA_SUCCESS;
    }
  }
  return ODLA_FAILURE;
}

odla_status odla_CreateComputation(odla_computation* computation) {
  g_comps.push_back(
      std::unique_ptr<_odla_computation>(new _odla_computation()));
  g_comp = g_comps.back().get();
  *computation = g_comp;

  const char aclConfigPath[32] = {0};
  // 1.acl init
  aclError acl_ret = aclInit(aclConfigPath);
  if (acl_ret != ACL_ERROR_NONE) {
    ERROR_LOG("acl init failed");
    return ODLA_FAILURE;
  }
  INFO_LOG("acl init success");

  acl_ret = aclrtSetDevice(0);
  if (acl_ret != ACL_ERROR_NONE) {
    ERROR_LOG("odla_CreateContext aclrtSetDevice %d", acl_ret);
    return ODLA_FAILURE;
  }

  acl_ret = aclrtCreateContext(&(g_comp->acl_ctx), 0);
  if (acl_ret != ACL_ERROR_NONE) {
    ERROR_LOG("odla_CreateContext aclrtCreateContext %d", acl_ret);
    return ODLA_FAILURE;
  }
  INFO_LOG("create acl context success");

  return ODLA_SUCCESS;
}

// graph inputs
odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  vector<int64_t> shape_data0 = {};
  // create input tensor
  for (auto i = 0; i < type.shape.size; i++) {
    shape_data0.emplace_back(type.shape.dims[i]);
  }
  const char* name = reinterpret_cast<const char*>(id);
  ge::DataType dType = GetAscendType(type);
  TensorDesc desc_data0(ge::Shape(shape_data0), FORMAT_ND, dType);

  // create data operator
  auto data = op::Data(name);
  data.update_input_desc_x(desc_data0);
  data.update_output_desc_y(desc_data0);
  std::vector<Operator> inputs{data};

  odla_value v = CreateValue(data, type, id);
  g_comp->inputs[name] = v;

  return v;
}

// graph output
odla_status odla_SetValueAsOutput(const odla_value val) {
  std::vector<ge::Operator> outputs{(val->op)};
  if (g_comp) {
    std::vector<ge::Operator> inputs;
    unordered_map<string, odla_value>::iterator iter;
    for (iter = g_comp->inputs.begin(); iter != g_comp->inputs.end(); iter++) {
      auto input = (g_comp->inputs[iter->first]->op);
      inputs.push_back(input);
    }

    g_comp->graph.SetInputs(inputs).SetOutputs(outputs);
    g_comp->outputs[val->name] = val;
  }

  return ODLA_SUCCESS;
}

void PrepareOptions(std::map<AscendString, AscendString>& options) {}

odla_status odla_CreateContext(odla_context* context) {
  *context = new _odla_context(g_comp);

  // 2. system init
  std::map<ge::AscendString, ge::AscendString> global_options = {
      {AscendString(ge::ir_option::SOC_VERSION), "Ascend310P3"},
  };
  auto status = aclgrphBuildInitialize(global_options);
  if (status == ACL_ERROR_NONE) {
    INFO_LOG("aclgrphBuildInitialize success");
  } else {
    ERROR_LOG("aclgrphBuildInitialize failed");
  }

  // 3. Build Ir Model
  std::map<ge::AscendString, ge::AscendString> options;
  PrepareOptions(options);

  // create model, get modelid
  status = aclgrphBuildModel(g_comp->graph, options, g_comp->ModelBufferData_);
  if (status == ACL_ERROR_NONE) {
    INFO_LOG("Build Model success");
  } else {
    ERROR_LOG("Build Model failed");
  }

  // 3.5 Save Model
  status = aclgrphSaveModel("model.om", g_comp->ModelBufferData_);
  if (status == ACL_ERROR_NONE) {
    INFO_LOG("Save Model success");
  } else {
    ERROR_LOG("Save Model failed");
  }

  // 4. Dump Graph
  status = aclgrphDumpGraph(g_comp->graph, "test_graph", 10);
  if (status == ACL_ERROR_NONE) {
    INFO_LOG("Dump Graph success");
  } else {
    ERROR_LOG("Dump Graph failed");
  }

  odla_status ret = model.LoadModelFromWithMem(g_comp->ModelBufferData_);
  if (ret == ODLA_FAILURE) {
    ERROR_LOG("odla LoadModelFromWithMem failed");
    return ODLA_FAILURE;
  }

  ret = model.CreateDesc();
  if (ret == ODLA_FAILURE) {
    ERROR_LOG("odla CreateDesc failed");
    return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context context) {
  delete context;
  return ODLA_SUCCESS;
}

// bind input address
odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  std::vector<Tensor> input_tensors;
  const char* name = reinterpret_cast<const char*>(value_id);
  odla_value input_val = context->comp->inputs[name];
  auto bs_size_weight = 1;
  vector<int64_t> input_shape;

  for (auto i = 0; i < input_val->type.shape.size; i++) {
    bs_size_weight *= input_val->type.shape.dims[i];
    input_shape.emplace_back(input_val->type.shape.dims[i]);
  }

  size_t bufferSize = bs_size_weight * GetElementSize(input_val->type);
  g_comp->input_ptr = (uint8_t*)data_ptr;
  odla_status ret = model.CreateInput(g_comp->input_ptr, bufferSize);
  if (ret == ODLA_FAILURE) {
    ERROR_LOG("odla BindToArgumentById failed");
    return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

// bind output address
odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  std::vector<ge::Tensor> output_tensors;
  const char* name = reinterpret_cast<const char*>(value_id);

  odla_value output_val = context->comp->outputs[name];
  auto bs_size_weight = 1;
  vector<int64_t> output_shape;

  for (auto i = 0; i < output_val->type.shape.size; i++) {
    bs_size_weight *= output_val->type.shape.dims[i];
    output_shape.emplace_back(output_val->type.shape.dims[i]);
  }

  g_comp->output_ptr = (uint8_t*)data_ptr;

  odla_status ret = model.CreateOutput();
  if (ret == ODLA_FAILURE) {
    ERROR_LOG("odla BindToOutputById failed");
    return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

// model exec and free resource
odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  odla_status ret = model.Execute();
  if (ret == ODLA_FAILURE) {
    ERROR_LOG("odla_ExecuteComputation Execute failed");
    return ODLA_FAILURE;
  }

  model.DumpModelOutputResult(g_comp->output_ptr);
  model.DestroyInput();
  model.DestroyOutput();
  return ODLA_SUCCESS;
}
