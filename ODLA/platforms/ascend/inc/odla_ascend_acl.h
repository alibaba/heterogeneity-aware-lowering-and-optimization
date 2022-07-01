#pragma once
#include <ODLA/odla.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>

#include "acl/acl.h"
#include "all_ops.h"
#include "ge/ge_api.h"
#include "model_process.h"

using namespace std;
using namespace ge;
using ge::Operator;

extern const uint32_t G_UINT32_1;
extern thread_local odla_computation g_comp;
extern std::vector<std::unique_ptr<_odla_computation>> g_comps;
extern ModelProcess model;

struct _odla_value {
  odla_value_type type;
  ge::Operator op;
  const char* name;
  const char* outputname;
  _odla_value(ge::Operator& op, const odla_value_type& value_type,
              const char* name, const char* outputname)
      : op(op), name(name), outputname(outputname) {
    type.element_type = value_type.element_type;
    type.shape.size = value_type.shape.size;
    for (int i = 0; i < type.shape.size; i++) {
      type.shape.dims[i] = value_type.shape.dims[i];
    }
  }
};

struct _odla_computation {
  ge::Graph graph;
  std::vector<ge::Operator> ops;
  unordered_map<string, odla_value> inputs;
  unordered_map<string, odla_value> outputs;

  std::vector<std::unique_ptr<_odla_value>> vals;
  ge::ModelBufferData ModelBufferData_;
  uint8_t* input_ptr;
  uint8_t* output_ptr;
  aclrtContext acl_ctx = nullptr;

  _odla_computation() { graph = ge::Graph("IrGraph1"); }

  ~_odla_computation() {
    input_ptr = nullptr;
    output_ptr = nullptr;
    acl_ctx = nullptr;
  }
};

struct _odla_context {
  odla_computation comp;
  std::vector<ge::Tensor> input_tensors;
  std::vector<ge::Tensor> output_tensors;

  int graph_id;
  int run_batch_size = 0;
  _odla_context(odla_computation comp) : comp(comp) {}
  ~_odla_context() { comp = nullptr; }
};

ge::DataType GetAscendType(odla_value_type type);

ge::DataType GetAscendType(odla_element_type type);

size_t GetElementSize(odla_value_type type);

ge::Format GetAscendFormat(odla_memory_layout input_layout);

ge::Format GetKernelFormat(odla_memory_layout input_layout);

std::string GetAscendFmtString(odla_memory_layout input_layout);

int GetElementNums(const odla_value_shape shape);

// odla_value
template <typename T>
static odla_value CreateValue(T op, const odla_value_type type,
                              const odla_value_id id,
                              const char* outputname = "y") {
  const char* name = reinterpret_cast<const char*>(id);
  auto v =
      std::unique_ptr<_odla_value>(new _odla_value(op, type, name, outputname));
  auto ret = v.get();
  g_comp->vals.push_back(std::move(v));

  return ret;
};

odla_status odla_CreateContext(odla_context* context);
odla_status odla_CreateComputation(odla_computation* computation);
odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id);
odla_status odla_SetValueAsOutput(const odla_value val);

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context);
odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context);
odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode, odla_device device);

odla_value odla_CreateConstant(odla_value_type type, const odla_void* data_ptr,
                               const odla_value_id id);
odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id);

odla_value odla_BiasAdd(odla_value input_x, odla_value bias,
                        const odla_value_id id, const char* data_format);

odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id);

odla_value odla_BatchNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id value_id);
odla_value odla_Relu(odla_value input, const odla_value_id id);
odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id);
odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id);
odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id);
odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id);
