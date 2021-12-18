#include <ODLA/odla.h>
#include <dlfcn.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ODLA/odla_common.h"
#include "ODLA/odla_value.h"

static void* handle;
static std::ofstream ofs_;
static int rec_cnt_; // count the number of records in a run
static int run_cnt_; // count the number of runs in a profiling
static std::string dump_path_;

struct Init {
  Init() {
    static const std::string default_lib_file{"libodla_dnnl_interpreter.so"};
    // TODO(unknown): allow user to choose computation backend.
    handle = dlopen(default_lib_file.c_str(), RTLD_NOW);
  }
};

static Init init;

static int channel_axis = -1;
static bool enable_chs_prof = false;
static bool skip_weight = false;

static int64_t GetTotalElements(const odla_value_shape& dims) {
  return std::accumulate(dims.dims, dims.dims + dims.size, 1,
                         std::multiplies<size_t>());
}

static size_t GetSize(const odla_value_type type) {
  return sizeof(float) * GetTotalElements(type.shape);
}

template <typename T>
odla_value_id GetValueId(T arg) {
  return nullptr;
}
template <>
odla_value_id GetValueId<odla_value_id>(odla_value_id arg) {
  return arg;
}

template <typename T, typename... Targs>
static odla_value_id GetValueId(T arg, Targs... args) {
  if constexpr (std::is_same<T, odla_value_id>()) {
    return arg;
  }
  return GetValueId(args...);
}

template <const char* fn_name, typename RetTy, typename... Args>
static RetTy dispatch(Args... args) {
  RetTy (*fn)(Args...);
  static std::unordered_map<void*, decltype(fn)> fns;
  assert(handle != nullptr);
  if (!fns[handle]) {
    fns[handle] = (decltype(fn))dlsym(handle, fn_name);
  }
  fn = fns[handle];
  if (fn == nullptr) {
    std::cerr << dlerror();
    assert(0);
  }
  return fn(args...);
}

static std::ostream& OS() { return ofs_.good() ? ofs_ : std::cout; }

static odla_value_type GetValueType(const odla_value v) {
  odla_value_type vt;
  odla_GetValueType(v, &vt);
  return vt;
}

template <typename T>
static std::string to_list(const T& vals) {
  std::ostringstream oss;
  oss << "[";
  bool is_first = true;
  for (auto x : vals) {
    if (!is_first) {
      oss << ", ";
    }
    oss << x;
    is_first = false;
  }
  oss << "]";
  return oss.str();
};

static void DumpValue(const std::string& fn_name, const void* ptr,
                      odla_value_type type, const char* id) {
  std::ofstream ofs;
  auto file_name = dump_path_ + "/" + std::string(id) + ".dump";
  ofs.open(file_name, std::ios_base::trunc);
  if (!ofs.good()) {
    std::cerr << "Unable to dump to " << file_name << std::endl;
    return;
  }
  std::vector<int> s(type.shape.dims, type.shape.dims + type.shape.size);

  ofs << "// Type: " << fn_name << "\n";
  ofs << "// Shape: " << to_list(s) << "\n";
  const float* data = reinterpret_cast<const float*>(ptr);
  size_t len = GetTotalElements(type.shape);
  for (size_t i = 0; i < len; ++i) {
    ofs << data[i] << ",\n";
  }

  ofs.close();
}

odla_status odla_ProfileValue(const std::string& fn_name, const void* ptr,
                              odla_value_type type, const char* id) {
  if (skip_weight && fn_name == "odla_CreateConstant") {
    return ODLA_SUCCESS;
  }

  if (!dump_path_.empty()) {
    DumpValue(fn_name, ptr, type, id);
  }
  const float* data = reinterpret_cast<const float*>(ptr);
  size_t len = GetTotalElements(type.shape);
  float min_value = *std::min_element(data, data + len);
  float max_value = *std::max_element(data, data + len);
  auto dims = type.shape.size;
  const auto& shape = type.shape.dims;
  int axis = (channel_axis < 0) ? dims + channel_axis : channel_axis;

  // Do channel-wise profiling
  std::vector<float> ch_min;
  std::vector<float> ch_max;
  if (enable_chs_prof && axis >= 0 && axis < dims) {
    auto chs = shape[axis];
    ch_max = std::vector<float>(chs, std::numeric_limits<float>::lowest());
    ch_min = std::vector<float>(chs, std::numeric_limits<float>::max());
    size_t stride = 1;
    for (int i = axis + 1; i < dims; ++i) {
      stride *= shape[i];
    }
    for (size_t i = 0; i < len; ++i) {
      size_t ch = (i / stride) % chs;
      ch_min[ch] = std::min(ch_min[ch], data[i]);
      ch_max[ch] = std::max(ch_max[ch], data[i]);
    }
  }
  // int valid_range = 255;
  // float scale = (max_value - min_value) / (float)valid_range;
  // zp = 0;
  // if (scale != 0) {
  //  zp = -(int)((min_value) / scale);
  // }
  if (rec_cnt_ > 0) {
    OS() << ',';
  }
  auto to_vec = [](const odla_value_shape& shape) {
    std::vector<int> s(shape.dims, shape.dims + shape.size);
    return s;
  };

  OS() << "\n\"" << id << "\" : {";
  OS() << "\"op_type\": \"" << fn_name << "\", ";
  OS() << "\"shape\": " << to_list(to_vec(type.shape)) << ", ";
  OS() << "\"channels\": " << ch_min.size() << ", ";
  OS() << "\"min_value\": " << min_value << ", ";
  OS() << "\"max_value\": " << max_value << ", ";
  OS() << "\"channels_min\": " << to_list(ch_min) << ", ";
  OS() << "\"channels_max\": " << to_list(ch_max);
  OS() << "}";
  ++rec_cnt_;
  return ODLA_SUCCESS;
}

template <const char* fn_name, typename... Args>
static odla_value profile(Args... args) {
  odla_value ret = dispatch<fn_name, odla_value>(args...);
  const auto& vt = GetValueType(ret);
  odla_value_id id = GetValueId(args...);
  assert(id != nullptr);
  assert(vt.element_type == ODLA_FLOAT32);
  std::vector<float> buf(GetTotalElements(vt.shape));
  odla_GetValueData(ret, buf.data());
  odla_ProfileValue(fn_name, buf.data(), vt, reinterpret_cast<char*>(id));
  return ret;
}

extern "C" {

// If the axis is non-negative, it will profile data on specified axis.
void EnableChannelWiseProf() { enable_chs_prof = true; }
void SetChannelAxis(int axis) { channel_axis = axis; }
void SkipWeightsProfiling(bool skip) { skip_weight = skip; }

void StartProfiling(const char* data_path) {
  rec_cnt_ = 0;
  run_cnt_ = 0;
  if (data_path != nullptr) {
    ofs_.open(data_path, std::ios_base::trunc);
    if (!ofs_.good()) {
      std::cerr << "Unable to open profiling result file " << data_path
                << std::endl;
    }
  }

  if (const char* env_p = std::getenv("ODLA_PROFILER_DUMP_PATH")) {
    dump_path_ = std::string(env_p);
  }

  if (!dump_path_.empty()) {
    std::cerr << "Dump all results to " << dump_path_ << "\n";
  }

  OS() << "{\n\"ProfilingResults\": {\n";
}

void StartOneRun(const char* tag) {
  if (run_cnt_ > 0) {
    OS() << ',';
  }
  ++run_cnt_;
  OS() << '"' << tag << "\" : {\n";
  rec_cnt_ = 0;
}
void StopOneRun() { OS() << "}\n"; }

void StopProfiling() {
  OS() << "}}";
  if (ofs_.good()) {
    ofs_.close();
  }
}

static std::unordered_map<odla_value, odla_value_type> g_value_types;
static std::unordered_map<odla_value, const char*> g_value_ids;

static constexpr const char fn_cv[] = "odla_CreateValue";
odla_value odla_CreateValue(const odla_value_type value_type,
                            const odla_value_id value_id) {
  odla_value ret = dispatch<fn_cv, odla_value>(value_type, value_id);
  g_value_types[ret] = value_type;
  g_value_ids[ret] = reinterpret_cast<char*>(value_id);
  return ret;
}

static constexpr const char fn_sv[] = "odla_SetValueData";
odla_status odla_SetValueData(odla_value value, const odla_void* data_ptr) {
  odla_ProfileValue(fn_sv, data_ptr, g_value_types[value], g_value_ids[value]);
  return dispatch<fn_sv, odla_status>(value, data_ptr);
}

static constexpr const char fn_gt[] = "odla_GetValueType";
odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  return dispatch<fn_gt, odla_status>(value, value_type);
}

static constexpr const char fn_gvd[] = "odla_GetValueData";
odla_status odla_GetValueData(const odla_value value, odla_void* data_ptr) {
  return dispatch<fn_gvd, odla_status>(value, data_ptr);
}

static constexpr const char fn_add[] = "odla_Add";
odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return profile<fn_add>(lhs, rhs, id);
}

static constexpr const char fn_argmax[] = "odla_ArgMax";
odla_value odla_ArgMax(odla_value input, odla_int32 axis, odla_bool keep_dims,
                       odla_bool return_last_index,
                       odla_value_type output_value_type,
                       const odla_value_id value_id) {
  return profile<fn_argmax>(input, axis, keep_dims, return_last_index,
                            output_value_type, value_id);
}

static constexpr const char fn_ap[] = "odla_AveragePool";
odla_value odla_AveragePool(odla_value input, odla_memory_layout input_layout,
                            const odla_uint32* window_dims,
                            const odla_uint32* strides,
                            const odla_uint32* paddings_front,
                            const odla_uint32* paddings_back,
                            odla_value_shape output_dims,
                            const odla_value_id value_id) {
  return profile<fn_ap>(input, input_layout, window_dims, strides,
                        paddings_front, paddings_back, output_dims, value_id);
}

static constexpr const char fn_bn[] = "odla_BatchNormalization";
odla_value odla_BatchNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id value_id) {
  return profile<fn_bn>(input, input_layout, mean, var, epsilon, scale, offset,
                        scalar_scale, scalar_offset, value_id);
}

static constexpr const char fn_cast[] = "odla_Cast";
odla_value odla_Cast(odla_value input, odla_element_type target_type,
                     const odla_value_id id) {
  return profile<fn_cast>(input, target_type, id);
}

static constexpr const char fn_clamp[] = "odla_Clamp";
odla_value odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
                      const odla_value_id id) {
  return profile<fn_clamp>(input, lo, hi, id);
}

static constexpr const char fn_concat[] = "odla_Concat";
odla_value odla_Concat(odla_values inputs, odla_int32 axis,
                       odla_value_shape output_dims, const odla_value_id id) {
  return profile<fn_concat>(inputs, axis, output_dims, id);
}

static constexpr const char fn_conv[] = "odla_Conv";
odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  auto ret = profile<fn_conv>(input, input_layout, group, kernel, kernel_layout,
                              strides, dilations, paddings_front, paddings_back,
                              bias, output_dims, id);
  return ret;
}

static constexpr const char fn_ca[] = "odla_CreateArgument";
odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  return dispatch<fn_ca, odla_value>(type, id);
}

static constexpr const char fn_cn[] = "odla_CreateConstant";
odla_value odla_CreateConstant(odla_value_type type, const void* ptr,
                               const odla_value_id id) {
  return profile<fn_cn>(type, ptr, id);
}

static constexpr const char fn_deconv[] = "odla_DeConv";
odla_value odla_DeConv(odla_value input, odla_memory_layout input_layout,
                       odla_uint32 group, odla_value kernel,
                       odla_memory_layout kernel_layout,
                       const odla_uint32* strides, const odla_uint32* dilations,
                       const odla_uint32* paddings_front,
                       const odla_uint32* paddings_back, odla_value bias,
                       odla_value_shape output_dims, const odla_value_id id) {
  return profile<fn_deconv>(input, input_layout, group, kernel, kernel_layout,
                            strides, dilations, paddings_front, paddings_back,
                            bias, output_dims, id);
}

static constexpr const char fn_div[] = "odla_Div";
odla_value odla_Div(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return profile<fn_div>(lhs, rhs, id);
}

static constexpr const char fn_exp[] = "odla_Exp";
odla_value odla_Exp(odla_value input, const odla_value_id id) {
  return profile<fn_exp>(input, id);
}

static constexpr const char fn_erf[] = "odla_Erf";
odla_value odla_Erf(odla_value input, const odla_value_id id) {
  return profile<fn_erf>(input, id);
}

static constexpr const char fn_floor[] = "odla_Floor";
odla_value odla_Floor(odla_value input, const odla_value_id id) {
  return profile<fn_floor>(input, id);
}

static constexpr const char fn_gather[] = "odla_Gather";
odla_value odla_Gather(odla_value params, const odla_value indices,
                       odla_int32 axis, odla_value_shape output_dims,
                       const odla_value_id id) {
  return profile<fn_gather>(params, indices, axis, output_dims, id);
}

static constexpr const char fn_gemm[] = "odla_Gemm";
odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  return profile<fn_gemm>(lhs, transpose_lhs, rhs, transpose_rhs, alpha, beta,
                          bias, output_dims, id);
}

static constexpr const char fn_leakyrelu[] = "odla_LeakyRelu";
odla_value odla_LeakyRelu(odla_value input, odla_float32 alpha,
                          const odla_value_id id) {
  return profile<fn_leakyrelu>(input, alpha, id);
}

static constexpr const char fn_lrn[] = "odla_LRN";
odla_value odla_LRN(odla_value input, odla_memory_layout input_layout,
                    odla_int32 window_size, odla_float32 alpha,
                    odla_float32 beta, odla_float32 bias,
                    const odla_value_id value_id) {
  return profile<fn_lrn>(input, input_layout, window_size, alpha, beta, bias,
                         value_id);
}

static constexpr const char fn_mp[] = "odla_MaxPool";
odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id) {
  return profile<fn_mp>(input, input_layout, window_dims, strides,
                        paddings_front, paddings_back, output_dims, value_id);
}

static constexpr const char fn_mul[] = "odla_Mul";
odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return profile<fn_mul>(lhs, rhs, id);
}

static constexpr const char fn_nms[] = "odla_NMS";
odla_value odla_NMS(odla_value input_boxes, odla_value input_scores,
                    odla_uint32 max_num_outputs, odla_float32 iou_threshold,
                    odla_float32 score_threshold,
                    odla_value_type output_value_type,
                    const odla_value_id value_id) {
  return profile<fn_nms>(input_boxes, input_scores, max_num_outputs,
                         iou_threshold, score_threshold, output_value_type,
                         value_id);
}

static constexpr const char fn_prelu[] = "odla_PRelu";
odla_value odla_PRelu(odla_value input, odla_value slope,
                      const odla_value_id id) {
  return profile<fn_prelu>(input, slope, id);
}

static constexpr const char fn_relu[] = "odla_Relu";
odla_value odla_Relu(odla_value input, const odla_value_id id) {
  return profile<fn_relu>(input, id);
}

static constexpr const char fn_reshape[] = "odla_Reshape";
odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id) {
  return profile<fn_reshape>(input, output_dims, id);
}

static constexpr const char fn_reduce_max[] = "odla_ReduceMax";
odla_value odla_ReduceMax(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return profile<fn_reduce_max>(input, num_of_axes, axes, keep_dims,
                                output_dims, id);
}

static constexpr const char fn_rm[] = "odla_ReduceMean";
odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  return profile<fn_rm>(input, num_of_axes, axes, keep_dims, output_dims, id);
}

static constexpr const char fn_reduce_min[] = "odla_ReduceMin";
odla_value odla_ReduceMin(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return profile<fn_reduce_min>(input, num_of_axes, axes, keep_dims,
                                output_dims, id);
}

static constexpr const char fn_reduce_sum[] = "odla_ReduceSum";
odla_value odla_ReduceSum(odla_value input, odla_size_t num_of_axes,
                          const odla_uint32* axes, odla_bool keep_dims,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return profile<fn_reduce_sum>(input, num_of_axes, axes, keep_dims,
                                output_dims, id);
}

static constexpr const char fn_resize[] = "odla_Resize";
odla_value odla_Resize(odla_value input, odla_interpolation_mode interpolation,
                       odla_resize_coordinate_mode mode, odla_uint32 axes_mask,
                       odla_value_shape output_dims, const odla_value_id id) {
  return profile<fn_resize>(input, interpolation, mode, axes_mask, output_dims,
                            id);
}

static constexpr const char fn_round[] = "odla_Round";
odla_value odla_Round(odla_value input, const odla_value_id id) {
  return profile<fn_round>(input, id);
}

static constexpr const char fn_rsqrt[] = "odla_Rsqrt";
odla_value odla_Rsqrt(odla_value input, const odla_value_id id) {
  return profile<fn_rsqrt>(input, id);
}

static constexpr const char fn_slice[] = "odla_Slice";
odla_value odla_Slice(odla_value input, const odla_int32* start,
                      const odla_int32* end, const odla_int32* strides,
                      odla_value_shape output_dims, const odla_value_id id) {
  return profile<fn_slice>(input, start, end, strides, output_dims, id);
}

static constexpr const char fn_sqrt[] = "odla_Sqrt";
odla_value odla_Sqrt(odla_value input, const odla_value_id id) {
  return profile<fn_sqrt>(input, id);
}

static constexpr const char fn_sm[] = "odla_Softmax";
odla_value odla_Softmax(odla_value input, odla_int32 axis,
                        const odla_value_id id) {
  return profile<fn_sm>(input, axis, id);
}

static constexpr const char fn_sg[] = "odla_Sigmoid";
odla_value odla_Sigmoid(odla_value input, const odla_value_id id) {
  return profile<fn_sg>(input, id);
}

static constexpr const char fn_sub[] = "odla_Sub";
odla_value odla_Sub(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return profile<fn_sub>(lhs, rhs, id);
}

static constexpr const char fn_tile[] = "odla_Tile";
odla_value odla_Tile(odla_value input, const odla_uint32* repeat,
                     odla_value_shape output_dims,
                     const odla_value_id value_id) {
  return profile<fn_tile>(input, repeat, output_dims, value_id);
}

static constexpr const char fn_tr[] = "odla_Transpose";
odla_value odla_Transpose(odla_value input, odla_value_shape permutations,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  return profile<fn_tr>(input, permutations, output_dims, id);
}

} // end of extern
