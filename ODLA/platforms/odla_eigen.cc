//===- odla_eigen.cc ------------------------------------------------------===//
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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

#if __cplusplus < 201103L
#error This library needs at least a C++11 compliant compiler
#endif
#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#endif

//#define EIGEN_DEFAULT_TO_ROW_MAJOR
//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE
#if defined(NDEBUG) && !defined(EIGEN_NO_DEBUG)
#define EIGEN_NO_DEBUG
#endif
//#define EIGEN_USE_MKL_ALL
#include <ODLA/odla.h>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

struct _odla_value {
  _odla_value(const odla_value_type& type, void* p)
      : type(type), ptr(p), needs_free(false) {}
  _odla_value(const odla_value_type& type, size_t t)
      : type(type), ptr(new char[t]), needs_free(true) {}

  ~_odla_value() {
    if (needs_free) {
      delete[](char*) ptr;
    }
  }
  odla_value_type type;
  void* ptr;
  bool needs_free;
};

std::vector<std::unique_ptr<_odla_value>> Vals;

static int64_t GetTotalElements(const odla_value_shape& dims) {
  return std::accumulate(dims.dims, dims.dims + dims.size, 1,
                         std::multiplies<size_t>());
}

static unsigned GetElementSize(odla_element_type type) {
  switch (type) {
    case ODLA_FLOAT32:
      return sizeof(float);
    case ODLA_FLOAT16:
      return sizeof(int16_t);
    case ODLA_INT32:
      return sizeof(int32_t);
    case ODLA_INT8:
      return 1;
    default:
      return 0;
  }
}

static size_t GetValueSize(odla_element_type type,
                           const odla_value_shape& dims) {
  return GetElementSize(type) * GetTotalElements(dims);
}

static size_t GetValueSize(odla_value_type type) {
  return GetElementSize(type.element_type) * GetTotalElements(type.shape);
}

static odla_value GetValue(const odla_value_type& type, void* ptr) {
  auto v = std::make_unique<_odla_value>(type, ptr);
  Vals.push_back(std::move(v));
  return Vals.back().get();
}

static odla_value GetValue(const odla_value_type& type) {
  auto v = std::make_unique<_odla_value>(type, GetValueSize(type));
  Vals.push_back(std::move(v));
  return Vals.back().get();
}

template <typename T, int RANK>
struct EigenTensorHelper {};

template <typename T>
struct EigenTensorHelper<T, 4> {
  using EigenTensorMap = typename Eigen::TensorMap<
      Eigen::Tensor<T, 4, Eigen::RowMajor, Eigen::DenseIndex>>;
  static EigenTensorMap GetEigenTensorMap(void* ptr,
                                          const odla_value_shape& dims) {
    assert(dims.size == 4);
    return {static_cast<T*>(ptr), static_cast<int>(dims.dims[0]),
            static_cast<int>(dims.dims[1]), static_cast<int>(dims.dims[2]),
            static_cast<int>(dims.dims[3])};
  }
};

template <typename T>
struct EigenTensorHelper<T, 2> {
  using EigenTensorMap = typename Eigen::TensorMap<
      Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>>;
  static EigenTensorMap GetEigenTensorMap(void* ptr,
                                          const odla_value_shape& dims) {
    assert(dims.size == 2);
    return {static_cast<T*>(ptr), static_cast<int>(dims.dims[0]),
            static_cast<int>(dims.dims[1])};
  }
};

template <typename T>
struct EigenTensorHelper<T, 1> {
  using EigenTensorMap = typename Eigen::TensorMap<
      Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static EigenTensorMap GetEigenTensorMap(void* ptr,
                                          const odla_value_shape& dims) {
    return {static_cast<T*>(ptr), static_cast<int>(GetTotalElements(dims))};
  }
  static EigenTensorMap GetEigenTensorMap(void* ptr, size_t len) {
    return {static_cast<T*>(ptr), static_cast<int>(len)};
  }
};

extern "C" {
odla_value odla_CreateValue(odla_value_type type, const odla_value_id id) {
  return GetValue(type, nullptr);
}
odla_status odla_SetValueData(odla_value val, const void* ptr) {
  val->ptr = const_cast<void*>(ptr); // FIXME
  return ODLA_SUCCESS;
}

static odla_value DepthwiseConvolution(
    odla_element_type type, odla_value_shape& input_dims,
    odla_memory_layout input_layout, odla_value input,
    odla_value_shape kernel_dims, odla_memory_layout kernel_layout,
    odla_value kernel, const unsigned* strides, const unsigned* dilations,
    const unsigned* paddings_front, const unsigned* paddings_back,
    unsigned group, odla_value bias, odla_value_shape& output_dims) {
  auto v = GetValue({type, output_dims});
  int data_ch_idx = (input_layout == ODLA_CHANNELS_LAST) ? 3 : 1;
  // assert(input_layout == ODLA_CHANNELS_FIRST && kernel_layout == ODLA_OIS);

  int batch = input_dims.dims[0];
  auto h = input_dims.dims[input_layout == ODLA_CHANNELS_LAST ? 1 : 2];
  auto w = input_dims.dims[input_layout == ODLA_CHANNELS_LAST ? 2 : 3];
  auto out_h = output_dims.dims[(input_layout == ODLA_CHANNELS_LAST) ? 1 : 2];
  auto out_w = output_dims.dims[(input_layout == ODLA_CHANNELS_LAST) ? 2 : 3];

  auto k_h = kernel_dims.dims[(kernel_layout == ODLA_SIO) ? 0 : 2];
  auto k_w = kernel_dims.dims[(kernel_layout == ODLA_SIO) ? 1 : 3];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_t = paddings_front[0];
  int pad_b = paddings_back[0];
  int pad_l = paddings_front[1];
  int pad_r = paddings_back[1];
  int in_ch = input_dims.dims[data_ch_idx];
  int out_ch = output_dims.dims[data_ch_idx];
  assert(group = in_ch);
  // assert(kernel_dims.dims[(kernel_layout == ODLA_SIO) ? 2 : 1] == 1);
  assert(in_ch == out_ch);

  float* in_ptr = static_cast<float*>(input->ptr);
  float* out_ptr = static_cast<float*>(v->ptr);
  if (input_layout == ODLA_CHANNELS_LAST) { // NHWC
    auto in =
        EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, input_dims);
    auto kn = EigenTensorHelper<float, 4>::GetEigenTensorMap(kernel->ptr,
                                                             kernel_dims);
    auto ret =
        EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, output_dims);

    auto out = in.extract_image_patches(k_w, k_h, stride_h, stride_w, 1, 1, 1,
                                        1, pad_l, pad_r, pad_t, pad_b, 0)
                   .reshape(Eigen::array<long, 2>{out_h * out_w * batch,
                                                  k_w * k_h * in_ch});
    // Element wise multiplication with kernel
    auto out_elt_mul =
        out * ((kn.reshape(Eigen::array<long, 2>{1, k_w * k_w * in_ch}))
                   .broadcast(Eigen::array<long, 2>{out_h * out_w * batch, 1}));
    // Reduced sum on every kernel spatial dims
    auto out_reduce =
        out_elt_mul
            .reshape(
                Eigen::array<long, 3>{out_h * out_w * batch, k_w * k_h, in_ch})
            .sum(Eigen::array<int, 1>{1})
            .reshape(Eigen::array<long, 4>{batch, out_h, out_w, out_ch});
    if (bias != nullptr) {
      ret = out_reduce + EigenTensorHelper<float, 4>::GetEigenTensorMap(
                             bias->ptr, {1, 1, 1, out_ch});
    } else {
      ret = out_reduce;
    }
    return Vals.back().get();
  }

  odla_value_shape local_output_dims{.size = 2, {out_h, out_w}};
  odla_value_shape local_input_dims{.size = 4, {1, h, w, 1}};
  odla_value_shape local_kernel_dims{.size = 2, {k_h * k_w, 1}};

  for (int b = 0; b < batch; ++b) {
    float* kn_ptr = static_cast<float*>(kernel->ptr);
    //#pragma omp parallel
    for (int c = 0; c < in_ch; ++c) {
      auto ret = EigenTensorHelper<float, 2>::GetEigenTensorMap(
          out_ptr, local_output_dims);
      auto in = EigenTensorHelper<float, 4>::GetEigenTensorMap(
          in_ptr, local_input_dims);
      auto kn = EigenTensorHelper<float, 2>::GetEigenTensorMap(
          kn_ptr, local_kernel_dims);

      ret = in.extract_image_patches(k_w, k_h, stride_h, stride_w, 1, 1, 1, 1,
                                     pad_l, pad_r, pad_t, pad_b, 0)
                .reshape(Eigen::array<int, 2>{
                    static_cast<int>(out_h) * static_cast<int>(out_w),
                    static_cast<int>(k_w) * static_cast<int>(k_h)})
                .contract(kn,
                          Eigen::array<Eigen::IndexPair<int>, 1>{
                              Eigen::IndexPair<int>(1, 0)})
                .reshape(Eigen::array<int, 2>{static_cast<int>(out_h),
                                              static_cast<int>(out_w)});
      kn_ptr += k_h * k_w;
      in_ptr += h * w;
      out_ptr += out_h * out_w;
    }
  }
  if (bias != nullptr) {
    auto ret =
        EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, output_dims);
    odla_value_shape bias_shape{.size = 4, .dims = {1, out_ch, 1, 1}};

    ret = ret +
          EigenTensorHelper<float, 4>::GetEigenTensorMap(bias->ptr, bias_shape)
              .broadcast(Eigen::array<int, 4>{batch, 1, static_cast<int>(out_h),
                                              static_cast<int>(out_w)});
  }

  return Vals.back().get();
}
odla_value odla_Conv(odla_value input, odla_memory_layout input_layout,
                     odla_uint32 group, odla_value kernel,
                     odla_memory_layout kernel_layout,
                     const odla_uint32* strides, const odla_uint32* dilations,
                     const odla_uint32* paddings_front,
                     const odla_uint32* paddings_back, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  kernel_layout = (input_layout == odla_memory_layout::ODLA_CHANNELS_FIRST)
                      ? odla_memory_layout::ODLA_OIS
                      : odla_memory_layout::ODLA_SIO;

  auto& input_dims = input->type.shape;
  auto& kernel_dims = kernel->type.shape;
  if (group > 1) {
    return DepthwiseConvolution(input->type.element_type, input_dims,
                                input_layout, input, kernel_dims, kernel_layout,
                                kernel, strides, dilations, paddings_front,
                                paddings_back, group, bias, output_dims);
  }

  auto v = GetValue({input->type.element_type, output_dims});
  int data_ch_idx = (input_layout == ODLA_CHANNELS_LAST) ? 3 : 1;
  // assert(input_layout == ODLA_CHANNELS_LAST && kernel_layout == SIO);

  auto in =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, input_dims);
  auto kn =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(kernel->ptr, kernel_dims);
  int k_h = kernel_dims.dims[(kernel_layout == ODLA_SIO) ? 0 : 2];
  int k_w = kernel_dims.dims[(kernel_layout == ODLA_SIO) ? 1 : 3];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_t = paddings_front[0];
  int pad_b = paddings_back[0];
  int pad_l = paddings_front[1];
  int pad_r = paddings_back[1];
  int in_ch = input_dims.dims[data_ch_idx];
  int out_ch = output_dims.dims[data_ch_idx];
  int batch = input_dims.dims[0];
  int out_h = output_dims.dims[(input_layout == ODLA_CHANNELS_LAST) ? 1 : 2];
  int out_w = output_dims.dims[(input_layout == ODLA_CHANNELS_LAST) ? 2 : 3];

  auto ret =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, output_dims);
  if (input_layout == ODLA_CHANNELS_FIRST && k_h == 1 && k_w == 1 &&
      stride_h == 1 && stride_w == 1) {
    Eigen::array<int, 2> perm{1, 0};
    auto out = in.reshape(Eigen::array<int, 2>{in_ch, out_h * out_w})
                   .contract(kn.reshape(Eigen::array<int, 2>({out_ch, in_ch})),
                             Eigen::array<Eigen::IndexPair<int>, 1>{
                                 Eigen::IndexPair<int>(0, 1)})
                   .shuffle(perm)
                   .reshape(Eigen::array<int, 4>{batch, out_ch, out_h, out_w});
    if (bias != nullptr) {
      ret = out + EigenTensorHelper<float, 4>::GetEigenTensorMap(
                      bias->ptr, {.size = 4, .dims = {1, out_ch, 1, 1}})
                      .broadcast(Eigen::array<int, 4>{batch, 1, out_h, out_w});
    } else {
      ret = out;
    }
    return Vals.back().get();
  }
  if (input_layout == ODLA_CHANNELS_LAST) { // NHWC
    auto out =
        in.extract_image_patches(k_w, k_h, stride_h, stride_w, 1, 1, 1, 1,
                                 pad_l, pad_r, pad_t, pad_b, 0)
            .reshape(
                Eigen::array<int, 2>{out_h * out_w * batch, in_ch * k_w * k_h})
            .contract(
                kn.reshape(Eigen::array<int, 2>({in_ch * k_w * k_h, out_ch})),
                Eigen::array<Eigen::IndexPair<int>, 1>{
                    Eigen::IndexPair<int>(1, 0)})
            .reshape(Eigen::array<int, 4>{batch, out_h, out_w, out_ch});
    if (bias != nullptr) {
      ret = out + EigenTensorHelper<float, 4>::GetEigenTensorMap(
                      bias->ptr, {1, 1, 1, out_ch});
    } else {
      ret = out;
    }
  } else {
    Eigen::array<int, 4> kernel_shuffles{2, 3, 1, 0};
    Eigen::array<int, 4> input_shuffles{0, 2, 3, 1};
    Eigen::array<int, 4> output_shuffles{0, 3, 1, 2};
    auto out =
        in.shuffle(input_shuffles)
            .extract_image_patches(k_w, k_h, stride_h, stride_w, 1, 1, 1, 1,
                                   pad_l, pad_r, pad_t, pad_b, 0)
            .reshape(
                Eigen::array<int, 2>{out_h * out_w * batch, in_ch * k_w * k_h})
            .contract(
                kn.shuffle(kernel_shuffles)
                    .reshape(Eigen::array<int, 2>({in_ch * k_w * k_h, out_ch})),
                Eigen::array<Eigen::IndexPair<int>, 1>{
                    Eigen::IndexPair<int>(1, 0)})
            .reshape(Eigen::array<int, 4>{batch, out_h, out_w, out_ch});
    if (bias != nullptr) {
      odla_value_shape bias_shape{.size = 4, .dims = {1, out_ch, 1, 1}};
      ret =
          out.shuffle(output_shuffles) +
          EigenTensorHelper<float, 4>::GetEigenTensorMap(bias->ptr, bias_shape)
              .broadcast(Eigen::array<int, 4>{batch, 1, out_h, out_w});
    } else {
      ret = out.shuffle(output_shuffles);
    }
  }
  return Vals.back().get();
}

odla_value odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
                      const odla_value_id id) {
  const auto& dims = input->type.shape;
  auto in = EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, dims);
  auto v = GetValue(input->type);

  auto ret = EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, dims);
  ret = in.cwiseMax(static_cast<float>(lo)).cwiseMin(static_cast<float>(hi));
  return v;
}

odla_value odla_Relu(odla_value input, const odla_value_id id) {
  const auto& dims = input->type.shape;
  auto in = EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, dims);
  auto v = GetValue(input->type);

  auto ret = EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, dims);
  ret = in.cwiseMax(static_cast<float>(0));
  return v;
}

odla_value odla_LeakyRelu(odla_value input, odla_float32 alpha,
                          const odla_value_id id) {
  const auto& dims = input->type.shape;
  auto in = EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, dims);
  auto v = GetValue(input->type);

  auto ret = EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, dims);
  ret = in.cwiseMax(in * alpha);
  return v;
}

enum class Op { ADD, MUL };

static odla_value odla_binary(Op op, odla_value lhs, odla_value rhs,
                              const odla_value_id id) {
  const auto& dims_lhs = lhs->type.shape;
  auto dims_rhs = rhs->type.shape;

  auto v = GetValue(lhs->type);
  auto l = EigenTensorHelper<float, 4>::GetEigenTensorMap(lhs->ptr, dims_lhs);
  auto ret = EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, dims_lhs);

  if (GetTotalElements(dims_lhs) != GetTotalElements(dims_rhs)) {
    assert(dims_lhs.size == 4);
    if (dims_rhs.size == 1) {
      dims_rhs.size = 4;
      dims_rhs.dims[3] = dims_rhs.dims[0];
      dims_rhs.dims[0] = dims_rhs.dims[1] = dims_rhs.dims[2] = 1;
    } else if (dims_rhs.size != 4) {
      assert(0 && "invalid dim");
    }
    auto r = EigenTensorHelper<float, 4>::GetEigenTensorMap(rhs->ptr, dims_rhs);
    int d0 = dims_lhs.dims[0];
    int d1 = dims_lhs.dims[1];
    int d2 = dims_lhs.dims[2];
    int d3 = dims_lhs.dims[3];

    auto r2 = r.broadcast(Eigen::array<int, 4>{
        d0 / (int)dims_rhs.dims[0], d1 / (int)dims_rhs.dims[1],
        d2 / (int)dims_rhs.dims[2], d3 / (int)dims_rhs.dims[3]});
    if (op == Op::ADD) {
      ret = l + r2;
    } else if (op == Op::MUL) {
      ret = l * r2;
    }
  } else {
    auto r = EigenTensorHelper<float, 4>::GetEigenTensorMap(rhs->ptr, dims_lhs);
    if (op == Op::ADD) {
      ret = l + r;
    } else if (op == Op::MUL) {
      ret = l * r;
    }
  }
  return v;
}

odla_value odla_Add(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return odla_binary(Op::ADD, lhs, rhs, id);
}

odla_value odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id id) {
  return odla_binary(Op::MUL, lhs, rhs, id);
}

odla_value odla_BatchNormalization(odla_value input,
                                   odla_memory_layout input_layout,
                                   odla_value mean, odla_value var,
                                   odla_float32 epsilon, odla_value scale,
                                   odla_value offset, odla_float32 scalar_scale,
                                   odla_float32 scalar_offset,
                                   const odla_value_id value_id) {
  const auto& input_dims = input->type.shape;
  // assert(input_layout == ODLA_CHANNELS_LAST);
  int d0 = input_dims.dims[0];
  int d1 = input_dims.dims[1];
  int d2 = input_dims.dims[2];
  int d3 = input_dims.dims[3];

  auto v = GetValue(input->type);
  auto ret = EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, input_dims);

  auto input_v =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, input_dims);

  Eigen::DSizes<int, 4> bd_s(d0, d1, d2, 1);
  odla_value_shape dim{.size = 4, .dims = {1, 1, 1, input_dims.dims[3]}};
  if (input_layout == ODLA_CHANNELS_FIRST) {
    bd_s = Eigen::DSizes<int, 4>(d0, 1, d2, d3);
    dim = odla_value_shape{.size = 4, .dims = {1, input_dims.dims[1], 1, 1}};
  }

  auto mean_v = EigenTensorHelper<float, 4>::GetEigenTensorMap(mean->ptr, dim);

  auto var_v = EigenTensorHelper<float, 4>::GetEigenTensorMap(var->ptr, dim);
  auto r = (input_v - mean_v.broadcast(bd_s)) *
           ((var_v + epsilon).rsqrt().broadcast(bd_s));
  assert(scale);
  assert(offset);
  if (scale) {
    auto scale_v =
        EigenTensorHelper<float, 4>::GetEigenTensorMap(scale->ptr, dim);
    auto offset_v =
        EigenTensorHelper<float, 4>::GetEigenTensorMap(offset->ptr, dim);
    ret = r * scale_v.broadcast(bd_s) + offset_v.broadcast(bd_s);
  } else {
    ret = r * scalar_scale + scalar_offset;
  }
  return v;
}

static odla_value odla_Pooling(
    bool is_max, odla_value input, odla_memory_layout input_layout,
    const odla_uint32* window_dims, const odla_uint32* strides,
    const odla_uint32* paddings_front, const odla_uint32* paddings_back,
    odla_value_shape output_dims, const odla_value_id value_id) {
  const auto& input_dims = input->type.shape;
  auto v = GetValue({input->type.element_type, output_dims});
  auto ret =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, output_dims);

  auto input_v =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, input_dims);

  int win_h = window_dims[0];
  int win_w = window_dims[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_top = paddings_front[0];
  int pad_b = paddings_back[0];
  int pad_l = paddings_front[1];
  int pad_r = paddings_back[1];
  int chs = input_dims.dims[3];
  int batch = input_dims.dims[0];
  int out_h = output_dims.dims[1];
  int out_w = output_dims.dims[2];

  if (input_layout == ODLA_CHANNELS_LAST) {
    auto t = input_v
                 .extract_image_patches(win_w, win_h, stride_w, stride_h, 1, 1,
                                        1, 1, pad_l, pad_r, pad_top, pad_b,
                                        std::numeric_limits<float>::lowest())
                 .reshape(Eigen::array<int, 5>{batch, out_h, out_w,
                                               win_h * win_w, chs});
    if (is_max) {
      ret = t.maximum(Eigen::array<int, 1>{3});
    } else {
      ret = t.mean(Eigen::array<int, 1>{3});
    }
  } else {
    chs = input_dims.dims[1];
    int h = input_dims.dims[2];
    int w = input_dims.dims[3];
    out_h = output_dims.dims[2];
    out_w = output_dims.dims[3];
    auto out = input_v.reshape(Eigen::array<int, 5>{batch, chs, h, w, 1})
                   .extract_image_patches(win_w, win_h, stride_w, stride_h, 1,
                                          1, 1, 1, pad_l, pad_r, pad_top, pad_b,
                                          std::numeric_limits<float>::lowest())
                   .reshape(Eigen::array<int, 6>{batch, chs, out_h, out_w,
                                                 win_h * win_w, 1});
    if (is_max) {
      ret = out.maximum(Eigen::array<int, 1>{4})
                .reshape(Eigen::array<int, 4>{batch, chs, out_h, out_w});
    } else {
      ret = out.mean(Eigen::array<int, 1>{4})
                .reshape(Eigen::array<int, 4>{batch, chs, out_h, out_w});
    }
  }
  return v;
}

odla_value odla_Concat(odla_values inputs, odla_int32 axis,
                       odla_value_shape output_dims, const odla_value_id id) {
  auto val = GetValue({inputs.values[0]->type.element_type, output_dims});
  assert(inputs.values[0]->type.element_type == ODLA_FLOAT32);
  assert(inputs.size == 2);
  auto ret =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(val->ptr, output_dims);

  auto input_a = EigenTensorHelper<float, 4>::GetEigenTensorMap(
      inputs.values[0]->ptr, inputs.values[0]->type.shape);
  auto input_b = EigenTensorHelper<float, 4>::GetEigenTensorMap(
      inputs.values[1]->ptr, inputs.values[1]->type.shape);

  ret = input_a.concatenate(input_b, axis);

  /*
  assert(axis == 3 && output_dims.size == 4);
  int s = 1;
  for (int i = 0; i < inputs.values[0]->type.shape.size - 1; ++i) {
    s *= inputs.values[0]->type.shape.dims[i];
  }
  float* dst = reinterpret_cast<float*>(val->ptr);
  for (int i = 0; i < s; ++i) {
    for (int j = 0; j < inputs.size; ++j) {
      int ch = inputs.values[j]->type.shape.dims[3];
      const float* src = reinterpret_cast<const
  float*>(inputs.values[j]->ptr); memcpy(dst, src + i * ch, sizeof(float) *
  ch); dst += ch;
    }
  }
  */
  return val;
}

odla_value odla_Resize(odla_value input, odla_interpolation_mode interpolation,
                       odla_resize_coordinate_mode mode, odla_uint32 axes_mask,
                       odla_value_shape output_dims,
                       const odla_value_id value_id) {
  assert(input->type.element_type == ODLA_FLOAT32);
  odla_value val = GetValue({input->type.element_type, output_dims});
  assert(interpolation == ODLA_NEAREST);
  assert(input->type.shape.size == 4 && axes_mask == -1);
  int out_h = output_dims.dims[1];
  int out_w = output_dims.dims[2];
  int ch = output_dims.dims[3];
  int in_h = input->type.shape.dims[1];
  int in_w = input->type.shape.dims[2];

  float* dst_ptr = (float*)val->ptr;
  assert(ch == input->type.shape.dims[3]);
  size_t copy_size = sizeof(float) * ch;
  const float* src_ptr = (float*)input->ptr;
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

odla_value odla_MaxPool(odla_value input, odla_memory_layout input_layout,
                        const odla_uint32* window_dims,
                        const odla_uint32* strides,
                        const odla_uint32* paddings_front,
                        const odla_uint32* paddings_back,
                        odla_value_shape output_dims,
                        const odla_value_id value_id) {
  return odla_Pooling(true, input, input_layout, window_dims, strides,
                      paddings_front, paddings_back, output_dims, value_id);
}

odla_value odla_AveragePool(odla_value input, odla_memory_layout input_layout,
                            const odla_uint32* window_dims,
                            const odla_uint32* strides,
                            const odla_uint32* paddings_front,
                            const odla_uint32* paddings_back,
                            odla_value_shape output_dims,
                            const odla_value_id value_id) {
  return odla_Pooling(false, input, input_layout, window_dims, strides,
                      paddings_front, paddings_back, output_dims, value_id);
}

odla_value odla_Softmax(odla_value input, odla_int32 axis,
                        const odla_value_id id) {
  auto v = GetValue(input->type);
  const auto& dims = input->type.shape;
  auto ret = EigenTensorHelper<float, 2>::GetEigenTensorMap(v->ptr, dims);

  auto input_v =
      EigenTensorHelper<float, 2>::GetEigenTensorMap(input->ptr, dims);

  Eigen::DSizes<int, 2> shape(dims.dims[0], 1);
  Eigen::DSizes<int, 2> bd(1, dims.dims[1]);

  auto r = (input_v - input_v.maximum(Eigen::DSizes<int, 1>{1})
                          .eval()
                          .reshape(shape)
                          .broadcast(bd))
               .exp();
  ret = r * (r.sum(Eigen::DSizes<int, 1>{1})
                 .inverse()
                 .eval()
                 .reshape(shape)
                 .broadcast(bd));
  return v;
}

odla_value odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                           const odla_uint32* axes, odla_bool keep_dims,
                           odla_value_shape output_dims,
                           const odla_value_id id) {
  auto v = GetValue({input->type.element_type, output_dims});
  const auto& dims = input->type.shape;
  auto input_v =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, dims);
  Eigen::array<int, 2> reduction_axes;
  for (int i = 0; i < 2; ++i) {
    reduction_axes[i] = axes[i];
  }
  auto r = input_v.mean(reduction_axes);
  if (output_dims.size == dims.size) {
    int d0 = output_dims.dims[0];
    int d1 = output_dims.dims[1];
    int d2 = output_dims.dims[2];
    int d3 = output_dims.dims[3];
    auto ret =
        EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, output_dims);

    ret = r.reshape(Eigen::array<int, 4>{d0, d1, d2, d3});
  } else {
    auto ret =
        EigenTensorHelper<float, 2>::GetEigenTensorMap(v->ptr, output_dims);

    ret = r;
  }
  return v;
}

odla_value odla_Reshape(odla_value input, odla_value_shape output_dims,
                        const odla_value_id id) {
  return GetValue({input->type.element_type, output_dims}, input->ptr);
}

odla_value odla_Gemm(odla_value lhs, odla_bool transpose_lhs, odla_value rhs,
                     odla_bool transpose_rhs, odla_float32 alpha,
                     odla_float32 beta, odla_value bias,
                     odla_value_shape output_dims, const odla_value_id id) {
  const auto& lhs_dims = lhs->type.shape;
  const auto& rhs_dims = rhs->type.shape;
  assert(lhs_dims.size == 2);
  auto A = EigenTensorHelper<float, 2>::GetEigenTensorMap(lhs->ptr, lhs_dims);
  auto B = EigenTensorHelper<float, 2>::GetEigenTensorMap(rhs->ptr, rhs_dims);

  Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};
  if (transpose_lhs && transpose_rhs) {
    dims[0] = Eigen::IndexPair<int>{0, 1};
  } else if (transpose_lhs) {
    dims[0] = Eigen::IndexPair<int>{0, 0};
  } else if (transpose_rhs) {
    dims[0] = Eigen::IndexPair<int>{1, 1};
  }

  auto v = GetValue({lhs->type.element_type, output_dims});
  auto ret =
      EigenTensorHelper<float, 2>::GetEigenTensorMap(v->ptr, output_dims);
  if (bias) {
    auto C =
        EigenTensorHelper<float, 2>::GetEigenTensorMap(bias->ptr, output_dims);
    ret = A.contract(B, dims) + C;
  } else {
    ret = A.contract(B, dims);
  }
  return v;
}

odla_value odla_Transpose(odla_value input, odla_value_shape permutations,
                          odla_value_shape output_dims,
                          const odla_value_id id) {
  const auto& input_dims = input->type.shape;
  assert(input_dims.size == 4);
  auto v = GetValue({input->type.element_type, output_dims});
  auto in =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(input->ptr, input_dims);
  auto ret =
      EigenTensorHelper<float, 4>::GetEigenTensorMap(v->ptr, output_dims);
  assert(permutations.size == 4);
  Eigen::array<size_t, 4> perm{static_cast<size_t>(permutations.dims[0]),
                               static_cast<size_t>(permutations.dims[1]),
                               static_cast<size_t>(permutations.dims[2]),
                               static_cast<size_t>(permutations.dims[3])};

  ret = in.shuffle(perm);
  return v;
}

odla_value odla_CreateConstant(odla_value_type type, const void* ptr,
                               const odla_value_id id) {
  return GetValue(type, const_cast<void*>(ptr));
}

odla_status odla_GetValueData(const odla_value value, odla_void* data_ptr) {
  memcpy(data_ptr, value->ptr, GetValueSize(value->type));
}

void odla_Dump(odla_value val) {
  int t = 1; // dims.dims[dims.size - 1];
  float* data = static_cast<float*>(val->ptr);
  const odla_value_shape& dims = val->type.shape;
  for (int i = 0, e = GetTotalElements(dims) / t; i < e; ++i) {
    for (int j = 0; j < t; ++j) printf("%.3f", *data++);
    printf("\n");
  }
}

} // C extern