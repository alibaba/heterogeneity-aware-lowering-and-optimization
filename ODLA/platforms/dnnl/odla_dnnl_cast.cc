//===- odla_dnnl_cast.cc --------------------------------------------------===//
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

#include <climits>
#include <cstring>
#include <string>

#include "ODLA/odla_common.h"
#include "odla_dnnl.h"

dnnl::memory cast_odla_mem(dnnl::memory src_mem, const odla_value_shape shape,
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

dnnl::memory cast_op(odla_value& input, dnnl::memory::data_type dt) {
  auto src_md = dnnl::memory::desc(getDims(input->shape),
                                   input->mem.get_desc().data_type(),
                                   getFormatTag(input->shape));
  auto src_mem =
      dnnl::memory(src_md, g_comp->eng, input->mem.get_data_handle());
  return cast_odla_mem(input->mem, input->shape, dt, input->is_const);
}

struct Float {
  // TODO(unknown): no infinity, underflow/overflow handling.
 private:
  Float() = delete;
  static constexpr int BitsPerInt = CHAR_BIT * sizeof(int);
  template <typename T, int exp, int mantissa>
  static std::array<int, 3> Extract(T x) {
    static_assert(exp + mantissa + 1 == sizeof(T) * CHAR_BIT);
    int sign = x >> (exp + mantissa);
    int m = x & ((1 << mantissa) - 1);
    int e = (x >> mantissa) & ((1 << exp) - 1);
    return {sign, e, m};
  }

  template <typename T, int exp, int mantissa>
  static T Combine(int sign, int e, int m) {
    static_assert(exp + mantissa + 1 == sizeof(T) * CHAR_BIT);
    T x{0};
    x = sign ? 1U << (exp + mantissa) : 0;
    m >>= BitsPerInt - mantissa;
    x |= m & ((1U << mantissa) - 1);
    x |= (e & ((1U << exp) - 1)) << mantissa;
    return x;
  }
  static constexpr int FP32Exp = 8;
  static constexpr int FP32Mantissa = 23;
  static constexpr int FP32ExpBias = 127;
  static constexpr int FP16Exp = 5;
  static constexpr int FP16Mantissa = 10;
  static constexpr int FP16ExpBias = 15;

  static inline float GetFP32(uint8_t sign, int32_t e, uint32_t m) {
    uint32_t x =
        Combine<uint32_t, FP32Exp, FP32Mantissa>(sign, e + FP32ExpBias, m);
    return *(reinterpret_cast<float*>(&x)); // NOLINT.
  }
  static inline uint16_t GetFP16(uint8_t sign, int32_t e, uint32_t m) {
    return Combine<uint16_t, FP16Exp, FP16Mantissa>(sign, e + FP16ExpBias, m);
  }

 public:
  static inline uint16_t GetFP16(float x) {
    uint32_t v = *(reinterpret_cast<int*>(&x)); // NOLINT.
    auto components = Extract<uint32_t, FP32Exp, FP32Mantissa>(v);
    components[1] -= FP32ExpBias;
    components[2] <<= BitsPerInt - FP32Mantissa;
    return GetFP16(components[0], components[1], components[2]);
  }

  static inline float GetFP32(uint16_t x) {
    auto components = Extract<uint16_t, FP16Exp, FP16Mantissa>(x);
    components[1] -= FP16ExpBias;
    components[2] <<= BitsPerInt - FP16Mantissa;
    return GetFP32(components[0], components[1], components[2]);
  }
};

static auto unsupported_cast = []() { assert(0 && "UNSUPPORTED"); };

static odla_value cast_to_string(odla_value input, odla_value output,
                                 size_t n) {
  void* buf = output->mem.get_data_handle();
  auto to_str = [input, buf, n]() {
    const char** dst = static_cast<const char**>(buf);
    if (input->elem_type == ODLA_FLOAT32) {
      const float* src =
          static_cast<const float*>(input->mem.get_data_handle());
      for (int i = 0; i < n; ++i) {
        const std::string& str = std::to_string((src[i])); // NOLINT.
        auto str_buf =
            static_cast<char*>(g_comp->CreateBuffer(str.size() + 1)); // NOLINT.
        strcpy(str_buf, str.c_str());                                 // NOLINT.
        dst[i] = str_buf;
      }
    } else {
      for (int i = 0; i < n; ++i) {
        dst[i] = "UNSUPPORTED"; // NOLINT.
      }
    }
  };
  add_op(to_str);
  InterpretIfNeeded();
  return output;
}

static odla_value cast_from_string(odla_value input, odla_value output,
                                   size_t n) {
  if (output->elem_type == ODLA_FLOAT32) {
    float* dst = static_cast<float*>(output->mem.get_data_handle());
    auto str_to_val = [input, dst, n]() {
      const char** strs =
          static_cast<const char**>(input->mem.get_data_handle());
      float* vals = static_cast<float*>(dst);
      for (int i = 0; i < n; ++i) {
        dst[i] = std::stof(strs[i]); // NOLINT.
      }
    };
    add_op(str_to_val);
  } else {
    add_op(unsupported_cast);
  }
  InterpretIfNeeded();
  return output;
}

static odla_value cast_to_f32(odla_value input, odla_value output, size_t n) {
  float* dst = static_cast<float*>(output->mem.get_data_handle());
  if (input->elem_type == ODLA_FLOAT16) {
    add_op([input, dst, n]() {
      const uint16_t* src =
          static_cast<const uint16_t*>(input->mem.get_data_handle());
      for (int i = 0; i < n; ++i) {
        dst[i] = Float::GetFP32(src[i]); // NOLINT.
      }
    });
  } else if (input->elem_type == ODLA_FLOAT64) {
    add_op([input, dst, n]() {
      const double* src =
          static_cast<const double*>(input->mem.get_data_handle());
      float* vals = static_cast<float*>(dst);
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i]; // NOLINT.
      }
    });
  } else {
    add_op(unsupported_cast);
  }
  InterpretIfNeeded();
  return output;
}

static odla_value cast_to_f64(odla_value input, odla_value output, size_t n) {
  double* dst = static_cast<double*>(output->mem.get_data_handle());
  if (input->elem_type == ODLA_FLOAT16) {
    add_op([input, dst, n]() {
      const uint16_t* src =
          static_cast<const uint16_t*>(input->mem.get_data_handle());
      for (int i = 0; i < n; ++i) {
        // TOOD: change to cast(cast(fp32), fp64).
        dst[i] = Float::GetFP32(src[i]); // NOLINT.
      }
    });
  } else if (input->elem_type == ODLA_FLOAT32) {
    add_op([input, dst, n]() {
      const float* src =
          static_cast<const float*>(input->mem.get_data_handle());
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i]; // NOLINT.
      }
    });
  } else {
    add_op(unsupported_cast);
  }
  InterpretIfNeeded();
  return output;
}

static odla_value cast_to_f16(odla_value input, odla_value output, size_t n) {
  uint16_t* dst = static_cast<uint16_t*>(output->mem.get_data_handle());
  if (input->elem_type == ODLA_FLOAT32) {
    add_op([input, dst, n]() {
      const float* src =
          static_cast<const float*>(input->mem.get_data_handle());
      for (int i = 0; i < n; ++i) {
        dst[i] = Float::GetFP16(src[i]); // NOLINT.
      }
    });
  } else if (input->elem_type == ODLA_FLOAT64) {
    add_op([input, dst, n]() {
      const double* src =
          static_cast<const double*>(input->mem.get_data_handle());
      for (int i = 0; i < n; ++i) {
        // Narrowing first. TODO://
        dst[i] = Float::GetFP16(src[i]); // NOLINT.
      }
    });
  } else {
    add_op(unsupported_cast);
  }
  InterpretIfNeeded();
  return output;
}

odla_value odla_Cast(odla_value input, odla_element_type target_type,
                     const odla_value_id id) {
  int n = GetTotalElements(input->shape);
  // Prepare dest memory.
  dnnl::memory dst_mem;
  dnnl::memory::desc dst_md = getMemoryDesc({target_type, input->shape});
  if (hasDNNLMemorySupport(target_type) || target_type == ODLA_FLOAT16) {
    dst_mem = dnnl::memory(dst_md, g_comp->eng);
  } else {
    auto buf = g_comp->CreateBuffer(getElementStorageSize(target_type) * n);
    dst_mem = dnnl::memory(dst_md, g_comp->eng, buf);
  }
  auto v = CreateValue(dst_mem, input->shape, id);
  v->elem_type = target_type;

  // Use DNNL built-in cast.
  if (hasDNNLMemorySupport(input->elem_type) &&
      hasDNNLMemorySupport(target_type)) {
    add_op(dnnl::reorder(input->mem, dst_mem),
           {{DNNL_ARG_FROM, input->mem}, {DNNL_ARG_TO, dst_mem}});
    InterpretIfNeeded();
    return v;
  }

  if (target_type == ODLA_STRING) {
    return cast_to_string(input, v, n);
  }

  if (input->elem_type == ODLA_STRING) {
    return cast_from_string(input, v, n);
  }
  if (target_type == ODLA_FLOAT16) {
    return cast_to_f16(input, v, n);
  }

  if (target_type == ODLA_FLOAT32) {
    return cast_to_f32(input, v, n);
  }

  if (target_type == ODLA_FLOAT64) {
    return cast_to_f64(input, v, n);
  }
  assert(0 && "unsupported");
  return v;
}
