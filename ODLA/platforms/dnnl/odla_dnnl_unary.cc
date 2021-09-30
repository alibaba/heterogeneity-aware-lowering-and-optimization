//===- odla_dnnl_unary.cc ---------------------------------------------===//
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

#include <Eigen/Core>

#include "odla_dnnl.h"

enum class alg_unary_eltwise {
  isnan,
  isinf,
  isinf_pos,
  isinf_neg,
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
  logic_not,
};

static void unary_eltwise_logic(alg_unary_eltwise alg, void* dst,
                                const void* data, int n) {
  const bool* data_t = static_cast<const bool*>(data);
  Eigen::Map<const Eigen::Array<bool, Eigen::Dynamic, 1>> in(data_t, n);
  bool* dst_t = static_cast<bool*>(dst);
  Eigen::Map<Eigen::Array<bool, Eigen::Dynamic, 1>> out(dst_t, n);
  switch (alg) {
    case alg_unary_eltwise::logic_not:
      out = !in;
      break;
    default:
      assert(0);
  }
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
    case alg_unary_eltwise::isinf_neg:
      out = in.isInf() && (in < 0);
      break;
    case alg_unary_eltwise::isinf_pos:
      out = in.isInf() && (in > 0);
      break;
    default:
      assert(0);
  }
}

bool is_unary_bool(alg_unary_eltwise alg) {
  return (alg == alg_unary_eltwise::isnan || alg == alg_unary_eltwise::isinf ||
          alg == alg_unary_eltwise::isinf_neg ||
          alg == alg_unary_eltwise::isinf_pos ||
          alg == alg_unary_eltwise::logic_not);
}

bool is_unary_logic(alg_unary_eltwise alg) {
  return (alg == alg_unary_eltwise::logic_not);
}

static odla_value odla_unary_eltwise(alg_unary_eltwise alg, odla_value input,
                                     const odla_value_id value_id) {
  // Extract type and size
  auto elem_type = input->elem_type;
  bool ret_bool = is_unary_bool(alg);
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
    if (is_unary_logic(alg)) {
      unary_eltwise_logic(alg, dst, data, n);
    } else if (input->elem_type == ODLA_FLOAT32) {
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

odla_value odla_Abs(odla_value input, const odla_value_id value_id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_abs, input, 0.f, 0.f,
                          value_id);
}

odla_value odla_IsNaN(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::isnan, input, value_id);
}

odla_value odla_IsInf(odla_value input, odla_bool detect_pos,
                      odla_bool detect_neg, const odla_value_id value_id) {
  if (detect_pos != 0 && detect_neg != 0) {
    return odla_unary_eltwise(alg_unary_eltwise::isinf, input, value_id);
  }
  if (detect_pos != 0) {
    return odla_unary_eltwise(alg_unary_eltwise::isinf_pos, input, value_id);
  }
  return odla_unary_eltwise(alg_unary_eltwise::isinf_neg, input, value_id);
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

odla_value odla_Tanh(odla_value input, const odla_value_id value_id) {
  return unary_eltwise_op(dnnl::algorithm::eltwise_tanh, input, 0.f, 0.f,
                          value_id);
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

odla_value odla_Not(odla_value input, const odla_value_id value_id) {
  return odla_unary_eltwise(alg_unary_eltwise::logic_not, input, value_id);
}