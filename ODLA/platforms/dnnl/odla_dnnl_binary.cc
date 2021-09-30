//===- odla_dnnl_binary.cc ---------------------------------------------===//
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

enum class alg_binary_eltwise {
  logic_or,
  logic_and,
  logic_xor,
  cmp_equal,
  cmp_less,
  cmp_less_equal,
  cmp_greater,
  cmp_greater_equal,
  max,
  min,
  mod,
  pow,
  sum,
};

// Input and output are both bool type.
static void binary_eltwise_logic(alg_binary_eltwise alg, void* dst,
                                 const void* data_l, const void* data_r,
                                 int n) {
  const bool* input_l = static_cast<const bool*>(data_l);
  Eigen::Map<const Eigen::Array<bool, Eigen::Dynamic, 1>> in_l(input_l, n);
  const bool* input_r = static_cast<const bool*>(data_r);
  Eigen::Map<const Eigen::Array<bool, Eigen::Dynamic, 1>> in_r(input_r, n);
  bool* dst_t = static_cast<bool*>(dst);
  Eigen::Map<Eigen::Array<bool, Eigen::Dynamic, 1>> out(dst_t, n);
  switch (alg) {
    case alg_binary_eltwise::logic_and:
      out = in_l && in_r;
      break;
    case alg_binary_eltwise::logic_or:
      out = in_l || in_r;
      break;
    case alg_binary_eltwise::logic_xor:
      out = in_l != in_r;
      break;
    default:
      assert(0);
  }
}

// Binary elementwise helper with return type bool
template <typename T>
static void binary_eltwise_bool(alg_binary_eltwise alg, void* dst,
                                const void* data_l, const void* data_r, int n) {
  const T* input_l = static_cast<const T*>(data_l);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> in_l(input_l, n);
  const T* input_r = static_cast<const T*>(data_r);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> in_r(input_r, n);
  bool* dst_t = static_cast<bool*>(dst);
  Eigen::Map<Eigen::Array<bool, Eigen::Dynamic, 1>> out(dst_t, n);
  switch (alg) {
    case alg_binary_eltwise::cmp_equal:
      out = (in_l == in_r);
      break;
    case alg_binary_eltwise::cmp_less:
      out = (in_l < in_r);
      break;
    case alg_binary_eltwise::cmp_less_equal:
      out = (in_l <= in_r);
      break;
    case alg_binary_eltwise::cmp_greater:
      out = (in_l > in_r);
      break;
    case alg_binary_eltwise::cmp_greater_equal:
      out = (in_l >= in_r);
      break;
    default:
      assert(0);
  }
}

// Binary elementwise helper with return type T
template <typename T>
static void binary_eltwise_T(alg_binary_eltwise alg, void* dst,
                             const void* data_l, const void* data_r, int n) {
  const T* input_l = static_cast<const T*>(data_l);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> in_l(input_l, n);
  const T* input_r = static_cast<const T*>(data_r);
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> in_r(input_r, n);
  T* dst_t = static_cast<T*>(dst);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> out(dst_t, n);
  switch (alg) {
    case alg_binary_eltwise::mod:
      out = in_l - (in_r * (in_l / in_r));
      break;
    default:
      assert(0);
  }
}

bool binary_ret_bool(alg_binary_eltwise alg) {
  return (alg == alg_binary_eltwise::logic_or) ||
         (alg == alg_binary_eltwise::logic_and) ||
         (alg == alg_binary_eltwise::logic_xor) ||
         (alg == alg_binary_eltwise::cmp_equal) ||
         (alg == alg_binary_eltwise::cmp_less) ||
         (alg == alg_binary_eltwise::cmp_less_equal) ||
         (alg == alg_binary_eltwise::cmp_greater) ||
         (alg == alg_binary_eltwise::cmp_greater_equal);
}

bool is_binary_logic(alg_binary_eltwise alg) {
  return (alg == alg_binary_eltwise::logic_or) ||
         (alg == alg_binary_eltwise::logic_and) ||
         (alg == alg_binary_eltwise::logic_xor);
}

static odla_value odla_binary_eltwise_bool(alg_binary_eltwise alg,
                                           odla_value lhs, odla_value rhs,
                                           const odla_value_id value_id) {
  assert(lhs->elem_type == rhs->elem_type);
  // Broadcast lhs and rhs
  auto input_type = lhs->elem_type;
  // std::cout << input_type << std::endl;
  // std::cout << ODLA_INT32 << std::endl;
  auto ret_type = binary_ret_bool(alg) ? ODLA_BOOL : input_type;
  odla_value_shape ret_shape;
  auto new_mems = broadcast_operands(lhs, rhs, &ret_shape);
  auto lhs_m = new_mems.first;
  auto rhs_m = new_mems.second;
  // Prepare ret memory
  auto ret_md = getMemoryDesc(ret_shape, ret_type);
  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto n = GetTotalElements(ret_shape);
  // Create lambda func as op
  auto op = [alg, input_type, ret_type, lhs_m, rhs_m, ret_mem, n] {
    void* dst = ret_mem.get_data_handle();
    const void* data_l = lhs_m.get_data_handle();
    const void* data_r = rhs_m.get_data_handle();
    if (is_binary_logic(alg)) {
      binary_eltwise_logic(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_FLOAT32) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<float>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<float>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_FLOAT64) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<double>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<double>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_UINT8) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<uint8_t>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<uint8_t>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_UINT16) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<uint16_t>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<uint16_t>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_UINT32) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<uint32_t>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<uint32_t>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_UINT64) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<uint64_t>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<uint64_t>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_INT8) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<int8_t>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<int8_t>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_INT16) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<int16_t>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<int16_t>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_INT32) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<int32_t>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<int32_t>(alg, dst, data_l, data_r, n);
    } else if (input_type == ODLA_INT64) {
      ret_type == ODLA_BOOL
          ? binary_eltwise_bool<int64_t>(alg, dst, data_l, data_r, n)
          : binary_eltwise_T<int64_t>(alg, dst, data_l, data_r, n);
    } else {
      assert(0);
    }
  };
  // Postprocess
  add_op(op);
  odla_value v = CreateValue(ret_mem, ret_shape, value_id);
  v->elem_type = ret_type;
  InterpretIfNeeded();
  return v;
}

odla_value odla_And(odla_value lhs, odla_value rhs,
                    const odla_value_id value_id) {
  return odla_binary_eltwise_bool(alg_binary_eltwise::logic_and, lhs, rhs,
                                  value_id);
}

odla_value odla_Or(odla_value lhs, odla_value rhs,
                   const odla_value_id value_id) {
  return odla_binary_eltwise_bool(alg_binary_eltwise::logic_or, lhs, rhs,
                                  value_id);
}

odla_value odla_Xor(odla_value lhs, odla_value rhs,
                    const odla_value_id value_id) {
  return odla_binary_eltwise_bool(alg_binary_eltwise::logic_xor, lhs, rhs,
                                  value_id);
}

odla_value odla_Equal(odla_value lhs, odla_value rhs,
                      const odla_value_id value_id) {
  return odla_binary_eltwise_bool(alg_binary_eltwise::cmp_equal, lhs, rhs,
                                  value_id);
}

odla_value odla_Less(odla_value lhs, odla_value rhs,
                     const odla_value_id value_id) {
  return odla_binary_eltwise_bool(alg_binary_eltwise::cmp_less, lhs, rhs,
                                  value_id);
}

odla_value odla_LessOrEqual(odla_value lhs, odla_value rhs,
                            const odla_value_id value_id) {
  return odla_binary_eltwise_bool(alg_binary_eltwise::cmp_less_equal, lhs, rhs,
                                  value_id);
}

odla_value odla_Greater(odla_value lhs, odla_value rhs,
                        const odla_value_id value_id) {
  return odla_binary_eltwise_bool(alg_binary_eltwise::cmp_greater, lhs, rhs,
                                  value_id);
}

odla_value odla_GreaterOrEqual(odla_value lhs, odla_value rhs,
                               const odla_value_id value_id) {
  return odla_binary_eltwise_bool(alg_binary_eltwise::cmp_greater_equal, lhs,
                                  rhs, value_id);
}
