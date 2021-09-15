//===- odla_dnnl_rnn.cc ---------------------------------------------------===//
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

#include <cstring>

#include "ODLA/odla_common.h"
#include "ODLA/ops/odla_ops_nn.h"
#include "odla_dnnl.h"

static std::tuple<dnnl::rnn_direction, int> GetDirection(
    odla_rnn_direction odla_dir) {
  dnnl::rnn_direction dir;
  int d = 1;
  switch (odla_dir) {
    case ODLA_RNN_FORWARD:
      dir = dnnl::rnn_direction::unidirectional_left2right;
      break;
    case ODLA_RNN_REVERSE:
      dir = dnnl::rnn_direction::unidirectional_right2left;
      break;
    default:
      d = 2;
      dir = dnnl::rnn_direction::bidirectional_concat;
  }
  return {dir, d};
}

odla_values odla_LSTM(odla_value input, odla_rnn_weight_format weight_format,
                      odla_rnn_gate_order gate_order,
                      odla_value_shape weight_dims, odla_value W, odla_value R,
                      odla_value B, odla_value sequence_lens,
                      odla_value initial_h, odla_value initial_c, odla_value P,
                      odla_int32 hidden_size, odla_rnn_direction direction,
                      odla_rnn_outputs outputs,
                      const odla_value_ids value_ids) {
  dnnl::rnn_direction dir;
  int d = 1;
  std::tie(dir, d) = GetDirection(direction);

  assert(weight_format == ODLA_RNN_LDGOI);
  assert(gate_order == ODLA_RNN_IOFC);

  // DNNL assumes the following layout:
  // src : { T (Seq), N (Batch), SLC(Input)}  tag::tnc
  // src_iter: {L, D, N, SIC} tag::ldnc
  // src_iter_c: {L, D, N, DIC} tag:ldnc
  // W: {L, D, SLC, 4, DIC}  tag:ldigo
  // R(W_it): {L, D, SIC, 4, DIC}  tag:ldigo
  // dst : {T, N, DIC} tag:tnc
  // dst_it: {L, D, N, DIC} tag:ldnc
  // dst_it_c: {L, D, N, DIC} tag:ldnc
  //
  constexpr int64_t l = 1;
  auto t = input->shape.dims[0]; // sequence.
  const auto n = input->shape.dims[1];
  const auto slc = input->shape.dims[2];
  constexpr int num_gates = 4;
  assert(d == W->shape.dims[0]);
  assert(W->shape.dims[1] == num_gates * hidden_size);
  assert(W->shape.dims[2] == slc);
  assert(R->shape.dims[1] == num_gates * hidden_size);
  assert(R->shape.dims[2] == hidden_size);

  if (B != nullptr) {
    assert(B->shape.dims[0] == d);
    assert(B->shape.dims[1] == num_gates * hidden_size ||
           B->shape.dims[1] == 2 * num_gates * hidden_size);
  }

  auto dt = getDataType(input->elem_type);
  dnnl::memory::desc nil;

  dnnl::memory::desc src_md({t, n, slc}, dt, dnnl::memory::format_tag::tnc);

  // Either both are empty or both are valid.
  assert((initial_h == nullptr && initial_c == nullptr) ||
         (initial_c != nullptr && initial_h != nullptr));
  auto src_iter_desc = initial_h != nullptr
                           ? dnnl::memory::desc({l, d, n, hidden_size}, dt,
                                                dnnl::memory::format_tag::ldnc)
                           : nil;
  auto src_iter_c_desc =
      initial_c != nullptr ? dnnl::memory::desc({l, d, n, hidden_size}, dt,
                                                dnnl::memory::format_tag::ldnc)
                           : nil;

  dnnl::memory::desc w_desc({l, d, slc, num_gates, hidden_size}, dt,
                            dnnl::memory::format_tag::ldigo);
  dnnl::memory::desc w_it_desc({l, d, hidden_size, num_gates, hidden_size}, dt,
                               dnnl::memory::format_tag::ldigo);

  dnnl::memory::desc ret_md({t, n, d * hidden_size}, dt,
                            dnnl::memory::format_tag::tnc);
  dnnl::memory::desc ret_it_md({l, d, n, hidden_size}, dt,
                               dnnl::memory::format_tag::ldnc);

  odla_value_shape ret_shape{4, {t, d, n, hidden_size}};
  odla_value_shape ret_iter_shape{3, {d, n, hidden_size}};

  const auto& w_peephole_desc =
      P != nullptr ? dnnl::memory::desc({l, d, 3, hidden_size}, dt,
                                        dnnl::memory::format_tag::ldnc)
                   : nil;
  auto bias_desc = B != nullptr
                       ? dnnl::memory::desc({l, d, num_gates, hidden_size}, dt,
                                            dnnl::memory::format_tag::ldgo)
                       : nil;

  dnnl::lstm_forward::desc desc(dnnl::prop_kind::forward_inference, dir, src_md,
                                src_iter_desc, src_iter_c_desc, w_desc,
                                w_it_desc, w_peephole_desc, bias_desc, ret_md,
                                ret_it_md, ret_it_md);
  auto pd = dnnl::lstm_forward::primitive_desc(desc, g_comp->eng);
  auto prim = dnnl::lstm_forward(pd);

  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto ret_h_mem = dnnl::memory(ret_it_md, g_comp->eng);
  auto ret_c_mem = dnnl::memory(ret_it_md, g_comp->eng);

  dnnl::memory zero;

  auto w_mem = W->mem;
  auto r_mem = R->mem;
  auto b_mem = (B == nullptr) ? zero : B->mem;
  if (W->is_const && R->is_const && (B == nullptr || B->is_const)) {
    // W/R/B are in [iofc] but DNNL assumes [ifco] for weights and [ifo] for
    // Peephole. See https://oneapi-src.github.io/oneDNN/dev_guide_rnn.html
    // For W/R, we also need to transpose from ldgoi to ldigo.
    w_mem = dnnl::memory(w_desc, g_comp->eng);
    r_mem = dnnl::memory(w_it_desc, g_comp->eng);
    b_mem = dnnl::memory(bias_desc, g_comp->eng);
    auto shuffle = [num_gates](float* dst, const float* src, int dir,
                               int output_chs, int input_chs) {
      const int gate_mapping[4] = {0, 3, 1, 2}; // iofc -> ifco
      for (int d = 0; d < dir; ++d) {
        size_t offset_base = d * num_gates * output_chs * input_chs;
        for (int src_gate = 0; src_gate < num_gates; ++src_gate) {
          auto dst_gate = gate_mapping[src_gate];
          for (int o_ch = 0; o_ch < output_chs; ++o_ch) {
            for (int i_ch = 0; i_ch < input_chs; ++i_ch) {
              size_t dst_offset = offset_base + i_ch * num_gates * output_chs +
                                  dst_gate * output_chs + o_ch;
              dst[dst_offset] = *src;
              ++src;
            }
          }
        }
      }
    };
    shuffle(static_cast<float*>(w_mem.get_data_handle()),
            static_cast<const float*>(W->mem.get_data_handle()), d, hidden_size,
            slc);
    shuffle(static_cast<float*>(r_mem.get_data_handle()),
            static_cast<const float*>(R->mem.get_data_handle()), d, hidden_size,
            hidden_size);
    if (B != nullptr) {
      auto shuffle_bias = [num_gates](float* dst, const float* src, int dir,
                                      int chs) {
        const int gate_mapping[4] = {0, 3, 1, 2}; // iofc -> ifco
        for (int d = 0, s = num_gates * chs; d < dir; ++d) {
          for (int src_gate = 0; src_gate < num_gates; ++src_gate) {
            auto dst_gate = gate_mapping[src_gate];
            memcpy(&dst[d * s + dst_gate * chs], &src[d * s + src_gate * chs],
                   sizeof(dst[0]) * chs);
          }
        }
      };

      shuffle_bias(static_cast<float*>(b_mem.get_data_handle()),
                   static_cast<const float*>(B->mem.get_data_handle()), d,
                   hidden_size);
    }
  } else {
  }

  // Primitive arguments
  std::unordered_map<int, dnnl::memory> args;
  args[DNNL_ARG_SRC_LAYER] = input->mem;
  args[DNNL_ARG_SRC_ITER] = (initial_h != nullptr) ? initial_h->mem : zero;
  args[DNNL_ARG_SRC_ITER_C] = (initial_h != nullptr) ? initial_c->mem : zero;
  args[DNNL_ARG_WEIGHTS_LAYER] = w_mem;
  args[DNNL_ARG_WEIGHTS_ITER] = r_mem;
  args[DNNL_ARG_WEIGHTS_PEEPHOLE] = (P != nullptr) ? P->mem : zero;
  args[DNNL_ARG_BIAS] = b_mem;
  args[DNNL_ARG_DST_LAYER] = ret_mem;
  args[DNNL_ARG_DST_ITER] = ret_h_mem;
  args[DNNL_ARG_DST_ITER_C] = ret_c_mem;

  add_op(prim, args);
  InterpretIfNeeded();

  auto ret = CreateValue(ret_mem, ret_shape, value_ids.value_ids[0]);
  auto ret_h = CreateValue(ret_h_mem, ret_iter_shape, value_ids.value_ids[1]);
  auto ret_c = CreateValue(ret_c_mem, ret_iter_shape, value_ids.value_ids[2]);

  return {.size = 3, .values = {ret, ret_h, ret_c}};
}

odla_values odla_GRU(odla_value input, odla_rnn_weight_format weight_format,
                     odla_rnn_gate_order gate_order,
                     odla_value_shape weight_dims, odla_value W, odla_value R,
                     odla_value B, odla_value sequence_lens,
                     odla_value initial_h, odla_int32 hidden_size,
                     odla_rnn_direction direction,
                     odla_bool linear_before_reset, odla_rnn_outputs outputs,
                     const odla_value_ids value_ids) {
  dnnl::rnn_direction dir;
  int d = 1;
  std::tie(dir, d) = GetDirection(direction);

  assert(gate_order == ODLA_RNN_URO);

  constexpr int64_t l = 1;
  auto t = input->shape.dims[0]; // sequence.
  const auto n = input->shape.dims[1];
  const auto slc = input->shape.dims[2];
  constexpr int num_gates = 3;
  assert(d == W->shape.dims[0]);
  assert(W->shape.dims[1] == num_gates * hidden_size);
  assert(W->shape.dims[2] == slc);
  assert(R->shape.dims[1] == num_gates * hidden_size);
  assert(R->shape.dims[2] == hidden_size);

  if (B != nullptr) {
    assert(B->shape.dims[0] == d);
    assert(B->shape.dims[1] == num_gates * hidden_size ||
           B->shape.dims[1] == 2 * num_gates * hidden_size);
  }

  auto dt = getDataType(input->elem_type);
  dnnl::memory::desc nil;

  dnnl::memory::desc src_md({t, n, slc}, dt, dnnl::memory::format_tag::tnc);

  auto src_iter_desc = initial_h != nullptr
                           ? dnnl::memory::desc({l, d, n, hidden_size}, dt,
                                                dnnl::memory::format_tag::ldnc)
                           : nil;
  dnnl::memory::desc w_desc({l, d, slc, num_gates, hidden_size}, dt,
                            dnnl::memory::format_tag::ldigo);
  dnnl::memory::desc w_it_desc({l, d, hidden_size, num_gates, hidden_size}, dt,
                               dnnl::memory::format_tag::ldigo);

  dnnl::memory::desc ret_md({t, n, d * hidden_size}, dt,
                            dnnl::memory::format_tag::tnc);
  dnnl::memory::desc ret_it_md({l, d, n, hidden_size}, dt,
                               dnnl::memory::format_tag::ldnc);

  odla_value_shape ret_shape{4, {t, d, n, hidden_size}};
  odla_value_shape ret_iter_shape{3, {d, n, hidden_size}};

  auto bias_desc = B != nullptr
                       ? dnnl::memory::desc({l, d, num_gates, hidden_size}, dt,
                                            dnnl::memory::format_tag::ldgo)
                       : nil;
  dnnl::primitive prim;
  if (linear_before_reset != 0) {
    dnnl::lbr_gru_forward::desc desc(dnnl::prop_kind::forward_inference, dir,
                                     src_md, src_iter_desc, w_desc, w_it_desc,
                                     bias_desc, ret_md, ret_it_md);
    auto pd = dnnl::lbr_gru_forward::primitive_desc(desc, g_comp->eng);
    prim = dnnl::lbr_gru_forward(pd);
  } else {
    dnnl::gru_forward::desc desc(dnnl::prop_kind::forward_inference, dir,
                                 src_md, src_iter_desc, w_desc, w_it_desc,
                                 bias_desc, ret_md, ret_it_md);
    auto pd = dnnl::gru_forward::primitive_desc(desc, g_comp->eng);
    prim = dnnl::gru_forward(pd);
  }

  auto ret_mem = dnnl::memory(ret_md, g_comp->eng);
  auto ret_h_mem = dnnl::memory(ret_it_md, g_comp->eng);

  dnnl::memory zero;

  auto w_mem = W->mem;
  auto r_mem = R->mem;
  auto b_mem = (B == nullptr) ? zero : B->mem;
  auto initial_h_mem = (initial_h == nullptr) ? zero : initial_h->mem;

  if (weight_format == ODLA_RNN_LDGOI) {
    if (W->is_const && R->is_const) {
      // W/R/B are in [URO].
      // For W/R, we also need to transpose from ldgoi to ldigo.
      w_mem = dnnl::memory(w_desc, g_comp->eng);
      r_mem = dnnl::memory(w_it_desc, g_comp->eng);
      b_mem = dnnl::memory(bias_desc, g_comp->eng);
      auto reorder = [num_gates](float* dst, const float* src, int dir,
                                 int output_chs, int input_chs) {
        for (int d = 0; d < dir; ++d) {
          size_t offset_base = d * num_gates * output_chs * input_chs;
          for (int gate = 0; gate < num_gates; ++gate) {
            for (int o_ch = 0; o_ch < output_chs; ++o_ch) {
              for (int i_ch = 0; i_ch < input_chs; ++i_ch) {
                size_t dst_offset = offset_base +
                                    i_ch * num_gates * output_chs +
                                    gate * output_chs + o_ch;
                dst[dst_offset] = *src;
                ++src;
              }
            }
          }
        }
      };
      reorder(static_cast<float*>(w_mem.get_data_handle()),
              static_cast<const float*>(W->mem.get_data_handle()), d,
              hidden_size, slc);
      reorder(static_cast<float*>(r_mem.get_data_handle()),
              static_cast<const float*>(R->mem.get_data_handle()), d,
              hidden_size, hidden_size);
    } else { // transpose from ldgoi to ldigo.
      odla_value_shape perm{5, {0, 1, 4, 2, 3}};
      auto wt = odla_Transpose(
          odla_Reshape(W,
                       odla_value_shape{5, {l, d, num_gates, hidden_size, slc}},
                       nullptr),
          perm, odla_value_shape{5, {l, d, slc, num_gates, hidden_size}},
          nullptr);
      auto rt = odla_Transpose(
          odla_Reshape(
              R,
              odla_value_shape{5, {l, d, hidden_size, num_gates, hidden_size}},
              nullptr),
          perm,
          odla_value_shape{5, {l, d, hidden_size, num_gates, hidden_size}},
          nullptr);
      w_mem = wt->mem;
      r_mem = rt->mem;
    }
  }

  // Primitive arguments
  std::unordered_map<int, dnnl::memory> args;
  args[DNNL_ARG_SRC_LAYER] = input->mem;
  args[DNNL_ARG_SRC_ITER] = initial_h_mem;
  args[DNNL_ARG_WEIGHTS_LAYER] = w_mem;
  args[DNNL_ARG_WEIGHTS_ITER] = r_mem;
  args[DNNL_ARG_BIAS] = b_mem;
  args[DNNL_ARG_DST_LAYER] = ret_mem;
  args[DNNL_ARG_DST_ITER] = ret_h_mem;
  add_op(prim, args);
  InterpretIfNeeded();

  auto ret = CreateValue(ret_mem, ret_shape, value_ids.value_ids[0]);
  auto ret_h = CreateValue(ret_h_mem, ret_iter_shape, value_ids.value_ids[1]);

  return {.size = 2, .values = {ret, ret_h}};
}
