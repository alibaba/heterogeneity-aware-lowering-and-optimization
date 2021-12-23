//===- nn_cnn_instructions.td --------------------------------*- tblgen -*-===//
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

#ifdef INSTRUCTION_BASE
#else
include "instruction_base.td"
#endif

defvar RNN_common_attrs = [
      Attr<"Weight data format.",
        EnumRNNWeightFormat, "weight_format", "INVALID">,
      Attr<"Optional scaling values used by some activation functions.",
           FloatList, "activation_alpha">,
      Attr<"Optional scaling values used by some activation functions.",
           FloatList, "activation_beta">,
      Attr<"Cell clip threshold.", Float, "clip">,
      Attr<"Specify if the RNN is forward, reverse, or bidirectional.",
           EnumDirection, "direction">,
      Attr<"Number of neurons in the hidden layer", Integer, "hidden_size">,
      Attr<"Couple the input and forget gates if 1.", Integer, "input_forget">,
      Attr<"The shape format of inputs X, initial_h, initial_c and outputs Y, "
           "Y_h, Y_c.",
           Integer, "layout">
];

defvar RNN_common_ins = [
      Arg<"The input sequences.", ArgType<[F16, F32]>, 3D>,
      Arg<"The weight tensor for the gates.", MatchArgType<0>, 3D>,
      Arg<"The recurrence weight tensor.", MatchArgType<0>, 3D>,
      OptionalArg<"The bias tensor for input gate.", MatchArgType<0>, 2D>,
      OptionalArg<
          "Optional tensor specifying lengths of the sequences in a batch",
          ArgType<[Integer]>, 1D>,
      OptionalArg<"Optional initial value of the hidden.", MatchArgType<0>, 3D>,
];

defvar RNN_common_outs = [
      OptionalArg<
          "A tensor that concats all the intermediate output values of the "
          "hidden.",
          MatchArgType<0>, 4D>,
      OptionalArg<"The last output value of the hidden.", MatchArgType<0>, 3D>,
];

let cat_ = cat_nn_rnn in {
  def LSTM : Inst<"Long Short-Term Memory"> {
    let attrs_ = !listconcat(RNN_common_attrs, [
      Attr<"Gate order.",
        EnumRNNGateOrder, "gate_order", "INVALID">,
      Attr<"A list of 3 (or 6 if bidirectional) activation functions for "
           "input, output, forget, cell, and hidden.",
           StringList, "activations">,
    ]);
    let ins_ = !listconcat(RNN_common_ins, [
      OptionalArg<"Optional initial value of the cell.", MatchArgType<0>, 3D>,
      OptionalArg<"The weight tensor for peepholes.", MatchArgType<0>, 2D>
    ]);
    let outs_ = !listconcat(RNN_common_outs, [
      OptionalArg<"The last output value of the cell.", MatchArgType<0>, 3D>
    ]);
  }

  def GRU : Inst<"Gated Recurrent Unit"> {
    let attrs_ = !listconcat(RNN_common_attrs, [
      Attr<"Gate order.",
        EnumRNNGateOrder, "gate_order", "INVALID">,
      Attr<"A list of 2 (or 4 if bidirectional) activation functions for "
           "update, reset, and hidden.",
           StringList, "activations">,
      Attr<"Whether to apply linear transformation before multiplying by the"
           " output of reset gate",
           Bool, "linear_before_reset">,
    ]);
    let ins_ = RNN_common_ins;
    let outs_ = RNN_common_outs;
  }

  def RNN : Inst<"Recurrent Neural Network"> {
    let attrs_ = !listconcat(RNN_common_attrs, [
      Attr<"A list of 1 (or 2 if bidirectional) activation functions input gate.",
           StringList, "activations">
    ]);
    let ins_ = RNN_common_ins;
    let outs_ = RNN_common_outs;
  }
}
