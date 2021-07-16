//===- nn_cnn_instructions.td --------------------------------*- tblgen -*-===//
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

#ifdef INSTRUCTION_BASE
#else
include "instruction_base.td"
#endif

let cat_ = cat_nn_rnn in {
  // TODO: state LSTM's functionality.
  def LSTM : Inst<"xxx"> {
    let attrs_ = [
      Attr<"Optional scaling values used by some activation functions.",
           FloatList, "activation_alpha">,
      Attr<"Optional scaling values used by some activation functions.",
           FloatList, "activation_beta">,
      Attr<"A list of 3 (or 6 if bidirectional) activation functions for "
           "input, output, forget, cell, and hidden.",
           StringList, "activations">,
      Attr<"Cell clip threshold.", Float, "clip">,
      Attr<"Specify if the RNN is forward, reverse, or bidirectional.",
           EnumDirection, "direction">,
      Attr<"Number of neurons in the hidden layer", Integer, "hidden_size">,
      Attr<"Couple the input and forget gates if 1.", Integer, "input_forget">,
      Attr<"The shape format of inputs X, initial_h, initial_c and outputs Y, "
           "Y_h, Y_c.",
           Integer, "layout">
    ];
    let ins_ = [
      Arg<"The input sequences.", ArgType<[F16, F32]>, 3D>,
      Arg<"The weight tensor for the gates.", MatchArgType<0>, 3D>,
      Arg<"The recurrence weight tensor.", MatchArgType<0>, 3D>,
      OptionalArg<"The bias tensor for input gate.", MatchArgType<0>, 2D>,
      OptionalArg<
          "Optional tensor specifying lengths of the sequences in a batch",
          ArgType<[Integer]>, 1D>,
      OptionalArg<"Optional initial value of the hidden.", MatchArgType<0>, 3D>,
      OptionalArg<"Optional initial value of the cell.", MatchArgType<0>, 3D>,
      OptionalArg<"The weight tensor for peepholes.", MatchArgType<0>, 2D>
    ];
    let outs_ = [
      OptionalArg<
          "A tensor that concats all the intermediate output values of the "
          "hidden.",
          MatchArgType<0>, 4D>,
      OptionalArg<"The last output value of the hidden.", MatchArgType<0>, 3D>,
      OptionalArg<"The last output value of the cell.", MatchArgType<0>, 3D>
    ];
  }
}