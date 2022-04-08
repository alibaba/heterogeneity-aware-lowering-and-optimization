//===- nn_activation_instructions.td -------------------------*- tblgen -*-===//
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

let cat_ = cat_nn_activation,
    ins_ = [Arg<"The unary input", ArgType<[F16,F32]> >],
    outs_ = [Arg<"The result.", MatchArgType<0> >] in {

  def Celu : Inst<"Compute celu activation, element-wise. Return "
                 "alpha * (exp(X1 / alpha) - 1) if X1 < 0, return X1 else."> {
    let attrs_ = [Attr<"Coefficient of celu", Float, "alpha", "1.0">];
  }

  def Elu : Inst<"Compute elu activation, element-wise. Return "
                 "alpha * (exp(X1) - 1) if X1 < 0, return X1 else."> {
    let attrs_ = [Attr<"Coefficient of elu", Float, "alpha", "1.0">];
  }

  def LeakyRelu : Inst<"Compute leaky relu activation, element-wise. Return"
                       " max(X1, X1 * alpha)."> {
    let attrs_ = [Attr<"Slope at X1 < 0", Float, "alpha", "0.2">];
  }

  def LogSoftmax : Inst<"Compute the log of softmax values."> {
    let attrs_ = [Attr<"The dimension the LogSoftmax would be performed on.",
                       Integer, "axis", "-1">];
  }

  def Selu : Inst<"Compute selu activation, element-wise. Return lambda * X1, "
                  "if X1 > 0; lambda * (alpha * exp(X1) - alpha), if X1 <= 0."> {
    let attrs_ = [
      Attr<"Scale factor", Float, "lambda", "1.0507">,
      Attr<"Slope at X1 < 0", Float, "alpha", "1.67326">
    ];
  }

  def ThresholdedRelu : Inst<"Compute thresholded relu activation, element-wise. Return"
                       " y = X1 > alpha ? X1 : 0."> {
    let attrs_ = [ Attr<"Threshold", Float, "alpha", "1.0"> ];
  }

  def Relu : Inst<"Compute the relu activation, element-wise."
                  " Return max(0, X1)">;

  def Relu6 : Inst<"Compute the relu6 activation, element-wise."
                   " Return min(max(0, X1), 6).">;

  def Tanh : Inst<"The unary opertion returns hyperbolic tangent of X1,"
                  " element-wise.">;

  def Sigmoid : Inst<"Compute the sigmoid activation, element-wise."
                     " Return 1 / (1 + exp(-X1)).">;

  def Swish : Inst<"Compute the swish activation, element-wise."
                     " Return x * sigmoid(x).">;

  def HardSigmoid : Inst<"A faster approximation of the sigmoid activation."> {
    let attrs_ = [
      Attr<"Value of alpha.", Float, "alpha", "0.2">,
      Attr<"Value of beta.", Float, "beta", "0.5">
    ];
  }

  def HardSwish : Inst<"Compute the h-swish activation"
                       "Return x * relu6(x + 3) / 6.">;

  def Gelu : Inst<"Compute the gelu (Gaussian error linear unit) activation, element-wise."
                  " Return 0.5 * x *(1 + erf(x / sqrt(2)))"> {
    let attrs_ = [Attr<"Use approximated Gaussian cumulative distribution function: "
                       " 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x^3)))",
                       Bool, "use_approximation", "false">];
  }

  def Softmax : Inst<"Compute the softmax activation."> {
    let attrs_ = [Attr<"The dimension the softmax would be performed on.",
                       Integer, "axis", "-1">];
  }

  def Softplus : Inst<"Compute the softplus activation, element-wise."
                      " Return ln(exp(x) + 1).">;

  def Softsign : Inst<"Compute the softsign activation, element-wise."
                      " Return x / (1 + |x|).">;

  def Hardmax : Inst<"Compute the hardmax."
                    " Return one-hot tensor for the location of the highest value alone with specified axis."> {
    let attrs_ = [Attr<"The dimension the hardmax would be performed on.",
                       Integer, "axis", "-1">];
  }

}

let cat_ = cat_nn_activation,
    ins_ = [Arg<"The input data.", ArgType<[F16, F32]> >,
            Arg<"The slope param, support broadcast", ArgType<[F16, F32]> >],
    outs_ = [Arg<"The result.", MatchArgType<0> >] in {
  def PRelu : Inst<"Compute prelu activation, element-wise. Return"
                   "f(x) = slope * x for x < 0, f(x) = x for x >= 0.">;
}

let cat_ = cat_nn_activation,
    ins_ = [Arg<"The input data.", ArgType<[I8,I16,I32,I64,UI8,UI16,UI32,UI64,F16,F32,F64]> >],
    outs_ = [Arg<"The result.", MatchArgType<0> >] in {
  def Shrink : Inst<"Compute Shrink activation, element-wise. Return"
                   "y = x + bias for x < -lambd, y = x - bias for x > lambd, "
                   "y = 0 , otherwise.">{
    let attrs_ = [
      Attr<"The bias value added to output", Float, "bias", "0.0">,
      Attr<"The lambd value for the Shrink formulation", Float, "lambd", "0.5">
    ];
  }
}