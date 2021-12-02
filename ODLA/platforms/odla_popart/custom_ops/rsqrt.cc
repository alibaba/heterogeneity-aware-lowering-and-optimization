//
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>

#include <memory>
#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popops/ElementWise.hpp>
#include <vector>

namespace CustomOperators {
const static popart::OperatorIdentifier Rsqrt_1(popart::Domain::ai_graphcore,
                                                "Rsqrt", 1, 1, 1);
} // namespace CustomOperators

class RsqrtOp : public popart::ElementWiseUnaryOp {
 public:
  RsqrtOp(const popart::OperatorIdentifier& _opid,
          const popart::Op::Settings& settings_)
      : popart::ElementWiseUnaryOp(_opid, settings_) {}

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<RsqrtOp>(*this);
  }
};

static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};

static popart::OpDefinition rsqrtOpDef(
    {popart::OpDefinition::Inputs({{"X", T}}),
     popart::OpDefinition::Outputs({{"Y", T}}),
     popart::OpDefinition::Attributes({})});

static popart::OpCreator<RsqrtOp> rsqrtOpCreator(popart::OpDefinitions({
    {CustomOperators::Rsqrt_1, rsqrtOpDef},
}));

class RsqrtOpx : public popart::popx::ElementWiseUnaryOpx {
 public:
  RsqrtOpx(popart::Op*, popart::popx::Devicex*);
  void grow(snap::program::Sequence&) const final;
};

RsqrtOpx::RsqrtOpx(popart::Op* op, popart::popx::Devicex* devicex)
    : popart::popx::ElementWiseUnaryOpx(op, devicex) {
  verifyOp<RsqrtOp>(op, CustomOperators::Rsqrt_1);
}

void RsqrtOpx::grow(snap::program::Sequence& prog) const {
  auto result =
      popops::map(graph().getPoplarGraph(), popops::expr::UnaryOpType::RSQRT,
                  getInTensor(0).getPoplarTensor(), prog.getPoplarSequence(), debugContext());
  setOutTensor(0, snap::Tensor{result, graph()});
}

namespace {
popart::popx::OpxCreator<RsqrtOpx> rsqrtOpxCreator(CustomOperators::Rsqrt_1);
} // namespace

namespace ONNX_NAMESPACE {
void RsqrtShapeInference(InferenceContext& ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

static const char RsqrtDoc[] =
    "Rsqrt returns reciprocal square root of the tensor.";

ONNX_OPERATOR_SET_SCHEMA_EX(
    Rsqrt, AiGraphcore, popart::Domain::ai_graphcore, 1, false,
    OpSchema()
        .SetDoc(RsqrtDoc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T", {"tensor(float)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(RsqrtShapeInference));

static bool registerOps() {
  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1,
                                                      Rsqrt)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE
