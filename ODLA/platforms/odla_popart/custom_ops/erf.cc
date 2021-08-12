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

// This file contains the implementation of the erf() function on IPU as a
// float32 PopART custom op.
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
#include <poplar/Graph.hpp>
#include <popops/ElementWise.hpp>
#include <vector>

using namespace poplar;

class ErfOp;
class ErfOpx;

namespace CustomOperators {
const popart::OperatorIdentifier Erf = {popart::Domain::ai_graphcore, "Erf", 1,
                                        1, 1};
}

// Forward op
class ErfOp : public popart::ElementWiseUnaryOp {
 public:
  ErfOp(const popart::OperatorIdentifier& _opid,
        const popart::Op::Settings& settings_)
      : popart::ElementWiseUnaryOp(_opid, settings_) {}

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<ErfOp>(*this);
  }
};

static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};

static popart::OpDefinition ErfOpDef({popart::OpDefinition::Inputs({{"X", T}}),
                                      popart::OpDefinition::Outputs({{"Y", T}}),
                                      popart::OpDefinition::Attributes({})});

static popart::OpCreator<ErfOp> ErfOpCreator(popart::OpDefinitions({
    {CustomOperators::Erf, ErfOpDef},
}));

// Forward Opx, poplar implementation of forward Op
class ErfOpx : public popart::popx::ElementWiseUnaryOpx {
 public:
  ErfOpx(popart::Op* op, popart::popx::Devicex* devicex)
      : popart::popx::ElementWiseUnaryOpx(op, devicex) {
    verifyOp<ErfOp>(op, CustomOperators::Erf);
  }

  void grow(poplar::program::Sequence& prog) const final {
    Tensor x = getInTensor(0);

    Tensor sign = popops::signum(graph().getPoplarGraph(), x, prog);
    Tensor y = popops::abs(graph().getPoplarGraph(), x, prog);
    popops::mulInPlace(graph().getPoplarGraph(), y, 0.3275911f, prog);
    popops::addInPlace(graph().getPoplarGraph(), y, 1.0f, prog);
    popops::invInPlace(graph().getPoplarGraph(), y, prog);

    static const std::array<float, 4> coeff{-1.453152027f, 1.421413741f,
                                            -0.284496736f, 0.254829592f};
    Tensor poly = popops::mul(graph().getPoplarGraph(), y, 1.061405429f, prog);
    for (float c : coeff) {
      popops::addInPlace(graph().getPoplarGraph(), poly, c, prog);
      popops::mulInPlace(graph().getPoplarGraph(), poly, y, prog);
    }

    y = popops::square(graph().getPoplarGraph(), x, prog);
    popops::negInPlace(graph().getPoplarGraph(), y, prog);
    popops::expInPlace(graph().getPoplarGraph(), y, prog);
    popops::mulInPlace(graph().getPoplarGraph(), y, poly, prog);
    popops::negInPlace(graph().getPoplarGraph(), y, prog);
    popops::addInPlace(graph().getPoplarGraph(), y, 1.0f, prog);
    popops::mulInPlace(graph().getPoplarGraph(), y, sign, prog);

    setOutTensor(0, y);
  }
};

namespace {
popart::popx::OpxCreator<ErfOpx> ErfOpxCreator(CustomOperators::Erf);
} // namespace

namespace ONNX_NAMESPACE {
static std::string ErfDoc("Erf:: ");
ONNX_OPERATOR_SET_SCHEMA_EX(
    Erf, AiGraphcore, popart::Domain::ai_graphcore, 1, false,
    OpSchema("Erf", __FILE__, __LINE__)
        .SetDoc(ErfDoc)
        .Input(0, std::string("X"), std::string("Input tensor"),
               std::string("T"))
        .Output(0, std::string("Y"), std::string("Output tensor"),
                std::string("T"))
        .TypeConstraint(
            std::string("T"),
            {std::string("tensor(float16)"), std::string("tensor(uint32)"),
             std::string("tensor(uint64)"), std::string("tensor(int32)"),
             std::string("tensor(int64)"), std::string("tensor(float)")},
            "")
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
              ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
            }));

static bool registerOps() {
  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Erf)>());

  return true;
}
static bool ret = registerOps();
} // namespace ONNX_NAMESPACE
