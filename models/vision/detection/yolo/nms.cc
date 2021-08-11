//===- nms.cc --------------------------------------------------------===//
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
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/ProfileValue.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Sort.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace popops::expr;

namespace CustomOperators {
const static popart::OperatorIdentifier NMS_1(popart::Domain::ai_graphcore,
                                              "NMS", 1, 3, 2);
} // namespace CustomOperators

class NMSOp : public popart::Op {
 public:
  NMSOp(const popart::OperatorIdentifier& _opid,
        const popart::Op::Settings& settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<NMSOp>(*this);
  }
  void setup() final {
    unsigned n = inInfo(0).shape()[0];
    popart::Shape shape1{80, n, 5};
    popart::Shape shape2{80};
    outInfo(0) = {inInfo(0).data_type(), shape1};
    outInfo(1) = {popart::DataType::UINT32, shape2};
  }

  void appendAttributes(popart::OpSerialiserBase& os) const override {
    Op::appendAttributes(os);
  }

  void appendOutlineAttributes(popart::OpSerialiserBase& os) const override {
    Op::appendOutlineAttributes(os);
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }
};

static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16,
                                            popart::DataType::FLOAT};
static popart::OpDefinition::DataTypes T2 = {popart::DataType::UINT32};

static popart::OpDefinition nmsOpDef(
    {popart::OpDefinition::Inputs(
         {{"bb", T}, {"iou_threshold", T}, {"score_threshold", T}}),
     popart::OpDefinition::Outputs({{"selected_info", T},
                                    {"selected_num", T2}}),
     popart::OpDefinition::Attributes({})});

static popart::OpCreator<NMSOp> nmsOpCreator(
    popart::OpDefinitions({
        {CustomOperators::NMS_1, nmsOpDef},
    }),
    [](const popart::OpCreatorInfo& info) {
      return std::make_unique<NMSOp>(info.opid, info.settings);
    },
    true);

class NMSOpx : public popart::popx::Opx {
  unsigned numTiles_;
  unsigned numWorkers_;

 public:
  NMSOpx(popart::Op*, popart::popx::Devicex*);
  void grow(poplar::program::Sequence&) const final;
};

NMSOpx::NMSOpx(popart::Op* op, popart::popx::Devicex* devicex)
    : popart::popx::Opx(op, devicex) {
  verifyOp<NMSOp>(op, CustomOperators::NMS_1);
  const auto& target = graph().getTarget();
  numTiles_ = target.getTilesPerIPU();
  numWorkers_ = target.getNumWorkerContexts();
  graph().addCodelets("vision/detection/yolo/codelets.cpp");
}

void NMSOpx::grow(poplar::program::Sequence& prog) const {
  Tensor bb = getInTensor(0).dimShuffle({1, 0});
  Tensor boxes_input = bb.slice(0, 4, 0).dimShuffle({1, 0});
  Tensor iou_threshold_input = getInTensor(1);
  Tensor score_threshold_input = getInTensor(2);
  Tensor selected_info_output = graph().addVariable(
      poplar::FLOAT, {80, bb.shape()[1], 5}, "selected_info_output");
  Tensor selected_num_output =
      graph().addVariable(poplar::UNSIGNED_INT, {80}, "selected_num_ouput");
  Tensor perm =
      graph().addVariable(poplar::UNSIGNED_INT, {bb.shape()[1]}, "perm");

  graph().setTileMapping(boxes_input, 0);
  graph().setTileMapping(iou_threshold_input, 0);
  graph().setTileMapping(score_threshold_input, 0);
  graph().setTileMapping(perm, 0);

  popops::iota(graph(), perm, 0u, prog);

  auto estimator = [](const poplar::VertexIntrospector& v,
                      const poplar::Target& device) {
    return std::uint64_t(50);
  };
  graph().registerPerfEstimator(
      poputil::templateVertex("popops::HeapSortVertexKV", UNSIGNED_INT, FLOAT),
      estimator);
  poplar::ComputeSet NMS_CS = graph().addComputeSet("NMS_CS");
  for (int cls = 0; cls < 80; ++cls) {
    graph().setTileMapping(bb[cls + 5], (cls / 5) % numTiles_);
    graph().setTileMapping(selected_info_output[cls], (cls / 5) % numTiles_);
    graph().setTileMapping(selected_num_output[cls], (cls / 5) % numTiles_);
    auto vtx = graph().addVertex(
        NMS_CS, "NonMaxSuppressionVertex",
        {{"boxes", boxes_input},
         {"scores", bb[cls + 5]},
         {"iou_threshold", iou_threshold_input},
         {"score_threshold", score_threshold_input},
         {"selected_info", selected_info_output[cls]},
         {"selected_indices",
          popops::sortKeyValue(graph(), bb[cls + 5], perm, 0, prog)},
         {"selected_num", selected_num_output[cls]}});
    graph().setTileMapping(vtx, (cls / 5) % numTiles_);
    graph().setPerfEstimate(vtx, 50);
  }
  prog.add(program::Execute(NMS_CS));
  setOutTensor(0, selected_info_output);
  setOutTensor(1, selected_num_output);
}

namespace {
popart::popx::OpxCreator<NMSOpx> nmsOpxCreator(CustomOperators::NMS_1);
} // namespace

namespace ONNX_NAMESPACE {
void NMSShapeInference(InferenceContext& ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

static const char NMSDoc[] = "Non-Maximum Suppression.";

ONNX_OPERATOR_SET_SCHEMA_EX(
    NMS, AiGraphcore, popart::Domain::ai_graphcore, 1, false,
    OpSchema()
        .SetDoc(NMSDoc)
        .Input(0, "bb", "Each Bbox Contains (cx, cy, w, h, conf, pred_cls(80))",
               "T")
        .Input(1, "iou_threshold", "Iou Threshold", "T")
        .Input(2, "score_threshold", "Score Threshold", "T")
        .Output(0, "selected_info",
                "Selected Info (cx, cy, w, h, pred_cls) of Each Class", "T")
        .Output(1, "selected_num", "Selected Num of Each Class", "T")
        .TypeConstraint(
            "T", {"tensor(float)", "tensor(float16)", "tensor(uint32)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(NMSShapeInference));

static bool registerOps() {
  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, NMS)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE
