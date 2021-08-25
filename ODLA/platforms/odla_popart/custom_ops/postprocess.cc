//===- postprocess.cc -----------------------------------------------------===//
//
// Copyright (C) 2020-2021 Alibaba Group Holding Limited.
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
#include <cassert>
#include <iostream>
#include <memory>
#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/op/elementwisex.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <poplar/ArrayRef.hpp>
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

poplar::ArrayRef<unsigned> AllAnchors{
#include "yolo_anchors.txt"
};

namespace CustomOperators {
const static popart::OperatorIdentifier PostProcess_1(
    popart::Domain::ai_graphcore, "PostProcess", 1, 5, 2);
} // namespace CustomOperators

class PostProcessOp : public popart::Op {
 public:
  PostProcessOp(const popart::OperatorIdentifier& _opid,
                const popart::Op::Settings& settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<PostProcessOp>(*this);
  }
  void setup() final {
    assert(inInfo(2).shape()[0] == 1);
    unsigned num_anchors = 3;
    unsigned dim_[3], n[3];
    for (int i = 0; i < num_anchors; i++) {
      dim_[i] = inInfo(2 + i).shape()[2];
      n[i] = dim_[i] * dim_[i] * num_anchors;
    }
    unsigned N = n[0] + n[1] + n[2];
    unsigned cls_num = inInfo(2).shape()[1] / num_anchors - 5;
    popart::Shape shape1{cls_num, N, 5};
    outInfo(0) = {inInfo(2).data_type(), shape1};
    popart::Shape shape2{cls_num};
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

static popart::OpDefinition postprocessOpDef(
    {popart::OpDefinition::Inputs({{"orig_img_w", T2},
                                   {"orig_img_h", T2},
                                   {"bb13", T},
                                   {"bb26", T},
                                   {"bb52", T}}),
     popart::OpDefinition::Outputs({{"selected_info", T},
                                    {"selected_num", T2}}),
     popart::OpDefinition::Attributes({})});

static popart::OpCreator<PostProcessOp> postprocessOpCreator(
    popart::OpDefinitions({
        {CustomOperators::PostProcess_1, postprocessOpDef},
    }),
    [](const popart::OpCreatorInfo& info) {
      return std::make_unique<PostProcessOp>(info.opid, info.settings);
    },
    true);

class PostProcessOpx : public popart::popx::Opx {
  unsigned numTiles_;
  unsigned numWorkers_;

 public:
  PostProcessOpx(popart::Op*, popart::popx::Devicex*);
  void grow(poplar::program::Sequence&) const final;
};

PostProcessOpx::PostProcessOpx(popart::Op* op, popart::popx::Devicex* devicex)
    : popart::popx::Opx(op, devicex) {
  verifyOp<PostProcessOp>(op, CustomOperators::PostProcess_1);
  const auto& target = graph().getTarget();
  numTiles_ = target.getTilesPerIPU();
  numWorkers_ = target.getNumWorkerContexts();
  graph().addCodelets("codelets_nms.cpp");
}

void PostProcessOpx::grow(poplar::program::Sequence& prog) const {
  Tensor orig_img_w = getInTensor(0);
  Tensor orig_img_h = getInTensor(1);
  unsigned num_anchors = 3;
  unsigned dim_[3], n[3];
  for (int i = 0; i < num_anchors; i++) {
    dim_[i] = inInfo(2 + i).shape()[2];
    n[i] = dim_[i] * dim_[i] * num_anchors;
  }
  unsigned N = n[0] + n[1] + n[2];
  unsigned cls_num = inInfo(2).shape()[1] / num_anchors - 5;
  // to nhwc
  Tensor bb13 = getInTensor(2)[0].dimShuffle({1, 2, 0}).reshape({n[0], 85});
  Tensor bb26 = getInTensor(3)[0].dimShuffle({1, 2, 0}).reshape({n[1], 85});
  Tensor bb52 = getInTensor(4)[0].dimShuffle({1, 2, 0}).reshape({n[2], 85});

  popops::expInPlace(graph(), bb13, prog);
  popops::expInPlace(graph(), bb26, prog);
  popops::expInPlace(graph(), bb52, prog);

  Tensor anchors =
      graph().addConstant<unsigned>(poplar::UNSIGNED_INT, {3, 6}, AllAnchors);
  poplar::ComputeSet Decode_CS = graph().addComputeSet("Decode_CS");

  graph().setTileMapping(bb13, 0);
  graph().setTileMapping(anchors[0], 0);
  graph().setTileMapping(orig_img_w, 0);
  graph().setTileMapping(orig_img_h, 0);
  Tensor dim13 = graph().addConstant<unsigned>(poplar::UNSIGNED_INT, {}, {13});
  graph().setTileMapping(dim13, 0);
  auto vtx_d0 = graph().addVertex(Decode_CS, "DecodeVertex",
                                  {{"boxes", bb13},
                                   {"anchors", anchors[0]},
                                   {"orig_img_w", orig_img_w},
                                   {"orig_img_h", orig_img_h},
                                   {"dim", dim13}});
  graph().setTileMapping(vtx_d0, 0);
  graph().setPerfEstimate(vtx_d0, 50);
  graph().setTileMapping(bb26, 1);
  graph().setTileMapping(anchors[1], 1);
  Tensor dim26 = graph().addConstant<unsigned>(poplar::UNSIGNED_INT, {}, {26});
  graph().setTileMapping(dim26, 1);
  auto vtx_d1 = graph().addVertex(Decode_CS, "DecodeVertex",
                                  {{"boxes", bb26},
                                   {"anchors", anchors[1]},
                                   {"orig_img_w", orig_img_w},
                                   {"orig_img_h", orig_img_h},
                                   {"dim", dim26}});
  graph().setTileMapping(vtx_d1, 1);
  graph().setPerfEstimate(vtx_d1, 50);
  graph().setTileMapping(bb52, 2);
  graph().setTileMapping(anchors[2], 2);
  Tensor dim52 = graph().addConstant<unsigned>(poplar::UNSIGNED_INT, {}, {52});
  graph().setTileMapping(dim52, 2);
  auto vtx_d2 = graph().addVertex(Decode_CS, "DecodeVertex",
                                  {{"boxes", bb52},
                                   {"anchors", anchors[2]},
                                   {"orig_img_w", orig_img_w},
                                   {"orig_img_h", orig_img_h},
                                   {"dim", dim52}});
  graph().setTileMapping(vtx_d2, 2);
  graph().setPerfEstimate(vtx_d2, 50);
  prog.add(program::Execute(Decode_CS));

  Tensor bb = concat({bb13, bb26, bb52});
  Tensor bb_rev = bb.dimShuffle({1, 0});
  Tensor iou_threshold_input =
      graph().addConstant<float>(poplar::FLOAT, {}, {0.45});
  Tensor score_threshold_input =
      graph().addConstant<float>(poplar::FLOAT, {}, {0.3});
  Tensor boxes_input = bb.slice(0, 4, 1);
  Tensor selected_info_output = graph().addVariable(
      poplar::FLOAT, {cls_num, bb.shape()[0], 5}, "selected_info_output");
  Tensor selected_num_output = graph().addVariable(
      poplar::UNSIGNED_INT, {cls_num}, "selected_num_ouput");
  Tensor perm =
      graph().addVariable(poplar::UNSIGNED_INT, {bb.shape()[0]}, "perm");
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
    graph().setTileMapping(bb_rev[cls + 5], (cls / 5) % numTiles_);
    graph().setTileMapping(selected_info_output[cls], (cls / 5) % numTiles_);
    graph().setTileMapping(selected_num_output[cls], (cls / 5) % numTiles_);
    auto vtx = graph().addVertex(
        NMS_CS, "NonMaxSuppressionVertex",
        {{"boxes", boxes_input},
         {"scores", bb_rev[cls + 5]},
         {"iou_threshold", iou_threshold_input},
         {"score_threshold", score_threshold_input},
         {"selected_info", selected_info_output[cls]},
         {"selected_indices",
          popops::sortKeyValue(graph(), bb_rev[cls + 5], perm, 0, prog)},
         {"selected_num", selected_num_output[cls]}});
    graph().setTileMapping(vtx, (cls / 5) % numTiles_);
    graph().setPerfEstimate(vtx, 50);
  }
  prog.add(program::Execute(NMS_CS));
  setOutTensor(0, selected_info_output);
  setOutTensor(1, selected_num_output);
}

namespace {
popart::popx::OpxCreator<PostProcessOpx> postprocessOpxCreator(
    CustomOperators::PostProcess_1);
} // namespace

namespace ONNX_NAMESPACE {
void PostProcessShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 1);
  propagateElemTypeFromInputToOutput(ctx, 2, 0);
  auto bb13_shape = ctx.getInputType(2)->tensor_type().shape();
  unsigned num_anchors = 3, dim_[3], n[3];
  for (int i = 0; i < 3; i++) {
    auto shape = ctx.getInputType(2 + i)->tensor_type().shape();
    if (shape.dim(0).dim_value() != 1) {
      fail_shape_inference("We assumed that dim0 = 1");
    } else if (shape.dim(2).dim_value() != shape.dim(3).dim_value()) {
      fail_shape_inference("BBox has incorrect shape");
    }
    dim_[i] = shape.dim(2).dim_value();
    n[i] = dim_[i] * dim_[i] * num_anchors;
  }
  unsigned N = n[0] + n[1] + n[2];
  unsigned cls_num = bb13_shape.dim(1).dim_value() / num_anchors - 5;
  auto* output_shape1 =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  output_shape1->add_dim()->set_dim_value(cls_num);
  output_shape1->add_dim()->set_dim_value(N);
  output_shape1->add_dim()->set_dim_value(5);
  auto* output_shape2 =
      ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
  output_shape2->add_dim()->set_dim_value(cls_num);
}

static const char PostProcessDoc[] = "Post Process of Yolov3.";

ONNX_OPERATOR_SET_SCHEMA_EX(
    PostProcess, AiGraphcore, popart::Domain::ai_graphcore, 1, false,
    OpSchema()
        .SetDoc(PostProcessDoc)
        .Input(0, "orig_img_w", "Width of Original Image", "T")
        .Input(1, "orig_img_h", "Height of Original Image", "T")
        .Input(2, "bb13", "BBoxes 13 x 13", "T")
        .Input(3, "bb26", "BBoxes 26 x 26", "T")
        .Input(4, "bb52", "BBoxes 52 x 52", "T")
        .Output(0, "selected_info",
                "Selected Info (cx, cy, w, h, pred_cls) of Each Class", "T")
        .Output(1, "selected_num", "Selected Num of Each Class", "T")
        .TypeConstraint(
            "T", {"tensor(float)", "tensor(float16)", "tensor(uint32)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(PostProcessShapeInference));

static bool registerOps() {
  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1,
                                                      PostProcess)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE
