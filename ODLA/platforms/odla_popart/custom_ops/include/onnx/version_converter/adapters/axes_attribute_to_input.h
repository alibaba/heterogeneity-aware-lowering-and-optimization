// Adapter for all ops that remove consumed_inputs

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class AxesAttributeToInput : public Adapter {
 public:
  explicit AxesAttributeToInput(
      const std::string& op_name,
      const OpSetID& initial,
      const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  void attrToInput(
      std::shared_ptr<Graph> graph,
      Node* node,
      std::vector<int64_t> axes) const {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes() = std::vector<int64_t>{static_cast<int64_t>(axes.size())};
    auto& data = t.int64s();
    for (auto a : axes) {
      data.emplace_back(a);
    }
    Value* v;
    v = graph->addInitializerAndInput(t);
    node->addInput(v);
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    if (node->hasAttribute(kaxes)){
      attrToInput(graph, node, node->is(kaxes));
      node->removeAttribute(kaxes);
    }
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
