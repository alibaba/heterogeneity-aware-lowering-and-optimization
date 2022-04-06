//===- serializer.cc ------------------------------------------------------===//
//
// Copyright (C) 2019-2022 Alibaba Group Holding Limited.
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

#include "halo/lib/serializer/serializer.h"

#include <unordered_set>

#include "halo/lib/ir/ir_builder.h"
#include "onnx.pb.h"

namespace halo {

static onnx::TensorProto::DataType GetONNXDataType(DataType dt) {
  static const std::unordered_map<DataType, onnx::TensorProto::DataType> types{
      {DataType::FLOAT32, onnx::TensorProto::FLOAT},
      {DataType::FLOAT16, onnx::TensorProto::FLOAT16},
      {DataType::BFLOAT16, onnx::TensorProto::BFLOAT16},
      {DataType::FLOAT64, onnx::TensorProto::DOUBLE},
      {DataType::INT64, onnx::TensorProto::INT64},
      {DataType::UINT64, onnx::TensorProto::UINT64},
      {DataType::INT32, onnx::TensorProto::INT32},
      {DataType::UINT32, onnx::TensorProto::UINT32},
      {DataType::INT16, onnx::TensorProto::INT16},
      {DataType::UINT16, onnx::TensorProto::UINT16},
      {DataType::INT8, onnx::TensorProto::INT8},
      {DataType::UINT8, onnx::TensorProto::UINT8},
      {DataType::BOOL, onnx::TensorProto::BOOL},
      {DataType::STRING, onnx::TensorProto::STRING},
  };
  auto it = types.find(dt);
  return it == types.end() ? onnx::TensorProto::UNDEFINED : it->second;
}

static onnx::TensorProto::DataType GetONNXDataType(const IRObject& ir,
                                                   unsigned idx = 0) {
  HLCHECK(idx < ir.GetResultsTypes().size());
  return GetONNXDataType(ir.GetResultsTypes()[idx].GetDataType());
}

static void SetData(onnx::TensorProto* tensor, const float* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    tensor->add_float_data(data[i]); // NOLINT
  }
}
static void SetData(onnx::TensorProto* tensor, const int64_t* data,
                    size_t size) {
  for (size_t i = 0; i < size; ++i) {
    tensor->add_int64_data(data[i]); // NOLINT
  }
}

static void SetData(onnx::TensorProto* tensor, const int16_t* data,
                    size_t size) {
  for (size_t i = 0; i < size; ++i) {
    tensor->add_int32_data(data[i]); // NOLINT
  }
}

static void SetData(onnx::TensorProto* tensor, const int32_t* data,
                    size_t size) {
  for (size_t i = 0; i < size; ++i) {
    tensor->add_int32_data(data[i]); // NOLINT
  }
}

static void SetData(onnx::TensorProto* tensor, const void* data, size_t size) {
  tensor->add_string_data(data, size);
}

bool Serializer::RunOnModule(Module* module) {
  HLCHECK(module->Functions().size() == 1);
  const auto& func = *module->Functions().front();
  onnx::ModelProto::InitAsDefaultInstance();
  onnx::ModelProto model_def;
  model_def.set_producer_name("Halo");
  model_def.set_producer_version(HALO_VERSION_STR);
  model_def.set_domain("");
  model_def.set_model_version(1);
  model_def.set_doc_string("");
  model_def.set_ir_version(1);

  onnx::GraphProto& graph_def = *model_def.mutable_graph();
  for (const auto& constant : func.Constants()) {
    auto node_def = graph_def.add_node();
    node_def->set_op_type("Constant");
    node_def->set_name(constant->GetName());
    node_def->add_output(constant->GetName());
    auto attr = node_def->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto_AttributeType::
                       AttributeProto_AttributeType_TENSOR);
    onnx::TensorProto& tensor = *attr->mutable_t();
    const auto& ty = constant->GetResultType();
    tensor.set_data_type(GetONNXDataType(*constant));
    auto dims = tensor.mutable_dims();
    int n = ty.GetTotalNumOfElements();
    for (auto x : ty.GetDimSizes()) {
      dims->Add(x);
    }
    if (emit_weights_) {
      switch (ty.GetDataType()) {
        case DataType::FLOAT32:
          SetData(&tensor, static_cast<const float*>(constant->GetRawDataPtr()),
                  n);
          break;
        case DataType::FLOAT16:
          SetData(&tensor,
                  static_cast<const int16_t*>(constant->GetRawDataPtr()), n);
          break;
        case DataType::INT8:
          SetData(&tensor, constant->GetRawDataPtr(), n);
          break;
        case DataType::INT32:
          SetData(&tensor,
                  static_cast<const int32_t*>(constant->GetRawDataPtr()), n);
          break;
        case DataType::INT64:
          SetData(&tensor,
                  static_cast<const int64_t*>(constant->GetRawDataPtr()), n);
          break;
        default:
          HLCHECK(false);
      }
    } else {
      tensor.set_data_location(
          onnx::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
      auto ext = tensor.mutable_external_data();
      auto kv = ext->Add();
      kv->set_key("location");
      kv->set_value("bin");
      kv = ext->Add();
      kv->set_key("length");
      kv->set_value(std::to_string(constant->GetElementSizeInBytes() * n));
    }
  }
  for (const auto& arg : func.Args()) {
    auto input = graph_def.add_input();
    input->set_name(arg->GetName());
    auto vt = input->mutable_type()->mutable_tensor_type();
    vt->set_elem_type(GetONNXDataType(*arg));
    auto dims = vt->mutable_shape()->mutable_dim();
    for (auto x : arg->GetResultType().GetDimSizes()) {
      onnx::TensorShapeProto_Dimension* dim = dims->Add();
      dim->set_dim_value(x);
    }
  }
  const auto ret = func.GetReturnInst();
  for (const auto& bb : func.BasicBlocks()) {
    for (const auto& inst : bb->Instructions()) {
      if (ret == inst.get()) {
        continue;
      }
      auto node_def = graph_def.add_node();
      std::ostringstream op_name;
      inst->PrintOpcode(op_name);
      node_def->set_op_type(op_name.str());
      node_def->set_name(inst->GetName());
      for (const auto& op : inst->GetOperands()) {
        node_def->add_input(op.GetDef()->GetName());
      }
      node_def->add_output(inst->GetName());
      for (const auto& attr : inst->GetAttributes()) {
        auto& a = *node_def->add_attribute();
        a.set_name(attr->GetName());
        switch (attr->GetKind()) {
          case Attribute::AttrKind::BOOL:
            a.add_ints(attr->GetValueAsBool() ? 1 : 0);
            break;
          case Attribute::AttrKind::FLOAT:
            a.add_floats(attr->GetValueAsFloat());
            break;
          case Attribute::AttrKind::INTEGER:
            a.add_ints(attr->GetValueAsInteger());
            break;
          default:
            std::stringstream ss;
            attr->Print(ss);
            a.add_strings(ss.str());
        }
      }
      auto& shape_attr = *(node_def->add_attribute());
      shape_attr.set_name("shape");
      for (auto& d : inst->GetResultType().GetDimSizes()) {
        shape_attr.add_ints(d);
      }
    }
  }
  for (const auto& ret : func.GetReturnInst()->GetOperands()) {
    graph_def.add_output()->set_name(ret.GetDef()->GetName());
  }
  os_ << model_def.SerializeAsString();
  return false;
}

} // end namespace halo
