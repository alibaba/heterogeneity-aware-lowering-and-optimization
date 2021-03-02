//===- weights_quantizer.cc
//---------------------------------------------------------===//
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

#include "halo/lib/quantizer/weights_quantizer.h"

#include <limits.h>
#include <math.h>

#include <algorithm>
#include <fstream>

#include "halo/lib/ir/ir_builder.h"
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/ir/nn_cnn_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

template <typename T>
void static QuantTo(Constant* val, DataType dt, float scale, float offset) {
  const auto& type = val->GetResultType();
  if (type.GetDataType() != DataType::FLOAT32) {
    return;
  }
  size_t len = type.GetTotalNumOfElements();
  std::vector<T> quant_data(len);
  float scale_r = (scale == 0) ? 1.F : 1.F / scale;
  for (size_t i = 0; i < len; ++i) {
    float q = round(val->GetDataAsFloat32(i) * scale_r + offset);
    float min = 0;
    float max = 0;
    if (dt == DataType::UINT8) {
      min = 0;
      max = UCHAR_MAX;
    } else if (dt == DataType::INT32) {
      min = INT_MIN;
      max = INT_MAX;
    }
    if (q < min) {
      q = min;
    } else if (q > max) {
      q = max;
    }
    quant_data[i] = q;
  }

  halo::Type new_type{dt, type.GetDimSizes()};
  val->SetData(new_type, quant_data.data());
}

void WeightsQuantizer::RunOnConstant(Constant* val) {
  const auto& type = val->GetResultType();
  // So far only quint8 is supported.
  HLCHECK(quant_ == CodeGen::Quantization::QUINT8);
  if (type.GetDataType() != DataType::FLOAT32) {
    return;
  }

  const float* data = val->GetDataPtr<float>();
  size_t len = type.GetTotalNumOfElements();
  constexpr int range = 255;
  auto get_odla_name = [](const IRObject& obj) {
    std::string ret = CodeGen::NormalizeVariableName(obj.GetName());
    return IsA<Constant>(&obj) ? ret + "_" : ret;
  };
  std::string emitted_name = get_odla_name(*val);

  auto it = quant_info_.find(emitted_name);
  if (!quant_info_.empty() && it == quant_info_.end()) {
    std::cerr << "Quant info not found for " << emitted_name << "\n";
  }
  float min_val = it != quant_info_.end()
                      ? it->second.min_val
                      : *std::min_element(data, data + len); // NOLINT.
  float max_val = it != quant_info_.end()
                      ? it->second.max_val
                      : *std::max_element(data, data + len); // NOLINT.
  if (min_val > 0) {
    min_val = 0;
  }
  float scale;
  float zp;
  if (it != quant_info_.end()) {
    scale = it->second.scale;
    zp = round(it->second.zp);
  } else {
    scale = (max_val - min_val) / range;
    if (scale != 0) {
      zp = 0 - min_val / scale;
    } else {
      scale = 1;
      zp = max_val;
    }
    zp = round(zp);
    constexpr float min_zp = 0;
    constexpr float max_zp = 255;
    zp = std::clamp(zp, min_zp, max_zp);
  }

  if (!quant_info_.empty()) {
    // Check if a constant is a bias of conv/matmul.
    if (val->GetNumberOfUses() == 1) {
      auto user = val->GetIthResultUses(0).begin();
      auto used_by = user->GetOwner();
      if (user->GetIdx() == 2 &&
          (IsA<Conv2DInst>(used_by) || IsA<MatMulInst>(used_by))) {
        // Quant to Int32.
        std::string input_name =
            get_odla_name(*used_by->GetOperand(0).GetOwner());
        std::string weight_name =
            get_odla_name(*used_by->GetOperand(1).GetOwner());

        auto input_it = quant_info_.find(input_name);
        if (input_it == quant_info_.end()) {
          std::cerr << "Missing quant info for " << input_name << "\n";
          return;
        }

        auto weight_it = quant_info_.find(weight_name);
        if (weight_it == quant_info_.end()) {
          std::cerr << "Missing quant info for " << weight_name << "\n";
          return;
        }
        float scale = input_it->second.scale * weight_it->second.scale;
        QuantTo<int32_t>(val, DataType::INT32, scale, 0);
        return;
      }
    }
  }
  QuantTo<uint8_t>(val, DataType::UINT8, scale, zp);
}

bool WeightsQuantizer::RunOnModule(Module* m) {
  if (quant_ == CodeGen::Quantization::None) {
    return false;
  }

  // Read quant info from profiling results.
  if (!pgq_file_.empty()) {
    std::ifstream ifs(pgq_file_);
    if (!ifs.good()) {
      std::cerr << "Unable to read PGQ file `" << pgq_file_ << "`\n";
      return false;
    }
    std::string name;
    char comma;
    for (std::string line; std::getline(ifs, line);) {
      auto first_comma = line.find_first_of(',');
      name = line.substr(0, first_comma);
      std::istringstream iss(line.substr(first_comma + 1));
      auto& info = quant_info_[name];
      iss >> info.min_val >> comma >> info.max_val >> comma >> info.scale >>
          comma >> info.zp;
    }
  }

  for (auto& func : *m) {
    for (auto& constant : func->Constants()) {
      RunOnConstant(constant.get());
    }
  }
  return true;
}

} // end namespace halo
