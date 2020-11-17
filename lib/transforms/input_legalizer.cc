//===- input_legalizer.cc -------------------------------------------------===//
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

#include "halo/lib/transforms/input_legalizer.h"

namespace halo {

static std::unordered_map<std::string, halo::Type> ParseInputShapes(
    const std::vector<std::string>& shapes, const std::string& default_name) {
  std::unordered_map<std::string, halo::Type> results;
  std::string type_name;

  for (auto str : shapes) {
    auto dt = DataType::INVALID;
    // format: name[:type]:1x20x14
    auto idx = str.find_last_of(':');
    std::string name = default_name;
    if (idx == std::string::npos || idx == 0) {
      idx = 0;
    } else {
      name = str.substr(0, idx);
      // parse type.
      auto type_idx = name.find_last_of(':');
      if (type_idx != std::string::npos) {
        type_name = name.substr(type_idx + 1);
        name = name.substr(0, type_idx);
      }
      ++idx;
    }
    if (!type_name.empty()) {
      dt = Type::StringToDataType(type_name);
    }
    HLCHECK(str.back() != 'x' && str.back() != 'X' && str.back() != '*');
    str.push_back('x'); // sentinel.

    std::vector<int64_t> dims;
    bool is_neg = false;
    for (int v = 0, i = idx, e = str.size(); i < e; ++i) {
      if (str[i] == 'x' || str[i] == 'X' || str[i] == '*') {
        if (i <= 0 || (std::isdigit(str[i - 1]) == 0)) {
          dims.clear();
          break;
        }
        dims.push_back(is_neg ? -v : v);
        v = 0;
        is_neg = false;
      } else if (str[i] == '-') {
        if (!(i == 0 || (str[i - 1] == 'x' || str[i - 1] == 'X' ||
                         str[i - 1] == '*' || str[i - 1] == ':'))) {
          dims.clear();
          break;
        }
        is_neg = true;
      } else if (isdigit(str[i]) != 0) {
        constexpr int d = 10;
        v = v * d + str[i] - '0';
      } else {
        dims.clear();
        break;
      }
    }
    HLCHECK(results.count(name) == 0);
    results[name] = halo::Type(dt, dims);
  }
  return results;
}

bool InputLegalizer::RunOnFunction(Function* func) {
  bool changed = false;
  auto specified_shapes = ParseInputShapes(
      inputs_shapes_,
      (func->Args().size() == 1) ? (*func->Args().begin())->GetName() : "");
  inputs_shapes_.clear(); // Avoid re-entry.
  for (auto& arg : func->Args()) {
    auto& ty = arg->GetResultType();
    if (auto it = specified_shapes.find(arg->GetName());
        it != specified_shapes.end()) {
      auto dt = it->second.GetDataType();
      if (dt == DataType::INVALID) {
        dt = ty.GetDataType();
      }
      auto dims = it->second.GetNumOfDims() == 0 ? ty.GetDimSizes()
                                                 : it->second.GetDimSizes();
      arg->GetResultsTypes()[0] = halo::Type(dt, dims);
      specified_shapes.erase(it);
    }

    auto dims = ty.GetDimSizes();
    if (!dims.empty() &&
        ((dims[0] < 0) || (batch_size_ == -1) || (batch_size_ > 1)) &&
        (dims[0] != batch_size_)) {
      dims[0] = batch_size_;
      arg->GetResultsTypes()[0] = halo::Type(ty.GetDataType(), dims);
      changed = true;
    }
    HLCHECK(ty.IsValid());
  }

  HLCHECK(specified_shapes.empty() && "Unclaimed specified shapes");

  return changed;
}

} // end namespace halo
