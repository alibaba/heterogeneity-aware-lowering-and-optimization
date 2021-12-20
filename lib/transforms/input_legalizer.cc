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

#include <functional>
#include <numeric>

#include "halo/lib/ir/ir_builder.h"

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

static std::vector<float> ParsePreprocessScale(const std::string& str) {
  std::istringstream iss(str);
  std::vector<float> values;
  constexpr int n = 6;
  values.reserve(n);
  float v;
  char comma;
  while (iss >> v && values.size() < n) {
    values.push_back(v);
    iss >> comma;
  }
  return values;
}

bool InputLegalizer::RunOnFunction(Function* func) {
  bool changed = false;
  if (handled_func_.count(func) != 0) {
    return false;
  }
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
      changed = true;
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

  if (!scale_str_.empty()) {
    constexpr int n = 6;
    auto v = ParsePreprocessScale(scale_str_);
    if (func->Args().size() != 1) {
      std::cerr << "Preprocess only works for model with exactly one input. "
                   "Ignore preprocessing.\n";
    } else if (v.size() != n) {
      std::cerr << "Invalid number of scale parameters (" << v.size()
                << " != " << n << ") Ignore preprocessing.\n";
    } else {
      // Insert scale.
      auto arg = func->Args().begin()->get();
      std::vector<float> bias(v.begin(), v.begin() + n / 2);
      std::vector<float> scale(v.begin() + n / 2, v.end());
      ConstantBuilder cb(func);
      const auto& arg_type = arg->GetResultType();
      size_t arg_dims = arg_type.GetNumOfDims();
      std::vector<int64_t> shape(arg_dims, 1);
      // FIXME if h=c or w =c, it not work
      for (int64_t i = arg_dims - 1; i >= 0; --i) {
        if (arg_type.GetNumOfElementsInDim(i) == n / 2) {
          shape[i] = n / 2;
          break;
        }
      }

      if (std::accumulate(shape.begin(), shape.end(), 1,
                          std::multiplies<int64_t>()) != n / 2) {
        std::cerr << "Preprocess element number does not match with input "
                     "type. Ignore preprocessing.\n";
        scale_str_.clear();
        return changed;
      }

      halo::Type dt{DataType::FLOAT32, shape};
      BasicBlock* bb = func->begin()->get();
      IRBuilder builder(bb);
      if (!bb->empty()) {
        builder.SetInsertBefore(bb->begin()->get());
      }
      Def input = *arg;
      auto addend = cb.CreateConstant("_pp_addend", dt, bias.data());
      if (!addend->HasSameValueOf(0)) {
        Def new_def = *builder.CreateAdd("pp_add", input, *addend);
        input.GetOwner()->ReplaceAllUsesWith(0, new_def);
        input = new_def;
        changed = true;
      }

      // it may be fused with conv
      auto coeff = cb.CreateConstant("_pp_coeff", dt, scale.data());
      if (!coeff->HasSameValueOf(1)) {
        Def new_def = *builder.CreateMul("_pp_scale", input, *coeff);
        input.GetOwner()->ReplaceAllUsesWith(0, new_def);
        changed = true;
      }
    }
    std::cout << v.size() << std::endl;
    for (auto x : v) {
      std::cout << x << "\n";
    }
    scale_str_.clear(); // Prevent re-entry.
  }
  handled_func_.insert(func);
  return changed;
}

} // end namespace halo
