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

#ifndef ODLA_POPART_COMMON_H_
#define ODLA_POPART_COMMON_H_

#include <ODLA/odla.h>
#include <vector>
#include <memory>
#include <popart/iarray.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>


popart::DataType GetPopartType(odla_value_type type);
popart::Shape GetPopartShape(odla_value_shape shape);
std::string&& GetDirectionName(odla_rnn_direction direction);
std::string&& GetTypeName(odla_element_type element_type);
std::string&& GetResizeInterpolationModeName(odla_interpolation_mode mode);
odla_element_type GetOdlaType(popart::DataType type);

std::unique_ptr<popart::IArray> MakeNDArrayWrapper(const odla_void *data_ptr,
                                                   popart::DataType data_type,
                                                   std::vector<int64_t> shape);
#endif
