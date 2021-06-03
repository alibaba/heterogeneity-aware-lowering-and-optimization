//===- models.td -----------------------------------*- tblgen -*-===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

/// Configure class
class Config<int batch_size> {
  int batch_size_ = batch_size;
}

/// Docs class
class Docs<string model_name, string model_src, string desc> {
  string model_name_ = model_name;
  string model_src_ = model_src;
  string model_desc_ = desc;
}

/// Model class
class Model<int num_inputs, int num_outputs> {
  int num_inputs_ = num_inputs;
  int num_outputs_ = num_outputs;
  list<Config> configs = [];
  Docs docs_ = ?;
}