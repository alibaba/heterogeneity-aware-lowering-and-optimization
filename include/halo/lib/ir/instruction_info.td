///===- instruction_info.td ----------------------------------*- tblgen -*-===//
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

// Instruction category
/// Naming convention: cat_##major category##_##minor category
class Category;

def cat_common : Category;
def cat_common_cast : Category;
def cat_common_reduction : Category;
def cat_controlFlow : Category;
def cat_creator : Category;
def cat_image : Category;
def cat_math : Category;
def cat_math_binary : Category;
def cat_math_unary : Category;
def cat_nn : Category;
def cat_nn_cnn : Category;
def cat_nn_rnn : Category;
def cat_nn_activation : Category;
def cat_objectDetection : Category;
def cat_quantization : Category;
def cat_customize : Category;