//===- instruction_base.td -----------------------------------*- tblgen -*-===//
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

#ifdef INSTRUCTION_BASE
#else
#define INSTRUCTION_BASE

include "arg_types.td"
include "attribute_types.td"
include "instruction_info.td"

/// pseudo value types for type constraints
def other_vt : ValueType<0>;

/// Attribute class
class Attr<string desc, ValueType type, string name, string init = ""> {
  ValueType type_ = type;
  string description_ = desc;
  string attr_name_ = name;
  string init_value_ = init;
}

/// Argument Type.
class ArgType<list<ValueType> vts> {
  list<ValueType> prime_types_ = vts;
}

/// Dimension class
class Dim<int min_rank, int max_rank = min_rank> {
  int min_rank_ = min_rank;
  int max_rank_ = max_rank;
}
/// Commonly used dimensions.
def AnyD : Dim<0, 6>;
def 0D : Dim<0>;
def 1D : Dim<1>;
def 2D : Dim<2>;
def 3D : Dim<3>;
def 4D : Dim<4>;

/// Pseudo ArgType to represent type match constraint
class MatchArgType<int num> : ArgType<[other_vt]> {
  int match_arg_id_ = num;
}

/// Base class for operand and result
class Arg<string desc, ArgType types, Dim dim = AnyD> {
  ArgType types_ = types;
  Dim dim_ = dim;
  string description_ = desc;
}

/// Special arguments
/// Optional arg & variable length arg
class OptionalArg<string desc, ArgType type, Dim dim = AnyD>
    : Arg<desc, type, dim>;
/// Variable length args
class VarArg<string desc, ArgType type, Dim dim = AnyD>
    : Arg<desc, type, dim>;

/// Instruction class
class Inst<string desc> {
  Category cat_ = ?;
  list<Attr> attrs_ = [];
  list<Arg> ins_ = [];
  list<Arg> outs_ = [];
  //list<dag> patterns_ = [];
  string description_ = desc;
}

#endif // INSTRUCTION_BASE