//===- fusion.td ---------------------------------------------*- tblgen -*-===//
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

include "instructions.td"

class Fusion<dag patternToMatch, dag result> {
    dag pattern_ = patternToMatch;
    dag result_ = result;
    string option_name_;
    string option_desc_;
    string copy_attrs_from_ = "";
}

def MatmulBias : Fusion<(Add(MatMul:$c $op0, $op1), $op2) , (MatMul $op0, $op1, $op2)> {
    let option_name_ = "fuse-matmul-bias";
    let option_desc_ = "Fuse bias into matmul/fc";
    let copy_attrs_from_ = "c";
}

def SigmoidMul: Fusion<(Mul$op0, (Sigmoid:$c $op0)), (Swish $op0)>{
    let option_name_ = "fuse-swish";
    let option_desc_ = "Fuse mul-sigmoid into swish";
}

