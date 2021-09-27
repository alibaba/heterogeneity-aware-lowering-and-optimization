# Copyright (C) 2019-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ctypes import *
from enum import Enum
import os
from pdb import set_trace

class Device(Enum):
    CUDA = 1
    IPU = 2
    X86 = 3


class ValueShape(Structure):
    _fields_ = [("size", c_int32), ("dims", c_int64 * 10)]


class ValueType(Structure):
    _fields_ = [("element_type", c_int32), ("shape", ValueShape)]


class ODLAModel:
    def __init__(self, so_file, files):
        self.so_file = so_file
        self.h = None
        self.files = files

    def Load(self):
        if self.h is None:
            self.h = CDLL(self.so_file)
        self.comp = c_void_p(0)
        self.device = c_void_p(0)
        self.h.odla_AllocateDevice(c_void_p(0), 0, pointer(self.device))
        self.h.odla_CreateComputation(pointer(self.comp))
        # TODO:
        #use_sim = c_bool(True)
        #self.h.odla_SetComputationItem(self.comp, 7, pointer(use_sim))
        cc_file = str(self.files[0]).encode("utf-8")
        bin_file = str(self.files[1]).encode("utf-8")
        self.comp = self.h.model_helper(cc_file, bin_file)
        self.ctx = c_void_p(0)
        self.h.odla_CreateContext(pointer(self.ctx))
        n = c_int32(-1)
        self.h.odla_GetNumOfArgsFromComputation(self.comp, pointer(n))
        #self.nr_args = n.value
        self.nr_args = 1
        #nr_args = c_int32(-1)
        self.h.odla_GetNumOfOutputsFromComputation(self.comp, pointer(n))
        #self.nr_outputs = n.value
        self.nr_outputs = 1
        self.in_vals = []
        for idx in range(0, self.nr_args):
            arg_v = c_void_p(0)
            self.h.odla_GetArgFromComputationByIdx(self.comp, idx, pointer(arg_v))
            #vt = ValueType()
            #vt = ValueType() # TODO!!! Type check odla_common.h
            #self.h.odla_GetValueType(arg_v, pointer(vt))
            #self.in_vals.append((arg_v.value, vt))

        self.out_vals = []
        for idx in range(0, self.nr_outputs):
            out = c_void_p(0)
            self.h.odla_GetOutputFromComputationByIdx(self.comp, idx, pointer(out))
            #vt = ValueType()
            #self.h.odla_GetValueType(out, pointer(vt))
            n = 1
            #for r in range(0, vt.shape.size):
                #n *= vt.shape.dims[r]
            #self.out_vals.append((out, vt, n))

    # (c_int_t * 1)(*[1000*4])
    def Execute(self, data, model):
#       ''' for idx, v in enumerate(self.in_vals):
#            self.h.odla_BindToArgument(
#                v[0], data[idx].ctypes.data_as(c_void_p), self.ctx
#        )'''
        self.h.odla_BindToArgument(c_void_p(0), data[0].ctypes.data_as(c_void_p), self.ctx)
        buffers = []
        # buf = (c_float * 1000)()
        if("resnet50" in model):
            # resnet50: 
            buf = (c_float * 1*1000)()
        elif("dbnet" in model):
            # dbnet: 
            buf = (c_float * 1*320*320)()
        elif("crnn" in model):
            # crnn: 
            buf = (c_float * 64*1*5331)()
        else:
            assert(False and f"unknown model.{model}")
        
        buffers.append(buf)
        self.h.odla_BindToOutput(c_void_p(0), buf, self.ctx)
        # self.h.model_data(self.ctx,  (c_float * 1)(*[224*224*3*4]), (c_int32 * 1)(*[1000*4]))
        if("resnet50" in model):
            # resnet50: 
            self.h.model_data(self.ctx,  (c_float * 1)(*[224*224*3*4]), (c_int32 * 1)(*[1000*4]))
        elif("dbnet" in model):
            # dbnet: 
            self.h.model_data(self.ctx,  (c_float * 1)(*[3*960*720*4]), (c_int32 * 1)(*[320*320*4]))
        elif("crnn" in model):
            # crnn: 
            self.h.model_data(self.ctx,  (c_float * 1)(*[3*32*665*4]), (c_int32 * 1)(*[64*1*5331*4]))
        else:
            assert(False and f"unknown model.{model}")
        self.h.odla_ExecuteComputation(self.comp, self.ctx, 0, self.device)
        return [buf[:] for buf in buffers]
