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
from time import time
import logging
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
        self.logger = logging.getLogger(__name__)
        self.so_file = so_file
        self.h = None
        self.buffers = []
        self.files = files

    def Load(self,model):
        if self.h is None:
            self.h = CDLL(self.so_file)
        self.comp = c_void_p(0)
        self.device = c_void_p(0)
        self.h.odla_AllocateDevice(c_void_p(0), 0, pointer(self.device))
        self.h.odla_CreateComputation(pointer(self.comp))
        # TODO:
        # use_sim = c_bool(True)
        # self.h.odla_SetComputationItem(self.comp, 7, pointer(use_sim))
        cc_file = str(self.files[0]).encode("utf-8")
        bin_file = str(self.files[1]).encode("utf-8")

        self.comp = self.h.model_helper(cc_file, bin_file)
        self.ctx = c_void_p(0)
        self.h.odla_CreateContext(pointer(self.ctx))
        n = c_int32(-1)
        self.h.odla_GetNumOfArgsFromComputation(self.comp, pointer(n))
        # self.nr_args = n.value
        if("bert" in model):
            print("bert_model input num.")
            self.nr_args = 3
        else:
            self.nr_args = 1

        # nr_args = c_int32(-1)
        self.h.odla_GetNumOfOutputsFromComputation(self.comp, pointer(n))
        # self.nr_outputs = n.value
        if("bert" in model):
            print("bert_model output num.")
            self.nr_outputs = 2
        else:
            self.nr_outputs = 1

        self.in_vals = []
        for idx in range(0, self.nr_args):
            arg_v = c_void_p(0)
            self.h.odla_GetArgFromComputationByIdx(self.comp, idx, pointer(arg_v))
            # vt = ValueType()
            # self.h.odla_GetValueType(arg_v, pointer(vt))
            # self.in_vals.append((arg_v.value, vt))

        # self.ctx = c_void_p(0)
        # self.h.odla_CreateContext(pointer(self.ctx))

        self.out_vals = []
        for idx in range(0, self.nr_outputs):
            out = c_void_p(0)
            self.h.odla_GetOutputFromComputationByIdx(self.comp, idx, pointer(out))
            # vt = ValueType()
            # self.h.odla_GetValueType(out, pointer(vt))
            # n = 1
            # for r in range(0, vt.shape.size):
            #     n *= vt.shape.dims[r]
            # self.out_vals.append((out, vt, n))
            # buf = (c_float * n)() # FIXME: handle types
            # self.h.odla_BindToOutput(out, buf, self.ctx)
            # self.buffers.append(buf)

    def Execute(self, data, model, batch):
        print(f"model:{model},batch:{batch}")
        # for idx, v in enumerate(self.in_vals):
        #     self.h.odla_BindToArgument(
        #         v[0], data[idx].ctypes.data_as(c_void_p), self.ctx
        #     )
        self.h.odla_BindToArgument(c_void_p(0), data[0].ctypes.data_as(c_void_p), self.ctx)
        # output buffer
        buffers = []
        if("bert" in model):
            buf1 = (c_float * 256 * batch)() 
            buffers.append(buf1)
            self.h.odla_BindToOutput(c_void_p(0), buf1, self.ctx)
            buf2 = (c_float * 256 * batch)()
            buffers.append(buf2)
            self.h.odla_BindToOutput(c_void_p(0), buf2, self.ctx)
        else:
            if("resnet50" in model):
                buf = (c_float * 1*1000 * batch)()
            elif("dbnet" in model):
                buf = (c_float * 1228800 * batch)()
            elif("crnn" in model):
                buf = (c_float * 918146 * batch)()
            else:
                assert(False and f"unknown model.{model}")
            buffers.append(buf)
            self.h.odla_BindToOutput(c_void_p(0), buf, self.ctx)

        if("resnet50" in model):
            self.h.model_data(self.ctx,  (c_int32 * 1)(*[224*224*3*4]), (c_int32 * 1)(*[1000*4]))
        elif("dbnet" in model):
            self.h.model_data(self.ctx,  (c_int32 * 1)(*[1*3*960*1280*4]), (c_int32 * 1)(*[1228800 * 4]))
        elif("crnn" in model):
            self.h.model_data(self.ctx,  (c_int32 * 1)(*[63840*4]), (c_int32 * 1)(*[918146*4]))
        elif("bert" in model):
            self.h.model_data(self.ctx,  (c_int32 * 1)(*[512*4, 256*4, 256*4]), (c_int32 * 1)(*[256*4,256*4]))
        else:
            assert(False and f"unknown model.{model}")
    
        s = time()
        # self.h.odla_ExecuteComputation(self.comp, self.ctx, 0, c_void_p(0))
        self.h.odla_ExecuteComputation(self.comp, self.ctx, 0, self.device)
        t = time()
        self.logger.info("Execution time:" + str(t - s) + " sec(s)")
        return buffers
