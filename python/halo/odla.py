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
            self.nr_args = 3
        elif("shoucai" in model):
            self.nr_args = 16
        else:
            self.nr_args = 1

        # nr_args = c_int32(-1)
        self.h.odla_GetNumOfOutputsFromComputation(self.comp, pointer(n))
        # self.nr_outputs = n.value
        if("bert" in model):
            self.nr_outputs = 2
        else:
            self.nr_outputs = 1

        self.in_vals = []
        for idx in range(0, self.nr_args):
            arg_v = c_void_p(0)
            self.h.odla_GetArgFromComputationByIdx(self.comp, idx, pointer(arg_v))

        self.out_vals = []
        for idx in range(0, self.nr_outputs):
            out = c_void_p(0)
            self.h.odla_GetOutputFromComputationByIdx(self.comp, idx, pointer(out))

    def Execute(self, data, model, batch):
        print(f"model:{model},batch:{batch}")
        # bind input
        if("bert" in model):
            self.h.odla_BindToArgument(c_void_p(0), data[0].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(1), data[1].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(2), data[2].ctypes.data_as(c_void_p), self.ctx)
        elif("shoucai" in model):
            self.h.odla_BindToArgument(c_void_p(0), data[0].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(1), data[1].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(2), data[2].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(3), data[3].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(4), data[4].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(5), data[5].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(6), data[6].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(7), data[7].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(8), data[8].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(9), data[9].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(10), data[10].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(11), data[11].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(12), data[12].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(13), data[13].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(14), data[14].ctypes.data_as(c_void_p), self.ctx)
            self.h.odla_BindToArgument(c_void_p(15), data[15].ctypes.data_as(c_void_p), self.ctx)
        else:
            self.h.odla_BindToArgument(c_void_p(0), data[0].ctypes.data_as(c_void_p), self.ctx)

        # output buffer
        buffers = []
        if("bert" in model):
            buf1 = (c_float * 256 * batch)()
            buffers.append(buf1)
            self.h.odla_BindToOutput(c_void_p(0), buf1, self.ctx)
            buf2 = (c_float * 256 * batch)()
            buffers.append(buf2)
            self.h.odla_BindToOutput(c_void_p(1), buf2, self.ctx)
        else:
            if("resnet50" in model):
                buf = (c_float * 1*1000 * batch)()
            elif("dbnet" in model):
                assert((batch==1) and "dbnet only support 1 batch.")
                buf = (c_float * 1228800 * batch)()
            elif("crnn" in model):
                assert((batch==1) and "crnn only support 1 batch.")
                buf = (c_float * 918146 * batch)()
            elif("shoucai" in model):
                #Add info for Shoucai Model
                assert((batch==1) and "shoucai only support 1 batch!")
                buf = (c_float * 425 * batch)()
            else:
                assert(False and f"unknown model.{model}")
            buffers.append(buf)
            self.h.odla_BindToOutput(c_void_p(0), buf, self.ctx)

        # send data
        if("resnet50" in model):
            self.h.model_data(self.ctx,  (c_int32 * 1)(*[224*224*3*4*batch]), (c_int32 * 1)(*[1000*4*batch]))
        elif("dbnet" in model):
            self.h.model_data(self.ctx,  (c_int32 * 1)(*[1*3*960*1280*4*batch]), (c_int32 * 1)(*[1228800 * 4*batch]))
        elif("crnn" in model):
            self.h.model_data(self.ctx,  (c_int32 * 1)(*[63840*4*batch]), (c_int32 * 1)(*[918146*4*batch]))
        elif("bert" in model):
            self.h.model_data(self.ctx,  (c_int32 * 3)(*[512*4*batch, 256*4*batch, 256*8*batch]), (c_int32 * 2)(*[256*4*batch,256*4*batch]))
        elif ("shoucai" in model):
            #per Input File
            #self.h.model_data(self.ctx,  (c_int32 * 16)(*[8*4*batch, 1*4*batch, 592*4*batch, 1*4*batch, 512*4*batch, 512*4*batch, 73728*4*batch, 27200*4*batch, 13600*4*batch, 13600*4*batch, 122400*4*batch, 350200*4*batch, 15552*4*batch, 1161984*4*batch, 178*4*batch, 425*8*batch]), (c_int32*1)(*[425*4*batch]))            
            
            #per Input Shape
            self.h.model_data(self.ctx, (c_int32 * 18)(*[425*4*batch, 8*4*batch, 1*4*batch, 592*4*batch, 1*4*batch, 512*4*batch, 512*4*batch, 73728*4*batch, 425*4*batch, 425*4*batch, 27200*4*batch,13600*4*batch, 13600*4*batch, 122400*4*batch, 350200*4*batch, 15552*4*batch, 1161984*4*batch, 178*4*batch]),  (c_int32*1)(*[425*4*batch]))            


            '''
            self.h.model_data(self._ctx, (c_int32 * 16)(*[8*4*batch,     \ # embedding_ui_oage_shared_embedding.txt
                                                         1*4*batch,      \ # input_from_feature_columns_concat_3.txt
                                                         592*4*batch,    \ # input_from_feature_columns_concat.txt
                                                         1*4*batch,      \ #input_from_feature_columns_concat_5.txt
                                                         512*4*batch,    \ #all_clk_seq_1_time.txt
                                                         512*4*batch,    \#all_clk_seq_1_st.txt
                                                         73728*4*batch,  \ #seq_input_from_feature_columns_concat_1.txt
                                                         27200*4*batch,  \#embedding_item_id_d_shard_embedding_2.txt
                                                         13600*4*batch,  \#embedding_item_cate_id_d_shared_embedding_2.txt
                                                         13600*4*batch,  \#embedding_item_seller_id_d_shared_embedding_2.txt
                                                         122400*4*batch, \#input_from_feature_columns_concat_4.txt
                                                         350200*4*batch, \#input_from_feature_columns_concat_1.txt
                                                         15552*4*batch,  \#seq_input_from_feature_columns_concat.txt
                                                         1161984*4*batch,\ #seq_input_from_feature_columns_concat_2.txt
                                                         178*4*batch,    \#input_from_feature_columns_concat_7.txt
                                                         425*8*batch]),  \#Unique_preprocess_int64.txt
                                         (c_int32*1)(*[425*4*batch]))   #output
            
            
            self.h.model_data(self._ctx, (c_int32 * 18)(*[425*4*batch, \# LookupPkOP
                                                          8*4*batch,     \ # embedding_ui_oage_shared_embedding.txt
                                                          1*4*batch,      \ # input_from_feature_columns_concat_3.txt
                                                          592*4*batch,    \ # input_from_feature_columns_concat.txt
                                                          1*4*batch,      \ #input_from_feature_columns_concat_5.txt
                                                          512*4*batch,    \ #all_clk_seq_1_time.txt
                                                          512*4*batch,    \#all_clk_seq_1_st.txt
                                                          73728*4*batch,  \ #seq_input_from_feature_columns_concat_1.txt
                                                          425*4*batch,    \ #batch_fill_attributes_for_gul_rank_item_feature
                                                          425*4*batch，   \ #batch_fill_attributes_for_gul_rank_item_feature_1
                                                          27200*4*batch,  \#embedding_item_id_d_shard_embedding_2.txt
                                                          13600*4*batch,  \#embedding_item_cate_id_d_shared_embedding_2.txt
                                                          13600*4*batch,  \#embedding_item_seller_id_d_shared_embedding_2.txt
                                                          122400*4*batch, \#input_from_feature_columns_concat_4.txt
                                                          350200*4*batch, \#input_from_feature_columns_concat_1.txt
                                                         15552*4*batch,  \#seq_input_from_feature_columns_concat.txt
                                                         1161984*4*batch,\ #seq_input_from_feature_columns_concat_2.txt
                                                         178*4*batch]),    \#input_from_feature_columns_concat_7.txt
                                                         
                                         (c_int32*1)(*[425*4*batch]))   #output
            '''        
        else:
            assert(False and f"unknown model.{model}")
    
        s = time()
        self.h.odla_ExecuteComputation(self.comp, self.ctx, 0, self.device)
        t = time()
        self.logger.info("Execution time:" + str(t - s) + " sec(s)")
        return buffers
