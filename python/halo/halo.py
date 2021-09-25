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
import tempfile
from pathlib import Path
import subprocess
import os

debug = os.environ.get("CANAL_DEBUG")
debug = debug is not None and debug != "0"

LIB_HALO = "libhalo.so"
lib_halo = CDLL(LIB_HALO)
Compile = lib_halo.halo_Compile
AnalyzeTFPbGraph = lib_halo.halo_AnalyzeTFPbGraph


class CXXCodeGenOpts(Structure):
    _fields_ = [
        ("dialect", c_int),
        ("print_mem_stats", c_bool),
        ("emit_value_reset", c_bool),
        ("emit_value_init", c_bool),
        ("emit_value_id_as_int", c_bool),
        ("bf16_mode", c_int),
        ("exec_mode", c_int),
        ("emit_inference_func_sig", c_bool),
        ("emit_model_info_apis", c_bool),
        ("emit_dynamic_batch", c_bool),
        ("fp16_mode = false", c_bool),
        ("max_batch_size", c_int),
        ("min_batch_size", c_int),
        ("opt_batch_size", c_int),
        ("enable_ipu_device", c_bool),
        ("use_ipu_model", c_bool),
        ("separate_constants", c_bool),
        ("disable_broadcasting", c_bool),
        ("enable_type_cast", c_bool),
        ("remove_input_transpose", c_bool),
        ("remove_output_transpose", c_bool),
        ("disable_conv_bn", c_bool),
        ("ipu_num", c_int64),
        ("batches_per_step", c_int64),
        ("check_model", c_bool),
        ("api", c_int),
        ("channel_order", c_int),
        ("format_code", c_bool),
        ("emit_header", c_bool),
        ("emit_obj", c_bool),
        ("emit_shared_lib", c_bool),
        ("linked_odla_lib", c_char_p),
        ("save_temps", c_bool),
    ]


"""
int halo_Compile(halo::ModelFormat model_format, unsigned num_models,
                 const char* const models[], size_t const model_sizes[],
                 const char* target, int batch, unsigned num_input_shapes,
                 const char* const input_shapes[], unsigned num_inputs,
                 const char* const inputs[], unsigned num_outputs,
                 const char* const outputs[], const HaloCodeGenOpts* cg_opts,
                 const char* main_output_file, HaloModelInfo* model_info);
"""

Compile.argtypes = [
    c_int,  # model_format
    c_uint,  # num_models
    c_void_p,  # models
    c_void_p,  # model_sizes
    c_char_p,  # target
    c_int,  # batch
    c_uint,  # num_input_shapes
    c_void_p,  # input_shapes
    c_uint,  # num_inputs
    c_void_p,  # inputs
    c_uint,  # num_outputs
    c_void_p,  # outputs
    c_void_p,  # cg_opts
    c_char_p,  # filename
    c_void_p,  # model_info
]


def exec(args):
    proc = subprocess.run(args)
    if proc.returncode != 0:
        print(proc.stderr)
        exit(proc.returncode)


def CompileModel(model_file, input_shapes, batch, format):
    output_file = tempfile.mktemp(".cc")
    output_bin = Path(output_file).with_suffix(".bin")
    odla_lib = cast(create_string_buffer(b""), c_char_p)
    opts = CXXCodeGenOpts()
    opts.linked_odla_lib = odla_lib
    opts.channel_order = 1
    opts.api = 1
    opts.emit_inference_func_sig = True
    format_vals = {
        "TENSORFLOW": 0,
        "CAFFE": 1,
        "ONNX": 2,
        "TFLITE": 3,
        "MXNET": 4,
        "INVALID": 5,
    }
    format_val = format_vals[format]
    model_data = []
    model_sizes = []
    with open(model_file, "rb") as f:
        bytes = f.read()
        model_data.append(bytes)
        model_sizes.append(len(bytes))
    model_num = len(model_data)
    if input_shapes is None:
        input_shapes = []
    num_input_shapes = len(input_shapes)
    input_shapes_ptrs = [s.encode("utf-8") for s in input_shapes]
    num_inputs = 0
    inputs = c_void_p(0)
    outputs = c_void_p(0)
    num_outputs = 0
    input_shapes = (c_char_p * num_input_shapes)(*input_shapes_ptrs)

    target = "cxx".encode("utf-8")
    output_filename = output_file.encode("utf-8")
    Compile(
        format_val,
        model_num,
        (c_char_p * model_num)(*model_data),
        (c_size_t * model_num)(*model_sizes),
        target,
        batch,
        num_input_shapes,
        input_shapes,
        num_inputs,
        inputs,
        num_outputs,
        outputs,
        pointer(opts),
        output_filename,
        0,
    )
    return [output_file, output_bin]


def AnalyzeModel(model_file, batch):
    output_file = ""
    odla_lib = cast(create_string_buffer(b""), c_char_p)
    opts = CXXCodeGenOpts()
    opts.linked_odla_lib = odla_lib
    opts.channel_order = 1
    opts.api = 1
    opts.emit_inference_func_sig = True
    with open(model_file, "rb") as f:
        bytes = f.read()
    num_input_shapes = 0
    input_shapes = c_void_p(0)
    AnalyzeTFPbGraph(
        bytes,
        len(bytes),
        num_input_shapes,
        input_shapes,
        batch,
        pointer(opts),
        output_file.encode("utf-8"),
        0,
    )


def CompileODLAModel(files, device):
    cc_file = files[0]
    bin_file = files[1]
    if not device or device == "auto":
        device = "odla_dnnl"
    so_file = Path(files[0]).with_suffix(".so")
    opt_flag = "-g" if debug else "-O2"
    exec(
        [
            "g++",
            "-shared",
            "-fPIC",
            opt_flag,
            "-o",
            so_file,
            cc_file,
            bin_file,
            "-l" + device,
            "-Wl,-rpath=/usr/local/lib",
        ]
    )
    return so_file


def LoadODLAModel(so_file):
    return CDLL(so_file)
