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

LIB_HALO = "libhalo.so"
lib_halo = CDLL(LIB_HALO)
CompileTFPbGraph = lib_halo.halo_CompileTFPbGraph
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


# int halo_CompileTFPbGraph(const char* pb_buf, size_t pb_buf_size,
#                          size_t num_input_shapes, const char* input_shapes[],
#                          int batch, const HaloCodeGenOpts* cg_opts,
#                          const char* main_output_file,
#                          HaloModelInfo* model_info) {

CompileTFPbGraph.argtypes = [
    c_char_p,
    c_size_t,
    c_size_t,
    c_void_p,
    c_int32,
    c_void_p,
    c_char_p,
    c_void_p,
]


def exec(args):
    proc = subprocess.run(args)
    if proc.returncode != 0:
        print(proc.stderr)
        exit(proc.returncode)


def CompileModel(model_file, batch):
    output_file = tempfile.mktemp(".cc")
    output_bin = Path(output_file).with_suffix(".bin")
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
    CompileTFPbGraph(
        bytes,
        len(bytes),
        num_input_shapes,
        input_shapes,
        batch,
        pointer(opts),
        output_file.encode("utf-8"),
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
    device = "odla_popart"
    so_file = Path(files[0]).with_suffix(".so")
    exec(
        [
            "g++",
            "-shared",
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
