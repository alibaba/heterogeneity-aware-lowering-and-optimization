#!/usr/bin/env python3
# RUN: %s --help

"""
Copyright (C) 2019-2021 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================
"""

from halo.inference import Inference
import argparse
from time import time
import logging
import sys


def test(
    model_file,
    device,
    batch_size,
    format,
    input_data_files,
    input_shape,
    output_name,
    ref_data_files,
    it,
    debug,
    log_level,
):
    import numpy as np

    if debug:
        log_level = logging.DEBUG
    inputs = []
    ref_outs = []
    for in_file in input_data_files:
        if str(in_file).find("_int64") >= 0:
            inputs.append(np.loadtxt(in_file, dtype=np.int64))
        else:
            inputs.append(np.loadtxt(in_file, dtype=np.float32))
    for in_file in ref_data_files:
        ref_outs.append(np.loadtxt(in_file, dtype=np.float32))

    service = Inference(
        model_file,
        input_shape,
        output_name,
        device,
        batch_size,
        format,
        debug,
        log_level,
    )
    service.Initialize()
    # Warm up, not count to it.
    results = service.Run(inputs)
    inference_time = 0
    ok = True
    for i in range(1, it + 1):
        s = time()
        results = service.Run(inputs)
        t = time()
        inference_time += t - s
        print(f"Overall iteration {i} inference time {(t-s)*1000:.2f} ms")
        for idx, data in enumerate(results):
            of = "/tmp/out." + str(idx) + ".txt"
            np.savetxt(of, data, delimiter="\n")
            ok = ok and np.allclose(ref_outs[idx], data, rtol=1e-1, atol=1e-1)
            if not ok:
                print("Mismatch of results " + str(idx))
                return False

    print(
        f"Results Verified, average inference time {(inference_time / it) * 1000:.2f} ms for {it} iterations"
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=open, required=True)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--batch", "-b", type=int, default="0")
    parser.add_argument("--format", "-f", type=str, default="")
    parser.add_argument("--input-data", "-i", nargs="+", type=open)
    parser.add_argument("--input-shape", nargs="*", type=str)
    parser.add_argument("--output-name", nargs="*", type=str)
    parser.add_argument("--ref-output", "-r", nargs="+", type=open)
    parser.add_argument("--iteration", "-it", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="log_level",
        const=logging.INFO,
        default=logging.WARNING,
    )

    args = parser.parse_args()

    ret = test(
        args.model.name,
        args.device,
        args.batch,
        args.format,
        args.input_data,
        args.input_shape,
        args.output_name,
        args.ref_output,
        args.iteration,
        args.debug,
        args.log_level,
    )

    sys.exit(0 if ret else -1)
