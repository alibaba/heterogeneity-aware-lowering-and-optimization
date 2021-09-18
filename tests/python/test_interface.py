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

def test(model_file, device, batch_size, format, input_data_files, ref_data_files):
    import numpy as np
    inputs = []
    ref_outs = []
    for in_file in input_data_files:
        inputs.append(np.loadtxt(in_file, dtype=np.float32))
    for in_file in ref_data_files:
        ref_outs.append(np.loadtxt(in_file, dtype=np.float32))

    service = Inference(model_file, device, batch_size, format)
    service.Initialize()

    for i in range(1, 2):
        print("iteration ", i)

        results = service.Run(inputs)
        for idx, data in enumerate(results):
            of = "/tmp/out." + str(idx) + ".txt"
            np.savetxt(of, data, delimiter="\n")
        ok = np.allclose(ref_outs[idx], results[idx], rtol=1e-3, atol=1e-3)
        if ok:
            print("Results Verified")
        else:
            print("Incorrect Results")
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=open, required=True)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--batch", "-b", type=int, default="0")
    parser.add_argument("--format", "-f", type=str, default="")
    parser.add_argument("--input-data", "-i", nargs="+", type=open)
    parser.add_argument("--ref-output", "-r", nargs="+", type=open)

    args = parser.parse_args()

    test(
        args.model.name,
        args.device,
        args.batch,
        args.format,
        args.input_data,
        args.ref_output,
    )
