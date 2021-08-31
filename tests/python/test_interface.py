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

def test(model_file, device, batch_size):
    import numpy as np
    service = Inference(model_file, device, batch_size)
    service.Initialize()
    import numpy
    for i in range(1, 5):
        print("iteration ", i)

        results = service.Run([np.random.rand(batch_size, 224, 224, 3) * 255.0])
        print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=open)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--batch", "-b", type=int, default="0")
    args = parser.parse_args()

    test(args.model.name, args.device, args.batch)
