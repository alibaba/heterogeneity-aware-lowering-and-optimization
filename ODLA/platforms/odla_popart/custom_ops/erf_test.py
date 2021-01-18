#===- erf_test.py --------------------------------------------------------===//
#
# Copyright (C) 2019-2020 Alibaba Group Holding Limited.
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import ctypes
import os

import numpy as np
import popart

# Define a function to build and run the rsqrt graph with
# specified input tensor data and alpha value
def build_and_run_graph(input_data, run_on_ipu):
    builder = popart.Builder()
    input_len = len(input_data)

    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", [input_len]))
    print("Shape of {}: {}".format(input_tensor, builder.getTensorShape(input_tensor)))

    output_tensor = builder.customOp(opName="Rsqrt",
                                     opVersion=1,
                                     domain="ai.graphcore",
                                     inputs=[input_tensor],
                                     attributes={})[0]

    print("Inputs: {}".format(builder.getInputTensorIds()))
    print("Outputs: {}".format(builder.getOutputTensorIds()))
    print("Values: {}".format(builder.getValueTensorIds()))
    print("Shape of {}: {}".format(output_tensor, builder.getTensorShape(output_tensor)))

    builder.addOutputTensor(output_tensor)

    proto = builder.getModelProto()

    anchors = {output_tensor: popart.AnchorReturnType("FINAL")}
    dataFlow = popart.DataFlow(1, anchors)

    if run_on_ipu:
        device = popart.DeviceManager().acquireAvailableDevice(1)
        print("IPU hardware device acquired")
    else:
        device = popart.DeviceManager().createIpuModelDevice({})
        print("Running on IPU Model")

    session = popart.InferenceSession(proto, dataFlow, device)

    session.prepareDevice()
    result = session.initAnchorArrays()

    X = (np.array(input_data)).astype(np.float32)
    print("X={}".format(X))

    stepio = popart.PyStepIO({input_tensor: X},
                             result)
    session.run(stepio)

    return result

def load_custom_ops_lib():
    so_path = os.path.join(os.path.dirname(__file__),
                           "build/custom_ops.so")

    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)

    ctypes.cdll.LoadLibrary(so_path)


# How to run: python rsqrt_test.py 16.0 16.0 16.0 16.0
if __name__ == '__main__':
    load_custom_ops_lib()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ipu", help="run on available IPU hardware device",
                        action='store_true')
    parser.add_argument('input_data', metavar='X', type=float, nargs='+',
                        help='input tensor data')

    args = parser.parse_args()

    result = build_and_run_graph(args.input_data, args.ipu)

    print("RESULT X")
    print(result)

