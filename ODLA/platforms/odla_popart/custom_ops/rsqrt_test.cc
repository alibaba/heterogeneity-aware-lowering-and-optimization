//===- rsqrt_test.cc ------------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

#include <dlfcn.h>

#include <iostream>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

// namespace CustomOperators {
//   extern const popart::OperatorIdentifier Rsqrt_1;
// }

int main(int argc, char const* argv[]) {
  void* handle = dlopen("build/libcustom_ops.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Cannot open library: " << dlerror() << std::endl;
    return 1;
  }

  auto builder = popart::Builder::create();

  // Add input tensors
  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};
  std::cout << "Adding input tensor a\n";
  auto input = builder->addInputTensor(inputInfo);

  // Add operation
  std::cout << "Adding custom operation rsqrt(input)\n";
  const popart::OperatorIdentifier rsqrt(popart::Domain::ai_graphcore, "Rsqrt",
                                         1, 1, 1);
  auto o = builder->customOp(rsqrt, 1, {input}, 1, {})[0];
  // auto o = builder->aiOnnxOpset10().sqrt({input});

  // Add output tensor
  std::cout << "Adding output tensor o\n";
  builder->addOutputTensor(o);

  std::cout << "Getting model proto\n";
  auto proto = builder->getModelProto();

  std::cout << "Constructing DataFlow\n";
  auto dataFlow = popart::DataFlow(1, {{o, popart::AnchorReturnType("ALL")}});

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};
  auto ipuModelDevice =
      popart::DeviceManager::createDeviceManager().createIpuModelDevice(
          deviceOpts);
  // or acquireAvailableDevice();

  std::cout << "Creating session from Onnx Model...\n";
  auto session = popart::InferenceSession::createFromOnnxModel(proto, dataFlow,
                                                               ipuModelDevice);
  std::cout << "Creating session from Onnx Model...done\n";

  // Prepare input tensor
  float rawInputData[2] = {16.0f, 16.0f};
  popart::NDArrayWrapper<float> inData(rawInputData, {2});
  std::map<popart::TensorId, popart::IArray&> inputs = {{input, inData}};

  // Prepare output tensor
  float rawOutputData[2] = {0, 0};
  popart::NDArrayWrapper<float> outData(rawOutputData, {2});
  std::map<popart::TensorId, popart::IArray&> anchors = {{o, outData}};

  std::cout << "Preparing session device...\n";
  session->prepareDevice();
  std::cout << "Preparing session device...done\n";

  popart::StepIO stepio(inputs, anchors);

  std::cout << "Running..."
            << "\n";
  session->run(stepio);
  std::cout << "Running...done"
            << "\n";

  std::cout << "Input Data:  " << inData << "\n";
  std::cout << "Output Data: " << outData << "\n";

  popart::logging::ir::err("inputs : {}", inData);
  popart::logging::ir::err("output : {}", outData);

  return 0;
}
