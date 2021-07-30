//
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
#include <iostream>

// namespace CustomOperators {
//   extern const popart::OperatorIdentifier Rsqrt_1;
// }

int main(int argc, char const* argv[]) {
  std::cout << "=====> 1, OK" << std::endl;

  void* handle = dlopen("build/libcustom_ops.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Cannot open library: " << dlerror() << std::endl;
    return 1;
  }
  std::cout << "=====> 2, OK" << std::endl;
  
  auto builder = popart::Builder::create();

  // Add input tensors
  popart::TensorInfo input_mask_info{popart::DataType::UINT32, std::vector<int64_t>{1, 384}};
  std::cout << "Adding input tensor input_mask\n";
  auto input_mask = builder->addInputTensor(input_mask_info);
  
  popart::TensorInfo data_info{popart::DataType::FLOAT, std::vector<int64_t>{1, 16, 384, 384}};
  std::cout << "Adding input tensor data\n";
  auto data = builder->addInputTensor(data_info);

  // Add operation
  
  std::cout << "Adding custom operation attention_mask(input_mask, data)\n";
  const popart::OperatorIdentifier attention_mask(popart::Domain::ai_graphcore, "AttentionMask", 1, 2, 1);
  auto o = builder->customOp(attention_mask, 1, {input_mask, data}, 1, {{"dataType", data_info.data_type()}})[0];

  std::cout << "Get the tensor type and tensor shape of the output of AttentionMask with tensorid: " << o << std::endl;
  builder->getTensorDataType(o);
  builder->getTensorShape(o);
  std::cout << "==================================================" << std::endl;
  
  std::cout << "The out of the customOp is: " << o << std::endl;
  auto out1 = builder->aiOnnxOpset10().add({o, o});
  //auto out1 = builder->aiOnnxOpset10().add({input_mask, input_mask});

  // Add output tensor
  std::cout << "Adding output tensor o\n";
  builder->addOutputTensor(out1);

  std::cout << "Getting model proto\n";
  auto proto = builder->getModelProto();
  builder->saveModelProto("attention_mask_test.onnx");
  
  std::cout << "Constructing DataFlow\n";
  auto dataFlow = popart::DataFlow(1, {{out1, popart::AnchorReturnType("ALL")}});
  
  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};
  auto ipuModelDevice =
      //popart::DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);
      popart::DeviceManager::createDeviceManager().acquireAvailableDevice(1);
  
  std::cout << "Creating session from Onnx Model...\n";
  auto session = popart::InferenceSession::createFromOnnxModel(
      proto, dataFlow, ipuModelDevice);
  std::cout << "Creating session from Onnx Model...done\n";
  
  // Prepare input tensor
  uint32_t  rawInputData[1 * 384] = {};
  std::fill_n(rawInputData, 384, 1);
  popart::NDArrayWrapper<uint32_t> input_mask_(rawInputData, {1, 384});
  float* rawInputData2 = new float[1 * 16 * 384 * 384];
  std::fill_n(rawInputData2, 1*16*384*384, 1.0);
  popart::NDArrayWrapper<float> data_(rawInputData2, {1, 16, 384, 384});
  std::map<popart::TensorId, popart::IArray &> inputs = {{input_mask, input_mask_}, {data, data_}};
  
  // Prepare output tensor
  float* rawOutputData = new float[1 * 1 * 384 * 384];
  std::fill_n(rawOutputData, 1*1*384*384, 2.0);
  popart::NDArrayWrapper<float> outData(rawOutputData, {1, 1, 384, 384});
  std::map<popart::TensorId, popart::IArray &> anchors = {{out1, outData}};

  std::cout << "Preparing session device...\n";
  session->prepareDevice();
  std::cout << "Preparing session device...done\n";
  
  popart::StepIO stepio(inputs, anchors);

  std::cout << "Running..."
            << "\n";
  session->run(stepio);
  std::cout << "Running...done"
            << "\n";

  std::cout << "input_mask:  " << input_mask << "\n";
  std::cout << "data" << data << std::endl;
  std::cout << "Output Data: " << outData << "\n";

  // popart::logging::ir::err("inputs : {}", input_mask);
  // popart::logging::ir::err("inputs : {}", data);
  // popart::logging::ir::err("output : {}", outData);
  
  return 0;
}
