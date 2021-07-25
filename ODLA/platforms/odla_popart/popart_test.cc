// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <thread>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>
#include <list>
#include <unistd.h>

using namespace popart;

namespace {

std::shared_ptr<popart::DeviceInfo> acquireIpu() {
  // keep trying to attach to a device until one is available (this may not
  // always be the case as other tests might be running in parallel).
  while (true) {
    if (auto d = createTestDevice(TEST_TARGET, 2, 1216)) {
      return d;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return nullptr;
}

} // unnamed namespace

std::string get_ModelProto()
{
  Shape inShape = {1};
  TensorInfo inInfo{"INT32", inShape};

  Shape constShape = {1};
  std::vector<int> rawConstInputData = {1};
//  std::iota(rawConstInputData.begin(), rawConstInputData.end(), 1);

  popart::NDArrayWrapper<int> constData(rawConstInputData.data(), {1});

  ConstVoidData constShapeData = {rawConstInputData.data(),
                                  {"INT32", constShape}};

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  auto constId = aiOnnx.constant(constShapeData, "out0ShapeData");
  auto inId    = builder->addInputTensor(inInfo);
  auto inId2   = builder->addInputTensor(inInfo); //--

  std::cout << "----> constId: " << constId << std::endl;
  builder->virtualGraph(constId, 0);
  builder->pipelineStage(constId, 0);

  //auto add        = aiOnnx.add({constId, inId});
  auto add        = aiOnnx.add({inId2, inId});
  std::cout << "-----> add: " << add << std::endl;
  builder->virtualGraph(add, 0);
  builder->pipelineStage(add, 0);

  auto outShapeId = aiOnnx.transpose({add}, {});
  std::cout << "----> outShapeId: " << outShapeId << std::endl;
  builder->virtualGraph(outShapeId, 1);
  builder->pipelineStage(outShapeId, 1);


  auto transpose = aiOnnx.transpose({outShapeId}, {});
  std::cout << "----> transpose: " << transpose << std::endl;
  //auto softmax = aiOnnx.softmax({add});
  builder->virtualGraph(transpose, 2);
  builder->pipelineStage(transpose, 2);
  
  
  auto out = aiOnnx.sqrt({transpose});
  std::cout << "----> out: " << out << std::endl;
  builder->virtualGraph(out, 3);
  builder->pipelineStage(out, 3);
 
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();

  //Print the pipline stage and ipu number

  builder->saveModelProto("test.onnx");
  //std::cout << "proto is: \n" << proto << std::endl;
  auto tensorIds = builder->getValueTensorIds();
  for(auto& tensorid : tensorIds){
    std::cout << "tensorid: " << tensorid << std::endl;
  }
  auto modelProto = io::getModelFromString(proto);

  return proto;
}

int main() {
  std::list<int> inputs;

  //auto proto = get_ModelProto();
  std::string proto = "test.onnx";
  // Create the IR, adding outId as an anchor
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(6, {{"Sqrt:0", art}});
  auto device =
      DeviceManager::createDeviceManager().acquireAvailableDevice(4);

//  auto device = popart::createTestDevice(TEST_TARGET);
  auto opts                   = SessionOptions();
  opts.enablePipelining = true;
  opts.virtualGraphMode = VirtualGraphMode::Manual;

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns({popart::PreAliasPatternType::PostNRepl})
          .enableRuntimeAsserts(false));

  // prepare the anchors
  int rawOutputData[10] = {0};
  popart::NDArrayWrapper<int> outData(rawOutputData, {2, 5});

  // std::map<popart::TensorId, popart::IArray &> anchors = {
  //     {out, outData},
  // };

  session->prepareDevice();

  int rawInputData[10] = {
   99, 
  };

  inputs.push_back(48); 
  inputs.push_back(1);
  inputs.push_back(3);
  inputs.push_back(1);
  inputs.push_back(8);
  inputs.push_back(1); 
  inputs.push_back(15);
  inputs.push_back(1); 
  inputs.push_back(24);
  inputs.push_back(1); 
  int i = 0;
  popart::StepIOCallback::InputCallback input_callback =
      [&](TensorId id, bool prefetch) -> ConstVoidData {
    popart::logging::info("input callback called {}", id);
    std::cout << "input callback called with tensorid: " << id << std::endl;
    (void)prefetch;
    int input_data = 0;
    if (inputs.size() > 0) {
        input_data = *(inputs.begin());
        std::cout << "The input data value is: " << input_data << std::endl;
    }
    else {
      std::cout << "empty queue" << std::endl;
      input_data = -1;
    }
    popart::NDArrayWrapper<int> inData(&input_data, {1});

    ConstVoidData data;
    data.data = inData.data();
    data.info = TensorInfo(DataType::INT32, {1});
    return data;
  };

  popart::StepIOCallback::InputCompleteCallback input_complete_callback =
      [&](TensorId id) -> void {
    popart::logging::info("input complete callback called {}", id);
    std::cout << "InputCompleteCallback called with tensorid: " << id << std::endl;
    if (inputs.size() > 0) {
        inputs.pop_front();
    }
  };

  popart::StepIOCallback::OutputCallback output_callback =
      [&](TensorId id) -> MutableVoidData {
    popart::logging::info("output callback called {}", id);
    std::cout << "OutputCallback called with tensorid: " << id << std::endl;
    popart::NDArrayWrapper<int> outData(rawOutputData, {1});

    MutableVoidData data;
    data.data = outData.data();
    data.info = TensorInfo(DataType::INT32, {1});
    return data;
  };

  popart::StepIOCallback::OutputCompleteCallback output_complete_callback =
      [&](TensorId id) -> void {
    std::cout << "OutputCompleteCallback called with tensorid: " << id << std::endl;
    for (int i=0; i < 1; ++i) {
      std::cout << rawOutputData[i] << " " << std::endl;
    }

    popart::logging::info("output complete callback called {}", id);
  };

  popart::StepIOCallback stepio(input_callback,
                                input_complete_callback,
                                output_callback,
                                output_complete_callback);

  for (int i = 0; i < 5; ++i) {
    session->run(stepio);
  }

  return 0;
}