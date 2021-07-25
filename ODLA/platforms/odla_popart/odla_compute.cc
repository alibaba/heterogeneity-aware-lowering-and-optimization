//===- odla_compute.cc ----------------------------------------------------===//
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

#include <ODLA/odla.h>
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>
#include <popart/names.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <array>
#include <fstream>
#include <sstream>

#include "ODLA/odla_common.h"
#include "common.h"
#include "odla_popart.h"

#if !defined(ODLA_VERSION_NUMBER) || (ODLA_VERSION_NUMBER < 50)
#error This library requires minimum ODLA version 0.5
#endif

/*
测试pipeline的使用，我们要做的事情如下：
1. 定义一个全局（或者单建）的queue，用来存储输入的请求；一个全局的输出queue，用来存放输出（或者直接讲输出写回给线程）
2. 开始就启动一个无限循环，用以执行stepIO (应该可以在create context之后就启动)
3. 定义一组callback的函数：取数据，取数据完毕，写输出，写输出完毕
4. 在odla_execution的时候，把数据放入输入的queue，然后阻塞 （需要看这部分调用是在odla内部创建线程池，还是直接使用调用者已有的线程。初步认为，原有调用是阻塞式调用，可以直接阻塞原有caller线程）
5. 一个线程专门遍历输出的Queue，然后去解阻塞caller的线程。假设stepIO的回调是顺序的，前面存储请求的同时，要顺序存储caller线程的对象或者ID
6. 
*/
struct SingleComp{
  odla_computation single_comp;
  static std::mutex instance_mutex;
  static SingleComp* instance;
  odla_computation get_comp(){return single_comp;}
  static SingleComp* get_instance()
  {
    if(nullptr == instance){
      std::lock_guard<std::mutex> guard(instance_mutex);
      if(nullptr == instance){
        instance = new SingleComp();
        //Create the single computation
        std::unique_ptr<popart::Builder> builder = popart::Builder::create();
        // Place Subgraph on IPU 0
        builder->setAttribute(popart::sVirtualGraphAttribute, 0);
        instance->single_comp = new _odla_computation(std::move(builder));
      }
    }
    return instance;
  }
};
SingleComp* SingleComp::instance = nullptr;
std::mutex SingleComp::instance_mutex;
//因为其他地方还在用这个thread_local的g_comp，所以还得保留这个
thread_local odla_computation g_comp = SingleComp::get_instance()->get_comp();
static bool pipeline_loop_running = true;
//static odla_context test_ctx = nullptr;
void pipeline_loop(odla_computation comp);

/* 这里的数据可以只创建一次，循环使用，反正输入就是创建0，输出随便覆盖，因为后面不会使用
   那，我们后面是把这个context，和这个queue的任务分开么？
   在put的时候用一个wrapper把ctx包起来，里面包含状态信息和mutex，condition_variable，在返回输出的时候拆掉wrapper。
   那这个wrapper是不是就会跟输入输出产生关联，需要加锁了？
   还是继续用context，让他们去delete吧。那这里生成的谁负责delete？自己玩的，是在一个线程里面么？
   因为输出要检查是不是所有的tensor都整完了，就是得有context和状态，
 */
odla_context create_empty_odla_context()
{
  //if(nullptr != test_ctx)
  //  return test_ctx;
  std::cout << "-----------------> create an empty input/output context" << std::endl;
  odla_context ctx = new _odla_context(SingleComp::get_instance()->get_comp());
  //initialize?
  for(auto& value : ctx->comp->input_values){
    std::size_t num_elements = 1;
    for(auto& shape : value->tensor_info.shape())
      num_elements *= shape;
    float* data = new float[num_elements];
    std::fill_n(data, num_elements, 0);
    odla_BindToArgument(value, data, ctx);
  }
  for(auto& value : ctx->comp->output_values){
    std::size_t num_elements = 1;
    for(auto& shape : value->tensor_info.shape())
      num_elements *= shape;
    float* data = new float[num_elements];
    std::fill_n(data, num_elements, 0);
    odla_BindToOutput(value, data, ctx);
  }
  return ctx;
}

struct ContextQueues{
  std::queue<odla_context> input_queue_1;
  std::queue<odla_context> input_queue_2;
  std::queue<odla_context> wait_output_queue;
  std::mutex write_mutex;
  std::queue<odla_context>* read_queue;
  std::queue<odla_context>* write_queue;
  ContextQueues():read_queue(&input_queue_1)
    ,write_queue(&input_queue_2)
    , input_ctx(nullptr), output_ctx(nullptr){}
  void put(odla_context ctx);
  odla_context get_input_context();
  void all_tensor_read(){
    std::cout << "ContextQueues::all_tensor_read(), ctx: " << input_ctx << " poped, and put into wait_output_queue" << std::endl;
    read_queue->pop();
    wait_output_queue.push(input_ctx);
    input_ctx = nullptr;
  }
  //void get_complete(TensorId id);  这个在context里面查看是不是所有的都visite了，如果是会在回调里面搞一下
  odla_context get_output_context();
  void all_tensor_written(){
    wait_output_queue.pop();
    output_ctx = nullptr;
  }
  odla_context input_ctx;  //当前正在被操作的，作为输入的context
  odla_context output_ctx; //当前正在被操作的，作为输出的context
  //std::set<std::string> tensor_visited;
  static ContextQueues* p_context_queues;
  static std::mutex instance_mutex;
  static ContextQueues* get_instance(){
    if(nullptr == p_context_queues){
      std::lock_guard<std::mutex> guard(instance_mutex);
      if(nullptr == p_context_queues){
        p_context_queues = new ContextQueues();
        std::cout << "Here is OK" << std::endl;
        //Create empty context

        //启动thread
        std::thread pipeline_thread(pipeline_loop, SingleComp::get_instance()->get_comp());
        // std::thread pipeline_thread(pipeline_loop);
        pipeline_thread.detach();
      }
    }
    return p_context_queues;
  }
};
ContextQueues* ContextQueues::p_context_queues = nullptr;
std::mutex ContextQueues::instance_mutex;

void ContextQueues::put(odla_context ctx)
{
  std::cout << "ContextQueues::put -> ctx: " << ctx << std::endl;
  std::lock_guard<std::mutex> guard(write_mutex);
  write_queue->push(ctx);
}

//这里先按照回调函数必须是串行的进行设计，那么get是回调时才会调用的，顺便把拿出来的ctx放到等待队列里面
odla_context ContextQueues::get_input_context()
{
  if(nullptr != input_ctx){
    return input_ctx;
    //tensor_visited.clear();
  }
  input_ctx = read_queue->front();
  if( nullptr == input_ctx)
  {
    std::lock_guard<std::mutex> guard(write_mutex);
    std::queue<odla_context>* tmp = read_queue;
    read_queue = write_queue;
    write_queue = tmp;
    input_ctx = read_queue->front();
    std::cout << "===============> switched the read write queue" << std::endl;
  }
  if(nullptr == input_ctx)
    input_ctx = create_empty_odla_context(); //test_ctx;  //TODO: 必须指向一个context结构，可以包含空数据，因为是按照tensor多次读取的。需要保持状态？
  // if(nullptr != ctx) //不能在这里直接pop，按tensor取的，得确保所有的tensor都取完了才行
  //   read_queue.pop();
  //   wait_output_queue.push(ctx);  //现在是如果没有数据就不push，后面如果没有数据的时候如果要补数据，跟这个context没关系？应该有关系吧，等数据的就是空吧，因为补的数据不会有别人等
  return input_ctx;
}

odla_context ContextQueues::get_output_context()
{
  output_ctx = wait_output_queue.front();
  if(nullptr == output_ctx)
    std::cerr << "No context in the queue when an output gotten" << std::endl; //严重错误了，会导致数据匹配补上了，是不是可以考虑把输入的数据也放到输出里面，比较一下MD5来确保对应关系
  return output_ctx;
  //解阻塞啥的逻辑在外面搞吧，这里只跟Queue相关
}

#define CB_NULL_QUEUE 100
#define CB_NULL_TENSOR 101

popart::StepIOCallback::InputCallback input_callback =
    [&](popart::TensorId id, bool prefetch) -> popart::ConstVoidData {
  popart::logging::info("input callback called {}", id);
  (void)prefetch;
  //Get the data from the context queues
  popart::ConstVoidData data;
  odla_context ctx = ContextQueues::get_instance()->get_input_context();
  if(nullptr != ctx)
  {
    std::cout << "input_callback got ctx: " << ctx << std::endl;
    popart::IArray* p_array = ctx->get_data_by_tensor_id(id);
    if(NULL != p_array)
    {
      data.data = p_array->data();
      std::cout << "input_callback -> the p_array dataType is: " << p_array->dataType() << ", shape is: " << p_array->shape() << std::endl;
      data.info = popart::TensorInfo(p_array->dataType(), p_array->shape());
    }
    else
    {
      std::cerr << "input_callback -> Can not find the tensor with id: " << id << " in ctx: " << ctx << std::endl;
      exit(CB_NULL_TENSOR);
    }
  }
  else
  {
    std::cout << "input_callback -> Queue is empty when try to get" << std::endl;
    //std::cout << "input_callback -> Use the test_ctx" << std::endl;
    std::cout << "input_callback -> return nullptr data" << std::endl;
    //ctx = test_ctx;
    // exit(CB_NULL_QUEUE);
    data.data = nullptr;
    odla_computation comp = SingleComp::get_instance()->get_comp();
    // auto search = comp->inputs_map.find(id);
    // if(comp->inputs_map.end() != search)
    //   data.info = search->second->tensor_info;
    data.info = comp->inputs_map[id]->tensor_info;
  }
  return data;
};

popart::StepIOCallback::InputCompleteCallback input_complete_callback =
    [&](popart::TensorId id) -> void {
  std::cout << "input_complete_callback -> called: " << id << std::endl;
  odla_context ctx = ContextQueues::get_instance()->get_input_context();
  if(nullptr == ctx)
  {
    std::cout << "input_complete_callback -> Queue is empty when try to get" << std::endl;
    exit(CB_NULL_QUEUE);
  }
  if(ctx->all_tensors_visited()){
    std::cout << "input_complete_callback -> All tensors read for current context:" << ctx << std::endl;
    ContextQueues::get_instance()->all_tensor_read();
  }
};

popart::StepIOCallback::OutputCallback output_callback =
    [&](popart::TensorId id) -> popart::MutableVoidData {
  popart::logging::info("output callback called {}", id);
  std::cout << "output_callback -> called with id: " << id << std::endl;
  odla_context ctx = ContextQueues::get_instance()->get_output_context();
  if(nullptr == ctx)
  {
    std::cout << "output_callback <- Queue is empty when try to get" << std::endl;
    exit(CB_NULL_QUEUE);
  }
  popart::IArray* p_array = ctx->write_data_by_tensor_id(id);
  if(NULL == p_array)
  {
    std::cerr << "output_callback <- Can not find the tensor with id: " << id << " in ctx: " << ctx << std::endl;
    exit(CB_NULL_TENSOR);
  }
  popart::MutableVoidData data;
  data.data = p_array->data();
  data.info = popart::TensorInfo(p_array->dataType(), p_array->shape());
  return data;
};

popart::StepIOCallback::OutputCompleteCallback output_complete_callback =
    [&](popart::TensorId id) -> void {
  popart::logging::info("output complete callback called {}", id);
  std::cout << "output_complete_callback -> called with id: " << id << std::endl;
  odla_context ctx = ContextQueues::get_instance()->get_output_context();
  if(nullptr == ctx)
  {
    std::cout << "output_complete_callback -> Queue is empty when try to get" << std::endl;
    exit(CB_NULL_QUEUE);
  }
  if(ctx->all_tensors_written()){
    std::cout << "output_complete_callback -> All tensors written for current context waiting output: " << ctx << std::endl;
    ContextQueues::get_instance()->all_tensor_written();
    ctx->clear_visited_and_written();
    ctx->notify();  //解阻塞，返回吧
  }
};

static std::shared_ptr<popart::DeviceInfo> AcquireAvailableDevice(
    int num_devices) {
  return popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
      num_devices);
}

static std::shared_ptr<popart::DeviceInfo> CreateIpuModelDevice(
    int num_devices) {
  std::cout << "---> CreateIpuModelDevice()" << std::endl;
  std::map<std::string, std::string> deviceOpts{
      {"numIPUs", std::to_string(num_devices)}, {"tilesPerIPU", "1216"}};
  std::cout << "<--- CreateIpuModelDevice()" << std::endl;
  return popart::DeviceManager::createDeviceManager().createIpuModelDevice(
      deviceOpts);
}

std::unique_ptr<popart::SessionOptions> SessionOptions() {
  std::cout << "---> SessionOptions()" << std::endl;
  auto opts =
      std::unique_ptr<popart::SessionOptions>(new popart::SessionOptions());
  
  // opts->virtualGraphMode = popart::VirtualGraphMode::Auto;
  // opts->enableStochasticRounding = true;
  opts->enablePipelining = true;
  opts->virtualGraphMode = popart::VirtualGraphMode::Manual;
  std::cout << "<--- SessionOptions()" << std::endl;
  return opts;
}

void pipeline_loop(odla_computation comp)
// void pipeline_loop()
{
  std::cout << "There is not any execution here." << std::endl;
  // odla_computation comp = g_comp;
  if(nullptr != comp->session)  //先放在这里吧
  {
    std::cerr << "||-______________________________________-||" << std::endl;
  }
  std::cout << "Creating the data flow" << std::endl;
  comp->opts.ipu_num = 2;
  comp->opts.batches_per_step = 1000;

  // Create dataflow
  std::vector<popart::TensorId> ids;
  for (const auto& output : comp->outputs_map) {
    std::cout << "dataflow tensorid: " << output.second->tensor_id << std::endl;
    ids.push_back(output.second->tensor_id);
  }

  // Batches per step is a compile time constant value
  popart::DataFlow data_flow(comp->opts.batches_per_step, ids,
                            popart::AnchorReturnType("All"));

  std::cout << "Data flow created" << std::endl;
  // Acquire IPU
  auto device = comp->opts.use_ipu_model
                    ? CreateIpuModelDevice(comp->opts.ipu_num)
                    : AcquireAvailableDevice(comp->opts.ipu_num);
  std::cout << "MUMUMUMU" << std::endl;
  // Create and config SessionOptions
  auto opts = SessionOptions(); //Manual & pipeline

  std::cout << "-------------------------------------- tensorIds" << std::endl;
  auto tensorIds = comp->builder->getValueTensorIds();
  int i = 0;
  for(auto& tensorid : tensorIds){
    std::cout << tensorid << std::endl;
  }
  // Create InferenceSession
  // auto proto = comp->builder->getModelProto();
  // // Save to onnx file
  // comp->builder->saveModelProto("test_mnist.onnx");
  // auto session = popart::InferenceSession::createFromOnnxModel(
  //     proto, data_flow, device, popart::InputShapeInfo(), *opts);
  auto session = popart::InferenceSession::createFromOnnxModel(
      "new_mnist.onnx", 
      data_flow, 
      device, 
      popart::InputShapeInfo(), 
      *opts//,
      //popart::Patterns({popart::PreAliasPatternType::PostNRepl})
      //    .enableRuntimeAsserts(false)
      );
  comp->session = std::move(session);
  // Compile graph, create engine and load into the IPU
  // use compileAndExport() to frozen engine to specified path
  comp->session->prepareDevice();
  // Init seed
  comp->session->setRandomSeed(0);
  // Copy weights from host to IPU
  comp->session->weightsFromHost();
  //这里就要启动thread循环读取数据了
  popart::StepIOCallback stepio(input_callback,
                              input_complete_callback,
                              output_callback,
                              output_complete_callback);
  //循环吧
  //std::this_thread::sleep_for(std::chrono::seconds(10));
  //int i=0;
  //while(pipeline_loop_running){
  for(int i=0; i<10; i++){
    std::cout << "LAIYA LAIYA, this is the " << i << " time for the inference" << std::endl;
    comp->session->run(stepio);
  }
  std::cout << "The run finished" << std::endl;
}

odla_status odla_SetComputationItem(odla_computation comp, odla_item_type type,
                                    odla_item_value value) {
  std::cout << "---> odla_SetComputationItem()" << std::endl;
  switch (type) {
    //case ODLA_USE_IPU_MODEL:
    case ODLA_USE_SIM_MODE:
      comp->opts.use_ipu_model = *(reinterpret_cast<bool*>(value));
      break;
    //case ODLA_IPU_NUM:
    case ODLA_PROCESSOR_NUM:
      comp->opts.ipu_num = *(reinterpret_cast<int*>(value));
      break;
    case ODLA_BATCHES_PER_STEP:
      comp->opts.batches_per_step = *(reinterpret_cast<int*>(value));
      break;
    //这是可以扩展的部分，比如PIPELINE的设置可以通过这个进行设置
    //这个_odla_item_value的定义在哪里？没有找到，好像可以转化成任意形式的指针，
    //那是不是就可以作为一个PIPELINE的结果传进来，halo编译的时候又怎么做切分呢？需要手动修改代码？
    //还是需要通过autopipeline获取到切分点的参数传进来？
    //应该是包括哪些stage，谁在哪个virtual group？
    default:
      std::cerr << "Unsupported property type: " << type << std::endl;
      return ODLA_FAILURE;
  }
  std::cout << "<--- odla_SetComputationItem()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_CreateComputation(odla_computation* comp) {
  // Create graph builder
  std::cout << "---> odla_CreateComputation()" << std::endl;
  static void* custom_op_handle = nullptr;
  // TODO(unknown) support shard mode
  // builder->virtualGraph(inst_tensor_id, 0/*device id*/);
  //大家都一样吧，同一个图,直接搞成静态成员就好了，可见范围是整个静态范围
  // if(nullptr == g_comp){
  //   std::unique_ptr<popart::Builder> builder = popart::Builder::create();
  //   // Place Subgraph on IPU 0
  //   builder->setAttribute(popart::sVirtualGraphAttribute, 0);
  //   g_comp = new _odla_computation(std::move(builder));
  //   //这里图还没创建，所以要在创建context的时候对图进行pipeline，Todo: 上面的Virutualgraph是不是就得注释掉了
  // }
  *comp = SingleComp::get_instance()->get_comp();
  if (custom_op_handle == nullptr) {
    custom_op_handle = dlopen("libcustom_ops.so", RTLD_NOW | RTLD_GLOBAL);
    if (custom_op_handle == nullptr) {
      std::cerr << "Unable to open libcustom_ops " << dlerror() << std::endl;
      assert(0);
      return ODLA_FAILURE;
    }
  }
  std::cout << "<--- odla_CreateComputation()" << std::endl;
  return ODLA_SUCCESS;
}

//Pipeline 的定义需要在这里确定，virtual graph是啥样的，节点的virtualGraph归属情况等
//创建pipeline，创建循环。从HALO生成的代码来看，Context也只被创建一次，而会被多次Bind，越发觉得ODLA这个不是线程安全的
//我希望的是并不限制这个数量，每次数据都会生成新的context，但session只有一个，context创建好之后会在执行时加入队列
//
//这个还是有问题，session应该只有一个，但不能在computation的时候创建，因为这个时候output，input等构图还没有完成，只能是在第一个context创建的时候去做，或者在第一次执行的时候去做
//Session应该只有一个吧，attach硬件应该只做一次吧，session应该和图对应
odla_status odla_CreateContext(odla_context* context) {
  std::cout << "---> odla_CreateContext()" << std::endl;
  *context = new _odla_context(SingleComp::get_instance()->get_comp());
  //if(nullptr == test_ctx)
  //  test_ctx = *context;
  std::cout << "<--- odla_CreateContext()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context ctx) {
  std::cout << "---> odla_DestroyContext()" << std::endl;
  if(nullptr != ctx)
    delete (ctx);
  else
    std::cerr << "Encounter a odla_DestroyContext with null ctx" << std::endl;
  std::cout << "<--- odla_DestroyContext()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation comp) {
  // g_comp.reset();
  return ODLA_SUCCESS;
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  std::cout << "---> odla_ExecuteComputation()" << std::endl;
  //在这里直接把context加到Queue里面，这里面的逻辑可以放到从Queue里面取数据的部分完成。
  ContextQueues::get_instance()->put(context);
  context->wait();
  // Config StepIO
  // std::map<popart::TensorId, popart::IArray&> inputs;
  // for (auto& input : context->inputs) {
  //   inputs.emplace(input.first, *input.second);
  // }
  // std::map<popart::TensorId, popart::IArray&> outputs;
  // for (auto& output : context->outputs) {
  //   outputs.emplace(output.first, *output.second);
  // }

  // popart::StepIO stepio(inputs, outputs);
  // // Run on ipu
  // comp->session->run(stepio);
  // std::cout << "<--- odla_ExecuteComputation()" << std::endl;
  return ODLA_SUCCESS;
}

odla_value odla_CreateArgument(odla_value_type type, const odla_value_id id) {
  std::cout << "---> odla_CreateArgument()" << std::endl;
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  popart::TensorInfo tensor_info(GetPopartType(type),
                                 GetPopartShape(type.shape));
  auto comp = SingleComp::get_instance()->get_comp();
  popart::TensorId tensor_id =
      comp->builder->addInputTensor(tensor_info, name);
  auto v = new _odla_value(tensor_id, tensor_info, name);
  comp->inputs_map[name] = v;
  comp->input_values.push_back(v);
  std::cout << "<--- odla_CreateArgument()" << std::endl;
  return v;
}

odla_status odla_GetNumOfArgsFromComputation(const odla_computation computation,
                                             odla_uint32* num_args) {
  std::cout << "---> odla_GetNumOfArgsFromComputation()" << std::endl;
  *num_args = computation->input_values.size();
  std::cout << "<--- odla_GetNumOfArgsFromComputation()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromComputationByIdx(const odla_computation computation,
                                            const odla_uint32 arg_idx,
                                            odla_value* arg_value) {
  std::cout << "---> odla_GetArgFromComputationByIdx()" << std::endl;
  *arg_value = nullptr;
  if (arg_idx >= computation->input_values.size()) {
    return ODLA_FAILURE;
  }
  *arg_value = computation->input_values[arg_idx];
  std::cout << "<--- odla_GetArgFromComputationByIdx()" << std::endl;
  return ODLA_SUCCESS;
}

odla_value odla_CreateConstant(odla_value_type type, const void* data_ptr,
                               const odla_value_id id) {
  std::cout << "---> odla_CreateConstant()" << std::endl;
  const auto& name = id ? std::string(reinterpret_cast<const char*>(id)) : "";
  popart::TensorInfo tensor_info(GetPopartType(type),
                                 GetPopartShape(type.shape));
  popart::ConstVoidData data = {
      data_ptr, {GetPopartType(type), GetPopartShape(type.shape)}};
  popart::TensorId tensor_id =
      SingleComp::get_instance()->get_comp()->builder->aiOnnxOpset10().constant(data, name);
  std::cout << "<--- odla_CreateConstant()" << std::endl;
  return new _odla_value(tensor_id, tensor_info, name);
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  std::cout << "---> odla_BindToArgument() : " << context << std::endl;
  // context->clear_visited_and_written();
  std::unique_ptr<popart::IArray> p_array = MakeNDArrayWrapper(
      data_ptr, context->comp->builder->getTensorDataType(value->tensor_id),
      context->comp->builder->getTensorShape(value->tensor_id));
  // context->comp->inputs[value->tensor_id] = std::move(p_array);
  context->inputs[value->tensor_id] = std::move(p_array);
  std::cout << "<--- odla_BindToArgument()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgumentById(const odla_value_id value_id,
                                    const odla_void* data_ptr,
                                    odla_context context) {
  std::cout << "---> odla_BindToArgumentById() : " << context << std::endl;
  std::string name(reinterpret_cast<const char*>(value_id));
  std::cout << "<--- odla_BindToArgumentById()" << std::endl;
  return odla_BindToArgument(context->comp->inputs_map[name], data_ptr,
                             context);
}

odla_status odla_SetValueAsOutput(const odla_value value) {
  std::cout << "---> odla_SetValueAsOutput()" << std::endl;
  auto comp = SingleComp::get_instance()->get_comp();
  comp->builder->addOutputTensor(value->tensor_id);
  comp->outputs_map[value->name] = value;
  comp->output_values.push_back(value);
  std::cout << "<--- odla_SetValueAsOutput()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_SetValuesAsOutput(const odla_values values) {
  std::cout << "---> odla_SetValuesAsOutput()" << std::endl;
  for (int i = 0; i < values.size; ++i) {
    odla_SetValueAsOutput(values.values[i]);
  }
  std::cout << "<--- odla_SetValuesAsOutput()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfOutputsFromComputation(
    const odla_computation computation, odla_uint32* num_outputs) {
  std::cout << "---> odla_GetNumOfOutputsFromComputation()" << std::endl;
  *num_outputs = computation->output_values.size();
  std::cout << "<--- odla_GetNumOfOutputsFromComputation()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_GetOutputFromComputationByIdx(
    const odla_computation computation, const odla_uint32 output_idx,
    odla_value* output_value) {
  std::cout << "---> odla_GetOutputFromComputationByIdx()" << std::endl;
  *output_value = nullptr;
  if (output_idx >= computation->output_values.size()) {
    return ODLA_FAILURE;
  }
  *output_value = computation->output_values[output_idx];
  std::cout << "<--- odla_GetOutputFromComputationByIdx()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  std::cout << "---> odla_BindToOutput()" << std::endl;
  std::unique_ptr<popart::IArray> p_array = MakeNDArrayWrapper(
      data_ptr, context->comp->builder->getTensorDataType(value->tensor_id),
      context->comp->builder->getTensorShape(value->tensor_id));
  context->outputs[value->tensor_id] = std::move(p_array);
  std::cout << "<--- odla_BindToOutput()" << std::endl;
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutputById(const odla_value_id value_id,
                                  odla_void* data_ptr, odla_context context) {
  std::cout << "---> odla_BindToOutputById()" << std::endl;
  std::string name(reinterpret_cast<const char*>(value_id));
  return odla_BindToOutput(context->comp->outputs_map[name], data_ptr, context);
  std::cout << "<--- odla_BindToOutputById()" << std::endl;
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  std::cout << "---> odla_GetValueType()" << std::endl;
  value_type->element_type = GetOdlaType(value->tensor_info.dataType());
  value_type->shape = GetOdlaShape(value->tensor_info.shape());
  std::cout << "<--- odla_GetValueType()" << std::endl;
  return ODLA_SUCCESS;
}
