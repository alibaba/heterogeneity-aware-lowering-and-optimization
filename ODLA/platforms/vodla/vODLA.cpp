#include <ODLA/odla.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#include <vodh_common.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#define gettid() syscall(SYS_gettid)
#define MAX_INPUT_TENSOR 256
#define MAX_OUTPUT_TENSOR 256

#ifndef LOOP_CNT
#define LOOP_CNT 10 // loop cnt for profiling
#endif

#ifndef USE_FILE_DMA
#define USE_FILE_DMA 0
#endif

// -DTIMING for timing, -DDEBUG for debug info print

enum ERR_CODE {
  ERR_TOTAL_CAP = 0,
  ERR_DEV_LIST,
  ERR_DEV_CAP_LIST,
  ERR_DEV_INFO,
  ERR_ONE_CAP,
  ERR_DEV_OPEN,
  ERR_CLEAN
};

struct _odla_device {
  void* vodh_hd = NULL;
  struct vodh_dev* vodh_dev_list = NULL;
  struct vodh_dev_cap* vodh_dev_cap_list = NULL;
  struct vodh_total_cap vodh_total_cap;
};

// input/output for a certain ctx
struct _odla_context {
  u32 ipNum = 0;
  u32 opNum = 0;
  u32 ipSizes[MAX_INPUT_TENSOR];
  u32 opSizes[MAX_OUTPUT_TENSOR];
  const void* ipPtr[MAX_INPUT_TENSOR];
  void* opPtr[MAX_OUTPUT_TENSOR];
  u32 batchSize = 1;
  u32 Id;
  u32 dataId = 0;
  bool dataChg = false; // indicate if ip data changed to increase dataId.
  vodla_context vodla_context_;
};

// model computation graph defined in ODLA files
struct _odla_computation {
  std::string ccFile = "";
  std::string wtFile = "";
  char* ccPtr = NULL;
  char* wtPtr = NULL;
  u32 ccFileSz;
  u32 wtFileSz;
  u32 Id;
};

struct _odla_value {
  u32 val;
};

static std::atomic<u32> g_ctxId(0);

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;
odla_device g_dev;

// read cc/bin files
u32 readFile(std::string fname, bool isBin, char*& ret) {
  std::ifstream f(fname, isBin ? std::ios::binary : std::ios::in);

  if (!f) {
    std::cout << "[vODLA] Error, Couldn't open file: " << fname << "\n";
    return 0;
  }

  f.seekg(0, std::ios::end);
  size_t file_sz = f.tellg();
  f.seekg(0, std::ios::beg);

  ret = reinterpret_cast<char*>(vodh_malloc(file_sz));
  if (ret == NULL) {
    std::cout << "[vODLA] Error allocating mem for file: " << fname << "\n";
  } else {
    f.seekg(0, std::ios::beg);
    f.read(ret, file_sz);
  }

  f.close();

  return file_sz;
}

void freeAllocs(odla_device dev, ERR_CODE cd) {
  switch (cd) {
    case ERR_CLEAN:
    case ERR_DEV_OPEN:
    case ERR_ONE_CAP:
    case ERR_DEV_INFO:
      free(dev->vodh_dev_cap_list);
      dev->vodh_dev_cap_list = NULL;
    case ERR_DEV_CAP_LIST:
      free(dev->vodh_dev_list);
      dev->vodh_dev_list = NULL;
    case ERR_DEV_LIST:
    case ERR_TOTAL_CAP:
    default:
      vodh_deinit(dev->vodh_hd);
  }
}

extern "C" {

void model_data(odla_context ctx, u32* ipSize, u32* opSize, int batchSize) {
  if (ctx == NULL) {
    std::cout << "[vODLA] ERROR: ctx is not created before model_data.\n";
    return;
  }

  memcpy(ctx->ipSizes, ipSize, ctx->ipNum * sizeof(u32));
  memcpy(ctx->opSizes, opSize, ctx->opNum * sizeof(u32));
  ctx->batchSize = batchSize;

  if (ctx->dataChg) {
    ctx->dataId++;
  }

#ifdef DEBUG
  std::cout << "[vODLA] DEBUG: Input num: " << ctx->ipNum << "\n";
  for (u32 i = 0; i < ctx->ipNum; i++) {
    std::cout << "[vODLA] DEBUG: Input " << i << " size: " << ctx->ipSizes[i]
              << "\n";
  }
  std::cout << "[vODLA] DEBUG: Output num: " << ctx->opNum << "\n";
  for (u32 i = 0; i < ctx->opNum; i++) {
    std::cout << "[vODLA] DEBUG: Output " << i << " size: " << ctx->opSizes[i]
              << "\n";
  }
#endif
}

odla_computation model_helper(const char* ccFile, const char* binFile) {
  odla_computation comp = nullptr;
  odla_status ret = odla_CreateComputation(&comp);
  if (ret != ODLA_SUCCESS) {
    std::cout << "[vODLA] ERROR: Failed creating ODLA computation!\n";
    return comp;
  }
#ifdef TIMING
  struct timeval t_s;
  struct timeval t_e;
  gettimeofday(&t_s, NULL);
#endif

  comp->ccFile = std::string(ccFile);
  comp->wtFile = std::string(binFile);

#if USE_FILE_DMA
  // open and read model cc/bin files, img file and ref file
  // read cc source file
  comp->ccFileSz = readFile(comp->ccFile, false, comp->ccPtr);
  if (comp->ccPtr == NULL) {
    std::cout << "[vODLA] ERROR: Error read model cc file: " << comp->ccFile
              << "\n";
    return comp;
  }

  // read weight bin file
  comp->wtFileSz = readFile(comp->wtFile, true, comp->wtPtr);
  if (comp->wtPtr == NULL) {
    std::cout << "[vODLA] ERROR: Error read model weight file: " << comp->wtFile
              << "\n";
    vodh_free(comp->ccPtr);
    return comp;
  }
#endif

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  u64 time_used =
      (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: read model files " << time_used << "us.\n";
  gettimeofday(&t_s, NULL);
#endif

  return comp;
}

odla_status odla_AllocateDevice(const odla_vendor vendor,
                                const odla_device_name device_name,
                                odla_device* device) {
  // Init, query and open vvodh devices
#ifdef DEBUG
  pid_t tid = gettid();
  std::cout << "[vODLA] INFO: thread " << tid << " allocate device\n";
#endif
  // create vODLA device
  odla_device dev = new _odla_device();
  if (dev == NULL) {
    std::cout << "[vODLA] ERROR: create odla device failed.\n";
    return ODLA_FAILURE;
  } else {
    *device = dev;
    g_dev = dev;
  }

#ifdef TIMING
  struct timeval t_s;
  struct timeval t_e;
  u64 time_used;
  gettimeofday(&t_s, NULL);
#endif

  vodh_ret ret = VODH_OK;

  /* Initialize vodla module */
  dev->vodh_hd = vodh_init();
  if (dev->vodh_hd == NULL) {
    std::cout << "[vODLA] ERROR: vodh_init failed!\n";
    return ODLA_FAILURE;
  }

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: Init VODH devices: " << time_used << "us.\n";
  gettimeofday(&t_s, NULL);
#endif

  /* Query total capability */
  memset(&(dev->vodh_total_cap), 0, sizeof(vodh_total_cap));
  ret = vodh_get_total_cap(dev->vodh_hd, &(dev->vodh_total_cap));
  if (ret) {
    std::cout << "[vODLA] ERROR: vodh_get_total_cap failed, ret=" << ret
              << "\n";
    freeAllocs(dev, ERR_TOTAL_CAP);
    return ODLA_FAILURE;
  }

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: Query VODH device total cap: " << time_used
            << "us.\n";
  gettimeofday(&t_s, NULL);
#endif

#ifdef DEBUG
  std::cout << "[vODLA] INFO: total capability:\n";
  std::cout << "[vODLA] INFO: xpu_num: " << dev->vodh_total_cap.vxpu_num
            << "\n";
  std::cout << "[vODLA] INFO: support: " << dev->vodh_total_cap.support << "\n";
  std::cout << "[vODLA] INFO: net_bw: " << dev->vodh_total_cap.net_bw
            << "Mbps\n";
  std::cout << "[vODLA] INFO: net_delay: " << dev->vodh_total_cap.net_delay
            << "ns\n";
  std::cout << "[vODLA] INFO: memory: " << dev->vodh_total_cap.memory
            << "Kbytes\n";
  std::cout << "[vODLA] INFO: compute: " << dev->vodh_total_cap.compute
            << "FLOPS\n";
#endif

  dev->vodh_dev_list =
      (vodh_dev*)malloc(sizeof(vodh_dev) * dev->vodh_total_cap.vxpu_num);
  if (dev->vodh_dev_list == NULL) {
    std::cout << "[vODLA] ERROR: Failed create device list.\n";
    freeAllocs(dev, ERR_DEV_LIST);
    return ODLA_FAILURE;
  } else {
    memset(dev->vodh_dev_list, 0,
           sizeof(vodh_dev) * dev->vodh_total_cap.vxpu_num);
    dev->vodh_dev_cap_list = (vodh_dev_cap*)malloc(
        sizeof(vodh_dev_cap) * dev->vodh_total_cap.vxpu_num);
    if (dev->vodh_dev_cap_list == NULL) {
      std::cout << "[vODLA] ERROR: Failed create dev cap list.\n";
      freeAllocs(dev, ERR_DEV_CAP_LIST);
      return ODLA_FAILURE;
    }
  }

  /* Get vodla device list and select one device */
  ret = vodh_get_all_dev_info(dev->vodh_hd, dev->vodh_dev_list);
  if (ret) {
    std::cout << "[vODLA] ERROR: No device found.\n";
    freeAllocs(dev, ERR_DEV_INFO);
    return ODLA_FAILURE;
  }

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: Get VODH devices info: " << time_used
            << "us.\n";
  gettimeofday(&t_s, NULL);
#endif

  for (unsigned i = 0; i < dev->vodh_total_cap.vxpu_num; i++) {
    /* Query device capability */
    ret = vodh_get_one_dev_cap(dev->vodh_hd, &(dev->vodh_dev_list[i]),
                               &(dev->vodh_dev_cap_list[i]));
    if (ret) {
      std::cout << "[vODLA] ERROR: vodh_get_dev_cap failed, ret=" << ret
                << "\n";
      freeAllocs(dev, ERR_ONE_CAP);
      return ODLA_FAILURE;
    }
#ifdef DEBUG
    std::cout << "[vODLA] INFO: device " << i << ":\n";
    std::cout << "[vODLA] INFO: device " << dev->vodh_dev_list[i].name << "\n";
    std::cout << "[vODLA] INFO: type: " << dev->vodh_dev_cap_list[i].type
              << "\n";
    std::cout << "[vODLA] INFO: memory: " << dev->vodh_dev_cap_list[i].memory
              << "Kbytes\n";
    // std::cout << "[vODLA] compute: " << vodh_dev_cap_list[i].comput <<
    // "eFLOPS\n";
#endif
    /*
            if ((vodh_dev_cap.compute < COMPUTE_NEED) ||
                    (vodh_dev_cap.memory < (INPUT_DATA_SIZE + g_out_size))) {
        std::cout << "[vODLA] Warning: device capability does not match
       requirements.\n"; std::cout << "[vODLA] Warning: Required: compute " <<
       COMPUTE_NEED << "FLOPS, memory "
            << (INPUT_DATA_SIZE + g_out_size) << "Kbytes.\n";
            }
    */
  }

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: Query each VODH device cap: " << time_used
            << "us.\n";
  gettimeofday(&t_s, NULL);
#endif

  // open dev[0] for remote infer temporarily, TODO: use multiple devices
  ret = vodh_dev_open(dev->vodh_hd, &(dev->vodh_dev_list[0]));
  if (ret) {
    std::cout << "[vODLA] ERROR: failed open vodh device, ret=" << ret << "\n";
    freeAllocs(dev, ERR_DEV_OPEN);
    return ODLA_FAILURE;
  }
#ifdef DEBUG
  std::cout << "[vODLA] INFO: Vodh device opened.\n";
#endif

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: Open VODH devices " << time_used << "us.\n";
  gettimeofday(&t_s, NULL);
#endif

  return ODLA_SUCCESS;
}

odla_status odla_DestroyDevice(odla_device device) {
  // allocate device failed
#ifdef DEBUG
  pid_t tid = gettid();
  std::cout << "[vODLA] INFO: thread " << tid << " destroy device\n";
#endif
  if (device == NULL) {
    std::cout << "[vODLA] ERROR: NULL device, nothing to destroy.\n";
    return ODLA_FAILURE;
  }
  if (device->vodh_hd == NULL || device->vodh_dev_list == NULL ||
      device->vodh_dev_cap_list == NULL) {
    std::cout << "[vODLA] ERROR: allocate device failed, nothing to destroy.\n";
    delete device;
    device = NULL;
    return ODLA_FAILURE;
  }
  if (device != g_dev) {
    std::cout << "[vODLA] ERROR: device doens't match current device inuse.\n";
  }

  odla_status ret = ODLA_SUCCESS;

  // close dev[0], TODO: use multiple devices
  vodh_ret rt = vodh_dev_close(device->vodh_hd, &(device->vodh_dev_list[0]));
  if (rt) {
    std::cout << "[vODLA] ERROR: failed close vodh device, ret=" << rt << "\n";
    ret = ODLA_FAILURE;
  }

  freeAllocs(device, ERR_CLEAN);
  delete device;
  device = NULL;
  g_dev = NULL;
#ifdef DEBUG
  std::cout << "[vODLA] INFO: vodh device closed.\n";
#endif
  return ret;
}

odla_status odla_GetNumOfArgsFromComputation(const odla_computation computation,
                                             odla_uint32* num_args) {
  *num_args = -1;
  return ODLA_SUCCESS;
}

odla_status odla_GetNumOfOutputsFromComputation(
    const odla_computation computation, odla_uint32* num_outputs) {
  *num_outputs = -1;
  return ODLA_SUCCESS;
}

odla_status odla_GetValueType(const odla_value value,
                              odla_value_type* value_type) {
  return ODLA_SUCCESS;
}

odla_status odla_GetArgFromComputationByIdx(const odla_computation computation,
                                            const odla_uint32 arg_idx,
                                            odla_value* arg_value) {
  *arg_value = reinterpret_cast<odla_value>(arg_idx);
  return ODLA_SUCCESS;
}

odla_status odla_GetOutputFromComputationByIdx(
    const odla_computation computation, const odla_uint32 output_idx,
    odla_value* output_value) {
  *output_value = reinterpret_cast<odla_value>(output_idx);
  return ODLA_SUCCESS;
}

odla_status odla_BindToArgument(odla_value value, const odla_void* data_ptr,
                                odla_context context) {
  if (context == NULL) {
    std::cout << "[vODLA] ERROR: NULL context to bind input args.\n";
    return ODLA_FAILURE;
  }

  auto idx = reinterpret_cast<std::uintptr_t>(value);
#ifdef DEBUG
  pid_t tid = gettid();
  std::cout << "[vODLA] DEBUG: thread " << tid << " bind argument\n";
  std::cout << "[vODLA] DEBUG: Binding input idx: " << idx << "\n";
#endif
  if (idx >= MAX_INPUT_TENSOR) {
    return ODLA_FAILURE;
  }
  if (data_ptr == NULL) {
    std::cout << "[vODLA] ERROR: input data " << idx << " pointer is NULL.\n";
    return ODLA_FAILURE;
  }
  context->ipPtr[idx] = data_ptr;
  context->dataChg = true;
  context->ipNum++;
  return ODLA_SUCCESS;
}

odla_status odla_BindToOutput(odla_value value, odla_void* data_ptr,
                              odla_context context) {
  if (context == NULL) {
    std::cout << "[vODLA] ERROR: NULL context to bind output args.\n";
    return ODLA_FAILURE;
  }

  auto idx = reinterpret_cast<std::uintptr_t>(value);
#ifdef DEBUG
  pid_t tid = gettid();
  std::cout << "[vODLA] DEBUG: thread " << tid << " bind output\n";
  std::cout << "[vODLA] DEBUG: Binding output idx: " << idx << "\n";
#endif
  if (idx >= MAX_OUTPUT_TENSOR) {
    return ODLA_FAILURE;
  }
  if (data_ptr == NULL) {
    std::cout << "[vODLA] ERROR: output data " << idx << " pointer is NULL.\n";
    return ODLA_FAILURE;
  }
  context->opPtr[idx] = data_ptr;
  context->dataChg = true;
  context->opNum++;
  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
#ifdef DEBUG
  pid_t tid = gettid();
  std::cout << "[vODLA] DEBUG: thread " << tid << " create context\n";
  std::cout << "[vODLA] DEBUG: odla_CreateContext handler " << g_dev->vodh_hd
            << " dev_list ptr " << &(g_dev->vodh_dev_list[0]) << " ctx "
            << context << ", *ctx " << *context << std::endl;
#endif
  *context = new _odla_context();
  vodh_ret ret = vodh_create_context(g_dev->vodh_hd, &(g_dev->vodh_dev_list[0]),
                                     &((*context)->vodla_context_));

  if (*context == nullptr || ret) {
#ifdef DEBUG
    std::cout << "[vODLA] ERROR: failed to create odla context. context "
              << context << " ret " << ret << "\n";
#endif
    return ODLA_FAILURE;
  }

  (*context)->Id = g_ctxId++;

  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context context) {
#ifdef DEBUG
  pid_t tid = gettid();
  std::cout << "[vODLA] DEBUG: odla_DestroyContext thread " << tid
            << " context " << context << std::endl;
#endif
  vodh_ret ret = vodh_destroy_context(
      g_dev->vodh_hd, &(g_dev->vodh_dev_list[0]), context->vodla_context_);
  if (ret) {
    std::cout << "[vODLA] ERROR: failed to call vodh_destroy_context"
              << std::endl;
    return ODLA_FAILURE;
  }
  if (context) {
    delete context;
    context = NULL;
  }
  return ODLA_SUCCESS;
}

odla_status odla_CreateComputation(odla_computation* computation) {
  g_comps.push_back(std::make_unique<_odla_computation>());
  g_comp = g_comps.back().get();
  g_comp->Id = g_comps.size();
  *computation = g_comp;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyComputation(odla_computation comp) {
  // call model_fini at backend
  if (g_dev == NULL) {
    std::cout << "[vODLA] Error: device is not created before model switch.\n";
  } else {
    if (g_dev->vodh_hd == NULL || g_dev->vodh_dev_list == NULL) {
      std::cout
          << "[vODLA] Error: device is not initiated before model switch.\n";
    } else {
      vodh_model_options mopt;
      mopt.opcode = TYPE_RELEASE;
      mopt.pad = 0;
      mopt.model.use_file = 1;
      mopt.model.model_id = comp->Id;
      mopt.model.weight_id = comp->Id;
      strcpy(mopt.model.model_file, comp->ccFile.c_str());
      strcpy(mopt.model.weight_file, comp->wtFile.c_str());
      vodh_model_op_result mret;
      if (vodh_model(g_dev->vodh_hd, &(g_dev->vodh_dev_list[0]), &mopt,
                     &mret)) {
        std::cout << "[vODLA] Error: failed to call model_fini at backend.\n";
      }
    }
  }

  for (auto it = g_comps.begin(), e = g_comps.end(); it != e; ++it) {
    if (it->get() == comp) {
      vodh_free(it->get()->ccPtr);
      vodh_free(it->get()->wtPtr);
      it->reset();
      g_comps.erase(it);
      return ODLA_SUCCESS;
    }
  }
  assert(0);
  return ODLA_FAILURE;
}

void deallocDMA(struct vodh_infer_options& vodh_infer_opt,
                struct vodh_infer_result& vodh_infer_res, bool ip, bool op) {
  /*DeAllocate DMA for inputs*/
  if (vodh_infer_opt.input && ip) {
    for (u32 i = 0; i < vodh_infer_opt.input_num; i++) {
#ifdef DEBUG
      printf("input %d data at %p\n", i, vodh_infer_opt.input[i]->data);
#endif

      vodh_free(vodh_infer_opt.input[i]->data);
      vodh_infer_opt.input[i]->data = NULL;
      delete vodh_infer_opt.input[i];
      vodh_infer_opt.input[i] = NULL;

#ifdef DEBUG
      std::cout << "[vODLA] DEBUG: released vodh input: " << i << "\n";
#endif
    }
    delete[] vodh_infer_opt.input;
    vodh_infer_opt.input = NULL;

#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: released input pointer\n";
#endif
  }
  /*DeAllocate DMA for outputs*/
  if (vodh_infer_res.output && op) {
    for (u32 i = 0; i < vodh_infer_res.output_num; i++) {
      vodh_free(vodh_infer_res.output[i]->data);
      vodh_infer_res.output[i]->data = NULL;
      delete vodh_infer_res.output[i];
      vodh_infer_res.output[i] = NULL;

#ifdef DEBUG
      std::cout << "[vODLA] DEBUG: released vodh output: " << i << "\n";
#endif
    }
    delete[] vodh_infer_res.output;
    vodh_infer_res.output = NULL;

#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: released output pointer\n";
#endif
  }
}

bool allocDMA(struct vodh_infer_options& vodh_infer_opt,
              struct vodh_infer_result& vodh_infer_res, odla_context context) {
  bool ret = true;

#ifdef TIMING
  struct timeval t_s;
  struct timeval t_e;
  double time_used;
#endif

  /* Alloc DMA memory for input data */
  u32 ip_mem_cap = 0;
#ifdef DEBUG
  std::cout << "[vODLA] DEBUG: vodh_infer_opt.input_num "
            << vodh_infer_opt.input_num << " context->ipNum " << context->ipNum
            << std::endl;
#endif
  if (vodh_infer_opt.input_num != context->ipNum) {
    deallocDMA(vodh_infer_opt, vodh_infer_res, true, false);
    vodh_infer_opt.input_num = context->ipNum;
    vodh_infer_opt.input = new vodh_input*[context->ipNum];
    if (vodh_infer_opt.input) {
      for (u32 i = 0; i < context->ipNum; i++) {
        vodh_infer_opt.input[i] = new vodh_input();
        if (vodh_infer_opt.input[i]) {
          vodh_infer_opt.input[i]->data = vodh_malloc(context->ipSizes[i]);
          if (vodh_infer_opt.input[i]->data && context->ipPtr[i]) {
#ifdef PRINTMEM
            std::cout << "vodh input " << i
                      << " malloc: " << context->ipSizes[i] << " bytes.\n";
            std::cout << "dst address: " << vodh_infer_opt.input[i]->data
                      << ".\n";
            std::cout << "src address: " << context->ipPtr[i] << ".\n";
            std::cout << "loop on dst mem:" << std::endl;
            for (int lp = 0; lp < context->ipSizes[i]; lp++) {
              char* tmp = (char*)(vodh_infer_opt.input[i]->data) + lp;
              if (lp == 0 || lp == (context->ipSizes[i] >> 1) ||
                  lp == context->ipSizes[i] - 1) {
                std::cout << "(" << lp << "," << (*tmp) << ")," << std::endl;
              }
            }
            std::cout << "\nloop on src mem:" << std::endl;
            for (int lp = 0; lp < context->ipSizes[i]; lp++) {
              char* tmp = (char*)(context->ipPtr[i]) + lp;
              if (lp == 0 || lp == (context->ipSizes[i] >> 1) ||
                  lp == context->ipSizes[i] - 1) {
                std::cout << "(" << lp << "," << (*tmp) << ")," << std::endl;
              }
            }
            std::cout << " \nstart memcpy ...\n";
#endif
#ifdef TIMING
            gettimeofday(&t_s, NULL);
#endif
            memcpy(vodh_infer_opt.input[i]->data, context->ipPtr[i],
                   context->ipSizes[i]);
#ifdef TIMING
            gettimeofday(&t_e, NULL);
            time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 +
                        (t_e.tv_usec - t_s.tv_usec);
            std::cout << "[vODLA] TIMING: CP DMA data for input " << i << ": "
                      << time_used << "us. (new)\n";
#endif

#ifdef PRINTMEM
            std::cout << "finished memcpy.\n";
#endif
          } else {
            std::cout << "[vODLA] ERROR: allocate infer input " << i
                      << " data failed.\n";
            ret = false;
          }
        } else {
          std::cout << "[vODLA] ERROR: allocate infer input: " << i
                    << " failed.\n";
          ret = false;
        }
        vodh_infer_opt.input[i]->size = context->ipSizes[i];
        ip_mem_cap += vodh_infer_opt.input[i]->size;
      }
    } else {
      std::cout << "[vODLA] ERROR: allocate infer input failed.\n";
      ret = false;
    }
  } else {
    if (vodh_infer_opt.input) {
      for (u32 i = 0; i < context->ipNum; i++) {
        if (vodh_infer_opt.input[i]) {
          if (vodh_infer_opt.input[i]->size != context->ipSizes[i]) {
            vodh_free(vodh_infer_opt.input[i]->data);
            vodh_infer_opt.input[i]->data = NULL;
            vodh_infer_opt.input[i]->data = vodh_malloc(context->ipSizes[i]);
          }
          if (vodh_infer_opt.input[i]->data && context->ipPtr[i]) {
#ifdef PRINTMEM
            std::cout << "vodh input " << i
                      << " malloc: " << context->ipSizes[i] << " bytes.\n";
            std::cout << "dst address: " << vodh_infer_opt.input[i]->data
                      << ".\n";
            std::cout << "src address: " << context->ipPtr[i] << ".\n";
            std::cout << "loop on dst mem:" << std::endl;
            for (int lp = 0; lp < context->ipSizes[i]; lp++) {
              char* tmp = (char*)(vodh_infer_opt.input[i]->data) + lp;
              if (lp == 0 || lp == (context->ipSizes[i] >> 1) ||
                  lp == context->ipSizes[i] - 1) {
                std::cout << "(" << lp << "," << (*tmp) << ")," << std::endl;
              }
            }
            std::cout << "\nloop on src mem:" << std::endl;
            for (int lp = 0; lp < context->ipSizes[i]; lp++) {
              char* tmp = (char*)(context->ipPtr[i]) + lp;
              if (lp == 0 || lp == (context->ipSizes[i] >> 1) ||
                  lp == context->ipSizes[i] - 1) {
                std::cout << "(" << lp << "," << (*tmp) << ")," << std::endl;
              }
            }
            std::cout << " \nstart memcpy ...\n";
#endif
#ifdef TIMING
            gettimeofday(&t_s, NULL);
#endif
            memcpy(vodh_infer_opt.input[i]->data, context->ipPtr[i],
                   context->ipSizes[i]);
#ifdef TIMING
            gettimeofday(&t_e, NULL);
            time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 +
                        (t_e.tv_usec - t_s.tv_usec);
            std::cout << "[vODLA] TIMING: CP DMA data for input " << i << ": "
                      << time_used << "us. (reuse)\n";
#endif
#ifdef PRINTMEM
            std::cout << "finished memcpy.\n";
#endif
          } else {
            std::cout << "[vODLA] ERROR: copy infer input " << i
                      << " data failed.\n";
            ret = false;
          }
        } else {
          std::cout << "[vODLA] ERROR: reuse infer input: " << i
                    << " failed.\n";
          ret = false;
        }
        vodh_infer_opt.input[i]->size = context->ipSizes[i];
        ip_mem_cap += vodh_infer_opt.input[i]->size;
      }
    } else {
      std::cout << "[vODLA] ERROR: reuse infer input failed.\n";
      ret = false;
    }
  }

  /* Alloc DMA memory for output data */
  u32 op_mem_cap = 0;
  if (vodh_infer_res.output_num != context->opNum) {
    deallocDMA(vodh_infer_opt, vodh_infer_res, false, true);
    vodh_infer_res.output_num = context->opNum;
    vodh_infer_res.output = new vodh_output*[context->opNum];
    if (vodh_infer_res.output) {
      for (u32 i = 0; i < context->opNum; i++) {
        vodh_infer_res.output[i] = new vodh_output();
        if (vodh_infer_res.output[i]) {
          vodh_infer_res.output[i]->data = vodh_malloc(context->opSizes[i]);
          if (vodh_infer_res.output[i]->data == NULL) {
            std::cout << "[vODLA] ERROR: allocate infer output " << i
                      << " data failed.\n";
            ret = false;
          }
        } else {
          std::cout << "[vODLA] ERROR: allocate infer output: " << i
                    << " failed.\n";
          ret = false;
        }
        vodh_infer_res.output[i]->size = context->opSizes[i];
        op_mem_cap += vodh_infer_res.output[i]->size;
      }
    } else {
      std::cout << "[vODLA] ERROR: allocate infer output failed.\n";
      ret = false;
    }
  } else {
    if (vodh_infer_res.output) {
      for (u32 i = 0; i < context->opNum; i++) {
        if (vodh_infer_res.output[i]) {
          if (vodh_infer_res.output[i]->size != context->opSizes[i]) {
            vodh_free(vodh_infer_res.output[i]->data);
            vodh_infer_res.output[i]->data = NULL;
            vodh_infer_res.output[i]->data = vodh_malloc(context->opSizes[i]);
          }
          if (vodh_infer_res.output[i]->data == NULL) {
            std::cout << "[vODLA] ERROR: allocate infer output " << i
                      << " data failed (reuse).\n";
            ret = false;
          }
        } else {
          std::cout << "[vODLA] ERROR: reuse infer output: " << i
                    << " failed.\n";
          ret = false;
        }
        vodh_infer_res.output[i]->size = context->opSizes[i];
        op_mem_cap += vodh_infer_res.output[i]->size;
      }
    } else {
      std::cout << "[vODLA] ERROR: reuse infer output failed.\n";
      ret = false;
    }
  }

  vodh_infer_opt.request_cap.input_memory = ip_mem_cap;
  vodh_infer_opt.request_cap.output_memory = op_mem_cap;

  return ret;
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
  pid_t tid = gettid();
  // init vODLA device failed
  if (context == NULL) {
    std::cout << "[vODLA] ERROR: odla device is NULL.\n";
    return ODLA_FAILURE;
  }
  if (device == NULL || comp == NULL || device->vodh_dev_list == NULL ||
      device->vodh_dev_cap_list == NULL) {
    if (device->vodh_dev_list == NULL || device->vodh_dev_cap_list == NULL) {
      std::cout << "[vODLA] ERROR: allocate device failed, skip executing "
                   "compution.\n";
    } else {
      std::cout << "[vODLA] ERROR: odla computation or device is NULL.\n";
    }

    // reset ip/op number for arg binding of the next inference
    context->ipNum = 0;
    context->opNum = 0;
    context->dataChg = false;

    return ODLA_FAILURE;
  }

  odla_status ret = ODLA_SUCCESS;

#ifdef TIMING
  struct timeval t_s;
  struct timeval t_e;
  double time_used;
  double time_max = 0;
  double time_min = std::numeric_limits<double>::max();
  double time_avg = 0;
  gettimeofday(&t_s, NULL);
#endif

  bool allocSuccess = true;
  thread_local static struct vodh_infer_options vodh_infer_opt = {
      .request_id = 0xDEADBEEF,
      .batch_size = 0,
      .input_num = 0,
      .pad = 0,
      .request_cap = {},
      .model = {},
      .input = NULL};
  thread_local static struct vodh_infer_result vodh_infer_res = {
      .request_id = 0xDEADBEEF,
      .status = 0,
      .output_num = 0,
      .pad = 0,
      .time_used = 0,
      .output = NULL};

#ifdef DEBUG
  pid_t tid = gettid();
  std::cout << "[vODLA] DEBUG: tid " << tid
            << " odla_ExecuteComputation use opt addr " << &vodh_infer_opt
            << "\nres addr " << &vodh_infer_res << std::endl;
#endif

  u64 rid = ((context->Id) << 16) + context->dataId;
  // context changed, new data mem to be allocated
  if (rid != vodh_infer_res.request_id) {
#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: infer input data changed.\n";
#endif

    allocSuccess = allocDMA(vodh_infer_opt, vodh_infer_res, context);

#ifdef TIMING
    gettimeofday(&t_e, NULL);
    time_used =
        (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
    std::cout << "[vODLA] TIMING: allocate DMAs: " << time_used << "us.\n";
    gettimeofday(&t_s, NULL);
#endif
  }

  // input/output allocation is successfull
  if (allocSuccess) {
    vodh_infer_res.request_id = rid;
    vodh_infer_opt.request_id = rid;

#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: infer request ID is "
              << vodh_infer_res.request_id << "\n";
#endif

    vodh_ret rt;

    /* Additional inference info */
    vodh_infer_opt.batch_size = context->batchSize;
    vodh_infer_opt.request_cap.type = device->vodh_dev_cap_list[0].type;
    vodh_infer_opt.request_cap.net_bw = device->vodh_total_cap.net_bw;
    vodh_infer_opt.request_cap.net_delay = device->vodh_total_cap.net_delay;
    vodh_infer_opt.request_cap.use_direct_rdma = true;
    vodh_infer_opt.request_cap.compute =
        0; // TODO: compute power required from cost model.
    /* Alloc DMA memory for model file */
    vodh_infer_opt.model.model_data = comp->ccPtr;
    vodh_infer_opt.model.model_size = comp->ccFileSz;
    vodh_infer_opt.model.model_id = comp->Id;
    strcpy(vodh_infer_opt.model.model_file, comp->ccFile.c_str());
    /* Alloc DMA memory for weight file */
    vodh_infer_opt.model.weight_data = comp->wtPtr;
    vodh_infer_opt.model.weight_size = comp->wtFileSz;
    vodh_infer_opt.model.weight_id = comp->Id;
    vodh_infer_opt.model.use_file = 1; // use file path instead of DMA
    vodh_infer_opt.context = context->vodla_context_;
    strcpy(vodh_infer_opt.model.weight_file, comp->wtFile.c_str());
#ifdef DEBUG
    std::cout << "[vODLA] INFO: thread " << tid
              << " start remote inference...\n";
#endif
#ifdef TIMING
    std::cout << "[vODLA] INFO: loop " << LOOP_CNT << " times.\n";

    double xpu_time_avg = 0;
    u64 xpu_time_max = 0;
    u64 xpu_time_min = std::numeric_limits<long long>::max();

    for (u32 lp = 0; lp < LOOP_CNT; lp++) {
      /* Remote infer */
      rt = vodh_infer(device->vodh_hd, &(device->vodh_dev_list[0]),
                      &vodh_infer_opt, &vodh_infer_res);
      if (rt) {
        break;
      }

      gettimeofday(&t_e, NULL);
      time_used =
          (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);

      time_max = std::max(time_used, time_max);
      time_min = std::min(time_used, time_min);
      time_avg += time_used;
      gettimeofday(&t_s, NULL);

      xpu_time_max = std::max(xpu_time_max, vodh_infer_res.time_used);
      xpu_time_min = std::min(xpu_time_min, vodh_infer_res.time_used);
      xpu_time_avg += vodh_infer_res.time_used;
    }
    xpu_time_avg /= LOOP_CNT;
    std::cout << "[vODLA] TIMING: Remote xPU inference avg time: "
              << xpu_time_avg << "us, max time: " << xpu_time_max
              << "us, min time: " << xpu_time_min << "us.\n";
    time_avg /= LOOP_CNT;
    std::cout << "[vODLA] TIMING: Inference avg time: " << time_avg << "us, "
              << "max time: " << time_max << " us,"
              << "min time: " << time_min << " us.\n";
    std::cout << "[vODLA] TIMING: Inference avg throuput: "
              << context->batchSize / time_avg * 1000000 << " imgs/s, "
              << "max throuput: " << context->batchSize / time_min * 1000000
              << " imgs/s,"
              << "min throuput: " << context->batchSize / time_max * 1000000
              << " imgs/s.\n";
#else
    /* Remote infer */
    rt = vodh_infer(device->vodh_hd, &(device->vodh_dev_list[0]),
                    &vodh_infer_opt, &vodh_infer_res);
#endif

    // cp back inference result
    for (u32 i = 0; i < context->opNum; i++) {
      if (context->opPtr[i]) {
#ifdef PRINTMEM
        std::cout << "vodh output " << i << " malloc: " << context->opSizes[i]
                  << " bytes.\n";
        std::cout << "src address: " << vodh_infer_res.output[i]->data << ".\n";
        std::cout << "dst address: " << context->opPtr[i] << ".\n";
        std::cout << "loop on src mem:" << std::endl;
        for (int lp = 0; lp < context->opSizes[i]; lp++) {
          char* tmp = (char*)(vodh_infer_res.output[i]->data) + lp;
          if (lp == 0 || lp == (context->opSizes[i] >> 1) ||
              lp == context->opSizes[i] - 1) {
            std::cout << "(" << lp << "," << (*tmp) << ")," << std::endl;
          }
        }
        std::cout << "\nloop on dst mem:" << std::endl;
        for (int lp = 0; lp < context->opSizes[i]; lp++) {
          char* tmp = (char*)(context->opPtr[i]) + lp;
          if (lp == 0 || lp == (context->opSizes[i] >> 1) ||
              lp == context->opSizes[i] - 1) {
            std::cout << "(" << lp << "," << (*tmp) << ")," << std::endl;
          }
        }
        std::cout << " \nstart memcpy ...\n";
#endif
        memcpy(context->opPtr[i], vodh_infer_res.output[i]->data,
               context->opSizes[i]);
#ifdef PRINTMEM
        std::cout << "finished memcpy.\n";
#endif
      } else {
        std::cout << "[vODLA] ERROR: copy output data failed, op#" << i << "\n";
        ret = ODLA_FAILURE;
      }
    }

    if (rt) {
      std::cout << "[vODLA] ERROR: infer failed, ret=" << rt << "\n";
      ret = ODLA_FAILURE;
    } else {
#ifdef DEBUG
      std::cout << "[vODLA] INFO: Run inference finished!\n";
#endif
      ret = static_cast<odla_status>(vodh_infer_res.status);
#ifdef DEBUG
      if (ret == ODLA_FAILURE) {
        std::cout << "[vODLA] INFO: Remote inference failed!\n";
      }
#endif
    }
  }

  // reset ip/op number for arg binding of the next inference
  context->ipNum = 0;
  context->opNum = 0;
  context->dataChg = false;

  return ret;
}

odla_status ODLA_API_CALL odla_LoadExecutable(const odla_char* file_name,
                                              odla_device device,
                                              odla_executable* executable,
                                              odla_context* context,
                                              odla_computation* computation) {
  // init vODLA device failed
  if (file_name == NULL) {
    std::cout << "[vODLA] ERROR: Cache file name is NULL.\n";
    return ODLA_FAILURE;
  }
  if (device == NULL) {
    std::cout << "[vODLA] ERROR: odla device is NULL.\n";
    return ODLA_FAILURE;
  }
  if (device->vodh_dev_list == NULL) {
    std::cout
        << "[vODLA] ERROR: allocation of device failed, skip loading cache.\n";
    return ODLA_FAILURE;
  }

  struct vodh_model_options moptions;
  struct vodh_model_op_result mresult;
  moptions.opcode = TYPE_LOAD;
  strcpy(moptions.modelname, file_name);
  vodh_ret ret = vodh_model(device->vodh_hd, &(device->vodh_dev_list[0]),
                            &moptions, &mresult);
  if (ret) {
    std::cout << "[vODLA] ERROR: load model cache failed, ret=" << ret << "\n";
    return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

} // extern "C"
