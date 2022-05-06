#include <ODLA/odla.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vodh_common.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "ODLA/odla_common.h"
#include "ODLA/odla_compute.h"

#define MAX_INPUT_TENSOR 256
#define MAX_OUTPUT_TENSOR 256

#ifndef LOOP_CNT
#define LOOP_CNT 10 // loop cnt for profiling
#endif

#ifndef USE_FILE_DMA
#define USE_FILE_DMA 0
#endif

#include <unistd.h>

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
  // mark
  u32 need_release = 0;
  u32 pad = 0;
  struct vodh_dev* vodh_dev_list = NULL;
  struct vodh_dev_cap* vodh_dev_cap_list = NULL;
  struct vodh_total_cap vodh_total_cap;
  struct vodh_infer_options vodh_infer_opt;
  struct vodh_infer_result vodh_infer_result;
  _odla_device() {
    vodh_infer_opt.request_id = 0xDEADBEEF;
    vodh_infer_opt.input = NULL;
    vodh_infer_opt.input_num = 0;
    vodh_infer_result.request_id = 0xDEADBEEF;
    vodh_infer_result.output = NULL;
    vodh_infer_result.output_num = 0;
  }
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

static u32 g_ctxId = 0;

thread_local odla_computation g_comp;
static std::vector<std::unique_ptr<_odla_computation>> g_comps;
odla_device g_dev;
bool is_new_cc = false;
std::mutex is_new_cc_mu_;

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
      if (dev->vodh_infer_opt.input) {
        for (u32 i = 0; i < dev->vodh_infer_opt.input_num; i++) {
          vodh_free(dev->vodh_infer_opt.input[i]->data);
          dev->vodh_infer_opt.input[i]->data = NULL;
          delete dev->vodh_infer_opt.input[i];
          dev->vodh_infer_opt.input[i] = NULL;
        }
        delete[] dev->vodh_infer_opt.input;
        dev->vodh_infer_opt.input = NULL;
      }
      if (dev->vodh_infer_result.output) {
        for (u32 i = 0; i < dev->vodh_infer_result.output_num; i++) {
          vodh_free(dev->vodh_infer_result.output[i]->data);
          dev->vodh_infer_result.output[i]->data = NULL;
          delete dev->vodh_infer_result.output[i];
          dev->vodh_infer_result.output[i] = NULL;
        }
        delete[] dev->vodh_infer_result.output;
        dev->vodh_infer_result.output = NULL;
      }
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
  {
    std::lock_guard<std::mutex> lock(is_new_cc_mu_);
    is_new_cc = true;
  }
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

odla_status odla_adaptiveRelease(odla_device dev) {
  sleep(3);
  std::cout << "[vODLA] Debug: starting odla_adaptiveRelease.\n";
#ifdef TIMING
  struct timeval t_s;
  struct timeval t_e;
  u64 time_used;
  gettimeofday(&t_s, NULL);
#endif

  char* pkey = NULL;
  u32 keylen = 0;
  vodh_ret ret = vodh_sa_get_apply_key(dev->vodh_hd, (void**)(&pkey), &keylen);
  if (ret) {
    std::cout << "[vODLA] ERROR: vodh get apply key failed.\n";
    return ODLA_FAILURE;
  }

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: VODH get apply key: " << time_used << "us.\n";
  gettimeofday(&t_s, NULL);
#endif
  std::string pkey_valid = pkey;
  pkey_valid = pkey_valid.substr(0, keylen);
  std::string s("");
  s += "{\"key\":";
  s += "\"";
  // s += pkey;
  s += pkey_valid;
  s += "\"";
  s += "}";
  int key_size = s.size();

#ifdef DEBUG
  std::cout << "[vODLA] INFO: resource release buffer:\n";
  std::cout << s << "\n";
#endif

  ret = vodh_sa_res_release(dev->vodh_hd, s.c_str(), key_size);
  if (ret) {
    std::cout << "[vODLA] ERROR: vodh vodh_sa_res_release failed.\n";
    return ODLA_FAILURE;
  } else {
    std::cout << "[vODLA] vodh vodh_sa_res_release succeeded.\n";
  }

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: VODH release resource: " << time_used
            << "us.\n";
#endif

  if (pkey != NULL) {
    vodh_sa_free_apply_key(dev->vodh_hd, (void*)pkey);
  }

  return ODLA_SUCCESS;
}

odla_status odla_adaptiveAlloc(odla_device dev, const char* cfg) {
// odla_adaptiveRelease(dev);
#ifdef TIMING
  struct timeval t_s;
  struct timeval t_e;
  u64 time_used;
  gettimeofday(&t_s, NULL);
#endif

  char* pkey = NULL;
  u32 keylen = 0;
  vodh_ret ret = vodh_sa_get_apply_key(dev->vodh_hd, (void**)(&pkey), &keylen);

  if (ret) {
    std::cout << "[vODLA] ERROR: vodh get apply key failed.\n";
    return ODLA_FAILURE;
  }

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: VODH get apply key: " << time_used << "us.\n";
  gettimeofday(&t_s, NULL);
#endif

  std::string pkey_valid = pkey;
  pkey_valid = pkey_valid.substr(0, keylen);
  std::string s("");
  s += "{\"key\":";
  s += "\"";
  // s += pkey;
  s += pkey_valid;
  s += "\"";
  s += ",";
  int key_size = s.size();
  s += cfg;

  // std::string s("");
  // s += "{\"key\":";
  // s +="\"";
  // s += pkey;
  // s +="\"";
  // s += ",";
  // int key_size = s.size();
  // s += cfg;

  int cfg_size = 0;
  while (cfg[cfg_size] != '\0') {
    cfg_size++;
  }

#ifdef DEBUG
  std::cout << "[vODLA] INFO: resource application buffer:\n";
  std::cout << s << "\n";
#endif

  ret = vodh_sa_res_apply(dev->vodh_hd, s.c_str(), key_size + cfg_size);
  if (ret) {
    std::cout << "[vODLA] ERROR: vodh sa resource application failed.\n";
    return ODLA_FAILURE;
  }

#ifdef TIMING
  gettimeofday(&t_e, NULL);
  time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 + (t_e.tv_usec - t_s.tv_usec);
  std::cout << "[vODLA] TIMING: VODH request resource: " << time_used
            << "us.\n";
#endif

  if (pkey != NULL) {
    vodh_sa_free_apply_key(dev->vodh_hd, (void*)pkey);
  }

  // mark
  dev->need_release = 1;
  return ODLA_SUCCESS;
}

odla_status odla_OpenDevice(odla_device dev) {
  if (dev == NULL || dev->vodh_hd == NULL) {
    std::cout << "[vODLA] ERROR: vodh device is not initialized!\n";
    return ODLA_FAILURE;
  }

#ifdef TIMING
  struct timeval t_s;
  struct timeval t_e;
  u64 time_used;
  gettimeofday(&t_s, NULL);
#endif

  /* Query total capability */
  memset(&(dev->vodh_total_cap), 0, sizeof(vodh_total_cap));
  vodh_ret ret = vodh_get_total_cap(dev->vodh_hd, &(dev->vodh_total_cap));
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
  std::cout << "[vODLA] INFO: net_bw: " << dev->vodh_total_cap.net_bw << "\n";
  std::cout << "[vODLA] INFO: net_delay: " << dev->vodh_total_cap.net_delay
            << "\n";
  std::cout << "[vODLA] INFO: memory: " << dev->vodh_total_cap.memory << "\n";
  std::cout << "[vODLA] INFO: compute: " << dev->vodh_total_cap.compute << "\n";
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
              << "\n";
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
#endif

  return ODLA_SUCCESS;
}

odla_status odla_AllocateDevice(const odla_vendor vendor,
                                const odla_device_name device_name,
                                odla_device* device, const char* config) {
  // Init, query and open vvodh devices
#ifdef DEBUG
  std::cout << "[vODLA] INFO: Start initializing vodh device.\n";
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
#endif

  // adaptive resource application, analyze model first
  if (config != NULL) {
    if (odla_adaptiveAlloc(dev, config) == ODLA_FAILURE) {
      std::cout << "[vODLA] ERROR: Failed to reuqest reources adaptively.\n";
      return ODLA_FAILURE;
    }
  }

  if (odla_OpenDevice(dev) == ODLA_FAILURE) {
    return ODLA_FAILURE;
  }

  return ODLA_SUCCESS;
}

odla_status odla_DestroyDevice(odla_device device) {
  // allocate device failed
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

  // vodh_ret vodh_sa_res_release(device->vodh_hd, const char *releaseinfo, u32
  // bufflen);
  if (device->need_release) odla_adaptiveRelease(device);

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
  std::cout << "[vODLA] DEBUG: Binding input idx: " << idx << "\n";
#endif
  if (idx >= MAX_INPUT_TENSOR) {
    return ODLA_FAILURE;
  }
  if (data_ptr == NULL) {
    std::cout << "[vODLA] ERROR: input data " << idx << " pointer is NULL.\n";
    return ODLA_FAILURE;
  }
  if (context->ipPtr[idx] != data_ptr) {
    context->ipPtr[idx] = data_ptr;
    context->dataChg = true;
#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: input " << idx << " address changed to "
              << data_ptr << "\n";
#endif
  }
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
  std::cout << "[vODLA] DEBUG: Binding output idx: " << idx << "\n";
#endif
  if (idx >= MAX_OUTPUT_TENSOR) {
    return ODLA_FAILURE;
  }
  if (data_ptr == NULL) {
    std::cout << "[vODLA] ERROR: output data " << idx << " pointer is NULL.\n";
    return ODLA_FAILURE;
  }
  if (context->opPtr[idx] != data_ptr) {
    context->opPtr[idx] = data_ptr;
    context->dataChg = true;
#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: output " << idx << " address changed to "
              << data_ptr << "\n";
#endif
  }
  context->opNum++;
  return ODLA_SUCCESS;
}

odla_status odla_CreateContext(odla_context* context) {
  *context = new _odla_context();

  if (context == NULL) {
    std::cout << "[vODLA] ERROR: failed to create odla context.\n";
    return ODLA_FAILURE;
  }

  (*context)->Id = g_ctxId++;
  return ODLA_SUCCESS;
}

odla_status odla_DestroyContext(odla_context context) {
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

// ----------enable model release when required----------//
odla_status odla_DestroyComputation(odla_computation comp) {
  // call model_fini at backend
  if (g_dev == NULL) {
    std::cout << "[vODLA] Error: device is not created before model switch.\n";
  } else {
    if (g_dev->vodh_hd == NULL || g_dev->vodh_dev_list == NULL) {
      std::cout
          << "[vODLA] Error: device is not initiated before model switch.\n";
    }
    // else{
    // vodh_model_options mopt;
    // mopt.opcode = TYPE_RELEASE;
    // mopt.pad = 0;
    // mopt.model.use_file = 1;
    // mopt.model.model_id = g_dev->vodh_infer_opt.model.model_id;
    // mopt.model.weight_id = g_dev->vodh_infer_opt.model.weight_id;
    // strcpy(mopt.model.model_file, g_dev->vodh_infer_opt.model.model_file);
    // strcpy(mopt.model.weight_file, g_dev->vodh_infer_opt.model.weight_file);
    // vodh_model_op_result mret;
    // hetong said this is for uc, not for zhaohang
    // if(vodh_model(g_dev->vodh_hd, &(g_dev->vodh_dev_list[0]), &mopt, &mret)){
    //     std::cout << "[vODLA] Error: failed to call model_fini at
    //     backend.\n";
    // }
    // }
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
//----------enable model release when required----------

void deallocDMA(odla_device device, bool ip, bool op) {
  /*DeAllocate DMA for inputs*/
  if (device->vodh_infer_opt.input && ip) {
    for (u32 i = 0; i < device->vodh_infer_opt.input_num; i++) {
#ifdef DEBUG
      printf("input %d data at %p\n", i, device->vodh_infer_opt.input[i]->data);
#endif

      vodh_free(device->vodh_infer_opt.input[i]->data);
      device->vodh_infer_opt.input[i]->data = NULL;
      delete device->vodh_infer_opt.input[i];
      device->vodh_infer_opt.input[i] = NULL;

#ifdef DEBUG
      std::cout << "[vODLA] DEBUG: released vodh input: " << i << "\n";
#endif
    }
    delete[] device->vodh_infer_opt.input;
    device->vodh_infer_opt.input = NULL;

#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: released input pointer\n";
#endif
  }
  /*DeAllocate DMA for outputs*/
  if (device->vodh_infer_result.output && op) {
    for (u32 i = 0; i < device->vodh_infer_result.output_num; i++) {
#ifdef DEBUG
      printf("output %d data at %p\n", i,
             device->vodh_infer_result.output[i]->data);
#endif

      vodh_free(device->vodh_infer_result.output[i]->data);
      device->vodh_infer_result.output[i]->data = NULL;
      delete device->vodh_infer_result.output[i];
      device->vodh_infer_result.output[i] = NULL;

#ifdef DEBUG
      std::cout << "[vODLA] DEBUG: released vodh output: " << i << "\n";
#endif
    }
    delete[] device->vodh_infer_result.output;
    device->vodh_infer_result.output = NULL;

#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: released output pointer\n";
#endif
  }
}

bool allocDMA(odla_device device, odla_context context) {
  bool ret = true;

#ifdef TIMING
  struct timeval t_s;
  struct timeval t_e;
  double time_used;
#endif

  /* Alloc DMA memory for input data */
  u32 ip_mem_cap = 0;
  if (device->vodh_infer_opt.input_num != context->ipNum) {
    deallocDMA(device, true, false);
    device->vodh_infer_opt.input_num = context->ipNum;
    device->vodh_infer_opt.input = new vodh_input*[context->ipNum];
    if (device->vodh_infer_opt.input) {
      for (u32 i = 0; i < context->ipNum; i++) {
        device->vodh_infer_opt.input[i] = new vodh_input();
        if (device->vodh_infer_opt.input[i]) {
          device->vodh_infer_opt.input[i]->data =
              vodh_malloc(context->ipSizes[i]);
          if (device->vodh_infer_opt.input[i]->data && context->ipPtr[i]) {
#ifdef TIMING
            gettimeofday(&t_s, NULL);
#endif
            memcpy(device->vodh_infer_opt.input[i]->data, context->ipPtr[i],
                   context->ipSizes[i]);
#ifdef TIMING
            gettimeofday(&t_e, NULL);
            time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 +
                        (t_e.tv_usec - t_s.tv_usec);
            std::cout << "[vODLA] TIMING: CP DMA data for input " << i << ": "
                      << time_used << "us. (new)\n";
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
        device->vodh_infer_opt.input[i]->size = context->ipSizes[i];
        ip_mem_cap += device->vodh_infer_opt.input[i]->size;
      }
    } else {
      std::cout << "[vODLA] ERROR: allocate infer input failed.\n";
      ret = false;
    }
  } else {
    if (device->vodh_infer_opt.input) {
      for (u32 i = 0; i < context->ipNum; i++) {
        if (device->vodh_infer_opt.input[i]) {
          if (device->vodh_infer_opt.input[i]->size != context->ipSizes[i]) {
            vodh_free(device->vodh_infer_opt.input[i]->data);
            device->vodh_infer_opt.input[i]->data = NULL;
            device->vodh_infer_opt.input[i]->data =
                vodh_malloc(context->ipSizes[i]);
          }
          if (device->vodh_infer_opt.input[i]->data && context->ipPtr[i]) {
#ifdef TIMING
            gettimeofday(&t_s, NULL);
#endif
            memcpy(device->vodh_infer_opt.input[i]->data, context->ipPtr[i],
                   context->ipSizes[i]);
#ifdef TIMING
            gettimeofday(&t_e, NULL);
            time_used = (t_e.tv_sec - t_s.tv_sec) * 1000000 +
                        (t_e.tv_usec - t_s.tv_usec);
            std::cout << "[vODLA] TIMING: CP DMA data for input " << i << ": "
                      << time_used << "us. (reuse)\n";
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
        device->vodh_infer_opt.input[i]->size = context->ipSizes[i];
        ip_mem_cap += device->vodh_infer_opt.input[i]->size;
      }
    } else {
      std::cout << "[vODLA] ERROR: reuse infer input failed.\n";
      ret = false;
    }
  }

  /* Alloc DMA memory for output data */
  u32 op_mem_cap = 0;
  if (device->vodh_infer_result.output_num != context->opNum) {
    deallocDMA(device, false, true);
    device->vodh_infer_result.output_num = context->opNum;
    device->vodh_infer_result.output = new vodh_output*[context->opNum];
    if (device->vodh_infer_result.output) {
      for (u32 i = 0; i < context->opNum; i++) {
        device->vodh_infer_result.output[i] = new vodh_output();
        if (device->vodh_infer_result.output[i]) {
          device->vodh_infer_result.output[i]->data =
              vodh_malloc(context->opSizes[i]);
          if (device->vodh_infer_result.output[i]->data == NULL) {
            std::cout << "[vODLA] ERROR: allocate infer output " << i
                      << " data failed.\n";
            ret = false;
          }
        } else {
          std::cout << "[vODLA] ERROR: allocate infer output: " << i
                    << " failed.\n";
          ret = false;
        }
        device->vodh_infer_result.output[i]->size = context->opSizes[i];
        op_mem_cap += device->vodh_infer_result.output[i]->size;
      }
    } else {
      std::cout << "[vODLA] ERROR: allocate infer output failed.\n";
      ret = false;
    }
  } else {
    if (device->vodh_infer_result.output) {
      for (u32 i = 0; i < context->opNum; i++) {
        if (device->vodh_infer_result.output[i]) {
          if (device->vodh_infer_result.output[i]->size !=
              context->opSizes[i]) {
            vodh_free(device->vodh_infer_result.output[i]->data);
            device->vodh_infer_result.output[i]->data = NULL;
            device->vodh_infer_result.output[i]->data =
                vodh_malloc(context->opSizes[i]);
          }
          if (device->vodh_infer_result.output[i]->data == NULL) {
            std::cout << "[vODLA] ERROR: allocate infer output " << i
                      << " data failed (reuse).\n";
            ret = false;
          }
        } else {
          std::cout << "[vODLA] ERROR: reuse infer output: " << i
                    << " failed.\n";
          ret = false;
        }
        device->vodh_infer_result.output[i]->size = context->opSizes[i];
        op_mem_cap += device->vodh_infer_result.output[i]->size;
      }
    } else {
      std::cout << "[vODLA] ERROR: reuse infer output failed.\n";
      ret = false;
    }
  }

  device->vodh_infer_opt.request_cap.input_memory = ip_mem_cap;
  device->vodh_infer_opt.request_cap.output_memory = op_mem_cap;

  return ret;
}

odla_status odla_ExecuteComputation(odla_computation comp, odla_context context,
                                    odla_compute_mode mode,
                                    odla_device device) {
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
  u64 rid = ((context->Id) << 16) + context->dataId;
  // context changed, new data mem to be allocated
  if (rid != device->vodh_infer_result.request_id) {
#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: infer input data changed.\n";
#endif
    allocSuccess = allocDMA(device, context);

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
    device->vodh_infer_result.request_id = rid;
#ifdef DEBUG
    std::cout << "[vODLA] DEBUG: infer request ID is "
              << device->vodh_infer_result.request_id << "\n";
#endif

    vodh_ret rt;

    /* Additional inference info */
    device->vodh_infer_opt.batch_size = context->batchSize;
    device->vodh_infer_opt.request_id = device->vodh_infer_result.request_id;
    device->vodh_infer_opt.request_cap.type = device->vodh_dev_cap_list[0].type;
    device->vodh_infer_opt.request_cap.net_bw = device->vodh_total_cap.net_bw;
    device->vodh_infer_opt.request_cap.net_delay =
        device->vodh_total_cap.net_delay;
    device->vodh_infer_opt.request_cap.use_direct_rdma = true;
    device->vodh_infer_opt.request_cap.compute =
        0; // TODO: compute power required from cost model.
    /* Alloc DMA memory for model file */
    device->vodh_infer_opt.model.model_data = comp->ccPtr;
    device->vodh_infer_opt.model.model_size = comp->ccFileSz;
    device->vodh_infer_opt.model.model_id = comp->Id;
    strcpy(device->vodh_infer_opt.model.model_file, comp->ccFile.c_str());
    /* Alloc DMA memory for weight file */
    device->vodh_infer_opt.model.weight_data = comp->wtPtr;
    device->vodh_infer_opt.model.weight_size = comp->wtFileSz;
    device->vodh_infer_opt.model.weight_id = comp->Id;
    device->vodh_infer_opt.model.use_file = 1; // use file path instead of DMA
    {
      std::lock_guard<std::mutex> lock(is_new_cc_mu_);
      if (is_new_cc) {
        device->vodh_infer_opt.model.is_needupdate = 1;
        is_new_cc = false;
      } else {
        device->vodh_infer_opt.model.is_needupdate = 0;
      }
    }
    strcpy(device->vodh_infer_opt.model.weight_file, comp->wtFile.c_str());

#ifdef DEBUG
    std::cout << "[vODLA] INFO: start remote inference...\n";
#endif

#ifdef TIMING
    std::cout << "[vODLA] INFO: loop " << LOOP_CNT << " times.\n";

    double xpu_time_avg = 0;
    u64 xpu_time_max = 0;
    u64 xpu_time_min = std::numeric_limits<long long>::max();

    for (u32 lp = 0; lp < LOOP_CNT; lp++) {
      /* Remote infer */
      rt = vodh_infer(device->vodh_hd, &(device->vodh_dev_list[0]),
                      &(device->vodh_infer_opt), &(device->vodh_infer_result));
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

      xpu_time_max =
          std::max(xpu_time_max, device->vodh_infer_result.time_used);
      xpu_time_min =
          std::min(xpu_time_min, device->vodh_infer_result.time_used);
      xpu_time_avg += device->vodh_infer_result.time_used;
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
                    &(device->vodh_infer_opt), &(device->vodh_infer_result));
#endif

    // cp back inference result
    for (u32 i = 0; i < context->opNum; i++) {
      if (context->opPtr[i]) {
        memcpy(context->opPtr[i], device->vodh_infer_result.output[i]->data,
               context->opSizes[i]);
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
      ret = static_cast<odla_status>(device->vodh_infer_result.status);
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

odla_status ODLA_API_CALL odla_LoadExecutable(odla_resource_location location,
                                              odla_device device,
                                              odla_executable* executable) {
  // init vODLA device failed
  if (location.location_type != ODLA_LOCATION_PATH) {
    std::cout << "[vODLA] ERROR: unsupported location type.\n";
    return ODLA_FAILURE;
  }
  const char* file_name = static_cast<const char*>(location.location);
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
