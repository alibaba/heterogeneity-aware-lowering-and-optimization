#ifndef __VODH_COMMON_H__
#define __VODH_COMMON_H__

typedef int vodh_ret;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

typedef unsigned short bufhd_t;

enum VODH_RET_ {
  VODH_OK = 0,
  VODH_EPERM = 1,    /* Operation not permitted */
  VODH_ENOENT = 2,   /* No such file or directory */
  VODH_ESRCH = 3,    /* No such process */
  VODH_EINTR = 4,    /* Interrupted system call */
  VODH_EIO = 5,      /* I/O error */
  VODH_ENXIO = 6,    /* No such device or address */
  VODH_E2BIG = 7,    /* Argument list too long */
  VODH_ENOEXEC = 8,  /* Exec format error */
  VODH_EBADF = 9,    /* Bad file number */
  VODH_ECHILD = 10,  /* No child processes */
  VODH_EAGAIN = 11,  /* Try again */
  VODH_ENOMEM = 12,  /* Out of memory */
  VODH_EACCES = 13,  /* Permission denied */
  VODH_EFAULT = 14,  /* Bad address */
  VODH_ENOTBLK = 15, /* Block device required */
  VODH_EBUSY = 16,   /* Device or resource busy */
  VODH_EEXIST = 17,  /* File exists */
  VODH_EXDEV = 18,   /* Cross-device link */
  VODH_ENODEV = 19,  /* No such device */
  VODH_ENOTDIR = 20, /* Not a directory */
  VODH_EISDIR = 21,  /* Is a directory */
  VODH_EINVAL = 22,  /* Invalid argument */
  VODH_ENFILE = 23,  /* File table overflow */
  VODH_EMFILE = 24,  /* Too many open files */
  VODH_ENOTTY = 25,  /* Not a typewriter */
  VODH_ETXTBSY = 26, /* Text file busy */
  VODH_EFBIG = 27,   /* File too large */
  VODH_ENOSPC = 28,  /* No space left on device */
  VODH_ESPIPE = 29,  /* Illegal seek */
  VODH_EROFS = 30,   /* Read-only file system */
  VODH_EMLINK = 31,  /* Too many links */
  VODH_EPIPE = 32,   /* Broken pipe */
  VODH_EDOM = 33,    /* Math argument out of domain of func */
  VODH_ERANGE = 34,  /* Math result not representable */
};

enum VODH_XPU_TYPE {
  TYPE_GPU,
  TYPE_IPU,
  TYPE_NPU,
  TYPE_X86_CPU,
  TYPE_ARM_CPU,
};

enum VODH_MODEL_OP_TYPE {
  TYPE_LOAD,
  TYPE_RELEASE,
  TYPE_QUERY,
};

enum VODLA_DEV_RUN_MODE {
  VODLA_DEV_RUN_REMOTE_MODE,
  VODLA_DEV_RUN_LOCAL_MODE,
  VODLA_DEV_RUN_MIXED_MODE, /* not support */
};

#define MAX_VODLA_XPU_TPTE_DES 256
struct vodla_xpu_attr {
  enum VODH_XPU_TYPE type;
  enum VODLA_DEV_RUN_MODE run_mode;
  char description[MAX_VODLA_XPU_TPTE_DES]; /*detail description*/
};

struct vodla_dev_key {
  u16 dev_name_len;
  char* dev_name;
  u16 token_len;
  char* token;
  u16 domain_len;
  char* domain;
  struct vodla_xpu_attr xpu_attr;
};

#define VODLA_TRANSFER_STEP_LEN 32 * 1024 * 1024
#define MAX_DEV_NAME_LEN 128
#define MAX_VODLA_FILE_NAME_LEN 255 /*include path*/

#define MAX_VODLA_MODEL_NAME_LEN 255

#define VODLA_DEV_RESERVED_BUFFER 256

#define VODH_ALIGN_SIZE 4096
#define CALC_ALIGN_LEN(size) \
  ((size + VODH_ALIGN_SIZE - 1) / VODH_ALIGN_SIZE * VODH_ALIGN_SIZE)

/* total vxpu cap for user */
struct vodh_total_cap {
  u32 vxpu_num;  /* total vxpus in one vodla device */
  u16 support;   // XPU types supported(bitmap of enum VODH_XPU_TYPE)
  u32 net_bw;    // unit: Mbps
  u32 net_delay; // unit: ns
  u64 memory;    // unit: Byte
  u64 compute;   // unit: FLOPS
};

struct vodh_dev_cap {
  enum VODH_XPU_TYPE type;
  u64 memory;  // unit: Byte
  u64 compute; // unit: FLOPS
};

struct vodh_dev {
  char name[MAX_DEV_NAME_LEN];
  void* _vodh_dev;
};

struct vodh_request_cap {
  enum VODH_XPU_TYPE type;
  u32 net_bw;          // unit: Mbps
  u32 net_delay;       // unit: ns
  u32 use_direct_rdma; // use direct rdma first if support
  u64 input_memory;    // unit: Byte
  u64 output_memory;   // unit: Byte
  u64 compute;         // unit: FLOPS
};

struct vodh_model {
  u8 use_file;      /* 0 use ram(default), 1 use file*/
  u8 is_needupdate; /* 0 not need, 1 need */
  u16 res2;
  u32 res3;
  u64 model_id;
  u64 model_size;
  void* model_data;
  u64 weight_id;
  u64 weight_size;
  void* weight_data;
  char weight_file[MAX_VODLA_FILE_NAME_LEN + 1]; /* inclue path */
  char model_file[MAX_VODLA_FILE_NAME_LEN + 1];  /* inclue path */
};

struct vodh_input {
  u64 size;
  void* data;
};

struct vodh_output {
  u64 size;
  void* data;
};

#define VODLA_MAX_INOUTPUT_NUM 64

struct vodh_infer_options {
  u64 request_id;
  u16 batch_size;
  u16 input_num;
  u32 pad;
  struct vodh_request_cap request_cap;
  struct vodh_model model;
  struct vodh_input** input;
  // struct vodh_output *output[VODLA_MAX_INOUTPUT_NUM];
};

struct vodh_infer_result {
  u64 request_id;  // will be return in the response
  vodh_ret status; // inference result status returned
  u16 output_num;
  u16 pad;
  u64 time_used; // unit: us
  struct vodh_output** output;
};

struct vodh_model_options {
  u32 opcode; // VODH_MODEL_OP_TYPE
  u32 pad;
  char modelname[MAX_VODLA_MODEL_NAME_LEN + 1];
  struct vodh_model model;
};

struct vodh_model_op_result {
  vodh_ret status;
  u32 pad;
};

#if 0
struct vodh_model_op_result {
	u32 odla_status;
	u32 odla_computation;
};

struct vodh_context_op_result {
	u32 odla_status;
	u32 odla_context;
};

struct vodh_context_options{
	u32 odla_computation;
	u32 odla_context;
};

struct vodh_exec_comp_options {
	u64 request_id;
	u32 odla_computation;
	u32 odla_context;
	u16 input_num;
	struct vodh_input **input;
};

struct vodh_exec_comp_result {
	u64 request_id;  //will be return in the response
	vodh_ret status; //inference result status returned
	u16 output_num;
	u16 pad;
	u64 time_used;   //unit: us
	struct vodh_output **output;
};

struct vodh_swap_odlaattr_options {
	u64 request_id;
	u32 odla_computation;
	u32 odla_context;
	u16 input_num;
	u16 function_id;
	struct vodh_input **input;
};

struct vodh_swap_odlaattr_result {
	u64 request_id;  //will be return in the response
	vodh_ret status; //inference result status returned
	u16 output_num;
	u16 function_id;
	u16 function_type;
	u16 pad1;
	u32 pad2;
	u64 time_used;   //unit: us
	struct vodh_output **output;
};
#endif

void* vodh_init(void);

vodh_ret vodh_get_total_cap(void* vodh_handle, struct vodh_total_cap* cap);

/* allvodh_dev num is vxpu_num in struct vodh_total_cap
 * vodh_handle is returned by vodh_init *     */
vodh_ret vodh_get_all_dev_info(void* vodh_handle, struct vodh_dev* allvodh_dev);

vodh_ret vodh_get_one_dev_cap(void* vodh_handle, struct vodh_dev* dev,
                              struct vodh_dev_cap* cap);

void vodh_deinit(void* vodh_handle);

vodh_ret vodh_dev_open(void* vodh_handle, struct vodh_dev* dev);

vodh_ret vodh_infer(void* vodh_handle, struct vodh_dev* dev,
                    struct vodh_infer_options* options,
                    struct vodh_infer_result* result);

vodh_ret vodh_model(void* vodh_handle, struct vodh_dev* dev,
                    struct vodh_model_options* moptions,
                    struct vodh_model_op_result* mresult);

vodh_ret vodh_dev_close(void* vodh_handle, struct vodh_dev* dev);

void* vodh_malloc(u32 size);

void vodh_free(void* ptr);

vodh_ret vodh_sa_get_apply_key(void* vodh_handle, void** outbuf,
                               u32* outbufflen);

void vodh_sa_free_apply_key(void* vodh_handle, void* keybuffer);

vodh_ret vodh_sa_res_apply(void* vodh_handle, const char* applyinfo,
                           u32 bufflen);

vodh_ret vodh_sa_res_release(void* vodh_handle, const char* releaseinfo,
                             u32 bufflen);

#endif
