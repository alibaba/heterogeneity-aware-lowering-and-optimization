
// RUN: g++ %s %flags -o %t.exe %link -lodla_tensorrt -I%odla_path/include

// RUN: %t.exe

#include <ODLA/odla.h>

#include <iostream>
#include <map>
#include <string>

#include "ODLA/odla_common.h"
#include "ODLA/odla_device.h"

#define str(x) #x
#define xstr(x) str(x)
#define ADD_ITEM(x) items[x] = xstr(x);

static void dump(const std::string& name, const odla_scalar_value& val) {
  std::cout << name << ":\t";
  if (name == "ODLA_DEVICE_INFO_DEV_UUID") {
    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << std::hex;
    const unsigned char* s =
        reinterpret_cast<const unsigned char*>(val.val_str);
    for (int i = 0; i < 16; ++i) {
      std::cout << (unsigned)s[i];
    }
    std::cout.flags(f);
    std::cout << "\n";
    return;
  }
  std::string suffix;
  suffix = name.find("_POWER_") == std::string::npos ? suffix : "Watt";
  bool is_percent = (name.substr(name.size() - 5) == "_UTIL");
  int scale = is_percent ? 100 : 1;
  suffix = is_percent ? "%" : suffix;
  switch (val.data_type) {
    case ODLA_INT32:
      std::cout << val.val_int32;
      break;
    case ODLA_UINT32:
      std::cout << val.val_uint32;
      break;
    case ODLA_FLOAT32:
      std::cout << scale * val.val_fp32;
      break;
    case ODLA_INT64:
      std::cout << val.val_int64;
      break;
    case ODLA_UINT64:
      std::cout << val.val_uint64;
      break;
    case ODLA_STRING:
      std::cout << val.val_str;
      break;
    default:
      std::cout << "--";
  }
  std::cout << " " << suffix << std::endl;
}

int main() {
  odla_device dev = nullptr;
  auto s = odla_AllocateDevice(nullptr, ODLA_DEVICE_DEFAULT, 0, &dev);
  std::map<odla_device_info, std::string> items;
  ADD_ITEM(ODLA_DEVICE_INFO_ODLA_LIB_VERSION);
  ADD_ITEM(ODLA_DEVICE_INFO_DEV_COUNT);
  ADD_ITEM(ODLA_DEVICE_INFO_DEV_INDEX);
  ADD_ITEM(ODLA_DEVICE_INFO_DEV_TYPE);
  ADD_ITEM(ODLA_DEVICE_INFO_DEV_UUID);
  ADD_ITEM(ODLA_DEVICE_INFO_PROCESSOR_UTIL);
  ADD_ITEM(ODLA_DEVICE_INFO_MEMORY_UTIL);
  ADD_ITEM(ODLA_DEVICE_INFO_TOTAL_MEMORY);
  ADD_ITEM(ODLA_DEVICE_INFO_USED_MEMORY);
  ADD_ITEM(ODLA_DEVICE_INFO_POWER_USAGE);
  ADD_ITEM(ODLA_DEVICE_INFO_POWER_LIMIT);
  ADD_ITEM(ODLA_DEVICE_INFO_DRIVER_VERSION);
  ADD_ITEM(ODLA_DEVICE_INFO_SDK_VERSION);

  for (const auto& kv : items) {
    odla_scalar_value val;
    auto s = odla_GetDeviceInfo(dev, kv.first, &val);
    if (s != ODLA_SUCCESS) {
      std::cout << "Failed to get " << kv.second << std::endl;
      return 1;
    }
    dump(kv.second, val);
  }
  return 0;
}