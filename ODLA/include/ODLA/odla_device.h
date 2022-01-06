//===- odla_device.h ------------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
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

#ifndef _ODLA_DEVICE_H_
#define _ODLA_DEVICE_H_

#include <ODLA/odla_common.h>

/*! \file
 * \details This file defines the ODLA device related APIs.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Supported vendors
typedef enum {
  ODLA_VENDOR_DEFAULT,
  ODLA_VENDOR_ALIBABA,
  ODLA_VENDOR_ARM,
  ODLA_VENDOR_CAMBRICON,
  ODLA_VENDOR_GRAPHCORE,
  ODLA_VENDOR_HABANA,
  ODLA_VENDOR_INTEL,
  ODLA_VENDOR_NVIDIA,
  ODLA_VENDOR_QUALCOMM,
} odla_vendor_name;

//! \brief Vendor info
typedef enum {
  OVI_DESCRIPTION,
  OVI_FEATURE_SET,
} odla_vendor_info;

//! \brief Supported devices
typedef enum {
  ODLA_DEVICE_DEFAULT,
  // Alibaba
  ODLA_DEVICE_ALIBABA_HANGUANG,
  // ARM
  ODLA_DEVICE_ARM_CORTEX_M,
  // Cambricon
  ODLA_DEVICE_CAMBRICON_MLU220,
  ODLA_DEVICE_CAMBRICON_MLU270,
  // GraphCore
  ODLA_DEVICE_GRAPHCORE_IPU,
  // Intel Habana
  ODLA_DEVICE_HABANA_GAUDI,
  ODLA_DEVICE_HABANA_GOYA,
  // Intel CPU
  ODLA_DEVICE_INTEL_X86,
  ODLA_DEVICE_INTEL_DNNL,
  // NVidia
  ODLA_DEVICE_NVIDIA_GPU,
  ODLA_DEVICE_NVIDIA_TENSORRT,
  // Qualcomm
  ODLA_DEVICE_QUALCOMM_AIC100,
} odla_device_name;

//! \brief Device info
typedef enum {
  ODLA_DEVICE_INFO_DESCRIPTION,
  ODLA_DEVICE_INFO_ODLA_VERSION,
  ODLA_DEVICE_INFO_NUM_CORES,
  ODLA_DEVICE_INFO_TOTAL_MEM_SIZE,
  ODLA_DEVICE_INFO_CORE_MEM_SIZE,
} odla_device_info;

//! \brief Vendor object
typedef struct _odla_vendor* odla_vendor;

//! \brief Device object
typedef struct _odla_device* odla_device;

//! \brief Device configuration object
typedef struct _odla_device_config* odla_device_config;

//! \brief Device configuration item object
typedef struct _odla_device_config_item* odla_device_config_item;

//! \brief Get the vendor object
/*!
  \param vendor_name the vendor name
  \param vendor the pointer to the accessed device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetVendor(const odla_vendor_name vendor_name, odla_vendor* vendor);

//! \brief Get the vendor info
/*!
  \param vendor the vendor object
  \param info_name the querying info name
  \param allocated_info_value_size the allocated info_value size
  \param info_value the pointer to the info_value
  \param retrieved_info_value_size the retrieved info_value size

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_GetVendorInfo(
    const odla_vendor vendor, const odla_vendor_info info_name,
    const odla_size_t allocated_info_value_size, odla_void* info_value,
    odla_size_t* retrieved_info_value_size);

//! \brief Allocate a device
/*!
  \param vendor the vendor object (can be NULL)
  \param device_name the device name
  \param device the pointer to the allocated device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_AllocateDevice(
    const odla_vendor vendor, const odla_device_name device_name,
    odla_device* device, const char* config);

//! \brief Get the device info
/*!
  \param device the device object
  \param info_name the querying info name
  \param allocated_info_value_size the allocated info_value size
  \param info_value the pointer to the info_value
  \param retrieved_info_value_size the retrieved info_value size

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_GetDeviceInfo(
    const odla_device device, const odla_device_info info_name,
    const odla_size_t allocated_info_value_size, odla_void* info_value,
    odla_size_t* retrieved_info_value_size);

//! \brief Initialize an allocated device
/*!
  \param device the device object
  \param config the device configuration object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_InitDevice(odla_device device, const odla_device_config config);

//! \brief Set an allocated device as the current device
/*!
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetCurrentDevice(odla_device device);

//! \brief Destroy an allocated device
/*!
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_DestroyDevice(odla_device device);

//! \brief Create a device config object
/*!
  \param device_config the pointer to the created config object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_CreateDeviceConfig(odla_device_config* device_config);

//! \brief Set the device config with a property item
/*!
  \param device_config the device config object
  \param device_config_item the item
  \param variadic

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_SetDeviceConfigItem(odla_device_config device_config,
                         odla_device_config_item device_config_item, ...);

//! \brief Destroy a device config object
/*!
  \param device_config the config object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_DestroyDeviceConfig(odla_device_config device_config);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_DEVICE_H_
