//===- odla_compute.h -----------------------------------------------------===//
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

#ifndef _ODLA_COMPUTE_H_
#define _ODLA_COMPUTE_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_device.h>
#include <ODLA/odla_value.h>

/*! \file
 * \details This file defines the ODLA compute related APIs.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Compute mode
typedef enum {
  ODLA_COMPUTE_DEFAULT,
  ODLA_COMPUTE_INFERENCE,
  ODLA_COMPUTE_TRAINING,
} odla_compute_mode;

//! \brief Property item type
typedef enum {
  ODLA_DYNAMIC_BATCH,
  ODLA_MIN_BATCH_SIZE,
  ODLA_MAX_BATCH_SIZE,
  ODLA_OPT_BATCH_SIZE,
  ODLA_RUN_BATCH_SIZE,
  ODLA_BF16_MODE,
  ODLA_FP16_MODE,
  ODLA_USE_SIM_MODE,
  ODLA_PROCESSOR_NUM,
  ODLA_BATCHES_PER_STEP,
  ODLA_USE_DATA_TYPE,
} odla_item_type;

//! \brief Computation object
typedef struct _odla_computation* odla_computation;

//! \brief Executable object
typedef struct _odla_executable* odla_executable;

//! \brief Context object
typedef struct _odla_context* odla_context;

//! \brief Property item value object
typedef struct _odla_item_value* odla_item_value;

//! \brief Constants array object
typedef struct _odla_constants_array* odla_constants_array;

//! \brief Create a computation object
/*!
  \param computation the pointer to the created computation object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_CreateComputation(odla_computation* computation);

//! \brief Compile a computation object into executable
/*!
  \param computation the computation object
  \param excecutable the pointer to the compiled excecutable object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_CompileComputation(
    const odla_computation computation, odla_executable* executable);

//! \brief Load a computation from the file system
/*!
  \param file_name the file name
  \param computation the pointer to the loaded computation object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_LoadComputation(const odla_char* file_name, odla_computation* computation);

//! \brief Store a computation object into the file system
/*!
  \param file_name the file name
  \param computation the computation object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_StoreComputation(
    const odla_char* file_name, const odla_computation computation);

//! \brief Differentiate a computation
/*!
  \param computation the computation object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_DifferentiateComputation(odla_computation computation);

//! \brief Execute a computation
/*!
  \param computation the computation object
  \param context the context object
  \param mode the compute mode
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_ExecuteComputation(
    const odla_computation computation, const odla_context context,
    const odla_compute_mode mode, odla_device device);

//! \brief Asynchronously execute a computation
/*!
  \param computation the computation object
  \param context the context object
  \param mode the compute mode
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_AsyncExecuteComputation(
    const odla_computation computation, const odla_context context,
    const odla_compute_mode mode, odla_device device);

//! \brief Get the number of arguments from a computation
/*!
  \param computation the computation object
  \param num_args the pointer to the retrieved number of args

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetNumOfArgsFromComputation(const odla_computation computation,
                                 odla_uint32* num_args);

//! \brief Get the #idx argument value from a computation
/*!
  \param computation the computation object
  \param arg_idx the index of argument
  \param arg_value the pointer to the retrieved argument value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetArgFromComputationByIdx(const odla_computation computation,
                                const odla_uint32 arg_idx,
                                odla_value* arg_value);

//! \brief Get the number of outputs from a computation
/*!
  \param computation the computation object
  \param num_outputs the pointer to the retrieved number of outputs

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetNumOfOutputsFromComputation(const odla_computation computation,
                                    odla_uint32* num_outputs);

//! \brief Get the #idx output value from a computation
/*!
  \param computation the computation object
  \param output_idx the index of output
  \param output_value the pointer to the retrieved output value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetOutputFromComputationByIdx(const odla_computation computation,
                                   const odla_uint32 output_idx,
                                   odla_value* output_value);

//! \brief Destroy a created computation
/*!
  \param computation the computation object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_DestroyComputation(odla_computation computation);

//! \brief Set the computation with a property item
/*!
  \param computation the computation object
  \param type the property item type
  \param value the property item value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_SetComputationItem(
    odla_computation computation, odla_item_type type, odla_item_value value);

//! \brief Create a constants array object
/*!
  \param constants_array the pointer to the created constants array object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_CreateConstantsArray(odla_constants_array* constants_array);

//! \brief Load a constants array from the file system
/*!
  \param file_name the file name
  \param constants_array the pointer to the loaded constants array object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_LoadConstantsArray(
    const odla_char* file_name, odla_constants_array* constants_array);

//! \brief Store a constants array into the file system
/*!
  \param file_name the file name
  \param constants_array the constants array object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_StoreConstantsArray(
    const odla_char* file_name, const odla_constants_array constants_array);

//! \brief Destroy a created constants array object
/*!
  \param constants_array the constants array object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_DestroyConstantsArray(odla_constants_array constants_array);

//! \brief Create an executable object
/*!
  \param executable the pointer to the created executable object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_CreateExecutable(odla_executable* executable);

//! \brief Load an executable from the file system
/*!
  \param file_name the file name
  \param executable the pointer to the loaded executable object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_LoadExecutable(const odla_char* file_name, odla_executable* executable);

//! \brief Store an executable object into the file system
/*!
  \param file_name the file name
  \param executable the executable object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_StoreExecutable(
    const odla_char* file_name, const odla_executable executable);

//! \brief Launch an executable
/*!
  \param executable the executable object
  \param constants_array the constants array object (can be NULL)
  \param context the context object
  \param mode the compute mode
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_LaunchExecutable(
    const odla_executable executable,
    const odla_constants_array constants_array, const odla_context context,
    const odla_compute_mode mode, odla_device device);

//! \brief Asynchronously launch an executable
/*!
  \param executable the executable object
  \param constants_array the constants array object (can be NULL)
  \param context the context object
  \param mode the compute mode
  \param device the device object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_AsyncLaunchExecutable(
    const odla_executable executable,
    const odla_constants_array constants_array, const odla_context context,
    const odla_compute_mode mode, odla_device device);

//! \brief Get the number of arguments from an Executable
/*!
  \param executable the executable object
  \param num_args the pointer to the retrieved number of args

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetNumOfArgsFromExecutable(const odla_executable executable,
                                odla_uint32* num_args);

//! \brief Get the #idx argument value from an executable
/*!
  \param executable the executable object
  \param arg_idx the index of argument
  \param arg_value the pointer to the retrieved argument value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_GetArgFromExecutableByIdx(
    const odla_executable executable, const odla_uint32 arg_idx,
    odla_value* arg_value);

//! \brief Get the number of outputs from an executable
/*!
  \param executable the executable object
  \param num_outputs the pointer to the retrieved number of outputs

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetNumOfOutputsFromExecutable(const odla_executable executable,
                                   odla_uint32* num_outputs);

//! \brief Get the #idx output value from an executable
/*!
  \param executable the executable object
  \param output_idx the index of output
  \param output_value the pointer to the retrieved output value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_GetOutputFromExecutableByIdx(const odla_executable executable,
                                  const odla_uint32 output_idx,
                                  odla_value* output_value);

//! \brief Destroy a created executable
/*!
  \param executable the executable object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_DestroyExecutable(odla_executable executable);

//! \brief Create a context object
/*!
  \param context the pointer to the created context object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_CreateContext(odla_context* context);

//! \brief Set the context with a property item
/*!
  \param context the context object
  \param type the property item type
  \param value the property item value

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_SetContextItem(
    odla_context context, odla_item_type type, odla_item_value value);

//! \brief Destroy a created context
/*!
  \param context the context object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_DestroyContext(odla_context context);

//! \brief Bind data to an argument
/*!
  Bind input data to an argument of computation. An error
  will be returned if `value` is not an argument.
  \param value the argument
  \param data_ptr the pointer to the host memory
  \param context the context object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_BindToArgument(
    odla_value value, const odla_void* data_ptr, odla_context context);

//! \brief Bind data to an argument by id
/*!
  Bind input data to an argument of computation. An error
  will be returned if the value specified by `value_id` is not an argument.
  \param value_id the value id for the argument
  \param data_ptr the pointer to the host memory
  \param context the context object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_BindToArgumentById(const odla_value_id value_id, const odla_void* data_ptr,
                        odla_context context);

extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_BindValueToArgumentById(
    const odla_value_id value_id, odla_value data, odla_context context);

//! \brief Bind memory to an output value
/*!
  Bind a memory buffer to an output of computation. An error
  will be returned if the value specified by `value_id` is not set as output
  value.
  \param value the output value
  \param data_ptr the pointer to the host data buffer
  \param context the context object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL
odla_BindToOutput(odla_value value, odla_void* data_ptr, odla_context context);

//! \brief Bind memory to an output value by id
/*!
  Bind a memory buffer to an output of computation by id. An error
  will be returned if the value specified by `value_id` is not set as output
  value.
  \param value_id the id for the output value
  \param data_ptr the pointer to the host data buffer
  \param context the context object

  \return odla_status
*/
extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_BindToOutputById(
    const odla_value_id value_id, odla_void* data_ptr, odla_context context);

extern ODLA_API_EXPORT odla_status ODLA_API_CALL odla_BindValueToOutputById(
    const odla_value_id value_id, odla_value data, odla_context context);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_COMPUTE_H_
