//===- odla_ops_process.h -------------------------------------------------===//
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

#ifndef _ODLA_OPERATOR_OPS_PROCESS_H_
#define _ODLA_OPERATOR_OPS_PROCESS_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_value.h>

/*! \file
 * \details This file defines the ODLA value process replated operators.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! Interpolation methods
typedef enum {
  ODLA_NEAREST, /*!< nearest-neighbor interpolation. */
  ODLA_LINEAR,  /*!< N-Linear interpolation. E.g. bilinear for 2D plane. */
  ODLA_CUBIC,   /*!< N-Cubic interpolation .E.g. bicubic for 2D plane. */
} odla_interpolation_mode;

//! Modes for coordinate transformation during resizing
typedef enum {
  ODLA_ASSYMMETRIC,   /*!< new_coord = orig_coord * scale */
  ODLA_HALF_PIXEL,    /*!< new_coord = (orig_coord + 0.5) * scale - 0.5 */
  ODLA_HALF_PIXEL_TF, /*!< new_coord = orig_coord * scale - 0.5 */
  ODLA_ALIGN_CORNERS, /*!< new_coord = orig_coord * (orig_dim - 1) / (new_dim -
                         1) */
} odla_resize_coordinate_mode;

//! Methods for filling a value
typedef enum {
  ODLA_EyeLike,       /*!< ones on the diagnoal and zeros elsewhere. */
  ODLA_RandomNormal,  /*!< normal distribution. */
  ODLA_RandomUniform, /*!< random uniform distrubution. */
} odla_fill_method;

//! \brief Broadcast the value
/*!
  Broadcast broadcasts the input based on \p output_shape.

  \param input the input value
  \param output_shape the shape after broadcasting
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Broadcast(const odla_value input, odla_value_shape output_shape,
               const odla_value_id value_id);

//! \brief cast the element data type of an input
/*!
  Cast casts the input element type to \p target_type.

  \param input the input value
  \param target_type the data type of casted value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Cast(odla_value input, odla_element_type target_type,
          const odla_value_id value_id);

//! \brief Select slices based on condition
/*!
  Select slices from input along specified axis based on condition vector.

  \param input the input value
  \param condition the condition value (1-D)
  \param axis the axis on which the slices to be selected
  \param max_output_shape the maximum result shape (assume all conditions are
  true)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Compress(odla_value input, odla_value condition, odla_int32 axis,
              odla_value_shape max_output_shape, const odla_value_id value_id);

//! \brief Concatenate multiple values into a single value
/*!
   Concat concatenates multiple values into single one. All inputs
   must have the same dimension except for the dimension size
   of the \p axis to concatenate on.

  \param inputs the input values
  \param axis the axis on which the inputs to be concatenated
  \param output_shape the result shape
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Concat(odla_values inputs, odla_int32 axis, odla_value_shape output_shape,
            const odla_value_id value_id);

//! \brief Broadcast the input tensor
/*!
  ExpandDims broadcast the \p input tensor into the shape of \p output_dims .

  \param input the input value
  \param output_dims the output shape
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ExpandDims(odla_value input, odla_value_shape output_dims,
                const odla_value_id value_id);

//! \brief Generate a value with data
/*!
  Fill genrates a value with type and dimensions sepecified by \p type and
  fill it with data as using the specified \p method. When filling with
  normal distribution, \p p0 and \p p1 are for mean and standard
  deviation, respectively. When filling with unform distribution,
  \p p0 and \p p1 are for the lower and upper bounds, respectively.
  For othere filling methods, \p p0 and \p p1 are ignored.

  \param type the type of generated odla_value
  \param method the method for filling the value
  \param p0 mean for normal distribution, or lower bound for uniform
         distribution
  \param p1 stddev for normal distribution, or upper bound for
         uniform distribution
  \param seed the seed to the random generator
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Fill(odla_value_type type, odla_fill_method method, odla_float32 p0,
          odla_float32 p1, odla_float32 seed, const odla_value_id value_id);

//! \brief Gather slices
/*!
  Gather slices from \p input according to \p indices.

  \param input the input value
  \param indices the indices value
  \param axis the axis on which the input is to gather
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Gather(odla_value input, odla_value indices, odla_int32 axis,
            odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Gather elements
/*!
  Gather slices from \p input according to \p indices.

  \param input the input value
  \param indices the indices value
  \param axis the axis on which the input is to gather
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_GatherElements(odla_value input, odla_value indices, odla_int32 axis,
                    odla_value_shape output_dims, const odla_value_id value_id);

//! \brief one-hot value
/*!
  OneHot returns a one-hot value from \p values based on \p indices
  and \p depth.

  \param indices the indices of "on value"
  \param depth the size of the new dimension
  \param values the pair of off and on values
  \param axis the axis of new dimension on
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_OneHot(
    odla_value indices, odla_int32 depth, odla_value values, odla_int32 axis,
    odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Pad the input
/*!
  Pad pads the \p input with given padding amount.

  \param input the input value
  \param padding_front the padding amount applied to the start of input
  \param padding_back the padding amount applied to the end of input
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Pad(odla_value input, const odla_uint32* padding_front,
         const odla_uint32* padding_back, odla_value_shape output_dims,
         const odla_value_id value_id);

//! \brief Reshape a value
/*!
  Reshape reshapes the input with a new dimension specified by \p output_dims.

  \param input the input value
  \param output_dims the output shape
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Reshape(odla_value input, odla_value_shape output_dims,
             const odla_value_id value_id);

//! \brief Resize by interpolating
/*!
  Resize resizes the input using specified interploation method.

  \param input the input value
  \param interpolation the interpolation method
  \param mode the coordinate transformation mode
  \param axes_mask the mask that indicates which axes need to be resized.
  The LSB corresponds to the shape dimension with stride of 1. For example,
  to resize a tensor in NHWC layout, the mask would be 0b0110.
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Resize(odla_value input, odla_interpolation_mode interpolation,
            odla_resize_coordinate_mode mode, odla_uint32 axes_mask,
            odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Resize by interpolating (dynamic shape)
/*!
  Resize resizes the input using specified interpolation method.
  The shape of output is determined by either `scales` or `sizes`.
  It is an error if both are specified (non-null).

  \param input the input value
  \param scales the scaling value
  \param sizes the size of output value
  \param interpolation the interpolation method
  \param mode the coordinate transformation mode
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_ResizeDynamic(
    odla_value input, odla_value scales, odla_value sizes,
    odla_interpolation_mode interpolation, odla_resize_coordinate_mode mode,
    odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Get the shape of input
/*!
  Shape returns the shape of \p input as a 1D odla_value. The element type of
  the result value is implementation determined.

  \param input the input value
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Shape(odla_value input, odla_value_shape output_dims,
           const odla_value_id value_id);

//! \brief Extract a slice
/*!
  Slice extracts a slice from \p input.

  \param input the input value
  \param start the offets at each slicing dimension
  \param end the ending indices(exclusive) at each slicing dimension
  \param stride the stride at each slicing dimension
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Slice(odla_value input, const odla_int32* start, const odla_int32* end,
           const odla_int32* stride, odla_value_shape output_dims,
           const odla_value_id value_id);

//! \brief Extract a dynamic slice
/*!
  SliceDynamic extracts a dynamic slice from \p input.

  \param input the input value
  \param start the offets at each slicing dimension
  \param size the number of elements at each slicing dimension
  \param stride the stride at each slicing dimension
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_SliceDynamic(
    odla_value input, odla_value start, odla_value size, odla_value stride,
    odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Splits a tensor into num_split tensors along one dimension.
/*!
  Split extracts a slice from \p input.

  \param input the input value
  \param split_dim the dimension along which to split
  \param num_split the number of ways to split
  \param value_ids an array of values ids (can be NULL)

  \return odla_values
*/
extern ODLA_API_EXPORT odla_values ODLA_API_CALL
odla_Split(odla_value input, odla_value split_dim, odla_int32 num_split,
           const odla_value_ids value_ids);

//! \brief Remove dimensions of size 1
/*!
  Squeeze removes dimensions of size 1 from the shape of \p input.
  All single dimensions will be squeezed if num_of_axes is zero.

  \param input the input value
  \param num_of_axes nubmer of axes to squeeze
  \param axes the axes to squeeze
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Squeeze(odla_value input, odla_size_t num_of_axes, const odla_uint32* axes,
             odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Join a sequence of Values along a new axis.
/*!
   Stack joins multiple values into single one along a new axis. All inputs
   must have the same dimension.

  \param inputs the input values
  \param axis the index of the new axis in the dimensions of the result
  \param output_shape the result shape
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Stack(odla_values inputs, odla_int32 axis, odla_value_shape output_shape,
           const odla_value_id value_id);

//! \brief Transpose the input
/*!
  Transpose returns a transposed value based on the \p permutation.

  \param input the input value
  \param permutations the axies for permutation. It should be the same size as
  input_dims and output_dims
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Transpose(odla_value input, odla_value_shape permutations,
               odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Tile input multiples times
/*!
  Replicate a given \p input value multiples times.

  \param input the input value
  \param repeat the dimension numbers of repeated copies along input's
  dimensions.
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Tile(odla_value input, const odla_uint32* repeat,
          odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Tile input multiples times dynamically
/*!
  Replicate a given \p input value multiples times dynamically.

  \param input the input value
  \param repeat the dimension numbers of repeated copies along input's
  dimensions.
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_TileDynamic(odla_value input, odla_value repeat,
                 odla_value_shape output_dims, const odla_value_id value_id);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_OPERATOR_OPS_PROCESS_H_