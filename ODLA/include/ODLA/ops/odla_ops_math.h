//===- odla_ops_math.h ----------------------------------------------------===//
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

#ifndef _ODLA_OPERATOR_OPS_MATH_H_
#define _ODLA_OPERATOR_OPS_MATH_H_

#include <ODLA/odla_common.h>
#include <ODLA/odla_value.h>

/*! \file
 * \details This file defines the ODLA mathematic operators.
 */

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Absolute value
/*!
  Abs returns the absolute value of \p input.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Abs(odla_value input, const odla_value_id value_id);

//! \brief ACos
/*!
  Computes acos of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ACos(odla_value x, const odla_value_id value_id);

//! \brief ACosh
/*!
  Computes acosh of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ACosh(odla_value x, const odla_value_id value_id);

//! \brief Addition
/*!
  Add returns the element-wise binary addition of \p lhs and \p rhs.
  It supports broadcasting to the same dimension as \p lhs.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Add(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Logical and
/*!
  And returns the element-wise binary logical \c and of \p lhs and \p rhs.
  It supports broadcasting to the same dimension as \p lhs.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_And(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Find the indices of the largest elements
/*!
  Argmax returns the indices of the largest elements of \p input alone the
  specified \p axis. If \p keep_dims is true, then the resulting value has the
  reduced dimension pruned. Otherwise the resulting value has the same rank as
  \p input. If the max occurs more than once, \p return_last_index specifies
  if the first index or the last index needs to be returned.

  \param input the input value
  \param axis the axis to compute the indices
  \param keep_dims keep the reduced dimension or not
  \param return_last_index which index to return when there is a tie
  \param output_value_type the output value type (an integer value type)
  \param value_id the value id assigned to the result

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ArgMax(odla_value input, odla_int32 axis, odla_bool keep_dims,
            odla_bool return_last_index, odla_value_type output_value_type,
            const odla_value_id value_id);

//! \brief Find the indices of the smallest elements
/*!
  ArgMin returns the indices of the smallest elements of \p input alone the
  specified \p axis. If \p keep_dims is true, then the resulting value has the
  reduced dimension pruned. Otherwise the resulting value has the same rank as
  \p input. If the min occurs more than once, \p return_last_index specifies
  if the first index or the last index needs to be returned.

  \param input the input value
  \param axis the axis to compute the indices
  \param keep_dims keep the reduced dimension or not
  \param return_last_index which index to return when there is a tie
  \param output_value_type the output value type (an integer value type)
  \param value_id the value id assigned to the result

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ArgMin(odla_value input, odla_int32 axis, odla_bool keep_dims,
            odla_bool return_last_index, odla_value_type output_value_type,
            const odla_value_id value_id);

//! \brief ASin
/*!
  Computes asin of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ASin(odla_value x, const odla_value_id value_id);

//! \brief ASinh
/*!
  Computes asinh of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ASinh(odla_value x, const odla_value_id value_id);

//! \brief ATan
/*!
  Computes atan of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ATan(odla_value x, const odla_value_id value_id);

//! \brief ATanh
/*!
  Computes atanh of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ATanh(odla_value x, const odla_value_id value_id);

//! \brief Round up a value
/*!
  Ceil rounds \p input upward, returning the smallest integral value that is not
  less than \p input.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Ceil(odla_value input, const odla_value_id value_id);

//! \brief Clamp a value to a given range
/*!
  If \p input is less than \p lo , returns \p lo. Otherwise
  if \p hi is less than \p input , returns \p hi. Otherwise
  returns \p input.

  \param input the input value to clamp
  \param lo the lower bound to clamp to
  \param hi the upper bound to clamp to
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Clamp(odla_value input, odla_float32 lo, odla_float32 hi,
           const odla_value_id value_id);

//! \brief Compute the determinant of a square matrix
/*!
  Det returns determinant of a squre matrix or batches of square matrices.
  The rank of resulting value is 1.

  \param input the input value
  \param output_shape the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Det(odla_value input, odla_value_shape output_shape,
         const odla_value_id value_id);

extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_CumSum(odla_value input, odla_value axis, odla_bool exclusion,
            odla_bool reverse, const odla_value_id value_id);

//! \brief Division
/*!
  Div returns the element-wise binary division of \p lhs and \p rhs.
  It supports broadcasting to the same dimension as \p lhs.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Div(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Compute Einstein summation.
/*!
  Einsum returns the Einstein summation convention on the inputs.
  \param inputs the input values
  \param equation the expression
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Einsum(odla_values inputs, const odla_char* equation,
            odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Compute guass error of the given input
/*!
  Erf returns element-wise guass error of \p input.
  The result value type is the same as input type.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Erf(odla_value input, const odla_value_id value_id);

//! \brief Equality test
/*!
  Equal tests if \p lhs and \p rhs are equal or not.
  The returning element type is implementation determined.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value.
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Equal(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Compute exponential function
/*!
  Exp returns the element-wise base-e exponential function of \p input.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Exp(odla_value input, const odla_value_id value_id);

//! \brief Round down a value
/*!
  Floor rounds \p input downward, returning the largest integral value that is
  not greater than \p input.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Floor(odla_value input, const odla_value_id value_id);

//! \brief General Batch Matrix Multiplication
/*!
  Gemm computes: \n
   \p A * \p B \n

  \param A the matrix A
  \param A_tranpose indicates if A needs to be transposed or not
  \param B the matrix B
  \param B_tranpose indicates if B needs to be transposed or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_BatchMatMul(
    odla_value A, odla_bool A_transpose, odla_value B, odla_bool B_transpose,
    odla_value_shape output_dims, const odla_value_id value_id);

//! \brief General Matrix Multiplication
/*!
  Gemm computes: \n
   \p alpha * \p A * \p B + \p beta * \p C \n
   \p alpha * \p A^T * \p B + \p beta * \p C \n
   \p alpha * \p A * \p B^T + \p beta * \p C \n
   \p alpha * \p A^T * \p B^T + \p beta * \p C \n

  \param A the matrix A
  \param A_tranpose indicates if A needs to be transposed or not
  \param B the matrix B
  \param B_tranpose indicates if B needs to be transposed or not
  \param alpha the alpha value
  \param beta the beta value
  \param C the optional matrix (can be NULL)
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_Gemm(
    odla_value A, odla_bool A_transpose, odla_value B, odla_bool B_transpose,
    odla_float32 alpha, odla_float32 beta, odla_value C,
    odla_value_shape output_dims, const odla_value_id value_id);

//! \brief "Greater Than" test
/*!
  Greater tests if \p lhs is greater than \p rhs.
  The result element type is implementation determined.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Greater(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief "Greater Than Or Equal" test
/*!
  Greater tests if \p lhs is greater than \p rhs.
  The result element type is implementation determined.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_GreaterOrEqual(
    odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Inverse of a square matrix
/*!
  Inverse returns the inverse of a square matrix or batches of square matrices.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Inverse(odla_value input, const odla_value_id value_id);

//! \brief Check infinity
/*!
  IsInf returns the bool array \p input.

  \param input the input value
  \param detect_pos whether detect positive infinity
  \param detect_neg whether detect negative infinity
  \param value_id a unique value id (can be NULL)

  \return odla_bool
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_IsInf(odla_value input, odla_bool detect_pos, odla_bool detect_neg,
           const odla_value_id value_id);

//! \brief Check whether element of input is a number
/*!
  IsNaN returns the bool array \p input.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_bool
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_IsNaN(odla_value input, const odla_value_id value_id);

//! \brief "Less Than" test
/*!
  Less tests if \p lhs is less than \p rhs.
  The result element type is implementation determined.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Less(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief "Less Than Or Equal" test
/*!
  Less tests if \p lhs is less than \p rhs.
  The result element type is implementation determined.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_LessOrEqual(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Compute the natural logrithm
/*!
  Log returns the natural logarithm of \p input.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Log(odla_value input, const odla_value_id value_id);

//! \brief Return the element-wise largest value
/*!
  Max returns the largest of \p lhs and \p rhs for each element.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Max(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Compute the element-wise mean value of inputs
/*!
  Mean returns the element-wise mean of each input values.
  All inputs must have the same type.

  \param inputs the input values
  \param value_id a unique value id (can be NULL)

  \return odla_value
 */
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Mean(odla_values inputs, const odla_value_id value_id);

//! \brief Returns the element-wise smallest value
/*!
  Min returns the smallest of \p lhs and \p rhs for each element.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Min(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Returns the element-wise modulus value
/*!
  Mod returns the modulus of \p lhs and \p rhs for each element.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Mod(odla_value lhs, odla_value rhs, odla_int64 fmod,
         const odla_value_id value_id);

//! \brief Multiplication
/*!
  Mul returns the element-wise binary mulitplication of \p lhs and \p rhs.
  It supports broadcasting to the same dimension as \p lhs.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Mul(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Flip the sign
/*!
  Neg returns the element-wise negation (y = -x).

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Neg(odla_value input, const odla_value_id value_id);

//! \brief Logical negation
/*!
  Not returns the element-wise negation (y = ! x).

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Not(odla_value input, const odla_value_id value_id);

//! \brief Logical or
/*!
  Or returns the element-wise binary logical \c or of \p lhs and \p rhs.
  It supports broadcasting to the same dimension as \p lhs.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/

//! \brief Inequality test
/*!
  Equal tests if \p lhs and \p rhs are equal or not.
  The returning element type is implementation determined.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value.
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_NotEqual(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Logic or test
/*!
  logic tests operator \p lhs or \p rhs .
  The returning element type is implementation determined.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value.
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Or(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Logic xor test
/*!
  logic tests operator \p lhs xor \p rhs .
  The returning element type is implementation determined.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value.
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Xor(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Raise to power
/*!
  Pow returns element-wise \p base raised to the power \p exponent.

  \param base the base value
  \param exponent the exponent value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Pow(odla_value base, odla_value exponent, const odla_value_id value_id);

//! \brief Compute reciprocal
/*!
  Reciprocal returns the element-wise reciprocal (y = 1 / x).

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Reciprocal(odla_value input, const odla_value_id value_id);

//! \brief Compute the L1 norm alone axes
/*!
  ReduceL1 returns the L1 norm of \p input alone \p axes.

  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param epsilon use to avoid division by zero
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_ReduceL1(
    odla_value input, odla_size_t num_of_axes, const odla_uint32* axes,
    odla_bool keep_dims, odla_float32 epsilon, odla_value_shape output_dims,
    const odla_value_id value_id);

//! \brief Compute the L2 norm alone axes
/*!
  ReduceL2 returns the L2 norm of \p input alone \p axes.

  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param epsilon use to avoid division by zero
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_ReduceL2(
    odla_value input, odla_size_t num_of_axes, const odla_uint32* axes,
    odla_bool keep_dims, odla_float32 epsilon, odla_value_shape output_dims,
    const odla_value_id value_id);

//! \brief Compute the log sum alone axes
/*!
  ReduceLogSum returns the log sum of \p input alone \p axes.

  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ReduceLogSum(odla_value input, odla_size_t num_of_axes,
                  const odla_uint32* axes, odla_bool keep_dims,
                  odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Compute the log sum exponent alone axes
/*!
  ReduceLogSumExp returns the log sum exponent of \p input alone \p axes.

  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_ReduceLogSumExp(
    odla_value input, odla_size_t num_of_axes, const odla_uint32* axes,
    odla_bool keep_dims, odla_value_shape output_dims,
    const odla_value_id value_id);

//! \brief Compute the max alone axes
/*!
  ReduceMax returns the max of \p input alone \p axes.

  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ReduceMax(odla_value input, odla_size_t num_of_axes,
               const odla_uint32* axes, odla_bool keep_dims,
               odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Compute the mean alone axes
/*!
  ReduceMean returns the mean of \p input alone \p axes.
  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)
  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ReduceMean(odla_value input, odla_size_t num_of_axes,
                const odla_uint32* axes, odla_bool keep_dims,
                odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Compute the min alone axes
/*!
  ReduceMin returns the min of \p input alone \p axes.
  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)
  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ReduceMin(odla_value input, odla_size_t num_of_axes,
               const odla_uint32* axes, odla_bool keep_dims,
               odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Compute the production alone axes
/*!
   ReduceProd returns the production of \p input alone \p axes.

  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ReduceProd(odla_value input, odla_size_t num_of_axes,
                const odla_uint32* axes, odla_bool keep_dims,
                odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Compute the sum alone axes
/*!
  ReduceSum returns the sum of \p input alone \p axes.

  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_ReduceSum(odla_value input, odla_size_t num_of_axes,
               const odla_uint32* axes, odla_bool keep_dims,
               odla_value_shape output_dims, const odla_value_id value_id);

//! \brief Compute the sum square alone axes
/*!
  ReduceSumSquare returns the sum square of \p input alone \p axes.

  \param input the input value
  \param num_of_axes nubmer of axes to reduce
  \param axes the axes to reduce
  \param keep_dims keep the reduced dimension or not
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_ReduceSumSquare(
    odla_value input, odla_size_t num_of_axes, const odla_uint32* axes,
    odla_bool keep_dims, odla_value_shape output_dims,
    const odla_value_id value_id);

//! \brief Round to nearest
/*!
  For each element of \p input , round returns the integral value that is
  nearest to \p input , with halfway cases rounded away from zero.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Round(odla_value input, const odla_value_id value_id);

//! \brief reciprocal square root
/*!
  Rsqrt returns the element-wise reciprocal square root of \p input.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Rsqrt(odla_value input, const odla_value_id value_id);

//! \brief conditional (ternary) operator
/*!
  Returns elements, either from A or B, based on the boolean elements of
  Condition.

  \param condition the condition value
  \param a the values to select from when condition is True
  \param b the values to select from when condition is False
  \param output_dims the optional output shape (can be undefined)
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/

extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Select(odla_value condition, odla_value a, odla_value b,
            odla_value_shape output_dims, const odla_value_id value_id);

//! \brief bit shift
/*!
 Shift returns the element-wise bit shift of \p input.
  \param input the input value
  \param shift_amount the shift amount
  \param is_left_shift the shift direction
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Shift(odla_value input, odla_value shift_amount, odla_bool is_left_shift,
           const odla_value_id value_id);

//! \brief Sign of input
/*!
  Sign returns the element-wise sign of \p input.
  The element type of returned value is implementation determined.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_value (odla_int32)
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Sign(odla_value input, const odla_value_id value_id);

//! \brief Square root
/*!
  Sqrt returns the element-wise square root of \p input.

  \param input the input value
  \param value_id a unique value id (can be NULL)

  \return odla_val
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Sqrt(odla_value input, const odla_value_id value_id);

//! \brief Squared Difference
/*!
  SquaredDifference returns ( \p lhs - \p rhs )( \p lhs - \p rhs ) element-wise.
  It supports broadcasting to the same dimension as \p lhs.

  \param lhs the first value
  \param rhs the second value
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL odla_SquaredDifference(
    odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Subtraction
/*!
  Sub returns the element-wise binary subtraction of \p lhs and \p rhs.
  It supports broadcasting to the same dimension as \p lhs.

  \param lhs the first value
  \param rhs the second value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Sub(odla_value lhs, odla_value rhs, const odla_value_id value_id);

//! \brief Cos
/*!
  Computes sine of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Cos(odla_value x, const odla_value_id value_id);

//! \brief Sin
/*!
  Computes sine of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Sin(odla_value x, const odla_value_id value_id);

//! \brief Sinh
/*!
  Computes sineh of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Sinh(odla_value x, const odla_value_id value_id);

//! \brief Cos
/*!
  Computes cosin of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Cos(odla_value x, const odla_value_id value_id);

//! \brief Cosh
/*!
  Computes cosinh of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Cosh(odla_value x, const odla_value_id value_id);

//! \brief Tan
/*!
  Computes tan of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Tan(odla_value x, const odla_value_id value_id);

//! \brief Tanh
/*!
  Computes tanh of \p x element-wise.

  \param x input value
  \param id the value id assigned to the result
  \param value_id a unique value id (can be NULL)

  \return odla_value
*/
extern ODLA_API_EXPORT odla_value ODLA_API_CALL
odla_Tanh(odla_value x, const odla_value_id value_id);

#ifdef __cplusplus
} // C extern
#endif

#endif // _ODLA_OPERATOR_OPS_MATH_H_
