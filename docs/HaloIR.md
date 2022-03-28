# Halo IR  
---
<!-- markdown-link-check-disable -->
## 1. [common](#common)  
* [Call](#Call)  
* [Concat](#Concat)  
* [Gather](#Gather)  
* [OneHot](#OneHot)  
* [Pad](#Pad)  
* [Reshape](#Reshape)  
* [Return](#Return)  
* [SetDiff1D](#SetDiff1D)  
* [Slice](#Slice)  
* [Stack](#Stack)  

## 2. [common.cast](#common.cast)  
* [FPtoSI](#FPtoSI)  
* [SItoFP](#SItoFP)  
* [ZExt](#ZExt)  

## 3. [common.reduction](#common.reduction)  
* [Argmax](#Argmax)  
* [Argmin](#Argmin)  
* [ReduceMax](#ReduceMax)  
* [ReduceMean](#ReduceMean)  
* [ReduceMin](#ReduceMin)  
* [ReduceProduct](#ReduceProduct)  
* [ReduceSum](#ReduceSum)  

## 4. [controlFlow](#controlFlow)  
* [If](#If)  
* [Jump](#Jump)  
* [Loop](#Loop)  

## 5. [creator](#creator)  
* [RandomUniform](#RandomUniform)  
* [Range](#Range)  

## 6. [image](#image)  

## 7. [math](#math)  
* [BatchMatMul](#BatchMatMul)  
* [Gemm](#Gemm)  
* [InnerProduct](#InnerProduct)  
* [MatMul](#MatMul)  
* [Transpose](#Transpose)  

## 8. [math.binary](#math.binary)  
* [Add](#Add)  
* [And](#And)  
* [Cmp](#Cmp)  
* [Div](#Div)  
* [Maximum](#Maximum)  
* [Minimum](#Minimum)  
* [Mul](#Mul)  
* [Or](#Or)  
* [Pow](#Pow)  
* [ShiftL](#ShiftL)  
* [ShiftR](#ShiftR)  
* [Sub](#Sub)  

## 9. [math.unary](#math.unary)  
* [Abs](#Abs)  
* [Ceil](#Ceil)  
* [Erf](#Erf)  
* [Exp](#Exp)  
* [Floor](#Floor)  
* [Neg](#Neg)  
* [Rcp](#Rcp)  
* [Rsqrt](#Rsqrt)  
* [Sign](#Sign)  
* [Sqrt](#Sqrt)  

## 10. [nn](#nn)  
* [BatchNorm](#BatchNorm)  
* [LRN](#LRN)  

## 11. [nn.activation](#nn.activation)  
* [Elu](#Elu)  
* [LeakyRelu](#LeakyRelu)  
* [Relu](#Relu)  
* [Relu6](#Relu6)  
* [Sigmoid](#Sigmoid)  
* [Softmax](#Softmax)  
* [Tanh](#Tanh)  

## 12. [nn.cnn](#nn.cnn)  
* [Conv2D](#Conv2D)  
* [Conv2DTranspose](#Conv2DTranspose)  
* [PoolingAvg](#PoolingAvg)  
* [PoolingMax](#PoolingMax)  

## 13. [nn.rnn](#nn.rnn)  

## 14. [objectDetection](#objectDetection)  
* [NonMaxSuppression](#NonMaxSuppression)  
* [Resize](#Resize)  
* [TopK](#TopK)  

## 15. [quantization](#quantization)  

---
<a id="common"></a>
# common  
---
<a id="Call"></a>
## Call  
Call a Halo Function  

**Attributes:**  
callee: fp0, default to nullptr, The callee function  

**Operands:**  
X1: (T1). The arguments passed to callee  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Concat"></a>
## Concat  
Concatenate a list of inputs along axis.  

**Attributes:**  
axis: int32, default to 0, The axis to concatenate on.  
N: int32, default to 0, The number of inputs.  

**Operands:**  
X1: (T1). The list of inputs.  
X2(OPTIONAL): (T2), scalar. The axis.  

**Results:**  
Y1: (T2). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
<a id="Gather"></a>
## Gather  
Gather data from input X1 along axis based on X2.  

**Attributes:**  
axis: int32, default to 0, The axis to gather on  

**Operands:**  
X1: (T1). The inpput data.  
X2: (T2). The indices.  
X3(OPTIONAL): (T3), scalar. The axis.  

**Results:**  
Y1: (T3). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32, int64  
T3: int32  
<a id="OneHot"></a>
## OneHot  
Generate a one-hot NDim array.  

**Attributes:**  
axis: int32, default to -1, The axis to fill data in the result.  

**Operands:**  
X1: (T1). Indices to fill in the on-value X2  
X2: (T2), scalar. The size of the one hot dimension specified by axis  
X3: (T3), scalar. The on-value.  
X4: (T3), scalar. The off-value.  

**Results:**  
Y1: (T3). The result.  

**Type Constraints:**  
T1: int32, int64  
T2: int32  
T3: int8, int16, int32, fp16, fp32  
<a id="Pad"></a>
## Pad  
Pad the input data based on various algorithms specified by modes along each spatial axis.  

**Attributes:**  
mode: Enum PadMode, default to CONSTANT, The padding algorithm such as CONSTANT, SYMMETRIC, REFLECT.  

**Operands:**  
X1: (T1). The input data.  
X2: (T2), 2D. The padding data indicating how many elements to add  before and after each axis of input, it has a shape of  [R, 2], where R is the rank of input.  
X3(OPTIONAL): (T2), scalar. The padding value used when modes is CONSTANT  

**Results:**  
Y1: (T2). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
<a id="Reshape"></a>
## Reshape  
Reshape the input X1 to create the result with the same number of elements and the shape specified by X2.  

**Operands:**  
X1: (T1). The input.  
X2: (T2), 1D. The shape.  

**Results:**  
Y1: (T2). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32, int64  
<a id="Return"></a>
## Return  
Specify the final output results.  

**Operands:**  
X1: (T1). The operands that will be used as final results  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="SetDiff1D"></a>
## SetDiff1D  
Return the data from X1 if it is not in X2 and preserve the original order.  

**Operands:**  
X1: (T1), 1D. Values to keep.  
X2: (T1), 1D. Values to remove.  

**Results:**  
Y1: (T1), 1D. The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Slice"></a>
## Slice  
Extract a slice from the input.  

**Operands:**  
X1: (T1). The input.  
X2: (T2), 1D. The starting indices, inclusive, along the specified axes.  
X3: (T3), 1D. The size of the slice, along the specified axes.  
X4(OPTIONAL): (T4), 1D. The slice steps along the specified axes.  
X5(OPTIONAL): (T5), 1D. The axes  

**Results:**  
Y1: (T5). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
T3: int32  
T4: int32  
T5: int32  
<a id="Stack"></a>
## Stack  
Join a list of NDArray inputs along a new axis  

**Attributes:**  
axis: int32, default to 0, The axis to stack on  

**Operands:**  
X1: (T1). The list of inputs.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="common.cast"></a>
# common.cast  
---
<a id="FPtoSI"></a>
## FPtoSI  
Cast the element of input X1 from floating pointto the integer type  

**Attributes:**  
data_type: Enum DataType, default to INVALID, The datatype to which the input data are cast  

**Operands:**  
X1: (T1). The input.  

**Results:**  
Y1: (T2). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int8, int16, int32, fp16, fp32  
<a id="SItoFP"></a>
## SItoFP  
Cast the element of input X1 from signed integerto floating point type  

**Attributes:**  
data_type: Enum DataType, default to INVALID, The datatype to which the input data are cast  

**Operands:**  
X1: (T1). The input.  

**Results:**  
Y1: (T2). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int8, int16, int32, fp16, fp32  
<a id="ZExt"></a>
## ZExt  
Perform zero-extension on X1  

**Attributes:**  
data_type: Enum DataType, default to INVALID, The datatype to which the input data are cast  

**Operands:**  
X1: (T1). The input.  

**Results:**  
Y1: (T2). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int8, int16, int32, fp16, fp32  
<a id="common.reduction"></a>
# common.reduction  
---
<a id="Argmax"></a>
## Argmax  
Computes the indices of the largest value of the input along the specified axis.  

**Attributes:**  
axis: int32, default to 0, Axis along which to reduce.  
keep_dims: bool, default to true, Indicate whether to keep the reduced dimension.  

**Operands:**  
X1: (T1). The input.  
X2(OPTIONAL): (T2), scalar. The axis.  

**Results:**  
Y1: (T3). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
T3: int32  
<a id="Argmin"></a>
## Argmin  
Computes the indices of the smallest value of the input along the specified axis.  

**Attributes:**  
axis: int32, default to 0, Axis along which to reduce.  
keep_dims: bool, default to true, Indicate whether to keep the reduced dimension.  

**Operands:**  
X1: (T1). The input.  
X2(OPTIONAL): (T2), scalar. The axis.  

**Results:**  
Y1: (T3). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
T3: int32  
<a id="ReduceMax"></a>
## ReduceMax  
Compute the maximum of the elements across dimensions  

**Attributes:**  
axis: int32 list, Axis along which to reduce, omitted if X2 exists.  
keep_dims: bool, default to true, Indicate whether to keep the reduced dimension.  

**Operands:**  
X1: (T1). The input.  
X2(OPTIONAL): (T2), 1D. The axis.  

**Results:**  
Y1: (T2). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
<a id="ReduceMean"></a>
## ReduceMean  
Compute the average of the elements across dimensions.  

**Attributes:**  
axis: int32 list, Axis along which to reduce, omitted if X2 exists.  
keep_dims: bool, default to true, Indicate whether to keep the reduced dimension.  

**Operands:**  
X1: (T1). The input.  
X2(OPTIONAL): (T2), 1D. The axis.  

**Results:**  
Y1: (T2). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
<a id="ReduceMin"></a>
## ReduceMin  
Compute the minimum of the elements across dimensions  

**Attributes:**  
axis: int32 list, Axis along which to reduce, omitted if X2 exists.  
keep_dims: bool, default to true, Indicate whether to keep the reduced dimension.  

**Operands:**  
X1: (T1). The input.  
X2(OPTIONAL): (T2), 1D. The axis.  

**Results:**  
Y1: (T2). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
<a id="ReduceProduct"></a>
## ReduceProduct  
Compute the product of the elements across dimensions  

**Attributes:**  
axis: int32 list, Axis along which to reduce, omitted if X2 exists.  
keep_dims: bool, default to true, Indicate whether to keep the reduced dimension.  

**Operands:**  
X1: (T1). The input.  
X2(OPTIONAL): (T2), 1D. The axis.  

**Results:**  
Y1: (T2). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
<a id="ReduceSum"></a>
## ReduceSum  
Compute the sum of the elements across dimensions  

**Attributes:**  
axis: int32 list, Axis along which to reduce, omitted if X2 exists.  
keep_dims: bool, default to true, Indicate whether to keep the reduced dimension.  

**Operands:**  
X1: (T1). The input.  
X2(OPTIONAL): (T2), 1D. The axis.  

**Results:**  
Y1: (T2). The result  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
<a id="controlFlow"></a>
# controlFlow  
---
<a id="If"></a>
## If  
Generate if conditional.  

**Attributes:**  
else_branch: fp0, default to nullptr, Else branch.  
then_branch: fp0, default to nullptr, Then branch.  

**Operands:**  
X1: (T1), scalar. The condition for if.  

**Results:**  
Y1: (T2). The result.  

**Type Constraints:**  
T1: bool  
T2: int8, int16, int32, int64, fp16, fp32  
<a id="Jump"></a>
## Jump  
Generate unconditional jump.  

**Attributes:**  
target: fp0, default to nullptr, Jump target address.  

**Operands:**  
X1: (T1), scalar. The condition for jump, this value is always true in unconditional case.  

**Results:**  
Y1: (T2). The result.  

**Type Constraints:**  
T1: bool  
T2: int8, int16, int32, int64, fp16, fp32  
<a id="Loop"></a>
## Loop  
Generate a loop.  

**Attributes:**  
body: fp0, default to nullptr, The loop body run each iteration.  

**Operands:**  
X1: (T1), scalar. The maximum loop count.  
X2: (T2), scalar. The termination condition.  
X3: (T3), scalar. The initial values across each loop iterations.  

**Results:**  
Y1: (T4). The result.  

**Type Constraints:**  
T1: int64  
T2: bool  
T3: int8, int16, int32, int64, fp16, fp32  
T4: int8, int16, int32, int64, fp16, fp32  
<a id="creator"></a>
# creator  
---
<a id="RandomUniform"></a>
## RandomUniform  
Generate an NDim array with random numbers to form a uniform distribution.  

**Attributes:**  
dtype: Enum DataType, default to FLOAT32, The result data type  
minval: fp32, default to 0.0, The lower bound of the result value, inclusive.  
maxval: fp32, default to 1.0, The upper bound of the result value, exclusive.  
shape: int32 list, default to {}, The result shape.  
seed: int32, default to 0, The seed to the random generator.  

**Operands:**  
X1(OPTIONAL): (T1), 1D. The result shape.  

**Results:**  
Y1: (T2). The result.  

**Type Constraints:**  
T1: int32  
T2: int8, int16, int32, fp16, fp32  
<a id="Range"></a>
## Range  
Generate a sequence of numbers, starting from X1, up to X2, by increments of X3.  

**Operands:**  
X1: (T1), scalar. The first number of the result.  
X2: (T1), scalar. The exclusive limit of the result.  
X3: (T1), scalar. The incrementing value.  

**Results:**  
Y1: (T1), 1D. The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="image"></a>
# image  
---
<a id="math"></a>
# math  
---
<a id="BatchMatMul"></a>
## BatchMatMul  
Batched matrix product such that each slice of X1 and X2 perform MatMul and the result has the same batch size as input.  

**Attributes:**  
transpose_a: bool, default to false, whether X1 needs transpose.  
transpose_b: bool, default to false, whether X2 needs transpose.  

**Operands:**  
X1: (T1). Shape of (b1, ..., bm, M, K), or (b1, ..., bm, K, M) if transpose_a is true.  
X2: (T1). Shape of (b1, ..., bm, K, N), or (b1, ..., bm, N, K) if transpose_b is true.  

**Results:**  
Y1: (T1). The result of the matrix multiplication.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Gemm"></a>
## Gemm  
General matrix multiplication, result value is alpha * X1 * X2 + beta * X3, where X1= transpose(X1) if transA else X1, X2 = transpose(X2) if transB else X2.  

**Attributes:**  
alpha: fp32, default to 1.0, multiplier of X1 * X2  
beta: fp32, default to 1.0, multiplier of X3  
transpose_a: bool, default to false, whether X1 needs transpose  
transpose_b: bool, default to false, whether X2 needs transpose  

**Operands:**  
X1: (T1), 2D. 2D Array of shape (M, K), or (K, M) if transA is non-zero.  
X2: (T1), 2D. 2D Array of shape (K, N), or (N, K) if transB is non-zero.  
X3: (T1), 2D. 2D Array of shape (M, N).  

**Results:**  
Y1: (T1), 2D. The result of shape (M, N).  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="InnerProduct"></a>
## InnerProduct  
Calculate inner product of two inputs, optionally add bias. The result value is  of X1 * X2' + X3, where X2' = transpose(X2) if transpose else X2.  

**Attributes:**  
transpose: bool, default to false, whether X2 needs transpose.  
num_output: int32, default to 0, the size of the result.  
axis: int32, default to 0, the first axis to be lumped into a single inner product computation.  
flatten: int32, default to 0, whether input X1 needs to be flattened.  

**Operands:**  
X1: (T1). Filter operand.  
X2: (T1). Weight operand.  
X3(OPTIONAL): (T1). Bias operand.  

**Results:**  
Y1: (T1). The result value.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="MatMul"></a>
## MatMul  
2D Matrix product, the result is computed as X1' * X2' where X1' = transpose(X1) if trans_a, else X1' = X1; and X2' = transpose(X2) if trans_b else X2' = X2.  

**Attributes:**  
transpose_a: bool, default to false, whether X1 needs transpose.  
transpose_b: bool, default to false, whether X2 needs transpose.  

**Operands:**  
X1: (T1), 2D. Shape of (M, K), or (K, M) if transpose_a is true.  
X2: (T1), 2D. Shape of (K, N), or (N, K) if transpose_b is true.  

**Results:**  
Y1: (T1), 2D. The result of the matrix multiplication.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Transpose"></a>
## Transpose  
Transpose the input such that the dimension is  permutated based on permutation.  

**Attributes:**  
permutation: int32 list, permutation of the dimension indices.  

**Operands:**  
X1: (T1). The input.  
X2(OPTIONAL): (T2), 1D. The permutation.  

**Results:**  
Y1: (T2). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
<a id="math.binary"></a>
# math.binary  
---
<a id="Add"></a>
## Add  
Element-wise addition, return X1 + X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="And"></a>
## And  
Logical AND operation, element-wise, return X1 && X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T2). The right hand side operand  

**Results:**  
Y1: (T3). The result.  

**Type Constraints:**  
T1: bool  
T2: bool  
T3: bool  
<a id="Cmp"></a>
## Cmp  
Compare operation, element-wise, support broadcast.  

**Attributes:**  
predicator: Enum KindPredicate, default to EQ, predicator  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T2). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: bool  
<a id="Div"></a>
## Div  
Element-wise division, return X1 / X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Maximum"></a>
## Maximum  
Compute maximum value of the inputs, element-wise, return max(X1, X2), support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Minimum"></a>
## Minimum  
Compute minimum value of the inputs, element-wise, return min(X1, X2), support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Mul"></a>
## Mul  
Binary multiplication, element-wise, return X1 * X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Or"></a>
## Or  
Logical OR operation, element-wise, return X1 || X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T2). The right hand side operand  

**Results:**  
Y1: (T3). The result.  

**Type Constraints:**  
T1: bool  
T2: bool  
T3: bool  
<a id="Pow"></a>
## Pow  
Power operation, element-wise, return X1 ^ X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="ShiftL"></a>
## ShiftL  
Bitwise shift left operation, element-wise, return X1 << X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32  
<a id="ShiftR"></a>
## ShiftR  
Bitwise logical shift right operation, element-wise, appending zeros as MSB, return X1 >> X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32  
<a id="Sub"></a>
## Sub  
Binary subtraction operation, element-wise, return X1 - X2, support broadcast.  

**Operands:**  
X1: (T1). The left hand side operand.  
X2: (T1). The right hand side operand  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="math.unary"></a>
# math.unary  
---
<a id="Abs"></a>
## Abs  
The unary opertion returns abs(X1), element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Ceil"></a>
## Ceil  
The unary opertion returns ceil(X1), element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Erf"></a>
## Erf  
The unary operation returns the error function of X1, element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Exp"></a>
## Exp  
The unary opertion returns exp(X1), element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Floor"></a>
## Floor  
The unary opertion returns floor(X1), element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Neg"></a>
## Neg  
The unary opertion returns -X1, element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Rcp"></a>
## Rcp  
The unary opertion returns 1 / X1, element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Rsqrt"></a>
## Rsqrt  
The unary opertion returns 1 / (X1 ^ 0.5), element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Sign"></a>
## Sign  
The unary opertion returns 1 if X1 > 0, return -1 if X1 < 0, return 0 if X1 == 0, element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="Sqrt"></a>
## Sqrt  
The unary opertion returns X1 ^ 0.5, element-wise.  

**Operands:**  
X1: (T1). The unary input.  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="nn"></a>
# nn  
---
<a id="BatchNorm"></a>
## BatchNorm  
Batch normalization.  

**Attributes:**  
offset: fp32, Offset coefficient, omitted if X4 exists.  
scale: fp32, Scale coefficient, omitted if X5 exists.  
epsilon: fp32, default to 0.00001, Use to avoid division by zero.  
data_format: Enum DataFormat, default to NHWC, Input data format.  
pre_scaling_factor: fp32, default to 1.0, Pre-scale coefficient to mean and variance.  

**Operands:**  
X1: (T1), 4D. The input image, shape is defined by data_format  
X2: (T1), 4D. The mean NDArray  
X3: (T1), 4D. The variance NDArray  
X4(OPTIONAL): (T1), 4D. The offset NDArray  
X5(OPTIONAL): (T1), 4D. The scale NDArray  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="LRN"></a>
## LRN  
Local response normalization.  

**Attributes:**  
size: int32, the size of local region.  
alpha: fp32, default to 1, scaling parameter.  
beta: fp32, default to 0.5, exponent.  
bias: fp32, default to 1.0, bias.  
data_format: Enum DataFormat, default to NCHW, Input data format.  

**Operands:**  
X1: (T1), 4D. The input image, shape is defined by data_format  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="nn.activation"></a>
# nn.activation  
---
<a id="Elu"></a>
## Elu  
Compute elu activation, element-wise. Return exp(X1) - 1 if X1 < 0, return X1 else.  

**Operands:**  
X1: (T1). The unary input  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="LeakyRelu"></a>
## LeakyRelu  
Compute leaky relu activation, element-wise. Return max(X1, X1 * alpha).  

**Attributes:**  
alpha: fp32, default to 0.2, Slope at X1 < 0  

**Operands:**  
X1: (T1). The unary input  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Relu"></a>
## Relu  
Compute the relu activation, element-wise. Return max(0, X1)  

**Operands:**  
X1: (T1). The unary input  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Relu6"></a>
## Relu6  
Compute the relu6 activation, element-wise. Return min(max(0, X1), 6).  

**Operands:**  
X1: (T1). The unary input  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Sigmoid"></a>
## Sigmoid  
Compute the sigmoid activation, element-wise. Return 1 / (1 + exp(-X1)).  

**Operands:**  
X1: (T1). The unary input  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Softmax"></a>
## Softmax  
Compute the softmax activation.  

**Attributes:**  
axis: int32, default to -1, The dimension the softmax would be performed on.  

**Operands:**  
X1: (T1). The unary input  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Tanh"></a>
## Tanh  
The unary opertion returns hyperbolic tangent of X1, element-wise.  

**Operands:**  
X1: (T1). The unary input  

**Results:**  
Y1: (T1). The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="nn.cnn"></a>
# nn.cnn  
---
<a id="Conv2D"></a>
## Conv2D  
Perform 2D convolution on input X1 with filter X2.  

**Attributes:**  
padding: Enum Padding, default to VALID, The padding algoririthm.  
strides: int32 list, default to {1,1,1,1}, The sliding window for each dimension of input, its order is determined by data_format.  
data_format: Enum DataFormat, default to NHWC, The input data format.  
filter_format: Enum DataFormat, default to NHWC, The filter data format.  
dilations: int32 list, default to {1,1,1,1}, The dilation factor for each dimension of input, its order is determined by data_format.  
padding_left: int32, default to 0, The explicit padding to the left of the input.  
padding_right: int32, default to 0, The explicit padding to the right of the input.  
padding_top: int32, default to 0, The explicit padding to the top of the input.  
padding_bottom: int32, default to 0, The explicit padding to the bottom of the input.  
group: int32, default to 1, The group size for depthwise conv  

**Operands:**  
X1: (T1), 4D. The input image.  
X2: (T1), 4D. The filter.  

**Results:**  
Y1: (T1), 4D. The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="Conv2DTranspose"></a>
## Conv2DTranspose  
Convolve input X1 with filter X2 by taking opposite computation meaning with the filter and attributes.  

**Attributes:**  
padding: Enum Padding, default to VALID, The padding algoririthm.  
strides: int32 list, default to {1,1,1,1}, The sliding window for each dimension of input, its order is determined by data_format.  
data_format: Enum DataFormat, default to NHWC, The input data format.  
filter_format: Enum DataFormat, default to NHWC, The filter data format.  
dilations: int32 list, default to {1,1,1,1}, The dilation factor for each dimension of input, its order is determined by data_format.  
padding_left: int32, default to 0, The explicit padding to the left of the input.  
padding_right: int32, default to 0, The explicit padding to the right of the input.  
padding_top: int32, default to 0, The explicit padding to the top of the input.  
padding_bottom: int32, default to 0, The explicit padding to the bottom of the input.  
kernel_shape: int32 list, default to {}, The spatial kernel size along each dimension, its order is based on the filter_format.  
output_shape: int32 list, default to {}, The output shape  
group: int32, default to 1, The group size for output channels are divided into.  

**Operands:**  
X1: (T1), 4D. The input image.  
X2: (T1), 4D. The filter.  

**Results:**  
Y1: (T1), 4D. The result.  

**Type Constraints:**  
T1: fp16, fp32  
<a id="PoolingAvg"></a>
## PoolingAvg  
Compute the down sampling on input such that the average value of the subset is retrieved.  

**Attributes:**  
strides: int32 list, default to {1,1,1,1}, The stride values along each dimension of input,its order is based on data_format.  
ksize: int32 list, default to {1,1,1,1}, The spatial kernel size along each dimension, its order is based on the data_format.  
padding: Enum Padding, default to VALID, The padding algoririthm.  
padding_left: int32, default to 0, The explicit padding to the left of the input.  
padding_right: int32, default to 0, The explicit padding to the right of the input.  
padding_top: int32, default to 0, The explicit padding to the top of the input.  
padding_bottom: int32, default to 0, The explicit padding to the bottom of the input.  
data_format: Enum DataFormat, default to NHWC, The input data format.  
pad_value: fp32, default to 0.0, The constant padding value  

**Operands:**  
X1: (T1), 4D. The input image.  

**Results:**  
Y1: (T1), 4D. The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="PoolingMax"></a>
## PoolingMax  
Compute the down sampling on input such that the largest value of the subset is retrieved.  

**Attributes:**  
strides: int32 list, default to {1,1,1,1}, The stride values along each dimension of input,its order is based on data_format.  
ksize: int32 list, default to {1,1,1,1}, The spatial kernel size along each dimension, its order is based on the data_format.  
padding: Enum Padding, default to VALID, The padding algoririthm.  
padding_left: int32, default to 0, The explicit padding to the left of the input.  
padding_right: int32, default to 0, The explicit padding to the right of the input.  
padding_top: int32, default to 0, The explicit padding to the top of the input.  
padding_bottom: int32, default to 0, The explicit padding to the bottom of the input.  
data_format: Enum DataFormat, default to NHWC, The input data format.  
pad_value: fp32, default to 0, The constant padding value  

**Operands:**  
X1: (T1), 4D. The input image.  

**Results:**  
Y1: (T1), 4D. The result.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
<a id="nn.rnn"></a>
# nn.rnn  
---
<a id="objectDetection"></a>
# objectDetection  
---
<a id="NonMaxSuppression"></a>
## NonMaxSuppression  
Select a subset of bounding box in descending order of scores.  

**Attributes:**  
eta: fp32, default to 1.0, Coefficient to adjust num_threshold in a fast nms algorithm.  
score_threshold: fp32, default to -1.0, The min score for a box to be selected.  

**Operands:**  
X1: (T1), 2D. Input box, 2-D of [num_boxes, 4].  
X2: (T1), 1D. Input score, 1-D.  
X3: (T2), scalar. Maximum number of boxes in the result  
X4: (T3), scalar. Input iou_threshold, the intersection over union threshold used to suppress boxes  
X5: (T4), scalar. Input score_threshold, the intersection over union threshold used to suppress scores  

**Results:**  
Y1: (T5), 1D. The result of the selected indices from input box, 1-D.  

**Type Constraints:**  
T1: fp32  
T2: int32  
T3: fp32  
T4: fp32  
T5: int32  
<a id="Resize"></a>
## Resize  
Resize the input tensor.  

**Attributes:**  
mode: Enum ResizeMode, default to HALF_PIXEL, Coordinating mode such as HALF_PIXEL, TF_HALF_PIXEL, ALIGN_CORNETS, ASYMMETRIC  
interpolation_mode: Enum Interpolation, default to NEAREST, Interpolation mode such as NEAREST, LINEAR and CUBIC  
axes_mask: int32, default to 0, The mask for axes to be resized  

**Operands:**  
X1: (T1). Input data.  
X2: (T2). Resized shape  

**Results:**  
Y1: (T2). The result value.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int64  
<a id="TopK"></a>
## TopK  
Calculate values and indices of the largest k elements of input along the last dimension.  

**Attributes:**  
axis: int32, default to -1, Dimension on which to do the sort.  
largest: bool, default to true, Whether to return the top-K largest or smallest elements.  
sorted: bool, default to true, Whether the input is sorted.  

**Operands:**  
X1: (T1). Input data.  
X2: (T2), scalar. Input k.  

**Results:**  
Y1: (T2). The result value.  
Y2: (T3). The result indices.  

**Type Constraints:**  
T1: int8, int16, int32, fp16, fp32  
T2: int32  
T3: int32  
<a id="quantization"></a>
# quantization  
---
