# How to run Unit Tests

## CI test:
we can configure the options in `config.csv` to test:

`test_case_name` : the same name as the file name in the `data/` directory without the prefix `test_`

`device_name` : ODLA supported back-end device name

`error_threshold` : upper limit of error between test result and golden

`enable_timeperf` : whether to perform time performance test

`support` : whether the device support the test case

`result` : whether the test passed


e.g the case `abs` runs on `tensorrt` backend device, and the result is verified with error
    threshold of `0.0001`, and the time consuming test is turned off, and the verification is passed
| test_case_name | device_name | error_threshold | enable_timeperf | support | result |
| -------------- | ----------- | --------------- | --------------- | ------- | ------ |
| abs            | tensorrt    | 0.0001          | FALSE           | yes     | PASS   |

[todo] support IPU unit tests

## single case test:

if CI result is failed, we can use single case test to debug

e.g the CI log as follows:
```
----------Test: [test_flatten_axis3:tensorrt]------
----------Test: [test_flatten_default_axis:tensorrt]------
----------Test: [test_flatten_negative_axis1:tensorrt]------
----------Test: [test_flatten_negative_axis2:tensorrt]------
----------Test: [test_flatten_negative_axis3:tensorrt]------
----------Test: [test_flatten_negative_axis4:tensorrt]------
----------Test: [test_floor:tensorrt]------
----------Test: [test_floor_example:tensorrt]------
----------Test: [test_gather_0:tensorrt]------
----------Test: [test_gather_1:tensorrt]------
----------Test: [test_gather_2d_indices:tensorrt]------
----------Test: [test_gemm_default_matrix_bias:dnnl]------

--
Command Output (stderr):
--
Cloning into 'onnx'...
Segmentation fault (core dumped)

--

********************
Testing Time: 442.35s
********************
Failing Tests (1):
    Halo Unit Tests :: run_list.sh
```
it means that the case `gemm_default_matrix_bias` run fail on the device `dnnl`.

we can use script `run_single.sh` to reproduce and debug.

./run_single.sh gemm_default_matrix_bias 0.0001 dnnl "your build directory"