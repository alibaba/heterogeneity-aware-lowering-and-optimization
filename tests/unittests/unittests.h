//===- unittests.h --------------------------------------------------------===//
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

#ifndef TEST_UNITTESTS_UNITTESTS_H_
#define TEST_UNITTESTS_UNITTESTS_H_

#if LOAD_DATA
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#endif
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#if LOAD_DATA
#include "onnx.pb.h"
#endif
#include "stdlib.h"

using namespace std;
#if LOAD_DATA
enum FileExtension : int { FE_PB = 0, FE_TXT = 1, FE_END = 2 };
#endif
class UnitTests {
 public:
  UnitTests(){};
  virtual ~UnitTests(){};
  // io function
#if LOAD_DATA
  template <typename T>
  vector<T> LoadInData(string test_case_dir, int data_set_id, int input_id);
  template <typename T>
  vector<T> LoadOutData(string test_case_dir, int data_set_id, int output_id);
#endif
  // time function
  void TimeBegin();
  void TimeEnd();
  long long GetMicroSeconds();
  // verify function
  template <typename T>
  void CheckResult(const std::vector<size_t>& num_elems, void* out[],
                   const void* out_ref[], string test_case_dir,
                   string device_namne, long long times, double thre);

 private:
#if LOAD_DATA
  template <typename T>
  vector<T> LoadSingleData(string test_case_name);
  template <typename T>
  vector<T> ConvertPbToVec(string test_case_name);
  template <typename T>
  vector<T> ConvertTxtToVec(string test_case_name);
  FileExtension GetFileExtension(string test_case_name);
#endif
  chrono::time_point<chrono::high_resolution_clock> _start;
  chrono::time_point<chrono::high_resolution_clock> _end;

  ifstream inputfile;
};

#include "unittests.tpp"
#endif // TEST_UNITTESTS_UNITTESTS_H_
