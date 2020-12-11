//===- test_util.tpp --------------------------------------------------------===//
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

FileExtension UnitTests::GetFileExtension(string test_case_name) {
  FileExtension file_ext = FE_END;
  size_t name_size = test_case_name.size();
  if (0 == test_case_name.compare(name_size - 3, 3, "txt")) {
    file_ext = FE_TXT;
  } else if (0 == test_case_name.compare(name_size - 2, 2, "pb")) {
    file_ext = FE_PB;
  }
  return file_ext;
}

template <typename T>
vector<T> UnitTests::ConvertTxtToVec(string test_case_name) {
  vector<T> ret_data;
  T elem;
  while (inputfile >> elem) {
    ret_data.push_back(elem);
  }
  if (!inputfile.eof()) {
    cerr << "Warning:: error to read: " << test_case_name << endl;
    ret_data.clear();
    return ret_data;
  }
  return ret_data;
}

template<typename T>
vector<T> UnitTests::ConvertPbToVec(string test_case_name) {

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  vector<T> res_data;
  onnx::TensorProto tensor_def;
  if (!tensor_def.ParseFromIstream(&inputfile)) {
    cerr << "Encountered error(s) when parsing" << test_case_name;
    res_data.clear();
    return res_data;
  }
  size_t raw_data_size = tensor_def.raw_data().size();
  if (raw_data_size % sizeof(T) != 0) {
    cerr << "not support data type" << endl;
    res_data.clear();
    return res_data;
  }
  const T* ptr = reinterpret_cast<const T*>(tensor_def.raw_data().c_str());

  for (size_t i = 0; i < raw_data_size; ++i) {
    res_data.push_back(*ptr++);
  }
  return res_data;
}

template <typename T>
vector<T> UnitTests::LoadSingleData(string test_case_name) {

  vector<T> ret_data;
  inputfile.open (test_case_name, std::ifstream::in);

  if (inputfile.fail()) {
    cerr << "Warning:: fail to open: "
              << test_case_name << endl;
    ret_data.clear();
    return ret_data;
  }
  switch(GetFileExtension(test_case_name)) {
    case FE_PB:
      ret_data = ConvertPbToVec<T>(test_case_name);
      break;
    case FE_TXT:
      ret_data = ConvertTxtToVec<T>(test_case_name);
      break;
    default:
      cerr << "Warning:: only support .pb & .txt: "
                << endl;
      ret_data.clear();
  }

  inputfile.close();
  return ret_data;
}

template <typename T>
vector<T> UnitTests::LoadInData(string test_case_dir,
                                  int data_set_id,
                                  int input_id) {
  ostringstream oss;
  oss << test_case_dir << "/test_data_set_"
      <<  data_set_id << "/input_" << input_id << ".pb";
  return LoadSingleData<T>(oss.str());
}

template <typename T>
vector<T> UnitTests::LoadOutData(string test_case_dir,
                                   int data_set_id,
                                   int output_id) {
  ostringstream oss;
  oss << test_case_dir << "/test_data_set_"
      <<  data_set_id << "/output_" << output_id << ".pb";
  return LoadSingleData<T>(oss.str());
}

template <typename T>
void UnitTests::CheckResult(size_t num_out,
                              void* out[],
                              const void* out_ref[],
                              string test_case_dir,
                              string device_name,
                              long long times,
                              double thre) {
  string test_case_name = test_case_dir.substr(
                          test_case_dir.find_last_of("/") + 1);
  string report_file_name = "tmp/" + test_case_name
                           + "_" + device_name + ".txt";
  ostringstream oss;
  ofstream outfile(report_file_name);

  oss << "time: " << times;

  size_t i = 0;
  for (; i < num_out; ++i) {
    T* out_data = reinterpret_cast<T*>(out[i]);
    const T* out_ref_data = reinterpret_cast<const T*>(out_ref[i]);
    size_t elem_size = sizeof(out_data) / sizeof(T);
    for (size_t j = 0; j < elem_size; ++j) {
      bool nan_mismatch = (isnan(out_data[j]) ^ isnan(out_ref_data[j]));
      if (nan_mismatch || fabs(out_data[j] - out_ref_data[j]) > thre) {
        oss << " result: FAIL  [" << i <<", "<< j << "]: " << out_data[j]
            << " expects: " << out_ref_data[j] << "\n";
        outfile << oss.str();
        outfile.close();
        return;
      }
    }
  }

  oss << " result: PASS";
  outfile << oss.str();
  outfile.close();
}

void UnitTests::TimeBegin() {
  _start = chrono::high_resolution_clock::now();
}

void UnitTests::TimeEnd() {
  _end = chrono::high_resolution_clock::now();
}

long long UnitTests::GetMicroSeconds() {
  return chrono::duration_cast<chrono::microseconds>(_end - _start).count();
}
