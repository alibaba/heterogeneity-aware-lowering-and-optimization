#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "interface_calibrator.h"

#define LOGINFO(fmt, args...) fprintf(stdout, "[MMINFO]  " fmt "\n", ##args)

template <typename T>
void getDataFromFile(const std::string path, std::vector<std::string> files,
                     T* data_ptr, int datasize) {
  char* temp_data = (char*)data_ptr;
  for (auto file : files) {
    std::string temp_path = path + file;
    std::ifstream inFile(temp_path, std::ios::in | std::ios::binary);
    if (!inFile) {
      LOGINFO("Open file %s failed.", temp_path.c_str());
      continue;
    }
    inFile.read(temp_data, datasize);
    inFile.close();
    temp_data += datasize;
  }
}

class FixedCalibData : public magicmind::CalibDataInterface {
 public:
  FixedCalibData(const magicmind::Dims& shape,
                 const magicmind::DataType& data_type, int max_samples,
                 std::vector<std::string>& data_paths) {
    shape_ = shape;
    data_type_ = data_type;
    batch_size_ = shape.GetDimValue(0);
    max_samples_ = max_samples;
    data_paths_ = data_paths;
    current_sample_ = 0;
    buffer_.resize(shape.GetElementCount());
  }

  magicmind::Dims GetShape() const { return shape_; }
  magicmind::DataType GetDataType() const { return data_type_; }
  void* GetSample() { return buffer_.data(); }

  magicmind::Status Next() {
    if (current_sample_ + batch_size_ > max_samples_) {
      std::string msg = "sample number is bigger than max sample number!\n";
      magicmind::Status status_(magicmind::error::Code::INTERNAL, msg);
      return status_;
    }

    auto data_size = sizeof(float) * shape_.GetElementCount();
    std::vector<std::string> temp_paths(
        data_paths_.begin() + current_sample_,
        data_paths_.begin() + current_sample_ + batch_size_);
    getDataFromFile("", temp_paths, buffer_.data(), data_size);

    current_sample_ += batch_size_;
    return magicmind::Status::OK();
  }

  magicmind::Status Reset() {
    current_sample_ = 0;
    return magicmind::Status::OK();
  }

 private:
  magicmind::Dims shape_;
  magicmind::DataType data_type_;
  int batch_size_;
  int max_samples_;
  int current_sample_;
  std::vector<std::string> data_paths_;
  std::vector<float> buffer_;
};

#endif
