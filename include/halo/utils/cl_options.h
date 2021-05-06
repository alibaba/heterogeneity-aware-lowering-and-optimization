//===- cl_options.h ---------------------------------------------*- C++ -*-===//
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
#ifndef HALO_UTILS_CL_OPTIONS_H_
#define HALO_UTILS_CL_OPTIONS_H_

#include "halo/halo.h"
#include "halo/lib/parser/parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace halo {

// Mark all our options with this category, everything else will be hidden.
static llvm::cl::OptionCategory HaloOptCat("Halo options");

static llvm::cl::list<std::string> ModelFiles(
    llvm::cl::Positional, llvm::cl::desc("model file name."),
    llvm::cl::OneOrMore, llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<ModelFormat> Format(
    "x",
    llvm::cl::values(
        clEnumValN(ModelFormat::CAFFE, "caffe", "CAFFE format"),
        clEnumValN(ModelFormat::ONNX, "onnx", "ONNX format"),
        clEnumValN(ModelFormat::TENSORFLOW, "tensorflow", "Tensorflow format"),
        clEnumValN(ModelFormat::TFLITE, "tflite", "TFLite format")),
    llvm::cl::desc("format of input model files. If unspecified, the format is "
                   "guessed base on file's extension."),
    llvm::cl::init(ModelFormat::INVALID), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<signed> Batch(
    "batch-size",
    llvm::cl::desc("Specify batch size if the first dim of input is negative"),
    llvm::cl::init(1), llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<std::string> EntryFunctionName(
    "entry-func-name", llvm::cl::desc("name of entry function"),
    llvm::cl::init(""), llvm::cl::cat(HaloOptCat));

static llvm::cl::list<std::string> InputsShape(
    "input-shape",
    llvm::cl::desc("Specify input names like -input-shape=foo:1x3x100x100 "
                   "-input-shape=bar:int8:-1x3x200x200"),
    llvm::cl::cat(HaloOptCat));

static llvm::cl::opt<std::string> PreprocessScale(
    "preproc-scale",
    llvm::cl::desc(
        "Insert scale operation to input. E.g. "
        "`-preproc-scale:1.0,-2.0,3.0,10,20,30`."
        "The number of values are comma separated and should be "
        "even. The first n values are for addition, the next n "
        "values are for multiplication. It should be broadcasting "
        "to input shape. Invalid if there are more than one inputs. "),
    llvm::cl::cat(HaloOptCat));

/// Guess the model format based on input file extension.gg
static ModelFormat InferFormat(const llvm::cl::list<std::string>& model_files,
                               size_t file_idx) {
  llvm::StringRef ext = llvm::sys::path::extension(model_files[file_idx]);
  auto format = llvm::StringSwitch<ModelFormat>(ext)
                    .Case(".pb", ModelFormat::TENSORFLOW)
                    .Case(".pbtxt", ModelFormat::TENSORFLOW)
                    .Case(".prototxt", ModelFormat::TENSORFLOW)
                    .Case(".onnx", ModelFormat::ONNX)
                    .Case(".json", ModelFormat::MXNET)
                    .Case(".tflite", ModelFormat::TFLITE)
                    .Default(ModelFormat::INVALID);
  // Check the next input file to see if it is caffe.
  if (format == ModelFormat::TENSORFLOW &&
      (file_idx + 1 < model_files.size()) &&
      llvm::sys::path::extension(model_files[file_idx + 1]) == ".caffemodel") {
    format = ModelFormat::CAFFE;
  }
  return format;
}

static Status ParseModels(const llvm::cl::list<std::string>& model_files,
                          const llvm::cl::opt<ModelFormat>& model_format,
                          const llvm::cl::opt<std::string>& entry_func_name,
                          const armory::Opts& opts, Module* module,
                          ModelFormat* f) {
  std::set<std::string> func_names;
  for (size_t i = 0, e = model_files.size(); i < e; ++i) {
    ModelFormat format = model_format;
    if (format == ModelFormat::INVALID) {
      format = InferFormat(model_files, i);
    }
    HLCHECK(format != ModelFormat::INVALID);
    *f = format;
    FunctionBuilder func_builder(module);
    // Use stem of the input model as function name.
    std::string func_name = entry_func_name.empty()
                                ? llvm::sys::path::stem(model_files[i]).str()
                                : entry_func_name.getValue();
    while (func_names.count(func_name) != 0) {
      func_name.append("_").append(std::to_string(i));
    }
    func_names.insert(func_name);
    Function* func = func_builder.CreateFunction(func_name);
    std::vector<std::string> files{model_files[i]};
    if (format == ModelFormat::CAFFE || format == ModelFormat::MXNET) {
      HLCHECK(i + 1 < e);
      files.push_back(model_files[++i]);
    }
    if (Status status = Parser::Parse(func, format, files, opts);
        status != Status::SUCCESS) {
      return status;
    }
  }
  return Status::SUCCESS;
}
} // namespace halo

#endif // HALO_UTILS_CL_OPTIONS_H_
