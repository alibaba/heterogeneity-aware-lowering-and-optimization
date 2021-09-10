//===- passes_helper.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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
#ifndef HALO_UTILS_PASSES_HELPER_H_
#define HALO_UTILS_PASSES_HELPER_H_

#include <iostream>

#include "halo/lib/framework/common.h"
#include "halo/lib/pass/pass_manager.h"
#include "llvm/ADT/Triple.h"

namespace halo {

HL_UNUSED static void PopulateCodeGenPasses(
    PassManager* pm, std::ostringstream* out_code,
    std::ostringstream* out_constants, std::ostringstream* out_header,
    std::ostream* out_dynamic_check, const std::string& target,
    bool is_c_or_cxx_output, bool is_binary_output, bool emit_data_as_c,
    bool emit_code_only, bool emit_llvm_ir, bool emit_triton_config,
    const std::string& triton_config_file, Quantization quant_weights,
    const std::string& pgq_file, bool riscv_opt, const CXXCodeGenOpts& opts,
    const std::string& output_file_name) {
  auto constant_storage = ConstantDataStorage::DefinedAsStatic;
  if (opts.separate_constants) {
    constant_storage = ConstantDataStorage::DeclaredAsExternal;
  }

  if (is_c_or_cxx_output) {
    pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
    if (emit_data_as_c) {
      pm->AddGenericCXXConstantWriterPass(*out_constants);
    } else {
      pm->AddX86ConstantWriterPass(*out_constants);
    }

    if (opts.template_file == nullptr) {
      pm->AddGenericCXXCodeGenPass(*out_code, *out_header, *out_dynamic_check,
                                   opts);
    } else {
      pm->AddTemplatedCXXCodeGenPass(*out_code, *out_header, opts);
    }
    if (emit_triton_config) {
      pm->AddTritonConfigWriterPass(
          triton_config_file,
          opts.emit_dynamic_batch ? opts.max_batch_size : 0);
    }
    if (is_c_or_cxx_output && opts.format_code) {
      pm->AddCodeFormatterPass(*out_code, *out_header, opts);
    }
    if (opts.emit_obj || opts.emit_shared_lib) {
      pm->AddObjEmitPass(*out_code, *out_code, {}, opts);
    }
    if (opts.emit_shared_lib) {
      pm->AddLinkPass(*out_code, *out_constants, output_file_name, opts);
    }
    return;
  }

  if (emit_llvm_ir) {
    pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
    pm->AddGenericLLVMIRCodeGenPass(constant_storage);
    pm->AddGenericLLVMIRWriterPass(*out_code, is_binary_output);
    if (opts.separate_constants && !emit_code_only) {
      pm->AddGenericConstantWriterPass(*out_constants, is_binary_output);
    }
  } else {
    llvm::Triple triple(target);
    switch (triple.getArch()) {
      case llvm::Triple::ArchType::x86:
      case llvm::Triple::ArchType::x86_64: {
        pm->AddX86LLVMIRCodeGenPass(ConstantDataStorage::DeclaredAsExternal);
        pm->AddX86BinaryWriterPass(*out_code);
        if (opts.separate_constants && !emit_code_only) {
          pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
          pm->AddX86ConstantWriterPass(*out_constants);
        }
        break;
      }
      case llvm::Triple::ArchType::aarch64: {
        pm->AddARMLLVMIRCodeGenPass(ConstantDataStorage::DeclaredAsExternal);
        pm->AddARMBinaryWriterPass(*out_code);
        if (opts.separate_constants && !emit_code_only) {
          pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
          pm->AddARMConstantWriterPass(*out_constants);
        }
        break;
      }
      case llvm::Triple::ArchType::riscv32:
      case llvm::Triple::ArchType::riscv64: {
        if (riscv_opt) {
          pm->AddRISCVLLVMIRCodeGenPass(ConstantDataStorage::DeclaredAsExternal,
                                        "libRT_RISCV.a");
        } else {
          pm->AddRISCVLLVMIRCodeGenPass(
              ConstantDataStorage::DeclaredAsExternal);
        }
        pm->AddRISCVBinaryWriterPass(std::ref(*out_code));
        if (opts.separate_constants && !emit_code_only) {
          pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
          pm->AddRISCVConstantWriterPass(*out_constants);
        }

        break;
      }

      default: {
        HLCHECK(0 && "Unsupported");
      }
    }
  }
}

static void PopulateOptPasses(PassManager* pm, const std::string& target,
                              const std::vector<std::string>& input_shapes,
                              const std::vector<std::string>& inputs,
                              const std::vector<std::string>& outputs,
                              int batch, const std::string& preprocess_scale,
                              bool split_function, ModelFormat format,
                              const CXXCodeGenOpts& opts,
                              const FusionOptions& fusion_opts) {
  pm->AddInputLegalizerPass(batch, input_shapes, preprocess_scale);
  if (!outputs.empty()) {
    pm->AddOutputRewriterPass(outputs);
  }
  if (format == ModelFormat::CAFFE) {
    pm->AddCAFFEExtensionLegalizerPass();
  } else if (format == ModelFormat::TENSORFLOW) {
    pm->AddTFExtensionLegalizerPass();
  } else if (format == ModelFormat::TFLITE) {
    HLCHECK(format == ModelFormat::TFLITE);
    pm->AddTFLiteExtensionLegalizerPass();
  } else {
    HLCHECK(format == ModelFormat::ONNX);
    pm->AddONNXExtensionLegalizerPass();
  }
  pm->AddDCEPass();
  pm->AddTypeLegalizerPass(true);
  if (!inputs.empty()) {
    pm->AddInputRewriterPass(inputs);
  }
  pm->AddInstSimplifyPass(
      target.substr(0, 3) == "cxx", opts.disable_broadcasting,
      opts.remove_input_transpose, opts.remove_output_transpose,
      opts.disable_conv_bn, fusion_opts.ConvBias);
  if (opts.channel_order != ChannelOrder::None) {
    pm->AddReorderChannelPass(opts.channel_order == ChannelOrder::ChannelFirst);
  }
  pm->AddFusionPass(fusion_opts);
  if (split_function) {
    pm->AddSplittingPass();
    pm->AddDevicePlacementPass();
  }
  if (opts.enable_type_cast) {
    pm->AddTypeCastPass();
  }
  if (opts.constant_decombine) {
    pm->AddConstantDecombinePass();
  }
}
} // namespace halo

#endif
