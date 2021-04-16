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
#include "halo/lib/parser/parser.h"
#include "halo/lib/pass/pass_manager.h"
#include "halo/lib/target/generic_llvmir/generic_llvmir_codegen.h"
#include "halo/lib/transforms/fusion.h"
#include "halo/lib/transforms/reorder_channel.h"
#include "llvm/ADT/Triple.h"

namespace halo {
__attribute__((unused)) static void PopulateCodeGenPasses(
    PassManager* pm, std::ostream* out_code, std::ostream* out_constants,
    std::ostream* out_header, std::ostream* out_dynamic_check,
    const std::string& target, bool is_c_or_cxx_output, bool is_binary_output,
    bool emit_data_as_c, bool emit_code_only, bool emit_llvm_ir,
    bool emit_triton_config, const std::string& triton_config_file,
    CodeGen::Quantization quant_weights, const std::string& pgq_file,
    bool riscv_opt, const Opts& opts) {
  auto constant_storage =
      GenericLLVMIRCodeGen::ConstantDataStorage::DefinedAsStatic;
  if (opts.separate_constants) {
    constant_storage =
        GenericLLVMIRCodeGen::ConstantDataStorage::DeclaredAsExternal;
  }

  if (is_c_or_cxx_output) {
    pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
    pm->AddGenericCXXCodeGenPass(std::ref(*out_code), std::ref(*out_header),
                                 std::ref(*out_dynamic_check), opts);
    if (emit_data_as_c) {
      pm->AddGenericCXXConstantWriterPass(std::ref(*out_constants));
    } else {
      pm->AddX86ConstantWriterPass(std::ref(*out_constants));
    }
    if (emit_triton_config) {
      pm->AddTritonConfigWriterPass(
          triton_config_file,
          opts.emit_dynamic_batch ? opts.max_batch_size : 0);
    }
    return;
  }

  if (emit_llvm_ir) {
    pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
    pm->AddGenericLLVMIRCodeGenPass(constant_storage);
    pm->AddGenericLLVMIRWriterPass(std::ref(*out_code), is_binary_output);
    if (opts.separate_constants && !emit_code_only) {
      pm->AddGenericConstantWriterPass(std::ref(*out_constants),
                                       is_binary_output);
    }
  } else {
    llvm::Triple triple(target);
    switch (triple.getArch()) {
      case llvm::Triple::ArchType::x86:
      case llvm::Triple::ArchType::x86_64: {
        pm->AddX86LLVMIRCodeGenPass(
            GenericLLVMIRCodeGen::ConstantDataStorage::DeclaredAsExternal);
        pm->AddX86BinaryWriterPass(std::ref(*out_code));
        if (opts.separate_constants && !emit_code_only) {
          pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
          pm->AddX86ConstantWriterPass(std::ref(*out_constants));
        }
        break;
      }
      case llvm::Triple::ArchType::aarch64: {
        pm->AddARMLLVMIRCodeGenPass(
            GenericLLVMIRCodeGen::ConstantDataStorage::DeclaredAsExternal);
        pm->AddARMBinaryWriterPass(std::ref(*out_code));
        if (opts.separate_constants && !emit_code_only) {
          pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
          pm->AddARMConstantWriterPass(std::ref(*out_constants));
        }
        break;
      }
      case llvm::Triple::ArchType::riscv32:
      case llvm::Triple::ArchType::riscv64: {
        if (riscv_opt) {
          pm->AddRISCVLLVMIRCodeGenPass(
              GenericLLVMIRCodeGen::ConstantDataStorage::DeclaredAsExternal,
              "libRT_RISCV.a");
        } else {
          pm->AddRISCVLLVMIRCodeGenPass(
              GenericLLVMIRCodeGen::ConstantDataStorage::DeclaredAsExternal);
        }
        pm->AddRISCVBinaryWriterPass(std::ref(*out_code));
        if (opts.separate_constants && !emit_code_only) {
          pm->AddWeightsQuantizerPass(quant_weights, pgq_file);
          pm->AddRISCVConstantWriterPass(std::ref(*out_constants));
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
                              ReorderChannel::ChannelOrder channel_order,
                              bool split_function, bool disable_type_cast,
                              Parser::Format format, const Opts& opts,
                              const Fusion::Options& fusion_opts) {
  pm->AddInputLegalizerPass(batch, input_shapes, preprocess_scale);
  if (!outputs.empty()) {
    pm->AddOutputRewriterPass(outputs);
  }
  if (format == Parser::Format::CAFFE) {
    pm->AddCAFFEExtensionLegalizerPass();
  } else if (format == Parser::Format::TENSORFLOW) {
    pm->AddTFExtensionLegalizerPass();
  } else if (format == Parser::Format::TFLITE) {
    HLCHECK(format == Parser::Format::TFLITE);
    pm->AddTFLiteExtensionLegalizerPass();
  } else {
    HLCHECK(format == Parser::Format::ONNX);
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
  if (channel_order != ReorderChannel::ChannelOrder::None) {
    pm->AddReorderChannelPass(channel_order ==
                              ReorderChannel::ChannelOrder::ChannelFirst);
  }
  pm->AddFusionPass(fusion_opts);
  if (split_function) {
    pm->AddSplittingPass();
    pm->AddDevicePlacementPass();
  }
  if (!disable_type_cast) {
    pm->AddTypeCastPass();
  }
}
} // namespace halo

#endif