//===- type_legalizer.h ---------------------------------------------------===//
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

#ifndef HALO_LIB_TRANSFORMS_TYPE_LEGALIZER_H_
#define HALO_LIB_TRANSFORMS_TYPE_LEGALIZER_H_

#include "halo/lib/ir/all_instructions.h"
#include "halo/lib/ir/math_instructions.h"
#include "halo/lib/pass/pass.h"

namespace halo {

/// This pass validates and infers types and dimensions of each operand.
class TypeLegalizer final : public BasicBlockPass {
 public:
  TypeLegalizer(bool relaxed)
      : BasicBlockPass("Type Legalizer"), relaxed_(relaxed) {}
  TypeLegalizer() : TypeLegalizer(false) {}

  bool RunOnBasicBlock(BasicBlock* bb) override;

 private:
  bool relaxed_; // Skip uninferable shape if true.
};

class ImageAxisInfo {
 public:
  const int batch_axis;
  const int data_channel_axis;
  const int data_depth_axis;
  const int data_height_axis;
  const int data_width_axis;
  const int data_spatial_axis;
  const int kernel_output_axis;
  const int kernel_input_axis;
  const int kernel_depth_axis;
  const int kernel_height_axis;
  const int kernel_width_axis;
  const int kernel_spatial_axis;
  ImageAxisInfo(int n, int c, int d, int h, int w, int s, int ko, int ki,
                int kd, int kh, int kw, int ks)
      : batch_axis(n),
        data_channel_axis(c),
        data_depth_axis(d),
        data_height_axis(h),
        data_width_axis(w),
        data_spatial_axis(s),
        kernel_output_axis(ko),
        kernel_input_axis(ki),
        kernel_depth_axis(kd),
        kernel_height_axis(kh),
        kernel_width_axis(kw),
        kernel_spatial_axis(ks) {}
  static const ImageAxisInfo& GetImageAxisInfo(DataFormat format,
                                               DataFormat filter_format);

 private:
  ImageAxisInfo(const ImageAxisInfo&) = delete;
  ImageAxisInfo& operator=(const ImageAxisInfo&) = delete;
};

} // end namespace halo.

#endif // HALO_LIB_TRANSFORMS_TYPE_LEGALIZER_H_