//===- common.h -------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_LIB_FRAMEWORK_COMMON_H_
#define HALO_LIB_FRAMEWORK_COMMON_H_

//! \brief API export directives
#if defined _WIN32 || defined __CYGWIN__
#define HL_API_EXPORT __declspec(dllexport)
#else
#define HL_API_EXPORT __attribute__((visibility("default")))
#endif

#if defined __has_attribute
#if __has_attribute(unused)
#define HL_UNUSED __attribute__((unused))
#endif
#endif

/// log with verbose less than GOOGLE_STRIP_LOG will not be displayed
// #define GOOGLE_STRIP_LOG 3
#include "glog/logging.h"
#include "glog/raw_logging.h"

#ifdef NDEBUG
#define HLCHECK(x) \
  if (!(x)) {      \
    exit(1);       \
  }
#else
#include <cassert>
#define HLCHECK(x) CHECK(x) // NOLINT
#endif

namespace halo {
#include "halo/lib/ir/attribute_enums.h.inc"

/// set glog global config
class GLogHelper {
 public:
  explicit GLogHelper(const char* cfg) {
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(cfg);
  }
  ~GLogHelper() { google::ShutdownGoogleLogging(); }

  GLogHelper(const GLogHelper&) = delete;
  GLogHelper& operator=(const GLogHelper&) = delete;
};

template <typename T_NEW, typename T_OLD>
T_NEW* Downcast(T_OLD* ptr) {
  return static_cast<T_NEW*>(ptr);
}

template <typename T_NEW, typename T_OLD>
const T_NEW* Downcast(const T_OLD* ptr) {
  return static_cast<const T_NEW*>(ptr);
}

template <typename T_TO, typename T_FROM>
bool IsA(const T_FROM* obj) {
  return T_TO::Classof(obj);
}

template <typename T_TO, typename T_FROM>
T_TO* DynCast(T_FROM* ptr) {
  if (!IsA<T_TO>(ptr)) {
    return nullptr;
  }
  return Downcast<T_TO>(ptr);
}

template <typename T_TO, typename T_FROM>
const T_TO* DynCast(const T_FROM* ptr) {
  return DynCast<T_TO>(const_cast<T_FROM*>(ptr));
}

} // namespace halo

#endif // HALO_LIB_FRAMEWORK_COMMON_H_
