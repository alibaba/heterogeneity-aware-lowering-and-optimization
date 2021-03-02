//===- version.h ------------------------------------------------*- C++ -*-===//
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

#ifndef HALO_VERSION_H_
#define HALO_VERSION_H_

#define HALO_MAJOR 0 // !< HALO major version.
#define HALO_MINOR 6 // !< HALO minor version.
#define HALO_PATCH 0 // !< HALO patch version.

//! \brief HALO version number.
#define HALO_VERSION_NUMBER ((HALO_MAJOR)*100 + (HALO_MINOR)*10 + (OLDA_PATCH))

#include "halo/version.inc"

#endif // HALO_VERSION_H_
