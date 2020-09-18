//===- halo_tblgen.h --------------------------------------------*- C++ -*-===//
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

#ifndef HALO_UTIL_TABLEGEN_HALO_TBLGEN_H_
#define HALO_UTIL_TABLEGEN_HALO_TBLGEN_H_

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

namespace halo {

extern void EmitAttrDecl(const llvm::RecordKeeper& records,
                         llvm::raw_ostream& os);
extern void EmitAttrDef(const llvm::RecordKeeper& records,
                        llvm::raw_ostream& os);
extern void EmitAttrEnum(const llvm::RecordKeeper& records,
                         llvm::raw_ostream& os);
extern void EmitConvertInfo(const llvm::RecordKeeper& records,
                            llvm::raw_ostream& os);
extern void EmitDataTypeEnum(const llvm::RecordKeeper& records,
                             llvm::raw_ostream& os);
extern void EmitDoc(const llvm::RecordKeeper& records, llvm::raw_ostream& os);
extern void EmitFusion(const llvm::RecordKeeper& records,
                       llvm::raw_ostream& os);
extern void EmitInstClass(const llvm::RecordKeeper& records,
                          llvm::raw_ostream& os);
extern void EmitInstInfo(const llvm::RecordKeeper& records,
                         llvm::raw_ostream& os);
extern void EmitSourceFileHeader(const std::string& filename,
                                 llvm::raw_ostream& os);
extern void EmitIRBuilder(const llvm::RecordKeeper& records,
                          llvm::raw_ostream& os, bool decl);
extern void EmitConverterDecl(const llvm::RecordKeeper& records,
                              llvm::raw_ostream&);
extern void EmitConverterDef(const llvm::RecordKeeper& records,
                             llvm::raw_ostream&);
extern void EmitRegisterOp(const llvm::RecordKeeper& records,
                           llvm::raw_ostream&);

} // namespace halo

#endif // HALO_UTIL_TABLEGEN_HALO_TBLGEN_H_