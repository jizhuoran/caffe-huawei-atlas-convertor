//===- IntrinsicsHIVM.td - Defines HIVM intrinsics ---------*- tablegen -*-===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the HIVM-specific intrinsics for use with HiIPU.
//
//===----------------------------------------------------------------------===//

//---------------------------------------------------------------------------//
// CCE address spaces are defined at HiIPUAddrSpaceMap@lib/Basic/Targets.cpp
//---------------------------------------------------------------------------//
def llvm_ptr_ty1 : LLVMQualPointerType<llvm_ptr_ty, 1>; // __gm__
def llvm_ptr_ty2 : LLVMQualPointerType<llvm_ptr_ty, 2>; // __cbuf__
def llvm_ptr_ty3 : LLVMQualPointerType<llvm_ptr_ty, 3>; // __ca__
def llvm_ptr_ty4 : LLVMQualPointerType<llvm_ptr_ty, 4>; // __cb__
def llvm_ptr_ty5 : LLVMQualPointerType<llvm_ptr_ty, 5>; // __cc__
def llvm_ptr_ty6 : LLVMQualPointerType<llvm_ptr_ty, 6>; // __ubuf__

let TargetPrefix = "hivm" in {
  include "llvm/IR/cce_aicore_intrinsics_ir.include"
}
