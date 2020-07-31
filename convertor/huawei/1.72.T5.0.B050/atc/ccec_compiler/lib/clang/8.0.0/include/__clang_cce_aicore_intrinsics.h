//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef __CCE_INTRINSICS_H__
#define __CCE_INTRINSICS_H__

#define CCE_INTRINSIC static __attribute__((always_inline))

#define PIPE_ID(v) __attribute__((__pipe_type__(v)))


#ifdef __CCE_AICORE__

#define CCE_TRUE 1
#define CCE_FALSE 0


#if TODO

CREATE_CA_MATRIX
CREATE_CB_MATRIX
#endif

CCE_SET_FLAG
CCE_WAIT_FLAG
CCE_PIPE_BARRIER

ABS_LONGLONG
ABS_LONG
ABS_INT
ABS_SHORT
ABS_CHAR
ABS_FLOAT
SQRT_LONGLONG
SQRT_LONG
SQRT_INT
SQRT_SHORT
SQRT_CHAR
SQRT_FLOAT
DUMMY_PIPE_S
DUMMY_PIPE_M
DUMMY_PIPE_V
DUMMY_PIPE_MTE1
DUMMY_PIPE_MTE2
DUMMY_PIPE_MTE3
MAX_LONGLONG
MAX_LONG
MAX_INT
MAX_SHORT
MAX_CHAR
MAX_FLOAT
MIN_LONGLONG
MIN_LONG
MIN_INT
MIN_SHORT
MIN_CHAR
MIN_FLOAT

SET_VA_REG_SB

SET_VA_REG_UB

CCE_INTRINSIC_VMS4_HALF
CCE_INTRINSIC_VMS4_HALF_UINT64
CCE_INTRINSIC_VMS4_HALF_ARRAY

CCE_INTRINSIC_VMS4_FLOAT
CCE_INTRINSIC_VMS4_FLOAT_UINT64
CCE_INTRINSIC_VMS4_FLOAT_ARRAY

// Vector Merge Sorter
#define UVEC_RP_O 0    // repeat offset
#define UVEC_LIST0_O 8 // number of region proposals in input list 0 offset
#define UVEC_LIST0_M 0xfffULL // number of region proposals in input list 0 mask
#define UVEC_LIST1_O 20 // number of region proposals in input list 1 offset
#define UVEC_LIST1_M 0xfffULL // number of region proposals in input list 1 mask
#define UVEC_LIST2_O 32 // number of region proposals in input list 2 offset
#define UVEC_LIST2_M 0xfffULL // number of region proposals in input list 2 mask
#define UVEC_LIST3_O 44 // number of region proposals in input list 3 offset
#define UVEC_LIST3_M 0xfffULL // number of region proposals in input list 3 mask
#define UVEC_EES_O 59         // enable input list exhausted suspension offset
#define UVEC_EES_M 0x1ULL     // enable input list exhausted suspension mask
#define UVEC_MASK_O 60        // 4-bit mask signal offset
#define UVEC_MASK_M 0xfULL    // 4-bit mask signal offset

SET_VEC_MRGSORT_CONFIG

CCE_INTRINSIC_VMRGSORT4_HALF_UINT64_CFG
CCE_INTRINSIC_VMRGSORT4_HALF_ARRAY_CFG

CCE_INTRINSIC_VMRGSORT4_FLOAT_UINT64_CFG
CCE_INTRINSIC_VMRGSORT4_FLOAT_ARRAY_CFG

#endif
#endif // defined(__CLANG_CCE_INTRINSICS_H__)
