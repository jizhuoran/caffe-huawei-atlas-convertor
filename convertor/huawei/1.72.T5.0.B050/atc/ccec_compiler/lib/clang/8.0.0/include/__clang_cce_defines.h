//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#if !defined(__CCE_DEFINES_H__)
#define __CCE_DEFINES_H__

#define __no_return__ __attribute__((noreturn))

#define __noinline__ __attribute__((noinline))

#define __forceinline__ __inline__ __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))

#define __global__ __attribute__((cce_kernel))

#define __gm__ __attribute__((cce_global))
#define __ca__ __attribute__((cce_cube_a))
#define __cb__ __attribute__((cce_cube_b))
#define __cc__ __attribute__((cce_cube_c))
#define __ubuf__ __attribute__((cce_unif_buff))
#define __cbuf__ __attribute__((cce_cube_buff))

/// CCE Read Only
/// This is to mark flowtable structs as read only to help guide alias analysis
#define __device_immutable__ __attribute__((device_immutable))

#define __device_builtin__

#endif /* !__CCE_DEFINES_H__ */
