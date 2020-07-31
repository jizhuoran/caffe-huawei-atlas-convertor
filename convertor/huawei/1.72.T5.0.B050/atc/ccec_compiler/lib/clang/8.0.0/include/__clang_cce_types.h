//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#if !defined(__CCE_TYPES_H__)
#define __CCE_TYPES_H__

#include "__clang_cce_defines.h"

struct __device_builtin__ dim3 {
  unsigned int x, y, z;
#if defined(__cplusplus)
  dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
#endif /* __cplusplus */
};

typedef __device_builtin__ struct dim3 dim3;

#define PIPE_LSU1 _Pragma("GCC warning \"'PIPE_LSU1' is deprecated\"") PIPE_MTE1
#define PIPE_LSU2 _Pragma("GCC warning \"'PIPE_LSU2' is deprecated\"") PIPE_MTE2
#define PIPE_LSU3 _Pragma("GCC warning \"'PIPE_LSU3' is deprecated\"") PIPE_MTE3

#if __CCE_AICORE__ == 100
#define PIPE_MTE4                                                              \
  _Pragma("GCC error \"'PIPE_MTE4' is only available in v200 targets\"")       \
      PIPE_MTE2
#define PIPE_MTE5                                                              \
  _Pragma("GCC error \"'PIPE_MTE5' is only available in v200 targets\"")       \
      PIPE_MTE3
#define PIPE_V2                                                                \
  _Pragma("GCC error \"'PIPE_V2' is only available in v200 targets\"") PIPE_V
#define EVENT_ID4                                                              \
  _Pragma("GCC error \"'EVENT_ID4' is only available in v200 targets\"")       \
      EVENT_ID0
#define EVENT_ID5                                                              \
  _Pragma("GCC error \"'EVENT_ID5' is only available in v200 targets\"")       \
      EVENT_ID1
#define EVENT_ID6                                                              \
  _Pragma("GCC error \"'EVENT_ID6' is only available in v200 targets\"")       \
      EVENT_ID2
#define EVENT_ID7                                                              \
  _Pragma("GCC error \"'EVENT_ID7' is only available in v200 targets\"")       \
      EVENT_ID3
#endif

#include "cce_aicore_intrinsics.h"

#endif /* !__CCE_TYPES_H__ */
