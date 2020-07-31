//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef fatbinaryctl_INCLUDED
#define fatbinaryctl_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#define AICORE_MAGIC_NUM 0x43554245 /*ASCII of "CUBE" */
// ASCII of "AARF", it is defined and used by runtime to
// check if it is vector core
#define AICORE_MAGIC_NUM_VEC 0x41415246
#define AICPU_MAGIC_NUM 0x41415243 /*ASCII of "AARC" */

typedef struct {
  int magic;
  int version;
  const unsigned long long *data;
  void *filename_or_fatbins; /* version 1: offline filename,
                              * version 2: array of prelinked fatbins */
} __fatBinC_Wrapper_t;

#ifdef __cplusplus
}
#endif

#endif /* fatbinaryctl_INCLUDED */
