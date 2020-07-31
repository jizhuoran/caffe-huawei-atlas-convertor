//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

/*
 * WARNING: This header is intended to be directly -include'd by
 * the compiler and is not supposed to be included by users.
 */

#ifndef __CCE_RUNTIME_WRAPPER_H__
#define __CCE_RUNTIME_WRAPPER_H__

#if defined(__CCE__) && defined(__clang__)

#include <stddef.h>
#include <stdint.h>

#ifndef __CCE_ARCH__
#define __CCE_ARCH__ 100
#endif

#ifndef __CCE_AICPU_NO_FIRMWARE__
#include "__clang_cce_types.h"

#include "__clang_cce_aicore_builtin_vars.h"

#include "__clang_cce_runtime.h"

#include "__clang_cce_aicore_intrinsics.h"

#include "__clang_cce_aicpu_neon.h"

#elif defined(__CCE_NON_AICPU_CODES_NO_FIRMWARE__)
#include "__clang_cce_types.h"

#include "__clang_cce_aicore_builtin_vars.h"

#include "__clang_cce_runtime.h"

#include "__clang_cce_aicore_intrinsics.h"

// CCE AICPU CODES in no firmware mode
#else
#include "__clang_cce_types.h"

#include "__clang_cce_aicore_builtin_vars.h"

#include "__clang_cce_runtime.h"

#include "__clang_cce_aicpu_neon.h"

#endif // __CCE_AICPU_NO_FIRMWARE__

#endif // __CCE__
#endif // __CCE_RUNTIME_WRAPPER_H__
