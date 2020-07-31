//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef __CCE_BUILTIN_VARS_H
#define __CCE_BUILTIN_VARS_H

#ifdef __CCE_AICORE__

// Forward declares from vector_types.h.
struct ulong_2;

// The file implements built-in CCE variables using __declspec(property).
// https://msdn.microsoft.com/en-us/library/yhfk0thd.aspx
// All read accesses of built-in variable fields get converted into calls to a
// getter function which in turn calls the appropriate builtin to fetch the
// value.
//
// Example:
//    int x = threadIdx.x;
// IR output:
//  %0 = call i32 @llvm.hivm.read.ptc.sreg.tid.x() #3
// PTC output:
//  mov.u32     %r2, %tid.x;

#define __CCE_DEVICE_BUILTIN(FIELD, INTRINSIC)                                 \
  __declspec(                                                                  \
      property(get = __fetch_builtin_##FIELD)) unsigned long long FIELD;       \
  static inline __attribute__((always_inline))[aicore] unsigned long long      \
      __fetch_builtin_##FIELD(void) {                                          \
    return INTRINSIC;                                                          \
  }

#if __cplusplus >= 201103L
#define __DELETE = delete
#else
#define __DELETE
#endif

// Make sure nobody can create instances of the special varible types.  nvcc
// also disallows taking address of special variables, so we disable address-of
// operator as well.
#define __CCE_DISALLOW_BUILTINVAR_ACCESS(TypeName)                             \
  [aicore] TypeName() __DELETE;                                                \
  [aicore] TypeName(const TypeName &) __DELETE;                                \
  [aicore] void operator=(const TypeName &) const __DELETE;                    \
  [aicore] TypeName *operator&() const __DELETE

struct __cce_builtin_block_t {
  __CCE_DEVICE_BUILTIN(idx, get_block_idx());
  __CCE_DEVICE_BUILTIN(num, get_block_num());

private:
  __CCE_DISALLOW_BUILTINVAR_ACCESS(__cce_builtin_block_t);
};

#define __CCE_BUILTIN_VAR extern const[aicore] __attribute__((weak))
__CCE_BUILTIN_VAR __cce_builtin_block_t block;

#define block_idx (block.idx)
#define block_num (block.num)

#define l2_vaddr_base get_l2_vaddr_base()
#define l2_in_main get_l2_in_main()
#define status get_status()

struct __device_builtin__ ulong_2 {
  unsigned long long idx, num;
};

typedef __device_builtin__ struct ulong_2 ulong_2;


#undef __CCE_DEVICE_BUILTIN
#undef __CCE_BUILTIN_VAR
#undef __CCE_DISALLOW_BUILTINVAR_ACCESS

#endif /* __CCE_AICORE__ */

#endif /* __CCE_BUILTIN_VARS_H */
