"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_vector_api.py
DESC:     provide vector instructions
CREATED:  2020-3-16 21:12:13
MODIFIED: 2020-3-16 21:12:45
"""
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import scope_ubuf
from te.platform.cce_conf import api_check_support
from .tik_tensor import Tensor
from .tik_scalar import mask_concat
from .. import debug
from ..tik_lib.tik_vector_api_ import TikVectorApi
from ..tik_lib.tik_source_info import source_info_decorator
from ..tik_lib.tik_check_util import TikCheckUtil
from ..tik_lib.tik_api_util import check_high_preci_param
from ..common.util import get_bit_len
from ..common.util import is_immediate_number
from ..common.tik_get_soc_name import get_soc_name
from ..common.common_util import DTYPE_SIZE
from ..tik_lib.tik_params import INSTR_DTYPE_SUPPORT_STATEMENT
from ..tik_lib.tik_params import ONE_BLK_SIZE
from ..common.common_util import check_over_high_preci
from ..common.common_util import check_mask_valid
from ..tik_lib.tik_expr import Expr

# round disable
_ROUND_TO_NEAREST_ENABLE = 0


class TikVectorApiv1(TikVectorApi):
    """
    Vector Api for open
    """
    # @cond
    def __init__(self):
        super(TikVectorApiv1, self).__init__()
    # @endcond

    def vec_add(self,  # pylint: disable=R0913
                mask,
                dst,
                src0,
                src1,
                repeat_times,
                dst_rep_stride,
                src0_rep_stride,
                src1_rep_stride):
        r'''
        Performs addition element-wise
        Description:
          Performs addition element-wise:
          \f$dst_i = src0_i+src1_i,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a
            Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst and src,
            \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src0 : Source operand 0, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          src1 : Source operand 1, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 0
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 1
          NOTICE :
            dst, src0, and src1 have the same data type:
            - Ascend 310 AI Processor: tensors of type float16/float32/int32
            - Ascend 910 AI Processor: tensors of type float16/float32/int32
            - HiSilicon SoC (ES): tensors of type float16/int32
            - Ascend 610 AI Processor (AI Core): tensors of type float16/int16
            /float32/int32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /int16/float32/int32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride, src0_rep_stride, or src1_rep_stride is within the
          range [0, 255], in the unit of 32 bytes.
          The argument is a scalar of type int16/int32/int64/uint16/uint32
          /uint64, an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the destination
            operand, that is, the destination operand of the Nth iteration is
            the source operand of the (N+1)th iteration,
            address overlapping is not supported. Address overlapping is not
            supported in the following cases: For the
            vec_add/vec_sub/vec_mul/vec_max/v_min/vec_and/vec_or instruction,
            (1) the data type is float16, int32, or
            float32, and the destination operand overlaps the second source
            operand; (2) src1_rep_stride = dst_rep_stride = 0;
            (3) The addresses of src0 and src1 cannot overlap.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_add(128, dst_ub, src0_ub, src1_ub, 1, 8, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_add"
        return self._vector_binary_tenary_elewise_func(
            'vadd', mask, dst, src0, src1, repeat_times, default_blk_stride,
            default_blk_stride, default_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride, stride_unit,
            print_name=print_name)

    def vec_sub(self,  # pylint: disable=R0913
                mask,
                dst,
                src0,
                src1,
                repeat_times,
                dst_rep_stride,
                src0_rep_stride,
                src1_rep_stride):
        r'''
        Performs subtraction element-wise
        Description:
          Performs subtraction element-wise:
          \f$dst_i = src0_i-src1_i,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a
            Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst and src,
            \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src0 : Source operand 0, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          src1 : Source operand 1, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 0
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 1
          NOTICE :
            dst, src0, and src1 have the same data type:
            - Ascend 310 AI Processor: tensors of type float16/float32/int32
            - Ascend 910 AI Processor: tensors of type float16/float32/int32
            - HiSilicon SoC (ES): tensors of type float16/int32
            - Ascend 610 AI Processor (AI Core): tensors of type float16/int16
            /float32/int32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /int16/float32/int32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride, src0_rep_stride, or src1_rep_stride is within the
          range [0, 255], in the unit of 32 bytes.
          The argument is a scalar of type int16/int32/int64/uint16/uint32
          /uint64, an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the destination
            operand, that is, the destination operand of the Nth iteration is
            the source operand of the (N+1)th iteration,
            address overlapping is not supported. Address overlapping is not
            supported in the following cases: For the
            vec_add/vec_sub/vec_mul/vec_max/v_min/vec_and/vec_or instruction,
            (1) the data type is float16, int32, or
            float32, and the destination operand overlaps the second source
            operand; (2) src1_rep_stride = dst_rep_stride = 0;
            (3) The addresses of src0 and src1 cannot overlap.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_sub(128, dst_ub, src0_ub, src1_ub, 1, 8, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_sub"
        return self._vector_binary_tenary_elewise_func(
            'vsub', mask, dst, src0, src1, repeat_times, default_blk_stride,
            default_blk_stride, default_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride, stride_unit,
            print_name=print_name)

    def vec_max(self,  # pylint: disable=R0913
                mask,
                dst,
                src0,
                src1,
                repeat_times,
                dst_rep_stride,
                src0_rep_stride,
                src1_rep_stride):
        r'''
        Computes the maximum element-wise
        Description:
          Computes the maximum element-wise:
          \f$dst_i = \max (src0_i,src1_i),i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a
            Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst and src,
            \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src0 : Source operand 0, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          src1 : Source operand 1, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 0
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 1
          NOTICE :
            dst, src0, and src1 have the same data type:
            - Ascend 310 AI Processor: tensors of type float16/float32/int32
            - Ascend 910 AI Processor: tensors of type float16/float32/int32
            - HiSilicon SoC (ES): tensors of type float16/int32
            - Ascend 610 AI Processor (AI Core): tensors of type float16/int16
            /float32/int32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /int16/float32/int32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride, src0_rep_stride, or src1_rep_stride is within the
          range [0, 255], in the unit of 32 bytes.
          The argument is a scalar of type int16/int32/int64/uint16/uint32
          /uint64, an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the destination
            operand, that is, the destination operand of the Nth iteration is
            the source operand of the (N+1)th iteration,
            address overlapping is not supported. Address overlapping is not
            supported in the following cases: For the
            vec_add/vec_sub/vec_mul/vec_max/v_min/vec_and/vec_or instruction,
            (1) the data type is float16, int32, or
            float32, and the destination operand overlaps the second source
            operand; (2) src1_rep_stride = dst_rep_stride = 0;
            (3) The addresses of src0 and src1 cannot overlap.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                        scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_max(128, dst_ub, src0_ub, src1_ub, 1, 8, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_max"
        return self._vector_binary_tenary_elewise_func(
            'vmax', mask, dst, src0, src1, repeat_times, default_blk_stride,
            default_blk_stride, default_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride, stride_unit,
            print_name=print_name)

    def vec_min(self,  # pylint: disable=R0913
                mask,
                dst,
                src0,
                src1,
                repeat_times,
                dst_rep_stride,
                src0_rep_stride,
                src1_rep_stride):
        r'''
        Computes the minimum element-wise
        Description:
          Computes the minimum element-wise:
          \f$dst_i = \min (src0_i,src1_i),i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a
            Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst and src,
            \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src0 : Source operand 0, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          src1 : Source operand 1, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 0
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 1
          NOTICE :
            dst, src0, and src1 have the same data type:
            - Ascend 310 AI Processor: tensors of type float16/float32/int32
            - Ascend 910 AI Processor: tensors of type float16/float32/int32
            - HiSilicon SoC (ES): tensors of type float16/int32
            - Ascend 610 AI Processor (AI Core): tensors of type float16/int16
            /float32/int32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /int16/float32/int32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride, src0_rep_stride, or src1_rep_stride is within the
          range [0, 255], in the unit of 32 bytes.
          The argument is a scalar of type int16/int32/int64/uint16/uint32
          /uint64, an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the destination
            operand, that is, the destination operand of the Nth iteration is
            the source operand of the (N+1)th iteration,
            address overlapping is not supported. Address overlapping is not
            supported in the following cases: For the
            vec_add/vec_sub/vec_mul/vec_max/v_min/vec_and/vec_or instruction,
            (1) the data type is float16, int32, or
            float32, and the destination operand overlaps the second source
            operand; (2) src1_rep_stride = dst_rep_stride = 0;
            (3) The addresses of src0 and src1 cannot overlap.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_min(128, dst_ub, src0_ub, src1_ub, 1, 8, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_min"
        return self._vector_binary_tenary_elewise_func(
            'vmin', mask, dst, src0, src1, repeat_times, default_blk_stride,
            default_blk_stride, default_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride, stride_unit,
            print_name=print_name)

    def vec_mul(self,  # pylint: disable=R0913
                mask,
                dst,
                src0,
                src1,
                repeat_times,
                dst_rep_stride,
                src0_rep_stride,
                src1_rep_stride):
        r'''
        Performs multiplication element-wise
        Description:
          Performs multiplication element-wise:
          \f$dst_i = src0_i*src1_i,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a
            Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst and src,
            \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src0 : Source operand 0, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          src1 : Source operand 1, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 0
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 1
          NOTICE :
            dst, src0, and src1 have the same data type:
            - Ascend 310 AI Processor: tensors of type float16/float32/int32
            - Ascend 910 AI Processor: tensors of type float16/float32/int32
            - HiSilicon SoC (ES): tensors of type float16/int32
            - Ascend 610 AI Processor (AI Core): tensors of type float16/int16
            /float32/int32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /int16/float32/int32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride, src0_rep_stride, or src1_rep_stride is within the
          range [0, 255], in the unit of 32 bytes.
          The argument is a scalar of type int16/int32/int64/uint16/uint32
          /uint64, an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the destination
            operand, that is, the destination operand of the Nth iteration is
            the source operand of the (N+1)th iteration,
            address overlapping is not supported. Address overlapping is not
            supported in the following cases: For the
            vec_add/vec_sub/vec_mul/vec_max/v_min/vec_and/vec_or instruction,
            (1) the data type is float16, int32, or
            float32, and the destination operand overlaps the second source
            operand; (2) src1_rep_stride = dst_rep_stride = 0;
            (3) The addresses of src0 and src1 cannot overlap.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_mul(128, dst_ub, src0_ub, src1_ub, 1, 8, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_mul"
        return self._vector_binary_tenary_elewise_func(
            'vmul', mask, dst, src0, src1, repeat_times, default_blk_stride,
            default_blk_stride, default_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride, stride_unit,
            print_name=print_name)

    def vec_and(self,  # pylint: disable=R0913
                mask,
                dst,
                src0,
                src1,
                repeat_times,
                dst_rep_stride,
                src0_rep_stride,
                src1_rep_stride):
        r'''
        Performs addition element-wise
        Description:
          Performs addition element-wise:
          \f$dst_i = src0_i\&src1_i,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a
            Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst and src,
            \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src0 : Source operand 0, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          src1 : Source operand 1, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 0
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 1
          NOTICE :
            dst, src0, and src1 have the same data type:
            - Tensors of type uint16 or int16

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride, src0_rep_stride, or src1_rep_stride is within the
          range [0, 255], in the unit of 32 bytes.
          The argument is a scalar of type int16/int32/int64/uint16/uint32
          /uint64, an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the destination
            operand, that is, the destination operand of the Nth iteration is
            the source operand of the (N+1)th iteration,
            address overlapping is not supported. Address overlapping is not
            supported in the following cases: For the
            vec_add/vec_sub/vec_mul/vec_max/v_min/vec_and/vec_or instruction,
            (1) the data type is float16, int32, or
            float32, and the destination operand overlaps the second source
            operand; (2) src1_rep_stride = dst_rep_stride = 0;
            (3) The addresses of src0 and src1 cannot overlap.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("uint16", (128,), name="src0_ub",
                                        scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("uint16", (128,), name="src1_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_and([0, 2**64-1], dst_ub, src0_ub, src1_ub, 1,
                                    8, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_and"
        return self._vector_binary_tenary_elewise_func(
            'vand', mask, dst, src0, src1, repeat_times, default_blk_stride,
            default_blk_stride, default_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride, stride_unit,
            print_name=print_name)

    def vec_or(self,  # pylint: disable=R0913
               mask,
               dst,
               src0,
               src1,
               repeat_times,
               dst_rep_stride,
               src0_rep_stride,
               src1_rep_stride):
        r'''
        Performs bit-wise OR element-wise
        Description:
          Performs bit-wise OR element-wise:
          \f$dst_i = src0_i|src1_i,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a
            Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst and src,
            \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src0 : Source operand 0, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          src1 : Source operand 1, which is the start element of the tensor.
          For details about the supported data type,
          see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 0
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand 1
          NOTICE :
            dst, src0, and src1 have the same data type:
            - Tensors of type uint16 or int16

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride, src0_rep_stride, or src1_rep_stride is within the
          range [0, 255], in the unit of 32 bytes.
          The argument is a scalar of type int16/int32/int64/uint16/uint32
          /uint64, an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the destination
            operand, that is, the destination operand of the Nth iteration is
            the source operand of the (N+1)th iteration,
            address overlapping is not supported. Address overlapping is not
            supported in the following cases: For the
            vec_add/vec_sub/vec_mul/vec_max/v_min/vec_and/vec_or instruction,
            (1) the data type is float16, int32, or
            float32, and the destination operand overlaps the second source
            operand; (2) src1_rep_stride = dst_rep_stride = 0;
            (3) The addresses of src0 and src1 cannot overlap.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("uint16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("uint16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            mask_h = tik_instance.Scalar(dtype="uint64", init_value=1)
            mask_l = tik_instance.Scalar(dtype="uint64", init_value=15)
            mask = [mask_h, mask_l]
            repeat_times = tik_instance.Scalar(dtype="int32", init_value=1)
            tik_instance.vec_or(mask, dst_ub, src0_ub, src1_ub, repeat_times,
                                8, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_or"
        return self._vector_binary_tenary_elewise_func(
            'vor', mask, dst, src0, src1, repeat_times, default_blk_stride,
            default_blk_stride, default_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride, stride_unit,
            print_name=print_name)

    def vec_relu(self,  # pylint: disable=R0913
                 mask,
                 dst,
                 src,
                 repeat_times,
                 dst_rep_stride,
                 src_rep_stride):
        r'''
        Performs a ReLU operation element-wise
        Description:
          Performs a ReLU operation element-wise:
          \f$dst_i = relu(src_i),i\in [0,PAR]\f$


          ReLU stands for rectified linear unit, and is the most used
          activation function in artificial neural networks.

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          NOTICE :
            dst has the same data type as src. Must be one
            of the following data types:
            - Ascend 310 AI Processor: tensors of type float16
            - Ascend 910 AI Processor: tensors of type float16
            - HiSilicon SoC (ES): tensors of type float16/int32
            - Ascend 610 AI Processor (AI Core): tensors of type float16
            /float32/int32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /float32/int32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_relu(128, dst_ub, src_ub, 1, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_relu"
        self._vector_single_elewise_func('vrelu', mask, dst, src, repeat_times,
                                         default_blk_stride, default_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit, print_name)

    def vec_ln(self,  # pylint: disable=R0913
               mask,
               dst,
               src,
               repeat_times,
               dst_rep_stride,
               src_rep_stride):
        r'''
        Computes the natural logarithm element-wise
        Description:
          Computes the natural logarithm element-wise:
          \f$dst_i = \log_{e}(src_i),i\in [0,PAR]\f$


          For the Ascend 310 AI Processor, the computation result using this
          API with float16 input in the range (0, 5/3)
          fails to meet the dual-0.1% error limit (the error ratio is
          within 0.1% and the relative error is within 0.1%).
          If the accuracy requirement is high, the vec_ln_high_preci
          API is preferred.

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          - NOTICE :
            dst has the same data type as src:
            - Ascend 310 AI Processor: tensors of type float16
            - Ascend 910 AI Processor: tensors of type float16 or float32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.
          - If any value of src is not positive, an unknown result may
          be produced.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            # Define the tensors.
            src_gm = tik_instance.Tensor("float16", (128,), tik.scope_gm,
                                        "src_gm")
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            dst_gm = tik_instance.Tensor("float16", (128,), tik.scope_gm,
                                        "dst_gm")
            # Move the user input to the source UB.
            tik_instance.data_move(src_ub, src_gm, 0, 1, 8, 0, 0)
            tik_instance.vec_ln(128, dst_ub, src_ub, 1, 8, 8)
            # Move the computation result to the destination GM.
            tik_instance.data_move(dst_gm, dst_ub, 0, 1, 8, 0, 0)
            tik_instance.BuildCCE("v100_mini_vec_ln_test", [src_gm], [dst_gm])

            #Inputs:
            #[1, 2, 3, 4, ......, 128]
            #Returns:
            #[0, 0.6931, 1.0986, 1.3863, ......, 4.8520]
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_ln"
        self._vector_single_elewise_func(
            'vln', mask, dst, src, repeat_times,
            default_blk_stride, default_blk_stride,
            dst_rep_stride, src_rep_stride, stride_unit, print_name)

    def vec_exp(self,  # pylint: disable=R0913
                mask,
                dst,
                src,
                repeat_times,
                dst_rep_stride,
                src_rep_stride):
        r'''
        Computes the natural exponential element-wise
        Description:
          Computes the natural exponential element-wise:
          \f$dst_i = e^{src_i},i\in [0,PAR]\f$


          Even if the \f$e^{x}\f$ computation result meets the
          accuracy requirement, the \f$e^{x}-1\f$ computation result using this
          API with float16 input fails to meet the dual-0.1% error
          limit (the error ratio is within 0.1% and the
          relative error is within 0.1%) due to the subtraction error. If the
          accuracy requirement for the \f$e^{x}-1\f$
          computation is high, the vec_expm1_high_preci API is preferred.

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          - NOTICE :
            dst has the same data type as src. Must be one
            of the following data types:
            - Ascend 310 AI Processor: tensors of type float16
            - Ascend 910 AI Processor: tensors of type float16 or float32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_gm = tik_instance.Tensor("float16", (128,), name="src_gm",
                                        scope=tik.scope_gm)
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_gm = tik_instance.Tensor("float16", (128,), name="dst_gm",
                                        scope=tik.scope_gm)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.data_move(src_ub, src_gm, 0, 1, 8, 0, 0)
            tik_instance.vec_exp(128, dst_ub, src_ub, 1, 8, 8)
            tik_instance.data_move(dst_gm, dst_ub, 0, 1, 8, 0, 0)
            tik_instance.BuildCCE(kernel_name="exp", inputs=[src_gm],
                                        outputs=[dst_gm])

            #Inputs:
            #[0, 1, 2, 3, ......]
            #Returns:
            #[1.0, 2.719, 7.391, 20.08, ......]
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_exp"
        self._vector_single_elewise_func(
            'vexp', mask, dst, src, repeat_times,
            default_blk_stride, default_blk_stride,
            dst_rep_stride, src_rep_stride, stride_unit, print_name)

    def vec_abs(self,  # pylint: disable=R0913
                mask,
                dst,
                src,
                repeat_times,
                dst_rep_stride,
                src_rep_stride):
        r'''
        Computes the absolute value element-wise
        Description:
          Computes the absolute value element-wise:
          \f$dst_i = abs(src_i),i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          - NOTICE :
            dst has the same data type as src.
            - Ascend 310 AI Processor: tensors of type float16 or float32
            - Ascend 910 AI Processor: tensors of type float16 or float32
            - HiSilicon SoC (ES): tensors of type float16
            - Ascend 610 AI Processor (AI Core): tensors
            of type float16 or float32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /float32/int16

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_abs(128, dst_ub, src_ub, 1, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_abs"
        self._vector_single_elewise_func('vabs', mask, dst, src, repeat_times,
                                         default_blk_stride, default_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit, print_name)

    @source_info_decorator()
    @debug.vexpm1_high_preci_decorator
    def vec_expm1_high_preci(self,  # pylint: disable=R0913, R0914
                             mask,
                             dst,
                             src,
                             work_tensor,
                             repeat_times,
                             dst_rep_stride,
                             src_rep_stride):
        r'''
        Computes the natural base element-wise
        Description:
          Computes the natural base element-wise:
          \f$dst_i = e^{src_i}-1,i\in [0,PAR]\f$


          The \f$e^{x}-1\f$ computation result using this API offers higher
          accuracy than the vec_exp API.

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          NOTICE :
            The following describes only the dst, src,
            and work_tensor parameters.
            dst, src, and work_tensor are tensors of
            the same data type, float16.
            - If the source operand tensor has an offset, the passing
            formats are as follows: tensor[offset1:offset2]
            means that starting from offset1 and ending
            at offset2. tensor[offset1:] means starting from offset1.
            tensor[offset] means that only one element is
            passed. (In this case, the tensor is impossible to be sliced
            and a runtime error will be reported. Therefore. this
            format is not allowed.)
            - If the source operand tensor does not have an offset, the tensor
            can be passed directly.
            - work_tensor:
              - work_tensor is a user-defined temporary buffer space for
              storing the intermediate result. The space is
              limited to scope_ubuf and is used for internal computation only.
              - work_tensor buffer space calculation:
                - Calculate the minimum buffer space required for src
                computation based on repeat_times and
                src_rep_stride as follows: src_extent_size =
                (repeat_times - 1) * src_rep_stride * 16 + 128 If
                0 < src_rep_stride <= 8, consider src_rep_stride as 8.
                Otherwise, retain its original value.
                - Round up the minimum buffer space required for src
                computation to the least multiple of 32 bytes:
                wk_size_unit = (src_extent_size + 15)//16 * 16
                - Calculate the size of work_tensor as follows:
                work_tensor = 11 * wk_size_unit
              - Example of work_tensor buffer space calculation:
                - If repeat_times = 1 and src_rep_stride = 8, then
                src_extent_size= 128 and work_tensor = 128 * 11.
                - If repeat_times = 2 and src_rep_stride = 4, then
                src_extent_size = (2 - 1) * 8 * 16 + 128 = 256 and
                work_tensor = 256 * 11.

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.
          - dst, src, and work_tensor must be declared in scope_ubuf.
          - The space of the dst, src, and work_tensor tensors cannot overlap.
          - The final computation result must be within the data range.
          Otherwise, an infinite or saturated result is yielded.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("V100", "mini"))
            src_gm = tik_instance.Tensor("float16", (128,), name="src_gm",
                                        scope=tik.scope_gm)
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_gm = tik_instance.Tensor("float16", (128,), name="dst_gm",
                                        scope=tik.scope_gm)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            # The required space is ((1 - 1) * 8 * 16 + 128) * 11 = 128 * 11.
            work_tensor_ub = tik_instance.Tensor("float16", (128*11,),
                        name="work_tensor_ub", scope=tik.scope_ubuf)
            tik_instance.data_move(src_ub, src_gm, 0, 1, 8, 0, 0)
            tik_instance.vec_expm1_high_preci(128, dst_ub, src_ub,
                        work_tensor_ub, 1, 8, 8)
            tik_instance.data_move(dst_gm, dst_ub, 0, 1, 8, 0, 0)
            tik_instance.BuildCCE(kernel_name="expm1", inputs=[src_gm],
                        outputs=[dst_gm])

            #Inputs:
            #[0, 1, 2, 3, ......]
            #Returns:
            #[0.0, 1.719, 6.391, 19.08, ......]
        '''
        def _gen_comparator():
            # default rep stride
            and_rep_stride = 8

            def _cmp_params_convert(param, target):
                result = param
                if is_immediate_number(src_rep_stride):
                    if src_rep_stride == 0:
                        result = target
                else:
                    result = self.Scalar_(  # pylint: disable=E1101
                        init_value=param)
                    with self.if_scope(src_rep_stride == 0):
                        result.set_as(target)
                return result

            # if src_rep_stride is 0, make cmp_repeat_times 1
            cmp_repeat_times = _cmp_params_convert(repeat_times, 1)
            # if src_rep_stride is 0, make and_rep_stride 0, else 8
            and_rep_stride = _cmp_params_convert(and_rep_stride, 0)

            self._vector_scalar_elewise_func(  # pylint: disable=E1101
                'vector_dup', mask, work_tensor[:extent], 1.7,
                repeat_times, 1, src_rep_stride, 0,
                print_name=instr_name, mask_o=mask_o)
            lt_tensor = \
                work_tensor[extent:2*extent].reinterpret_cast_to("uint16")
            self.vcmpv_lt(lt_tensor,  # pylint: disable=E1101
                          src, work_tensor[:extent], cmp_repeat_times,
                          1, 1, src_rep_stride, src_rep_stride)
            self._vector_scalar_elewise_func(  # pylint: disable=E1101
                'vector_dup', mask, work_tensor[:extent], -0.7,
                repeat_times, 1, src_rep_stride, 0,
                print_name=instr_name, mask_o=mask_o)
            gt_tensor = \
                work_tensor[2*extent:3*extent].reinterpret_cast_to("uint16")
            self.vcmpv_gt(gt_tensor,  # pylint: disable=E1101
                          src, work_tensor[:extent], cmp_repeat_times,
                          1, 1, src_rep_stride, src_rep_stride)

            and_tensor = work_tensor[:extent].reinterpret_cast_to("uint16")
            self._vector_binary_tenary_elewise_func(
                'vand', 128, and_tensor, lt_tensor, gt_tensor, repeat_times,
                1, 1, 1, and_rep_stride, and_rep_stride, and_rep_stride,
                0, print_name=instr_name, mask_o=mask_o)
            return and_tensor

        def _expm1(dst, work_tensor, src,  # pylint: disable=R0913
                   dst_rep_stride, wk_rep_stride, src_rep_stride):
            self._vector_single_elewise_func(
                'vexp', mask, work_tensor, src, repeat_times, 1, 1,
                wk_rep_stride, src_rep_stride, 0,
                print_name=instr_name, mask_o=mask_o)
            self._vector_scalar_single_elewise_func(
                'vadds', mask, dst, work_tensor, -1, repeat_times, 1, 1,
                dst_rep_stride, wk_rep_stride, 0, 0,
                print_name=instr_name, mask_o=mask_o)

        def _do_select():
            src_offset = src_rep_stride*ONE_BLK_SIZE // DTYPE_SIZE[src.dtype]
            dst_offset = dst_rep_stride*ONE_BLK_SIZE // DTYPE_SIZE[src.dtype]
            sel_offset = self.Scalar_(init_value=8)  # pylint: disable=E1101
            with self.if_scope(src_rep_stride == 0):
                sel_offset.set_as(0)

            with self.for_range(0, repeat_times) as index:
                self.vec_sel_(mask, 0,
                              dst[dst_offset*index:],
                              cmp_sel[sel_offset*index:],
                              work_tensor[2*extent + src_offset*index:],
                              work_tensor[extent + src_offset*index:], 1,
                              dst_rep_stride, src_rep_stride, src_rep_stride,
                              instr_name, mask_o)

        instr_name = "vec_expm1_high_preci"
        with self.context.freeze():  # pylint: disable=E1101
            # params check
            check_high_preci_param(dst, src, work_tensor, repeat_times,
                                   dst_rep_stride, src_rep_stride)
            TikCheckUtil.check_equality(work_tensor.dtype, dst.dtype,
                                        "work_tensor's dtype must be same "
                                        "with dst's dtype")
            TikCheckUtil.check_equality(api_check_support("tik."
                                                          + instr_name,
                                                          dst.dtype), True,
                                        INSTR_DTYPE_SUPPORT_STATEMENT.
                                        format(dst.dtype, instr_name))
            # check mask
            check_mask_valid(mask, tensor_bit_len=max(get_bit_len(src.dtype),
                                                      get_bit_len(dst.dtype)))
            multi_factor = 11
            check_over_high_preci(
                mask, dst, src, work_tensor, repeat_times, dst_rep_stride,
                src_rep_stride, Expr(dst.offset).eval_value(),
                Expr(src.offset).eval_value(),
                Expr(work_tensor.offset).eval_value(), multi_factor,
                name=instr_name)
            default_rep_stride = src_rep_stride
            if is_immediate_number(src_rep_stride):
                if 0 < src_rep_stride < 8:
                    default_rep_stride = 8
            else:
                default_rep_stride = self.Scalar_(  # pylint: disable=E1101
                    init_value=src_rep_stride)
                with self.if_scope(src_rep_stride > 0):
                    with self.if_scope(src_rep_stride < 8):
                        default_rep_stride.set_as(8)

            default_mask = 128
            extent = self.get_wk_tensor_extend(default_mask, src.dtype,
                                               repeat_times,
                                               default_rep_stride)
            mask_o = mask_concat(self, mask,
                                 tensor_bit_len=get_bit_len(src.dtype))
            cmp_sel = _gen_comparator()
            _expm1(work_tensor[extent:2*extent],
                   work_tensor[2*extent:3*extent], src, src_rep_stride,
                   src_rep_stride, src_rep_stride)
            self._fp162fp32_high_preci_func(
                self._expm1_taylor, instr_name, mask,
                work_tensor[2*extent:3*extent], src,
                work_tensor[3*extent:], repeat_times,
                src_rep_stride, src_rep_stride, 4)
            _do_select()

    def vec_rec(self,  # pylint: disable=R0913
                mask,
                dst,
                src,
                repeat_times,
                dst_rep_stride,
                src_rep_stride):
        r'''
        Computes the reciprocal element-wise
        Description:
          Computes the reciprocal element-wise:
          \f$dst_i = \frac{1}{src_i},i\in [0,PAR]\f$


          Using this API, the operator computation result fails to meet the
          dual-0.1% error limit (the error ratio is
          within 0.1% and the relative error is within 0.1%) with float16
          input, and fails to meet the dual-0.01% error
          limit with float32 input. If the accuracy requirement is high,
          the vec_rec_high_preci API is preferred

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          - NOTICE :
            dst must have the same data type as src. Must be one of the
            following data types:
            Tensors of type float16 or float32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.
          - If any value of src is 0, an unknown result may be produced.

        Examples:
          # Example 1
              from te import tik
              # Define a container.
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              # Define the tensors.
              src_gm = tik_instance.Tensor("float32", (128,), name="src_gm",
                                            scope=tik.scope_gm)
              dst_gm = tik_instance.Tensor("float32", (128,), name="dst_gm",
                                            scope=tik.scope_gm)
              src_ub = tik_instance.Tensor("float32", (128,), name="src_ub",
                                            scope=tik.scope_ubuf)
              dst_ub = tik_instance.Tensor("float32", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
              # Move data from the GM to the UB.
              tik_instance.data_move(src_ub, src_gm, 0, 1, 128*4 // 32, 0, 0)
              tik_instance.vec_rec(64, dst_ub, src_ub, 2, 8, 8)
              # Move data from the UB to the GM.
              tik_instance.data_move(dst_gm, dst_ub, 0, 1, 128*4 // 32, 0, 0)
              tik_instance.BuildCCE(kernel_name="vec_rec", inputs=[src_gm],
                                    outputs=[dst_gm])

            #Inputs:
            #[1.2017815 -8.758528 -3.9551935 ... -1.3599057 -2.319316]
            #Returns:
            #[0.83203125 -0.11401367 -0.2529297 ... -0.734375 -0.43164062]

          # Example 2
              from te import tik
              # Define a container.
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              # Define the tensors.
              src_gm = tik_instance.Tensor("float16", (128,), name="src_gm",
                                            scope=tik.scope_gm)
              dst_gm = tik_instance.Tensor("float16", (128,), name="dst_gm",
                                            scope=tik.scope_gm)
              src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                            scope=tik.scope_ubuf)
              dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
              # Move data from the GM to the UB.
              tik_instance.data_move(src_ub, src_gm, 0, 1, 128*2 // 32, 0, 0)
              tik_instance.vec_rec(128, dst_ub, src_ub, 1, 8, 8)
              # Move data from the UB to the GM.
              tik_instance.data_move(dst_gm, dst_ub, 0, 1, 128*2 // 32, 0, 0)
              tik_instance.BuildCCE(kernel_name="vec_rec", inputs=[src_gm],
                                    outputs=[dst_gm])

            #Inputs:
            #[-7.152 -7.24 1.771 ... -1.339 4.473]
            #Returns:
            #[-0.1396 -0.1382 0.5645 ... -0.748 0.2231]
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_rec"
        self._vector_single_elewise_func('vrec', mask, dst, src, repeat_times,
                                         default_blk_stride,
                                         default_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit, print_name)

    @source_info_decorator()
    @debug.vrec_high_preci_decorator
    def vec_rec_high_preci(self,  # pylint: disable=R0913
                           mask,
                           dst,
                           src,
                           work_tensor,
                           repeat_times,
                           dst_rep_stride,
                           src_rep_stride):
        r'''
        Computes the reciprocal element-wise
        Description:
          Computes the reciprocal element-wise:
          \f$dst_i = \frac{1}{src_i},i\in [0,PAR]\f$


          The computation result using this API offers higher accuracy
          than the vec_rec API.

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          NOTICE :
            The following describes only the dst, src, and
            work_tensor parameters.
            dst has the same data type as src. They are tensors
            of type float16 or float32. work_tensor is a tensor of
            type float32.
            - If the source operand tensor has an offset, the passing formats
            are as follows: tensor[offset1:offset2]
            means that starting from offset1 and ending at offset2.
            tensor[offset1:] means starting from offset1.
            tensor[offset] means that only one element is passed.
            (In this case, the tensor is impossible to be sliced
            and a runtime error will be reported. Therefore.
            this format is not allowed.)
            - If the source operand tensor does not have an offset, the tensor
            can be passed directly.
            - work_tensor:
              - work_tensor is a user-defined temporary buffer space for
              storing the intermediate result. The space is
              limited to scope_ubuf and is used for internal computation only.
              - work_tensor buffer space calculation:
                - Calculate the minimum buffer space required for src
                computation based on repeat_times, mask, and
                src_rep_stride as follows: src_extent_size =(repeat_times
                - 1)*src_rep_stride * block_len + mask_len;
                When the source operand is of type float16, block_len is 16;
                When the source operand is of type float32, block_len is 8;
                In consecutive mask mode, mask_len is the mask value itself;
                In bit-wise mask mode, mask_len is the mask value corresponding
                 to the most significant bit.
              - Round up the minimum buffer space required for src computation
              to the least multiple
              of 32 bytes:wk_size_unit =((src_extent_size+block_len-1)//
                block_len) * block_len
              - Calculate the size of work_tensor as follows:
              When the source operand is of type float16,work_tensor
              = 4 * wk_size_unit
              When the source operand is of type float32,work_tensor
              = 2 * wk_size_unit
            - Example of work_tensor buffer space calculation:
              - If src is of type fp16, mask is 128, repeat_times is 2, and
              src_rep_stride is 8, then block_len is 16,
              mask_len is 128, and src_extent_size = (2-1* 8 * 16 + 128 = 256.
              Round up src_extent_size to the least
              multiple of 32 bytes: src_extent_size
              (wk_size_unit = ((256+16-1)//16)*16 =256). Therefore, the size of
              work_tensor is 4 * 256 = 1024.
              - If src is of type fp32, mask is 64, repeat_times is 2, and
              src_rep_stride is 8, then block_len is 8,
              mask_len is 64, and src_extent_size = (2 - 1) * 8 * 8 + 64 = 128.
              Round up src_extent_size to the least
              multiple of 32 bytes: src_extent_size (wk_size_unit =
              ((128+8-1)//8) * 8 = 128). Therefore, the size of
              work_tensor is 2 * 128 = 256.


        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.
          - dst, src, and work_tensor must be declared in scope_ubuf.
          - The space of the dst, src, and work_tensor tensors cannot overlap.
          - If any value is 0, an unknown result may be produced.

        Examples:
          #Example 1
              from te import tik
              # Define a container.
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              # Define the tensors.
              src_gm = tik_instance.Tensor("float32", (128,), name="src_gm",
                                            scope=tik.scope_gm)
              dst_gm = tik_instance.Tensor("float32", (128,), name="dst_gm",
                                            scope=tik.scope_gm)
              src_ub = tik_instance.Tensor("float32", (128,), name="src_ub",
                                            scope=tik.scope_ubuf)
              dst_ub = tik_instance.Tensor("float32", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
              # Move data from the GM to the UB.
              tik_instance.data_move(src_ub, src_gm, 0, 1, 128*4 // 32, 0, 0)
              # Calculate the size of work_tensor.
              mask = [0, 2**64 - 1]
              mask_len = 64
              repeat_times = 2
              dst_rep_stride = 8
              src_rep_stride = 8
              block_len = 8  # src dtype is float32
              src_extent_size = (repeat_times - 1)*src_rep_stride*block_len  +
              mask_len
              wk_size_unit = ((src_extent_size + block_len - 1)//block_len) *
              block_len
              wk_size = 2*wk_size_unit
              # Define work_tensor.
              work_tensor_ub = tik_instance.Tensor("float32", (wk_size,),
              name="work_tensor_ub", scope=tik.scope_ubuf)
              # If the work_tensor has an index, use the
              # work_tensor[index:] format.
              tik_instance.vec_rec_high_preci(mask_len, dst_ub, src_ub,
              work_tensor_ub[0:], repeat_times, dst_rep_stride, src_rep_stride)
              # Move data from the UB to the GM.
              tik_instance.data_move(dst_gm, dst_ub, 0, 1, 128*4 // 32, 0, 0)
              tik_instance.BuildCCE(kernel_name="test_vec_rec_high_preci",
                                inputs=[src_gm], outputs=[dst_gm])

            #Inputs:
            #[-6.9427586 -3.5300326 1.176882 ... -6.196793 9.0379095]
            #Returns:
            #[-0.14403497 -0.2832835 0.8497028 ... -0.16137381 0.11064506]

          #Example 2
              from te import tik
              # Define a container.
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              # Define the tensors.
              src_gm = tik_instance.Tensor("float16", (128,), name="src_gm",
                                            scope=tik.scope_gm)
              dst_gm = tik_instance.Tensor("float16", (128,), name="dst_gm",
                                            scope=tik.scope_gm)
              src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                            scope=tik.scope_ubuf)
              dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
              # Move data from the GM to the UB.
              tik_instance.data_move(src_ub, src_gm, 0, 1, 128*2 // 32, 0, 0)
              # Calculate the size of work_tensor.
              mask = 128
              mask_len = mask
              repeat_times = 1
              dst_rep_stride = 8
              src_rep_stride = 8
              block_len = 16  # src dtype is float16
              src_extent_size = (repeat_times - 1)*src_rep_stride*block_len  +
              mask_len
              wk_size_unit = ((src_extent_size + block_len - 1) // block_len)*
              block_len
              wk_size = 4*wk_size_unit
              # Define work_tensor.
              work_tensor_ub = tik_instance.Tensor("float32", (wk_size,),
              name="work_tensor_ub", scope=tik.scope_ubuf)
              # If the work_tensor has an index, use the work_tensor[index:]
              #format.
              tik_instance.vec_rec_high_preci(mask_len, dst_ub, src_ub,
              work_tensor_ub[0:], repeat_times, dst_rep_stride, src_rep_stride)
              # Move data from the UB to the GM.
              tik_instance.data_move(dst_gm, dst_ub, 0, 1, 128*2 // 32, 0, 0)
              tik_instance.BuildCCE(kernel_name="test_vec_rec_high_preci",
              inputs=[src_gm], outputs=[dst_gm])

            #Inputs:
            #[-7.08 -4.434 1.294 ... 8.82 -2.854]
            #Returns:
            #[-0.1412 -0.2256 0.773 ... 0.1134 -0.3503]
        '''
        instr_name = "vec_rec_high_preci"

        # params check
        check_high_preci_param(dst, src, work_tensor, repeat_times,
                               dst_rep_stride, src_rep_stride)
        TikCheckUtil.check_equality(work_tensor.dtype, "float32",
                                    "work_tensor dtype should be float32, "
                                    "but input dtype: %s" % work_tensor.dtype)
        TikCheckUtil.check_equality(
            api_check_support("tik." + instr_name, dst.dtype), True,
            INSTR_DTYPE_SUPPORT_STATEMENT.format(dst.dtype, instr_name))

        src_bit_len = get_bit_len(src.dtype)
        dst_bit_len = get_bit_len(dst.dtype)
        check_mask_valid(mask, tensor_bit_len=max(src_bit_len, dst_bit_len))
        multi_factor = 2
        if src.dtype == "float16":
            # 4B of fp32, need keep 32B algin
            multi_factor += 2
        # check overflow and overlap
        check_over_high_preci(
            mask, dst, src, work_tensor,
            repeat_times, dst_rep_stride,
            src_rep_stride,
            Expr(dst.offset).eval_value(),
            Expr(src.offset).eval_value(),
            Expr(work_tensor.offset).eval_value(),
            multi_factor, name="vec_rec_high_preci")

        if src.dtype == "float16":
            # work_tensor need more!
            with self.context.freeze():  # pylint: disable=E1101
                self._fp162fp32_high_preci_func(
                    self._vec_rec_high_preci, instr_name, mask,
                    dst, src, work_tensor, repeat_times,
                    dst_rep_stride, src_rep_stride, multi_factor)
            return

        # get mask_o
        mask_o = mask_concat(self, mask, tensor_bit_len=src_bit_len)
        tensor_split_size = self.get_wk_tensor_extend(
            mask, src.dtype, repeat_times, src_rep_stride)
        with self.context.freeze():  # pylint: disable=E1101
            self._vec_rec_high_preci(instr_name, mask, dst, src, work_tensor,
                                     repeat_times, dst_rep_stride,
                                     src_rep_stride, tensor_split_size, mask_o)

    def vec_not(self,  # pylint: disable=R0913
                mask,
                dst,
                src,
                repeat_times,
                dst_rep_stride,
                src_rep_stride):
        r'''
        Performs bit-wise NOT element-wise
        Description:
          Performs bit-wise NOT element-wise:
          \f$dst_i = \sim src_i,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          - NOTICE :
            dst and src have the same data type:Tensors of type uint16 or int16

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("uint16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_not(128, dst_ub, src_ub, 1, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_not"
        self._vector_single_elewise_func('vnot', mask, dst, src, repeat_times,
                                         default_blk_stride, default_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit, print_name)

    def vec_rsqrt(self,  # pylint: disable=R0913
                  mask,
                  dst,
                  src,
                  repeat_times,
                  dst_rep_stride,
                  src_rep_stride):
        r'''
        Computes the reciprocal after extracting the square root element-wise
        Description:
          Computes the reciprocal after extracting the square root
          element-wise:\f$dst_i = \sqrt[-1]{src_i},i\in [0,PAR]\f$


          Using this API, the operator computation result fails to meet the
          dual-0.1% error limit (the error ratio is
          within 0.1% and the relative error is within 0.1%)
          with float16 input, and fails to meet the dual-0.01% error
          limit with float32 input. If the accuracy requirement is high,
          the vec_rsqrt_high_preci API is preferred.

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          - NOTICE :
            dst has the same data type as src:
            Tensors of type float16 or float32

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.
          - If any value of src is not positive, an unknown result
          may be produced.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            # Define the tensors.
            src_gm = tik_instance.Tensor("float16", (128,), name="src_gm",
                                        scope=tik.scope_gm)
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            dst_gm = tik_instance.Tensor("float16", (128,), name="dst_gm",
                                        scope=tik.scope_gm)
            # Move the user input to the source UB.
            tik_instance.data_move(src_ub, src_gm, 0, 1, 8, 0, 0)
            tik_instance.vec_rsqrt(128, dst_ub, src_ub, 1, 8, 8)
            # Move the computation result to the destination GM.
            tik_instance.data_move(dst_gm, dst_ub, 0, 1, 8, 0, 0)
            tik_instance.BuildCCE("test_vec_rsqrt", [src_gm], [dst_gm])

            #Inputs:
            #[1, 2, 3, 4, ......, 128]
            #Returns:
            #[0.998, 0.705, 0.576, 0.499, ......, 0.08813]
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_rsqrt"
        self._vector_single_elewise_func(
            'vrsqrt', mask, dst, src, repeat_times,
            default_blk_stride, default_blk_stride,
            dst_rep_stride, src_rep_stride, stride_unit, print_name)

    @source_info_decorator()
    @debug.vrsqrt_high_preci_decorator
    def vec_rsqrt_high_preci(    # pylint: disable=R0913
            self, mask, dst, src, work_tensor,
            repeat_times, dst_rep_stride, src_rep_stride):
        r'''
        Computes the reciprocal after extracting the square root element-wise
        Description:
          Computes the reciprocal after extracting the square root
          element-wise:\f$dst_i = \sqrt[-1]{src_i},i\in [0,PAR]\f$


          The computation result using this API offers higher accuracy than
          the vec_rsqrt API.

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          NOTICE :
            The following describes only the dst, src,
            and work_tensor parameters.
            dst has the same data type as src. They are tensors
            of type float16 or float32.
            work_tensor is a tensor of type float32.
            - If the source operand tensor has an offset, the passing formats
            are as follows: tensor[offset1:offset2]
            means that starting from offset1 and ending at offset2.
            tensor[offset1:] means starting from offset1.
            tensor[offset] means that only one element is passed.
            (In this case, the tensor is impossible to be sliced
            and a runtime error will be reported. Therefore.
            this format is not allowed.)
            - If the source operand tensor does not have an offset,
            the tensor can be passed directly.
            - work_tensor:
              - work_tensor is a user-defined temporary buffer space for
              storing the intermediate result. The space is limited
              to scope_ubuf and is used for internal computation only.
              - work_tensor buffer space calculation:
              - Calculate the minimum buffer space required for src computation
               based on repeat_times, mask,
                and src_rep_stride as follows: src_extent_size =(repeat_times
                - 1) * src_rep_stride * block_len + mask_len;
                When the source operand is of type float16, block_len is 16;
                When the source operand is of type float32, block_len is 8;
                In consecutive mask mode, mask_len is the mask value itself;
                In bit-wise mask mode, mask_len is the mask value corresponding
                 to the most significant bit.
              - Round up the minimum buffer space required for src computation
              to the least multiple
              of 32 bytes:wk_size_unit =((src_extent_size+block_len-1)//
                block_len) * block_len
              - Calculate the size of work_tensor as follows:
                - For the Ascend 310 AI Processor:
                  - When the source operand is
                  of type float16,work_tensor = 6 * wk_size_unit
                  - When the source operand is
                  of type float32,work_tensor = 4 * wk_size_unit
                - For the Ascend 910 AI Processor:
                  - When the source operand is
                  of type float16,work_tensor = 5 * wk_size_unit
                  - When the source operand is
                  of type float32,work_tensor = 3 * wk_size_unit
            - Example of work_tensor buffer space calculation:
              - For the Ascend 310 AI Processor:
            If src is of type fp16, mask is 128, repeat_times is 2,
            and src_rep_stride is 8, then block_len is 16,
            mask_len is 128, and src_extent_size = (2 - 1) * 8 * 16 + 128 =256.
             Round up src_extent_size to the least
            multiple of 32 bytes: wk_size_unit = 256. Therefore, the size of
            work_tensor is 6 * 256 = 1536.
              - For the Ascend 910 AI Processor:
            If src is of type float16, mask is 128, repeat_times is 2, and
            src_rep_stride is 8, then block_len is 16,
            mask_len is 128, and src_extent_size = (2 - 1) * 8 * 16 + 128 =256.
             Round up src_extent_size to the least
            multiple of 32 bytes: wk_size_unit = 256. Therefore, the size of
            work_tensor is 5 * 256 = 1280.

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.
          - dst, src, and work_tensor must be declared in scope_ubuf.
          - The space of the dst, src, and work_tensor tensors cannot overlap.
          - If any value of src is not positive, an unknown result
          may be produced.

        Examples:
          #Example 1
              from te import tik
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              # Define the tensors.
              dst_gm = tik_instance.Tensor("float16", (128,), name="dst_gm",
                                            scope=tik.scope_gm)
              src_gm = tik_instance.Tensor("float16", (128,), name="src_gm",
                                            scope=tik.scope_gm)
              src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                            scope=tik.scope_ubuf)
              dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
              # Move the user input to the source UB.
              tik_instance.data_move(src_ub, src_gm, 0, 1, 128*2 // 32, 0, 0)
              mask = 128
              #In consecutive mask mode, mask_len is the mask value itself.
              mask_len = mask
              repeat_times = 1
              dst_rep_stride = 8
              src_rep_stride = 8
              block_len = 16  # src dtype is float16
              src_extent_size = (repeat_times - 1)*src_rep_stride*
              block_len  + mask_len
              wk_size_unit = ((src_extent_size + block_len - 1) // block_len)
              *block_len
              wk_size = 6*wk_size_unit # Obtain the size of work_tensor.
              # Define work_tensor.
              work_tensor = tik_instance.Tensor("float32", (wk_size ,),
                            name="work_tensor", scope=tik.scope_ubuf)
              # If the tensor has an index offset, add a colon (:) after the
              # subscript in the following format. Otherwise, the program
              # will report an error.
              tik_instance.vec_rsqrt_high_preci(mask, dst_ub, src_ub,
              work_tesnor[0:], repeat_times, dst_rep_stride, src_rep_stride)
              # Move the computation result to the destination GM.
              tik_instance.data_move(dst_gm, dst_ub, 0, 1, 128*2 // 32, 0, 0)
              tik_instance.BuildCCE("test_vec_rsqrt_high_preci",
                                    inputs=[src_gm], outputs=[dst_gm])

            #Inputs:
            #src_gm= [6.996 1.381 5.996 7.902 ... 5.113  5.78  1.672  5.418  ]
            #Returns:
            #dst_gm:[0.3782 0.851 0.4084 0.3557 ... 0.4421 0.416 0.7734 0.4297]

          #Example 2
              from te import tik
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              # Define the tensors.
              dst_gm = tik_instance.Tensor("float32", (128,), name="dst_gm",
                                            scope=tik.scope_gm)
              src_gm = tik_instance.Tensor("float32", (128,), name="src_gm",
                                            scope=tik.scope_gm)
              src_ub = tik_instance.Tensor("float32", (128,), name="src_ub",
                                            scope=tik.scope_ubuf)
              dst_ub = tik_instance.Tensor("float32", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
              # Move the user input to the source UB.
              tik_instance.data_move(src_ub, src_gm, 0, 1, 128*4 // 32, 0, 0)
              mask = [0, 2**64 - 1]
              # In bit-wise mask mode, mask_len is the mask value corresponding
              #to the most significant bit.
              mask_len = 64
              repeat_times = 2
              dst_rep_stride = 8
              src_rep_stride = 8
              block_len = 8  # src dtype is float32
              src_extent_size = (repeat_times - 1)*src_rep_stride*block_len  +
              mask_len
              wk_size_unit = ((src_extent_size + block_len - 1)//block_len)*
              block_len
              wk_size = 4*wk_size_unit # Obtain the size of work_tensor.
              # Define work_tensor.
              work_tensor = tik_instance.Tensor("float32", (wk_size ,),
                            name="work_tensor", scope=tik.scope_ubuf)
              # If the tensor has an index offset, add a colon (:) after the
              #subscript in the following format. Otherwise, the program
              #will report an error.
              tik_instance.vec_rsqrt_high_preci(mask, dst_ub, src_ub,
              work_tesnor[0:], repeat_times, dst_rep_stride, src_rep_stride)
              # Move the computation result to the destination GM.
              tik_instance.data_move(dst_gm, dst_ub, 0, 1, 128*4 // 32, 0, 0)
              tik_instance.BuildCCE("test_vec_rsqrt_high_preci",
                                    inputs=[src_gm], outputs=[dst_gm])

              #Inputs:
              #  src_gm=   [5.349619, 0.4301902, 4.7152824, 9.539162, ...,
              #             5.7243876, 4.4785686, 7.030495, 7.489954]
              #Returns:
              #  dst_gm:   [0.43235308, 1.5246484, 0.46051747, 0.32377616, ...,
              #             0.41796073, 0.47253108, 0.37714386, 0.36539316]
        '''
        check_high_preci_param(dst, src, work_tensor,
                               repeat_times, dst_rep_stride, src_rep_stride)
        TikCheckUtil.check_equality(work_tensor.dtype, "float32",
                                    "work_tensor's dtype should be"
                                    " float32, input: %s" % work_tensor.dtype)
        instr_name = "vec_rsqrt_high_preci"
        TikCheckUtil.check_equality(api_check_support("tik."
                                                      + instr_name,
                                                      dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, instr_name))
        # check mask
        check_mask_valid(mask, tensor_bit_len=max(get_bit_len(src.dtype),
                                                  get_bit_len(dst.dtype)))
        multi_factor = 3
        if get_soc_name() == ASCEND_310:
            multi_factor = 4
        if src.dtype == "float16":
            # 4B of fp32, need keep 32B algin
            multi_factor += 2
        # check overflow and overlap
        check_over_high_preci(mask, dst, src, work_tensor,
                              repeat_times, dst_rep_stride,
                              src_rep_stride,
                              Expr(dst.offset).eval_value(),
                              Expr(src.offset).eval_value(),
                              Expr(work_tensor.offset).eval_value(),
                              multi_factor, name="vec_rsqrt_high_preci")

        if src.dtype == "float16":
            with self.context.freeze():    # pylint: disable=E1101
                # work_tensor need more!
                if get_soc_name() == ASCEND_310:
                    self._fp162fp32_high_preci_func(
                        self.high_rsqrt_mini, instr_name, mask, dst,
                        src, work_tensor, repeat_times,
                        dst_rep_stride, src_rep_stride, multi_factor)
                else:
                    self._fp162fp32_high_preci_func(
                        self.high_rsqrt_cloud, instr_name, mask, dst,
                        src, work_tensor, repeat_times,
                        dst_rep_stride, src_rep_stride, multi_factor)
            return
        # get mask_o
        mask_o = mask_concat(self, mask, tensor_bit_len=get_bit_len(src.dtype))
        # get split tensor size
        tensor_split_size = self.get_wk_tensor_extend(
            mask, src.dtype, repeat_times, src_rep_stride)
        with self.context.freeze():    # pylint: disable=E1101
            if get_soc_name() == ASCEND_310:
                self.high_rsqrt_mini(instr_name, mask, dst, src, work_tensor,
                                     repeat_times, dst_rep_stride,
                                     src_rep_stride, tensor_split_size, mask_o)
            else:
                self.high_rsqrt_cloud(instr_name, mask, dst, src,
                                      work_tensor, repeat_times,
                                      dst_rep_stride, src_rep_stride,
                                      tensor_split_size, mask_o)

    def vec_axpy(self,  # pylint: disable=R0913
                 mask,
                 dst,
                 src,
                 scalar,
                 repeat_times,
                 dst_rep_stride,
                 src_rep_stride):
        r"""
        Performs multiplication-accumulation between a vector and a scalar ...
        Description:
          Performs multiplication-accumulation between a vector and a scalar
          element-wise:\f$dst_i = src_i*scalar+dst_i,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst
            and src, \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is
            participated in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand
          - NOTICE :
            dst, src, and scalar are tensors of type float16 or float32. src
            and scalar must have the same data type.

            The supported precision combinations are as follows:
          |Type      |src.dtype  |scalar.dtype  |dst.dtype       |PAR/Repeat|
          |----      |----       |----          |----            |----      |
          |fp16      |float16    |float16       |float16         |   128    |
          |fp32      |float32    |float32       |float32         |    64    |
          |fmix      |float16    |float16       |float32         |    64    |

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are
          as follows. Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.
          - Note that mixed precision (fmix) is supported.
          - In fmix mode, only the first four blocks of src are computed
          every iteration.


        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            scalar = 2
            dst_ub = tik_instance.Tensor("float32", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_axpy(64, dst_ub, src_ub, scalar, 1, 8, 4)
        """
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_axpy"
        self._vector_scalar_single_elewise_func('vaxpy', mask, dst, src,
                                                scalar, repeat_times,
                                                default_blk_stride,
                                                default_blk_stride,
                                                dst_rep_stride, src_rep_stride,
                                                stride_unit,
                                                _ROUND_TO_NEAREST_ENABLE,
                                                print_name=print_name)

    def vec_adds(self,  # pylint: disable=R0913
                 mask,
                 dst,
                 src,
                 scalar,
                 repeat_times,
                 dst_rep_stride,
                 src_rep_stride,
                 mask_mode="normal"):
        r'''
        Performs addition between a vector and a scalar element-wise
        Description:
          Performs addition between a vector and a scalar element-wise:
          \f$dst_i = src_i+scalar,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a
            Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst and src,
            \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Vector destination operand, which is the start element
          of the tensor. For details about the supported
          data precision, see the specific instruction.
          src : Vector source operand, which is the start element
          of the tensor. For details about the supported data
          precision, see the specific instruction.
          scalar : A scalar or immediate, specifying the scalar source operand
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand
          NOTICE :
            The dst, src, and scalar operands have the same data type:
            - Ascend 310 AI Processor: tensors of type float16 or float32
            - Ascend 910 AI Processor: tensors of type float16 or float32
            - HiSilicon SoC (ES): tensors of type float16/int32
            - Ascend 610 AI Processor (AI Core): tensors of type float16/int16
            /float32/int32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /int16/float32/int32

        Kwargs:
          mask_mode : A string specifying the mask mode.
          The options are as follows:
            - normal: normal mode
            - counter: counter mode
              - For Ascend 310 AI Processor, this parameter has no effect.
              - For Ascend 910 AI Processor, this parameter has no effect.

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - The addresses of dst and src cannot overlap.
          - The argument of the scalar parameter is a scalar or an immediate
          of type int/float.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            scalar = tik_instance.Scalar(dtype="float16", init_value=2)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_adds(128, dst_ub, src_ub, scalar, 1, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_adds"
        self._vector_scalar_single_elewise_func(
            'vadds', mask, dst, src, scalar, repeat_times, default_blk_stride,
            default_blk_stride, dst_rep_stride, src_rep_stride, stride_unit,
            _ROUND_TO_NEAREST_ENABLE, mask_mode, print_name)

    def vec_muls(self,  # pylint: disable=R0913
                 mask,
                 dst,
                 src,
                 scalar,
                 repeat_times,
                 dst_rep_stride,
                 src_rep_stride,
                 mask_mode="normal"):
        r'''
        Performs multiplication between a vector and a scalar element-wise
        Description:
          Performs multiplication between a vector and a scalar element-wise:
          \f$dst_i = src_i*scalar,i\in [0,PAR]\f$

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          NOTICE :
            - mask_mode is an internal optional parameter
            The dst, src, and scalar operands have the same data type:
            - Ascend 310 AI Processor: tensors of type float16 or float32
            - Ascend 910 AI Processor: tensors of type float16 or float32
            - HiSilicon SoC (ES): tensors of type float16/int32
            - Ascend 610 AI Processor (AI Core): tensors of type float16/int16
            /float32/int32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            /int16/float32/int32

        Kwargs:
          mask_mode : A string specifying the mask mode.
          The options are as follows:
            - normal: normal mode
            - counter: counter mode
              - For Ascend 310 AI Processor, this parameter has no effect.
              - For Ascend 910 AI Processor, this parameter has no effect.

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            scalar = 2
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_muls(128, dst_ub, src_ub, scalar, 1, 8, 8)
        '''
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_muls"
        self._vector_scalar_single_elewise_func(
            'vmuls', mask, dst, src, scalar, repeat_times, default_blk_stride,
            default_blk_stride, dst_rep_stride, src_rep_stride, stride_unit,
            _ROUND_TO_NEAREST_ENABLE, mask_mode, print_name)

    @source_info_decorator()
    def vec_sel(self, mask, mode, dst,  # pylint: disable=R0913
                sel, src0, src1, repeat_times,
                dst_rep_stride=0, src0_rep_stride=0, src1_rep_stride=0):
        """
        Selects elements bit-wise
        Description:
          Selects elements bit-wise. 1'b1: selected from src0; other values:
          selected from src1.
        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or
            a Python immediate, specifying the number of
            elements participated in the computation. If mask = 16, the
            first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst
            and src, \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is
            participated in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          mode : Instruction mode
            - 0 : Select between two tensors based on sel. Multiple iterations
            are supported. Each iteration is based on
            the first 128 bits (if the destination operand is of type float16)
            or 64 bits (if the destination operand is
            of type float32) of sel.
            - 1 : Select between a tensor and a scalar bit-wise based on sel.
            Multiple iterations are supported.
            - 2 : Select between two tensors bit-wise based on sel. Multiple
            iterations are supported.
            - Ascend 310 AI Processor: Only mode 0 is supported.
            - Ascend 910 AI Processor: Only mode 0 is supported.
          dst : A tensor for the start element of the destination operand
            - Ascend 310 AI Processor: tensor of type float16
            - Ascend 910 AI Processor: tensor of type float16 or float32
            - Hi3796: tensor of type float16
            - Ascend 610 AI Processor (AI Core): tensor of type float16
            or float32
            - Ascend 610 AI Processor (Vector Core): tensor of type float16
            or float32
          sel : Mask selection. Each bit indicates the selection of an element.
            - In mode 0, 1, or 2, sel is a tensor of type uint8/uint16
            /uint32/uint64.
            - In mode 1 or 2, elements are consumed continuously
            between iterations.
          src0 : A tensor for the start element of source operand 0
            - Note: dst must have the same data type as src0 and src1.
          src1 : A tensor for the start element of source operand 1In mode 0
          or 2, the argument is a tensor.
          In mode 1, the argument is a scalar or and immediate
          of type int/float.
            - Note: dst must have the same data type as src0 and src1
          repeat_times : Number of iteration repeats.
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand.
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 0.
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 1.
            - Note: This parameter is invalid in mode 1.
        Returns:
          None
        Restrictions:
          - The mode argument must be an immediate.
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
            an immediate of type int, or an Expr of type int16/int32/int64
            /uint16/uint32/uint64.
            If an immediate is passed, 0 is not supported.
          - dst_rep_stride, src0_rep_stride, or src1_rep_stride is
          within the range [0, 255], in the unit of 32 bytes.
            The argument is a scalar of type int16/int32/int64/uint16/uint32
            /uint64, an immediate of type int, or an Expr
            of type int16/int32/int64/uint16/uint32/uint64.
          - dst and src0 must be different tensors or the same element
          of the same tensor, not different elements of the
            same tensor. This also applies to dst and src1.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address overlapping).
            The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the destination
              operand, that is, the destination operand of the Nth iteration
              is the source operand of the (N+1)th iteration,
              address overlapping is not supported.
        Example:
        @code
        #Ascend310
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 0, dst_ub, sel, src0_ub, src1_ub, 1,
                                    8, 8, 8)

            #Inputs:
            (float16)
            #src0_ub =
            {1,2,3,...,128}
            #src1_ub =
            {2,2,2,...,2}
            #sel:
            [2,0,0,0,0,0,0,0]
            #Returns:
            #dst_ub =
            {2,2,2,...,2}

        #Ascend910
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 0, dst_ub, sel, src0_ub, src1_ub, 1,
                                    8, 8, 8)

            #Inputs :
            (float16)
            #src0_ub =
            {1,2,3,...,128}
            #src1_ub =
            {2,2,2,...,2}
            #sel:
            [2,0,0,0,0,0,0,0]
            #Returns:
            #dst_ub =
            {2,2,2,...,2}

        #Hi3796
            # mode 0
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "hisi-es"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 0, dst_ub, sel, src0_ub, src1_ub, 1,
                                8, 8, 8)

            #Inputs :
            (float16)
            #src0_ub =
            {1,2,3,...,128}
            #src1_ub =
            {2,2,2,...,2}
            #sel:
            [2,0,0,0,0,0,0,0]
            #Returns:
            #dst_ub =
            {2,2,2,...,2}

            # mode 1
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "hisi-es"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                    scope=tik.scope_ubuf)
            src1 = tik_instance.Scalar(dtype="float16", init_value=5.2)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                    scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                    scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 1, dst_ub, sel, src0_ub, src1,1, 8, 8, 8)

            # mode 2
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "hisi-es"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                        scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 2, dst_ub, sel, src0_ub, src1_ub, 1,
                                8, 8, 8)

        #Ascend610-aic
            # mode 0
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "aic"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 0, dst_ub, sel, src0_ub, src1_ub, 1,
                                8, 8, 8)

            #Inputs :
            (float16)
            #src0_ub =
            {1,2,3,...,128}
            #src1_ub =
            {2,2,2,...,2}
            #sel :
            [2,0,0,0,0,0,0,0]
            #Returns:
            #dst_ub =
            {2,2,2,...,2}

            # mode 1
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "aic"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                        scope=tik.scope_ubuf)
            src1 = tik_instance.Scalar(dtype="float16", init_value=5.2)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 1, dst_ub, sel, src0_ub, src1, 1,8, 8, 8)

            # mode 2
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "aic"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 2, dst_ub, sel, src0_ub, src1_ub, 1,
                                8, 8, 8)

        #Ascend610-vec
            # mode 0
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "vec"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                            scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 0, dst_ub, sel, src0_ub, src1_ub, 1,
                                8, 8, 8)

            #Inputs :
            (float16)
            #src0_ub =
            {1,2,3,...,128}
            #src1_ub =
            {2,2,2,...,2}
            #sel:
            [2,0,0,0,0,0,0,0]
            #Returns:
            #dst_ub =
            {2,2,2,...,2}

            # mode 1
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "vec"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                        scope=tik.scope_ubuf)
            src1 = tik_instance.Scalar(dtype="float16", init_value=5.2)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 1, dst_ub, sel, src0_ub, src1, 1, 8,8, 8)

            # mode 2
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v200", "vec"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                        scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            sel = tik_instance.Tensor("uint16", (8,), name="sel",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_sel(128, 2, dst_ub, sel, src0_ub, src1_ub, 1,
                                8, 8, 8)
            @endcode
        """
        return self.vec_sel_(
            mask, mode, dst,
            sel, src0, src1, repeat_times,
            dst_rep_stride, src0_rep_stride, src1_rep_stride)

    @source_info_decorator()
    @debug.vec_conv_decorator
    def vec_conv(self, mask,  # pylint: disable=R0913
                 round_mode, dst, src, repeat_times,
                 dst_rep_stride, src_rep_stride, deqscale=None,
                 ldst_high_half=False):
        r"""
        Converts the precision based on the data types of the src and dst ...
        Description:
          Converts the precision based on the data types of the src and
          dst tensors.
        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or
            a Python immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst
            and src, \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars
            or immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element
            is participated in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h
              and mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          round_mode : Rounding mode. The following string-based
          configurations are supported:
            - '': no rounding
            - 'none': no rounding
            - 'round': rounding to the nearest even number (rint in C language)
            - 'floor': rounding down (floor in C language)
            - 'ceil' or 'ceiling': rounding up (ceil in C language)
            - 'away-zero': rounding away from 0 (round in C language)
            - 'to-zero': rounding to 0 (trunc in C language)
            - 'odd': rounding to the nearest odd number (Von Neumann rounding)
          dst : Destination operand.
          src : Source operand.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand.
          src_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand.
        Kwargs:
          deqscale : Quantization scale, which is an auxiliary conversion
          parameter. Defaults to None.
            The argument is a scalar of type float16 or an immediate
            of type float.
          ldst_high_half : A bool specifying whether dst_list or src_list
          stores or comes from the upper or lower half
            of each block. Defaults to False.True indicates the upper half,
            and False indicates the lower half.
            - Note: This parameter defines different functions for
            different combinations,
              indicating the storage and read of dst_list and
              src_list respectively.
              This parameter is supported only by some precision
              conversion modes.
              - Ascend 310 AI Processor does not support this parameter.
              - Ascend 910 AI Processor does not support this parameter.

        Returns:
          None
        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/int32/int64/uint16
          /uint32/uint64. If repeat_times is an immediate,
          0 is not supported.
          - The degree of parallelism of each repeat depends on the data
          precision and chip version. For example:
          64 source or destination elements are operated in each repeat
          during f32-to-f16 conversion.
          - Instructions dst_rep_stride and src_rep_stride are within
          the range [0, 255], in the unit of 32 bytes. The argument
          is a scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type
          int16/int32/int64/uint16/uint32/uint64.
          - The supported data types of dst and src are related to the chip
          version. If the data types are not supported,
          the tool reports an error.
          - dst and src must be the same element of different tensors or
          the same tensor, not different elements of the same tensor.
          - Ascend 610 AI Processor (AI Core):
            - When int16 is converted to uint8 or int8, 16 elements are
            processed for each block. The ldst_high_half
            parameter is supported, indicating that 16 result elements
            are stored in the upper or lower half of each
            block of dst. For example, if ldst_high_half is set to True,
            the result is written to the lower half of
            each block of dst.
            - When int16 is converted to uint8 or int8, the deqscale(uint64)
            parameter is supported. Bit[46] indicates
            whether the result contains the sign bit. For example, when
            src.dtype is set to int16 and dst.dtype is set
            to int8, make sure that deqscale[46] is set to 0b1. When
            src.dtype is set to int16 and dst.dtype is set
            to uint8, make sure that deqscale[46] is set to 0b0. Otherwise,
            the result is incorrect.
          - Ascend 610 AI Processor (Vector Core):
            - When int16 is converted to uint8 or int8, 16 elements are
            processed for each block. The ldst_high_half
            parameter is supported, indicating that 16 result elements are
            stored in the upper or lower half of each
            block of dst. For example, if ldst_high_half is set to True,
            the result is written to the lower half of
            each block of dst.
            - When int16 is converted to uint8 or int8, the deqscale(uint64)
            parameter is supported. Bit[46] indicates
            whether the result contains the sign bit. For example, when
            src.dtype is set to int16 and dst.dtype is set
            to int8, make sure that deqscale[46] is set to 0b1. When
            src.dtype is set to int16 and dst.dtype is set
            to uint8, make sure that deqscale[46] is set to 0b0. Otherwise,
            the result is incorrect.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address overlapping).
          The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the destination
            operand, that is, the destination operand of the Nth iteration is
            the source operand of the (N+1)th iteration,
            address overlapping is not supported.

        Example:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("int32", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_conv(64, "round", dst_ub, src_ub, 2, 8, 4)
        """
        default_blk_stride = 1
        stride_unit = 0
        instr_name = "vec_conv"
        with self.context.freeze():  # pylint: disable=E1101
            return self.vconv(
                mask, round_mode, dst, src, repeat_times, default_blk_stride,
                default_blk_stride, dst_rep_stride, src_rep_stride,
                deqscale, ldst_high_half, stride_unit, instr_name)

    @source_info_decorator()
    @debug.vec_ln_high_preci_decorator
    def vec_ln_high_preci(self, mask, dst, src,  # pylint: disable=R0913
                          work_tensor, repeat_times,
                          dst_rep_stride, src_rep_stride):
        r'''
        Computes the natural logarithm element-wise
        Description:
          Computes the natural logarithm element-wise:
          \f$dst_i = \log_{e}(src_i),i\in [0,PAR]\f$


          The computation result using this API has higher accuracy in certain
          ranges than the vec_ln API.

        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16,
            the first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.
              - Value range: For 16-bit dst and src, \f$mask\in[1, 128]\f$.
              For 32-bit dst and src, \f$mask\in[1, 64]\f$.
               For 64-bit dst and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$ \in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : Destination operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          src : Source operand, which is the start element of the tensor.
          For details about the supported data
          precision, see the specific instruction.
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations of
           the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations of
           the source operand
          NOTICE :
            The following describes only the dst, src, and work_tensor
            parameters.
            dst, src, and work_tensor are tensors of the
            same data type, float16.
            - If the source operand tensor has an offset, the passing formats
            are as follows: tensor[offset1:offset2]
            means that starting from offset1 and ending at offset2.
            tensor[offset1:] means starting from offset1.
            tensor[offset] means that only one element is passed.
            (In this case, the tensor is impossible to be sliced
            and a runtime error will be reported. Therefore. this format
            is not allowed.)
            - If the source operand tensor does not have an offset, the tensor
            can be passed directly.
            - work_tensor:
              - work_tensor is a user-defined temporary buffer space for
              storing the intermediate result. The space is
              limited to scope_ubuf and is used for internal computation only.
              - work_tensor buffer space calculation:
                - Calculate the minimum buffer space required for src
                computation based on repeat_times, mask, and
                src_rep_stride as follows: src_extent_size =
                (repeat_times - 1) * src_rep_stride * 16 + mask_len  In
                consecutive mask mode, mask_len is the mask value itself.
                In bit-wise mask mode, mask_len is the
                mask value corresponding to the most significant bit.
                - Round up the minimum buffer space required for src
                computation to the least multiple of 32 bytes:
                wk_size_unit = (src_extent_size + 15)//16 * 16
                - Calculate the size of work_tensor as follows:
                work_tensor = 10 * wk_size_unit
            - Example of work_tensor buffer space calculation:
              - If mask = 128, rep_times = 2, and src_rep_stride = 8,
              then mask_len = 128,
              src_extent_size = (2 - 1) * 8 * 16 + mask_len = 256 and
              wk_size_unit = (src_extent_size + 15)//16 * 16 = 256.
              Therefore, the size of work_tensor is 10 * wk_size_unit = 2560.
              - If mask = [3, 2**64-1], rep_times = 2, src_rep_stride = 8,
              then mask_len = 66. The most significant bit
              of mask is 3, corresponding to binary bit 11. Therefore,
              mask_len = 2 + 64,
              src_extent_size = (2 - 1) * 8 * 16 + mask_len = 194,
              wk_size_unit = (src_extent_size+ 15)//16 * 16 = 208.
              Therefore, the size of work_tensor is 10 * wk_size_unit = 2080.

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 255], in
          the unit of 32 bytes. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/
          int32/int64/uint16/uint32/uint64.
          - dst and src must be declared in scope_ubuf, and the supported
          data types are related to the chip version. If
          the data types are not supported, the tool reports an error.
          - dst has the same data type as src.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.
          - dst, src, and work_tensor must be declared in scope_ubuf.
          - The space of the dst, src, and work_tensor tensors cannot overlap.
          - If any value of src is not positive, an unknown result
          may be produced.

        Examples:
          #In consecutive mask mode
              from te import tik
              tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
              kernel_name = "vln_rep_8_src_rep_8"
              # Define the tensors.
              src_gm = tik_instance.Tensor("float16", (2, 128), tik.scope_gm,
                                            "src_gm")
              src_ub = tik_instance.Tensor("float16", (2, 128), tik.scope_ubuf,
                                            "src_ub")
              dst_ub = tik_instance.Tensor("float16", (2, 128), tik.scope_ubuf,
                                            "dst_ub")
              # The size of work_tensor is 10 times src.
              work_tensor = tik_instance.Tensor("float16", (10, 2, 128),
                                        tik.scope_ubuf, "work_tensor")
              dst_gm = tik_instance.Tensor("float16", (2, 128), tik.scope_gm,
                                            "dst_gm")
              # Move the user input to the source UB.
              tik_instance.data_move(src_ub, src_gm, 0, 1, 16, 0, 0)
              tik_instance.vec_dup(128, dst_ub, 0.0, 2, 8)
              mask = 128
              rep_times = 2
              src_rep_stride = 8
              dst_rep_stride = 8
              # If the input work_tensor has an index, use the
              # work_tensor[index:] format.
              tik_instance.vec_ln_high_preci(mask, dst_ub, src_ub,
              work_tensor[0:], rep_times, dst_rep_stride, src_rep_stride)
              # Move the computation result to the destination GM.
              tik_instance.data_move(dst_gm, dst_ub, 0, 1, 16, 0, 0)
              tik_instance.BuildCCE(kernel_name, [src_gm], [dst_gm])

              #Inputs:
              #[  [1, 2, 3, 4, ......, 128],  [1, 2, 3, 4, ......, 128]]
              #Returns:
              #[  [0, 0.6931, 1.0986, 1.3863, ......, 4.8520],
              # [0, 0.6931, 1.0986, 1.3863, ......, 4.8520]]

          #In bit-wise mask mode
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            kernel_name = "vln_rep_8_src_rep_8"
            # Define the tensors.
            src_gm = tik_instance.Tensor("float16", (2, 128), tik.scope_gm,
                                        "src_gm")
            src_ub = tik_instance.Tensor("float16", (2, 128), tik.scope_ubuf,
                                        "src_ub")
            dst_ub = tik_instance.Tensor("float16", (2, 128), tik.scope_ubuf,
                                        "dst_ub")
            # Calculate the size of work_tensor.
            rep_times = 2
            src_rep_stride = 8
            dst_rep_stride = 8
            mask = [3, 2**64-1]
            mask_len = 66
            src_extent_size = (rep_times - 1)*src_rep_stride*16 + mask_len
            wk_size_unit = (src_extent_size + 15)//16*16
            wk_size = 10*wk_size_unit
            work_tensor = tik_instance.Tensor("float16", (wk_size, ),
                                        tik.scope_ubuf, "work_tensor")
            dst_gm = tik_instance.Tensor("float16", (2, 128,), tik.scope_gm,
                                            "dst_gm")
            # Move the user input to the source UB.
            tik_instance.data_move(src_ub, src_gm, 0, 1, 16, 0, 0)
            # Initialize the destination UB.
            tik_instance.vec_dup(128, dst_ub, 0.0, 2, 8)
            tik_instance.vec_ln_high_preci(mask, dst_ub, src_ub, work_tensor,
                rep_times, dst_rep_stride, src_rep_stride)
            # Move the computation result to the destination GM.
            tik_instance.data_move(dst_gm, dst_ub, 0, 1, 16, 0, 0)
            tik_instance.BuildCCE(kernel_name, [src_gm], [dst_gm])

            #Inputs:
            #[  [1, 2, 3, 4, ......, 65, 66, 67, ......, 128],
            #[1, 2, 3, 4, ......, 65, 66, 67, ......, 128]]
            #Returns:
            #[[0, 0.6931, 1.0986, 1.3863, ......, 4.1744, 4.1897, 0, ......,0],
            #[0, 0.6931, 1.0986, 1.3863, ......, 4.1744, 4.1897, 0, ......, 0]]
        '''
        default_blk_stride = 1
        multi_factor = 10  # work_tensor size need more than 10 * src_extend
        instr_name = "vec_ln_high_preci"
        with self.context.freeze():  # pylint: disable=E1101
            # params check
            check_high_preci_param(dst, src, work_tensor, repeat_times,
                                   dst_rep_stride, src_rep_stride)
            TikCheckUtil.check_equality(work_tensor.dtype, dst.dtype,
                                        "work_tensor's dtype must be same "
                                        "with dst's dtype")
            TikCheckUtil.check_equality(api_check_support("tik."
                                                          + instr_name,
                                                          dst.dtype), True,
                                        INSTR_DTYPE_SUPPORT_STATEMENT.
                                        format(dst.dtype, instr_name))
            # check mask
            check_mask_valid(mask, tensor_bit_len=max(get_bit_len(src.dtype),
                                                      get_bit_len(dst.dtype)))

            check_over_high_preci(
                mask, dst, src, work_tensor, repeat_times, dst_rep_stride,
                src_rep_stride, Expr(dst.offset).eval_value(),
                Expr(src.offset).eval_value(),
                Expr(work_tensor.offset).eval_value(), multi_factor,
                name=instr_name)

            self.vln_calculate_by_taylor(
                instr_name, mask, dst, src, work_tensor,
                repeat_times, default_blk_stride,
                default_blk_stride, dst_rep_stride, src_rep_stride)
