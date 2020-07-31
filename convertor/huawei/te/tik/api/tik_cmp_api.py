"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_cmp_api.py
DESC:     make a module
CREATED:  2020-3-16 21:12:13
MODIFIED: 2020-3-16 21:12:45
"""
from ..tik_lib.tik_cmp_api_ import TikCompareApi


class TikCompareApiv1(TikCompareApi):
    """Provide compare api for open"""
    # @cond
    def __init__(self):
        super(TikCompareApiv1, self).__init__()
    # @endcond

    def vec_cmpv_lt(self, dst, src0, src1,  # pylint: disable=R0913
                    repeat_times, src0_rep_stride, src1_rep_stride):
        """
        Performs element-wise comparison to generate a 1-bit result
        Description:
          Performs element-wise comparison to generate a 1-bit result. 1'b1
          indicates true, and 1'b0 indicates false.
          Multiple comparison modes are supported.
        Args:
          dst : Destination operand, which is the start element of the tensor.
          Must be one of the following data types: uint64, uint32, uint16
          , uint8
          src0 : Source operand 0, which is the start element of the tensor.
          Must be one of the following data types:
            - Ascend 310 AI Processor: tensor of type float16
            - Ascend 910 AI Processor: tensor of type float16 or float32
          src1 : Source operand 1, which is the start element of the
          tensor.src1 has the same data type as src0.
          repeat_times : Number of iteration repeats:
            - When repeat_times = 1, the addresses of the source and
            destination operands can overlap.
            - When repeat_times > 1, the addresses of the source and
            destination operands cannot overlap.
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 0.
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 1.
          NOTICE:
          Instruction name. The following comparison modes are supported:
            - vec_cmpv_lt: indicates that src0 is less than src1.
            - vec_cmpv_gt: indicates that src0 is greater than src1.
            - vec_cmpv_ge: indicates that src0 is greater than or equal to src1
            - vec_cmpv_eq: indicates that src0 is equal to src1.
            - vec_cmpv_ne: indicates that src0 is not equal to src1.
            - vec_cmpv_le: indicates that src0 is less than or equal to src1.
        Return:
          None
        Restrictions:
          - The mask parameter is unavailable.
          - dst is generated continuously. For example, if the source operand
          is of type float16 while the destination
          operand is of type uint16, eight elements of dst are skipped between
          adjacent iterations.
          - src0_rep_stride or src1_rep_stride is within the range [0, 255],
          in the unit of blocks. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type
          int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the (N+1)th
            iteration, address overlapping is not supported.
        Example:
        @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (16,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            # Initializes dst_ub to all 5s.
            tik_instance.vec_dup(16, dst_ub, 5, 1, 1)
            tik_instance.vec_cmpv_lt(dst_ub, src0_ub, src1_ub, 1, 8, 8)

        #Inputs:
         (float16)
        #src0_ub =
        {1,2,3,...,128}
        #src1_ub =
        {2,2,2,...,2}
        #Returns:
        #dst_ub =
        {2,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5}
        @endcode
        """
        default_blk_stride = 1
        print_name = "vec_cmpv_lt"
        return self._vcmpv_elewise_func('vcmpv_lt', dst, src0, src1,
                                        repeat_times, default_blk_stride,
                                        default_blk_stride, src0_rep_stride,
                                        src1_rep_stride, print_name=print_name)

    def vec_cmpv_gt(self, dst, src0, src1,  # pylint: disable=R0913
                    repeat_times, src0_rep_stride, src1_rep_stride):
        """
        Performs element-wise comparison to generate a 1-bit result
        Description:
          Performs element-wise comparison to generate a 1-bit result. 1'b1
          indicates true, and 1'b0 indicates false.
          Multiple comparison modes are supported.
        Args:
          dst : Destination operand, which is the start element of the tensor.
          Must be one of the following data types: uint64, uint32, uint16
          , uint8
          src0 : Source operand 0, which is the start element of the tensor.
          Must be one of the following data types:
            - Ascend 310 AI Processor: tensor of type float16
            - Ascend 910 AI Processor: tensor of type float16 or float32
          src1 : Source operand 1, which is the start element of the
          tensor.src1 has the same data type as src0.
          repeat_times : Number of iteration repeats:
            - When repeat_times = 1, the addresses of the source and
            destination operands can overlap.
            - When repeat_times > 1, the addresses of the source and
            destination operands cannot overlap.
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 0.
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 1.
          NOTICE:
          Instruction name. The following comparison modes are supported:
            - vec_cmpv_lt: indicates that src0 is less than src1.
            - vec_cmpv_gt: indicates that src0 is greater than src1.
            - vec_cmpv_ge: indicates that src0 is greater than or equal to src1
            - vec_cmpv_eq: indicates that src0 is equal to src1.
            - vec_cmpv_ne: indicates that src0 is not equal to src1.
            - vec_cmpv_le: indicates that src0 is less than or equal to src1.
        Return:
          None
        Restrictions:
          - The mask parameter is unavailable.
          - dst is generated continuously. For example, if the source operand
          is of type float16 while the destination
          operand is of type uint16, eight elements of dst are skipped between
          adjacent iterations.
          - src0_rep_stride or src1_rep_stride is within the range [0, 255],
          in the unit of blocks. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type
          int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the (N+1)th
            iteration, address overlapping is not supported.
        Example:
        @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (16,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            # Initializes dst_ub to all 5s.
            tik_instance.vec_dup(16, dst_ub, 5, 1, 1)
            tik_instance.vec_cmpv_lt(dst_ub, src0_ub, src1_ub, 1, 8, 8)

        #Inputs:
         (float16)
        #src0_ub =
        {1,2,3,...,128}
        #src1_ub =
        {2,2,2,...,2}
        #Returns:
        #dst_ub =
        {2,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5}
        @endcode
        """
        default_blk_stride = 1
        print_name = "vec_cmpv_gt"
        return self._vcmpv_elewise_func('vcmpv_gt', dst, src0, src1,
                                        repeat_times, default_blk_stride,
                                        default_blk_stride, src0_rep_stride,
                                        src1_rep_stride, print_name=print_name)

    def vec_cmpv_ge(self, dst, src0, src1,  # pylint: disable=R0913
                    repeat_times, src0_rep_stride, src1_rep_stride):
        """
        Performs element-wise comparison to generate a 1-bit result
        Description:
          Performs element-wise comparison to generate a 1-bit result. 1'b1
          indicates true, and 1'b0 indicates false.
          Multiple comparison modes are supported.
        Args:
          dst : Destination operand, which is the start element of the tensor.
          Must be one of the following data types: uint64, uint32, uint16
          , uint8
          src0 : Source operand 0, which is the start element of the tensor.
          Must be one of the following data types:
            - Ascend 310 AI Processor: tensor of type float16
            - Ascend 910 AI Processor: tensor of type float16 or float32
          src1 : Source operand 1, which is the start element of the
          tensor.src1 has the same data type as src0.
          repeat_times : Number of iteration repeats:
            - When repeat_times = 1, the addresses of the source and
            destination operands can overlap.
            - When repeat_times > 1, the addresses of the source and
            destination operands cannot overlap.
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 0.
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 1.
          NOTICE:
          Instruction name. The following comparison modes are supported:
            - vec_cmpv_lt: indicates that src0 is less than src1.
            - vec_cmpv_gt: indicates that src0 is greater than src1.
            - vec_cmpv_ge: indicates that src0 is greater than or equal to src1
            - vec_cmpv_eq: indicates that src0 is equal to src1.
            - vec_cmpv_ne: indicates that src0 is not equal to src1.
            - vec_cmpv_le: indicates that src0 is less than or equal to src1.
        Return:
          None
        Restrictions:
          - The mask parameter is unavailable.
          - dst is generated continuously. For example, if the source operand
          is of type float16 while the destination
          operand is of type uint16, eight elements of dst are skipped between
          adjacent iterations.
          - src0_rep_stride or src1_rep_stride is within the range [0, 255],
          in the unit of blocks. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type
          int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the (N+1)th
            iteration, address overlapping is not supported.
        Example:
        @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (16,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            # Initializes dst_ub to all 5s.
            tik_instance.vec_dup(16, dst_ub, 5, 1, 1)
            tik_instance.vec_cmpv_lt(dst_ub, src0_ub, src1_ub, 1, 8, 8)

        #Inputs:
         (float16)
        #src0_ub =
        {1,2,3,...,128}
        #src1_ub =
        {2,2,2,...,2}
        #Returns:
        #dst_ub =
        {2,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5}
        @endcode
        """
        default_blk_stride = 1
        print_name = "vec_cmpv_ge"
        return self._vcmpv_elewise_func('vcmpv_ge', dst, src0, src1,
                                        repeat_times, default_blk_stride,
                                        default_blk_stride, src0_rep_stride,
                                        src1_rep_stride, print_name=print_name)

    def vec_cmpv_eq(self, dst, src0, src1,  # pylint: disable=R0913
                    repeat_times, src0_rep_stride, src1_rep_stride):
        """
        Performs element-wise comparison to generate a 1-bit result
        Description:
          Performs element-wise comparison to generate a 1-bit result. 1'b1
          indicates true, and 1'b0 indicates false.
          Multiple comparison modes are supported.
        Args:
          dst : Destination operand, which is the start element of the tensor.
          Must be one of the following data types: uint64, uint32, uint16
          , uint8
          src0 : Source operand 0, which is the start element of the tensor.
          Must be one of the following data types:
            - Ascend 310 AI Processor: tensor of type float16
            - Ascend 910 AI Processor: tensor of type float16 or float32
          src1 : Source operand 1, which is the start element of the
          tensor.src1 has the same data type as src0.
          repeat_times : Number of iteration repeats:
            - When repeat_times = 1, the addresses of the source and
            destination operands can overlap.
            - When repeat_times > 1, the addresses of the source and
            destination operands cannot overlap.
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 0.
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 1.
          NOTICE:
          Instruction name. The following comparison modes are supported:
            - vec_cmpv_lt: indicates that src0 is less than src1.
            - vec_cmpv_gt: indicates that src0 is greater than src1.
            - vec_cmpv_ge: indicates that src0 is greater than or equal to src1
            - vec_cmpv_eq: indicates that src0 is equal to src1.
            - vec_cmpv_ne: indicates that src0 is not equal to src1.
            - vec_cmpv_le: indicates that src0 is less than or equal to src1.
        Return:
          None
        Restrictions:
          - The mask parameter is unavailable.
          - dst is generated continuously. For example, if the source operand
          is of type float16 while the destination
          operand is of type uint16, eight elements of dst are skipped between
          adjacent iterations.
          - src0_rep_stride or src1_rep_stride is within the range [0, 255],
          in the unit of blocks. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type
          int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the (N+1)th
            iteration, address overlapping is not supported.
        Example:
        @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (16,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            # Initializes dst_ub to all 5s.
            tik_instance.vec_dup(16, dst_ub, 5, 1, 1)
            tik_instance.vec_cmpv_lt(dst_ub, src0_ub, src1_ub, 1, 8, 8)

        #Inputs:
         (float16)
        #src0_ub =
        {1,2,3,...,128}
        #src1_ub =
        {2,2,2,...,2}
        #Returns:
        #dst_ub =
        {2,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5}
        @endcode
        """
        default_blk_stride = 1
        print_name = "vec_cmpv_eq"
        return self._vcmpv_elewise_func('vcmpv_eq', dst, src0, src1,
                                        repeat_times, default_blk_stride,
                                        default_blk_stride, src0_rep_stride,
                                        src1_rep_stride, print_name=print_name)

    def vec_cmpv_ne(self, dst, src0, src1,  # pylint: disable=R0913
                    repeat_times, src0_rep_stride, src1_rep_stride):
        """
        Performs element-wise comparison to generate a 1-bit result
        Description:
          Performs element-wise comparison to generate a 1-bit result. 1'b1
          indicates true, and 1'b0 indicates false.
          Multiple comparison modes are supported.
        Args:
          dst : Destination operand, which is the start element of the tensor.
          Must be one of the following data types: uint64, uint32, uint16
          , uint8
          src0 : Source operand 0, which is the start element of the tensor.
          Must be one of the following data types:
            - Ascend 310 AI Processor: tensor of type float16
            - Ascend 910 AI Processor: tensor of type float16 or float32
          src1 : Source operand 1, which is the start element of the
          tensor.src1 has the same data type as src0.
          repeat_times : Number of iteration repeats:
            - When repeat_times = 1, the addresses of the source and
            destination operands can overlap.
            - When repeat_times > 1, the addresses of the source and
            destination operands cannot overlap.
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 0.
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 1.
          NOTICE:
          Instruction name. The following comparison modes are supported:
            - vec_cmpv_lt: indicates that src0 is less than src1.
            - vec_cmpv_gt: indicates that src0 is greater than src1.
            - vec_cmpv_ge: indicates that src0 is greater than or equal to src1
            - vec_cmpv_eq: indicates that src0 is equal to src1.
            - vec_cmpv_ne: indicates that src0 is not equal to src1.
            - vec_cmpv_le: indicates that src0 is less than or equal to src1.
        Return:
          None
        Restrictions:
          - The mask parameter is unavailable.
          - dst is generated continuously. For example, if the source operand
          is of type float16 while the destination
          operand is of type uint16, eight elements of dst are skipped between
          adjacent iterations.
          - src0_rep_stride or src1_rep_stride is within the range [0, 255],
          in the unit of blocks. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type
          int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the (N+1)th
            iteration, address overlapping is not supported.
        Example:
        @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (16,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            # Initializes dst_ub to all 5s.
            tik_instance.vec_dup(16, dst_ub, 5, 1, 1)
            tik_instance.vec_cmpv_lt(dst_ub, src0_ub, src1_ub, 1, 8, 8)

        #Inputs:
         (float16)
        #src0_ub =
        {1,2,3,...,128}
        #src1_ub =
        {2,2,2,...,2}
        #Returns:
        #dst_ub =
        {2,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5}
        @endcode
        """
        default_blk_stride = 1
        print_name = "vec_cmpv_ne"
        return self._vcmpv_elewise_func('vcmpv_ne', dst, src0, src1,
                                        repeat_times, default_blk_stride,
                                        default_blk_stride, src0_rep_stride,
                                        src1_rep_stride, print_name=print_name)

    def vec_cmpv_le(self, dst, src0, src1,  # pylint: disable=R0913
                    repeat_times, src0_rep_stride, src1_rep_stride):
        """
        Performs element-wise comparison to generate a 1-bit result
        Description:
          Performs element-wise comparison to generate a 1-bit result. 1'b1
          indicates true, and 1'b0 indicates false.
          Multiple comparison modes are supported.
        Args:
          dst : Destination operand, which is the start element of the tensor.
          Must be one of the following data types: uint64, uint32, uint16
          , uint8
          src0 : Source operand 0, which is the start element of the tensor.
          Must be one of the following data types:
            - Ascend 310 AI Processor: tensor of type float16
            - Ascend 910 AI Processor: tensor of type float16 or float32
          src1 : Source operand 1, which is the start element of the
          tensor.src1 has the same data type as src0.
          repeat_times : Number of iteration repeats:
            - When repeat_times = 1, the addresses of the source and
            destination operands can overlap.
            - When repeat_times > 1, the addresses of the source and
            destination operands cannot overlap.
          src0_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 0.
          src1_rep_stride : Block-to-block stride between adjacent iterations
          of source operand 1.
          NOTICE:
          Instruction name. The following comparison modes are supported:
            - vec_cmpv_lt: indicates that src0 is less than src1.
            - vec_cmpv_gt: indicates that src0 is greater than src1.
            - vec_cmpv_ge: indicates that src0 is greater than or equal to src1
            - vec_cmpv_eq: indicates that src0 is equal to src1.
            - vec_cmpv_ne: indicates that src0 is not equal to src1.
            - vec_cmpv_le: indicates that src0 is less than or equal to src1.
        Return:
          None
        Restrictions:
          - The mask parameter is unavailable.
          - dst is generated continuously. For example, if the source operand
          is of type float16 while the destination
          operand is of type uint16, eight elements of dst are skipped between
          adjacent iterations.
          - src0_rep_stride or src1_rep_stride is within the range [0, 255],
          in the unit of blocks. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type
          int16/int32/int64/uint16/uint32/uint64.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the (N+1)th
            iteration, address overlapping is not supported.
        Example:
        @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src0_ub = tik_instance.Tensor("float16", (128,), name="src0_ub",
                                            scope=tik.scope_ubuf)
            src1_ub = tik_instance.Tensor("float16", (128,), name="src1_ub",
                                            scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("uint16", (16,), name="dst_ub",
                                            scope=tik.scope_ubuf)
            # Initializes dst_ub to all 5s.
            tik_instance.vec_dup(16, dst_ub, 5, 1, 1)
            tik_instance.vec_cmpv_lt(dst_ub, src0_ub, src1_ub, 1, 8, 8)

        #Inputs:
         (float16)
        #src0_ub =
        {1,2,3,...,128}
        #src1_ub =
        {2,2,2,...,2}
        #Returns:
        #dst_ub =
        {2,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5}
        @endcode
        """
        default_blk_stride = 1
        print_name = "vec_cmpv_le"
        return self._vcmpv_elewise_func('vcmpv_le', dst, src0, src1,
                                        repeat_times, default_blk_stride,
                                        default_blk_stride, src0_rep_stride,
                                        src1_rep_stride, print_name=print_name)
