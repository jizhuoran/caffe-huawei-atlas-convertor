"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_data_operation_api.py
DESC:     provide data padding, data conversion, data move related instructions
CREATED:  2020-3-16 21:12:13
MODIFIED: 2020-3-16 21:12:45
"""
from te import tvm
from te.platform import cce_params

from .tik_tensor import Tensor
from .tik_scalar import Scalar
from .. import debug
from ..tik_lib.tik_expr import Expr
from ..tik_lib.tik_data_operation_api_ import TikDataOpApi, \
    check_dma_instr_params
from ..tik_lib.tik_source_info import source_info_decorator
from ..tik_lib.tik_check_util import TikCheckUtil
from ..common.common_util import check_vec_trans_params_range, \
    check_vec_trans_overflow
from ..common.util import check_integer_in_range, \
    is_immediate_number, check_scalar_dtype
from ..tik_lib.tik_params import PIPE_V, ONE_IR, PER_TRANSPOSE_DATA_SIZE, \
    MAX_SID
from ..tik_lib.tik_api_constants import DTYPE_MAP, SCOPE_MAP


class TikDataOpApiv1(TikDataOpApi):
    """
    Data convert, Data fill, Data move Api for open
    """
    # @cond
    def __init__(self):
        super(TikDataOpApiv1, self).__init__()
    # @endcond

    # source_info
    # decorator
    def vec_dup(self, mask, dst, scalar, repeat_times,  # pylint: disable=R0913
                dst_rep_stride):
        r"""
        Copies a Scalar variable or an immediate for multiple times ...
        Description:
          Copies a Scalar variable or an immediate for multiple times and fill
          it in the vector (PAR indicates the degree of parallelism).


          \f$dst_i = scalar_i ,i\in[0,repeat_times*PAR]\f$
        Args:
          mask : 128-bit mask. If a bit is set to 0, the corresponding element
          of the vector is masked in the
          computation. If a bit is set to 1, the corresponding element of the
          vector participates in the computation.
          The consecutive mode and bit-wise mode are supported.
            - In consecutive mode, the argument is a single scalar or a Python
            immediate, specifying the number of
            elements participated in the computation. If mask = 16, the
            first 16 elements participate in the computation.
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range:
            For 16-bit dst and src, \f$mask\in[1, 128]\f$. For 32-bit dst and
            src, \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to
            the lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.Value range:
            For 16-bit dst and src are 16 bits, mask_h and
            mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, \f$mask_l \in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, \f$mask_l \in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : A tensor for the start element of the destination operand.
          Must be one of the following data types: uint16, int16, float16
          , uint32, int32, float32
          scalar : A scalar or an immediate, for the source operand to be
          copied. Has the same dtype as dst.
          repeat_times : Number of iteration repeats. The addresses of the
          source and destination operands change upon
          every iteration. The value range is [0, 255]. If repeat_times is an
          immediate, 0 is not supported. The argument
          is a scalar of type int16/int32/int64/uint16/uint32/uint64, an
          immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64
          dst_rep_stride : Block-to-block stride in a single iteration of the
          destination operand. The value range is
          [0, 255], in the unit of 32 bytes. The argument is a scalar
          of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64
        Returns:
          None
        Restrictions:
          Ensure that the online scalar argument does not
          exceed the value range.
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            dst_ub = tik_instance.Tensor("float16", (128, ), tik.scope_ubuf,
                                        "dst_ub")
            src_scalar = tik_instance.Scalar(init_value=0, dtype="float16")
            tik_instance.vec_dup(128, dst_ub, src_scalar, 1, 8)
            @endcode
        """
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_dup"
        return self._vector_scalar_elewise_func('vector_dup', mask, dst,
                                                scalar, repeat_times,
                                                default_blk_stride,
                                                dst_rep_stride, stride_unit,
                                                print_name=print_name)

    @source_info_decorator()
    @debug.vec_trans_scatter_decorator
    def vec_trans_scatter(  # pylint: disable=R0913
            self, dst_high_half, src_high_half, dst_list, src_list,
            repeat_times, dst_rep_stride, src_rep_stride):
        """
        Converts NCHW into NC1HWC0.
        Description:
          Converts NCHW into NC1HWC0. If the data type is float32, int32
          , uint32, int16, unint16, or float16, then C0 is 16.
          If the data type is uint8 or int8, then C0 is 32.
        Args:
          dst_high_half : Whether to store the data of dst_list[*] to the upper
          or lower half of the block. Only the int8
          or uint8 data type is supported.
          The data type of this parameter can only be bool.
          The options are as follows:
            - True: upper half
            - False: lower half
          src_high_half : A bool specifying whether to read the data of
          src_list[*] from the upper or lower half of the block.
          Only the int8 or uint8 data type is supported.
          The options are as follows:
            - True: upper half
            - False: lower half
          dst_list : A list of elements, specifying the vector destination
          operand. Each element marks the start of a
          destination operand.The supported data types are as follows:
            - Ascend 310 AI Processor: tensor (int8/uint8/int16/uint16/float16)
            - Ascend 910 AI Processor: tensor (int8/uint8/int16/uint16/float16)
            - HiSilicon SoC (ES): tensor (int8/uint8/int16/uint16/float16/int32
            /uint32/float32)
            - Ascend 610 AI Processor (AI Core): tensor (int8/uint8/int16
            /uint16/float16/int32/uint32/float32)
            - Ascend 610 AI Processor (Vector Core): tensor (int8/uint8/int16
            /uint16/float16/int32/uint32/float32)
          src_list : A list of elements, specifying the vector source operand.
          Each element marks the start of a
          destination operand. Has the same data type as dst_list.
          repeat_times : Number of iteration repeats, in the unit of blocks.
          The value range is [0, 255]. The argument
          is a scalar of type int16/int32/int64/uint16/uint32/uint64, an
          immediate of type int, or an Expr of type int16/int32/int64/uint16
          /uint32/uint64.
            - Notes:
              - When repeat_times=1, the valid start of a destination or source
               operand is the start of dst_list or src_list
              plus dst_rep_stride or src_rep_stride.
              - When repeat_times > 1, the valid start of a destination or
              source operand in the first repeat is the start of
              dst_list or src_list. In the second repeat, dst_rep_stride or
              src_rep_stride needs to be added. This rule applies.
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand, in the unit of
          blocks. The value range is [0, 65535]. The argument is a scalar
          of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/int32/int64/uint16
          /uint32/uint64.
          src_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand, in the unit of
          blocks.The value range is [0, 65535]. The argument is a scalar
          of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
        Returns:
          None
        Restrictions:
          - Generally, each element in src_list or dst_list is configured as
          the start of each HW plane.
          - For better performance, it is recommended that dstHighHalf and
          srcHighHalf be fixed when the data type is int8
          or uint8, and be changed after the repeat in the H and W directions.
          - The mask value does not affect the execution of the API.
          - To save memory space, you can define a tensor shared by the
          source and destination operands (by address overlapping).
          The general instruction restrictions are as follows.
            - For a single repeat (repeat_times = 1), the source operand
            sequence and the target operand sequence must
            be completely the same. Partial overlapping is not supported.
            Instead, each block must be the same.
            - For multiple repeats (repeat_times > 1), if there is a dependency
             between the source operand sequence and
            the destination operand sequence, that is, the destination operand
            of the Nth iteration is the source operand
            of the (N+1)th iteration, address overlapping is not supported.
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            dst_ub = tik_instance.Tensor("float16", (256,), tik.scope_ubuf,
                                        "dst_ub")
            src_ub = tik_instance.Tensor("float16", (256,), tik.scope_ubuf,
                                        "src_ub")
            dst_list = [dst_ub[16 * i] for i in range(16)]
            src_list = [src_ub[16 * i] for i in range(16)]
            tik_instance.vec_trans_scatter(True, False, dst_list, src_list,
                                        1, 0, 0)
            @endcode
        """
        name = "vec_trans_scatter"
        with self.context.freeze():  # pylint: disable=E1101
            return self.vnchwconv(
                dst_high_half, src_high_half, dst_list, src_list,
                repeat_times, dst_rep_stride, src_rep_stride, name)

    @source_info_decorator()
    @debug.datamove_decorator
    def data_move(self, dst,  # pylint: disable=R0913
                  src, sid, nburst, burst, src_stride, dst_stride,
                  *args, **argv):
        """
        Moves data based on the data types of the src and dst tensors.
        Description:
          Moves data based on the data types of the src and dst tensors.
          The options are as follows:
            - UB->UB
            - UB->OUT
            - OUT->UB
            - OUT->L1
        Args:
          dst : Destination operand.
          src : Source operand.
          sid : A scalar, an immediate, or an Expr of type int32, specifying
          the SMMU ID, which is hardware-related.
          The value range is [0, 15]. The value 0 is recommended.
          nburst : A scalar, an immediate, or an Expr of type int32, specifying
           the number of the data segments to be
          transmitted. The value range is [1, 4095].
          burst : Burst length.
          The value range is [1, 65535], in the unit
          of 32 bytes.The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/int32/int64
          /uint16/uint32/uint64.
          src_stride : Burst-to-burst stride of the source tensor.
          The value range is [0, 65535]. The argument is a scalar of
          type int16/int32/int64/uint16/uint32/uint64, an immediate of type int
          , or an Expr of type int16/int32/int64/uint16/uint32/uint64.
          dst_stride : Burst-to-burst stride of the destination tensor.
          The value range is [0, 65535]. The argument is a scalar of type
          int16/int32/int64/uint16/uint32/uint64, an immediate of type int, or
          an Expr of type int16/int32/int64/uint16/uint32/uint64.
          NOTICE:
            - *args : Number of extended parameters
            - **args : Extended parameters

        Returns:
          None
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (128,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.data_move(dst_ub, src_ub, 0, 1, 128 // 16, 0, 0)
            @endcode
        """
        dma_list = {
            "L0C UB": (2, 'copy_matrix_cc_to_ubuf'),  # V
            "UB L0C": (2, 'copy_matrix_ubuf_to_cc'),  # special
            "UB UB": (2, 'copy_ubuf_to_ubuf'),
            "L1 UB": (4, 'copy_cbuf_to_ubuf'),
            "OUT L1": (5, 'copy_gm_to_cbuf'),  # LSU2
            "OUT UB": (5, 'copy_gm_to_ubuf'),
            "UB OUT": (6, 'copy_ubuf_to_gm'),  # LSU3
            "UB L1": (6, 'copy_ubuf_to_cbuf')
        }
        check_integer_in_range(sid, range(MAX_SID),
                               "sid should be in the range of [0, 15], input "
                               "value is {}".format(sid))
        check_dma_instr_params(dst, src, nburst, burst, src_stride, dst_stride)
        type_args = [sid, nburst, burst, src_stride, dst_stride]
        self._gen_data_move_code(src, dst, dma_list, type_args, args, argv)

    @staticmethod
    def _vec_trans_params_check(dst, src, repeat_times,  # pylint: disable=R0913
                                dst_rep_stride, src_rep_stride):
        """used to check the params of vec_trans

        Parameters
        ----------
        dst: the des address
        src: the src address
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator between adjacent iterations,
                         offset unit is 512B
        src_rep_stride : offset of src operator between adjacent iterations,
                         offset unit is 512B

        Returns
        -------
        Nones
        """
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "dst should be tensor, "
                                      "but input type: {}".format(type(dst)))
        TikCheckUtil.check_type_match(src, Tensor,
                                      "src should be tensor, "
                                      "but input type: {}".format(type(src)))
        TikCheckUtil.check_type_match(repeat_times, (int, Scalar, Expr),
                                      "repeat_times should be int, Scalar or "
                                      "Expr, but input type: {}".
                                      format(type(repeat_times)))
        TikCheckUtil.check_type_match(dst_rep_stride, (int, Scalar, Expr),
                                      "dst_rep_stride should be int, Scalar or "
                                      "Expr, but input type: {}".
                                      format(type(dst_rep_stride)))
        TikCheckUtil.check_type_match(src_rep_stride, (int, Scalar, Expr),
                                      "src_rep_stride should be int, Scalar or "
                                      "Expr, but input type: {}".
                                      format(type(src_rep_stride)))

        # check tensor scope
        dst_scope = SCOPE_MAP[dst.scope]
        TikCheckUtil.check_in_range(dst_scope, ['ubuf'],
                                    "dst tensor scope must be ubuf, "
                                    "but input scope: {}".format(dst_scope))
        src_scope = SCOPE_MAP[src.scope]
        TikCheckUtil.check_in_range(src_scope, ['ubuf'],
                                    "src tensor scope must be ubuf, "
                                    "but input scope: {}".format(dst_scope))
        check_scalar_dtype(repeat_times,
                           "repeat_times should be a scalar of int/uint")
        check_scalar_dtype(dst_rep_stride,
                           "dst_rep_stride should be a scalar of int/uint")
        check_scalar_dtype(src_rep_stride,
                           "src_rep_stride should be a scalar of int/uint")

        check_vec_trans_params_range(repeat_times, dst_rep_stride,
                                     src_rep_stride)

        # check data type, all supported data type
        dst_src_map = ["u16u16", "s16s16", "f16f16"]
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]
        if dtype_str not in dst_src_map:
            TikCheckUtil.raise_error(
                "dtype of dst and src should be u16u16, s16s16 or f16f16")

        # check tensor overflow
        if is_immediate_number([repeat_times, dst_rep_stride, src_rep_stride]):
            dst_offset = Expr(dst.indice.offset).eval_value()
            src_offset = Expr(src.indice.offset).eval_value()
            check_vec_trans_overflow(dst.indice.origin_shape,
                                     src.indice.origin_shape,
                                     dst_offset, src_offset, repeat_times,
                                     dst_rep_stride, src_rep_stride)

    @source_info_decorator()
    @debug.vec_trans_decorator
    def vec_trans(self, dst, src, repeat_times,  # pylint: disable=R0913
                  dst_rep_stride, src_rep_stride):
        """
        Consecutively transposes 16x16 two-dimensional matrix data blocks ...
        Description:
          Consecutively transposes 16x16 two-dimensional matrix data blocks for
           repeat_times times. Each iteration
          operates 256 consecutive address space data blocks. The addresses
          between different iterations can be
          inconsecutive. The address space between adjacent iterations is
          specified by dst_rep_stride and src_rep_stride.
        Args:
          dst : A tensor for the destination operand. Must be one of
          the following data types: int16, uint16, float16.
            - For the Ascend 310 AI Processor, the address
            must be 32-byte aligned.
            - For the Ascend 910 AI Processor, the address
            must be 32-byte aligned.
          src : A tensor for the source operand. Must be one
          of the following data types: int16, uint16, float16.
            - For the Ascend 310 AI Processor, the address
            must be 32-byte aligned.
            - For the Ascend 910 AI Processor, the address
            must be 32-byte aligned.
          repeat_times : Number of repeats. The argument is
          a scalar of type int/uint, an immediate of type int, or
          an Expr of type int/uint. The value range is [1, 4095].
          dst_rep_stride : dst address space between adjacent iterations (unit:
           512 bytes). The argument is a scalar
            of type int/uint, an immediate of type int, or
            an Expr of type int/uint. The value range is [0, 4095].
          src_rep_stride : src address space between adjacent iterations (unit:
           512 bytes). The argument is a scalar
            of type int/uint, an immediate of type int, or an Expr
            of type int/uint. The value range is [0, 4095].
        Returns:
          None
        Restrictions:
          To save memory space, you can define a tensor shared by the source
          and destination operands (by address overlapping).
          The general instruction restriction is that the source operand must
          completely overlap the destination operand.
        Example:
            @code
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            dst_ub = tik_instance.Tensor("float16", (1,16,16), name='dst_ub',
                                        scope=tik.scope_ubuf)
            src_ub = tik_instance.Tensor("float16", (1,16,16), name='src_ub',
                                        scope=tik.scope_ubuf)
            tik_instance.vec_trans(dst_ub, src_ub, 1, 1, 1)
            #Inputs:
            #src_ub=
            [1,2,3,4,...,256]
            #Return:
            #dst_ub=
            [1,17,33,49,...,256]
            @endcode
        """
        self._vec_trans_params_check(dst, src, repeat_times,
                                     dst_rep_stride, src_rep_stride)

        # data type is B16, 2Bytes
        extent_value = PER_TRANSPOSE_DATA_SIZE*2
        extent = Expr(extent_value)

        # deal with input params are scalar
        with self.new_scope():
            # 2 is size of b16, 2 Bytes
            # one ir is call_extern
            with self.for_range(0, repeat_times) as index:
                self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
                dst_offset = Expr(dst_rep_stride*PER_TRANSPOSE_DATA_SIZE*index)
                src_offset = Expr(src_rep_stride*PER_TRANSPOSE_DATA_SIZE*index)
                self.emit(tvm.call_extern(dst.dtype, "vtranspose",
                                          dst.reinterpret_cast_to("uint16").
                                          access_ptr("w", extent=extent.get(),
                                                     offset=dst_offset.get()),
                                          src.reinterpret_cast_to("uint16").
                                          access_ptr("r", extent=extent.get(),
                                                     offset=src_offset.get())),
                          ONE_IR)
