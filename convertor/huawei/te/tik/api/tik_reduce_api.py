"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_reduce_api.py
DESC:     provide reduce vector instructions
CREATED:  2020-3-16 21:12:13
MODIFIED: 2020-3-16 21:12:45
"""
from te import tvm  # pylint: disable=C0302
from te.platform import cce_params
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_910
from te.platform.cce_params import HI3796CV300ES
from te.platform.cce_params import ASCEND_610
from te.platform.cce_params import ASCEND_620
from te.platform.cce_conf import intrinsic_check_support
from te.tik.common.util import DTYPE_SIZE, reduce_mul, ceil_div, get_bit_len, \
    is_immediate_number, check_scalar_dtype, check_integer_in_range,\
    check_scalar_int32
from te.tik.common.common_util import check_tensor_overflow, \
    check_address_overlapping, align_start_pos, check_space_overflow, \
    vreduce_create_mask, check_vreduce_repeat_times, check_dtype_overflow
from ..tik_lib.tik_reduce_api_ import TikReduceApi
from ..tik_lib.tik_source_info import source_info_decorator
from ..tik_lib.tik_reduce_api_ import vec_reduce_add_check_decorator
from ..tik_lib.tik_expr import Expr
from ..api.tik_tensor import Tensor
from .. import debug
from ..api.tik_scalar import mask_concat, Scalar
from ..tik_lib.tik_check_util import TikCheckUtil
from ..tik_lib.tik_util import type_convert
from ..tik_lib.tik_api_util import set_ctrl_counter_mask, \
    reset_ctrl_counter_mask, check_repeat_times, check_address_align
from ..tik_lib.tik_params import ONE_BYTE_BIT_LEN, ONE_REP_BYTE_SIZE, \
    PIPE_V, ONE_BLK_SIZE, TWO_IR, BLK_NUM_PER_REP, MAX_REPEAT_TIMES, \
    MASK_VALUE_ZERO, MIN_REPEAT_TIMES, MIN_INDEX, VREDUCE_PER_REP_OUTPUT, \
    VREDUCE_DEFAULT_SRC_BLK_STRIDE, MAX_REP_STRIDE_DOUBLE_BYTE,\
    VREDUCE_MIN_REPEAT_TIMES, VREDUCE_DEFAULT_DST_REP_STRIDE, \
    VREDUCE_DEFAULT_SRC_REP_STRIDE, VREDUCE_DEFAULT_DST_BLK_STRIDE, \
    INSTR_DTYPE_SUPPORT_STATEMENT
from ..common.tik_get_soc_name import get_soc_name
_DEFAULT_STRIDE = 1


def _check_vec_reduce_add_two_operator_overlap(src,  # pylint: disable=R0913
                                               dst, mask, repeat_times,
                                               src_rep_stride, dst_rep_stride,
                                               msg):
    """check vec_reduce_add if two of the tensor overlapping"""
    src_blk_stride = 1
    default_dst_blk_stride = 0
    block_len = 1
    default_nblock = 1
    if src.buffer == dst.buffer:
        if all(isinstance(value, int) for
               value in (repeat_times, src_rep_stride)):
            if repeat_times == 1:
                _default_dst_blk_stride = src_blk_stride
            _default_dst_rep_stride = dst_rep_stride
            check_address_overlapping(
                "vec_reduce_add", mask, dst, src, default_nblock,
                ONE_REP_BYTE_SIZE // max(
                    get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                block_len, ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                repeat_times, default_dst_blk_stride, src_blk_stride,
                dst_rep_stride, src_rep_stride,
                Expr(dst.offset).eval_value(),
                Expr(src.offset).eval_value(), msg=msg)


def _check_vec_reduce_add_operator_overlap(dst, src,  # pylint: disable=R0913
                                           work_tensor, mask, repeat_times,
                                           src_rep_stride):
    default_dst_rep_stride = 1
    _check_vec_reduce_add_two_operator_overlap(src, work_tensor, mask,
                                               repeat_times, src_rep_stride,
                                               default_dst_rep_stride,
                                               "src and work_tensor")
    default_dst_rep_stride = 0
    _check_vec_reduce_add_two_operator_overlap(src, dst, mask, repeat_times,
                                               src_rep_stride,
                                               default_dst_rep_stride,
                                               "src and dst")
    if work_tensor.buffer == dst.buffer:
        work_tensor_start = Expr(work_tensor.offset).eval_value()
        work_tensor_stop = Expr(
            repeat_times + work_tensor.offset).eval_value()
        dst_start = Expr(dst.offset).eval_value()
        dst_stop = dst_start + 1
        if all((value is not None) for value in
               (work_tensor_start, work_tensor_stop, dst_start, dst_stop)):
            if max(work_tensor_start, dst_start) < \
                    min(work_tensor_stop, dst_stop):
                TikCheckUtil.raise_error("vec_reduce_add work_tensor and dst "
                                         "address overlapping error.")


def _vreduce_it1_compute(tik_instance, func, mask,  # pylint: disable=R0913
                         work_tensor, src, repeat_times, src_rep_stride):
    """
    run the first iteration of vreduce

    Parameters
    ----------
    tik_instance: Tik, tik_instance
    func: function, vcmax/vcmin
    mask: int/Scalar/list, effective operation on element.
    src: Tensor, the source tensor
    work_tensor: Tensor, temporary memory, for internal calculation
    repeat_times: int/Scalar, the times of instruction run
    src_rep_stride: int/Scalar, the stride between each repeat in src

    Returns
    -------

    """
    return tik_instance.run_reduce_func_(func, mask, work_tensor, src,
                                         repeat_times, src_rep_stride)


def _vreduce_it4_compute(tik_instance,  # pylint: disable=R0913
                         pre_output_count, func, work_tensor,
                         cur_start_pos, pre_start_pos):
    """
    run the third iteration of vreduce

    Parameters
    ----------
    tik_instance: Tik, tik_instance
    pre_output_count: int/Scalar, the output elements count in it2
    func: function, vcmax/vcmin
    work_tensor: Tensor, temporary memory, for internal calculation
    cur_start_pos: int/Scalar, the start position in work_tensor
    pre_start_pos: int/Scalar, the stride between each repeat in src

    Returns
    -------

    """
    it3_mask = tik_instance.creat_mask_(pre_output_count //
                                        VREDUCE_PER_REP_OUTPUT)
    tik_instance._run_vreduce_max_min(  # pylint: disable=W0212
        func, it3_mask, work_tensor,
        work_tensor, VREDUCE_MIN_REPEAT_TIMES,
        VREDUCE_DEFAULT_SRC_REP_STRIDE, cur_start_pos, pre_start_pos)


class TikReduceApiv1(TikReduceApi):
    """Provide whole-reduce group-reduce pair-reduce api for open"""

    # @cond
    def __init__(self):
        super(TikReduceApiv1, self).__init__()
    # @endcond

    def vec_cpadd(self,  # pylint: disable=R0913
                  mask,
                  dst,
                  src,
                  repeat_times,
                  dst_rep_stride,
                  src_rep_stride):
        r"""
        Adds elements (odd and even) between adjacent pairs
        Description:
          Adds elements (odd and even) between adjacent pairs:
          \f$dst_i = \sum_{k}^2src_{k+i*PAR/2} ,i\in [0,PAR/2]\f$
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
            The argument is an immediate of type int or a scalar
            of type int64/int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$. For 32-bit dst
            and src, \f$mask\in[1, 64]\f$. For 64-bit dst
            and src, \f$mask\in[1, 32]\f$.
            - In bit-wise mode, the argument can be a list of two scalars or
            immediates of type int64. The format is
            [mask_h, mask_l]. If a bit is set to 0, the corresponding element
            of the vector is masked in the computation.
            If a bit is set to 1, the corresponding element of the vector
            participates in the computation. mask_h
            corresponds to the upper 64 elements, and mask_l corresponds to the
             lower 64 elements. For example, if
            mask = [0, 8] (8 = 0b1000), only the fourth element is participated
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : A tensor for the start element of the destination operand
          src : A tensor for the start element of the source operand
          repeat_times : Number of iteration repeats
          dst_rep_stride : Block-to-block stride between adjacent iterations
          of the destination operand
          src_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand

        Returns:
          None

        Restrictions:
          - repeat_times is within the range [0, 255]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/
          uint64, an immediate of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64. If an immediate
          is passed, 0 is not supported.
          - The degree of parallelism of each repeat depends on the data type
          and chip version. The following uses PAR
          to describe the degree of parallelism.
          - dst_rep_stride or src_rep_stride is within the range [0, 65535].
          The argument is a scalar of type int16/int32/int64/uint16/uint32
          /uint64, an immediate of type int, or an Expr of type int16/int32
          /int64/uint16/uint32/uint64. If dst_rep_stride is set
          to 0, the value 1 is used.
          - Note that the implementation of dst_rep_stride is
          different. The unit is 128 bytes.
          - To save memory space, you can define a tensor shared by the source
          and destination operands (by address
          overlapping). The general instruction restrictions are as follows.
          Note that each instruction might have
          specific restrictions.
            - For a single repeat (repeat_times = 1), the source operand must
            completely overlap the destination operand.
            - For multiple repeats (repeat_times > 1), if there is a
            dependency between the source operand and the
            destination operand, that is, the destination operand of the Nth
            iteration is the source operand of the
            (N+1)th iteration, address overlapping is not supported.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            src_ub = tik_instance.Tensor("float16", (128,), name="src_ub",
                                        scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor("float16", (64,), name="dst_ub",
                                        scope=tik.scope_ubuf)
            tik_instance.vec_cpadd(128, dst_ub, src_ub, 1, 1, 8)
        """
        default_blk_stride = 1
        stride_unit = 0
        print_name = "vec_cpadd"
        return self._vector_pair_reduce_func('vcpadd', mask, dst, src,
                                             repeat_times, dst_rep_stride,
                                             default_blk_stride, src_rep_stride,
                                             stride_unit, print_name=print_name)

    def _gen_vec_reduce_add_vcadd_part_code(self,  # pylint: disable=R0913, R0914
                                            mask, mask_mode, dst, src,
                                            repeat_times, src_rep_stride,
                                            dst_offset=0, src_offset=0,
                                            is_mask_o=False):
        """generate vcadd code for instruction vec_reduce_add"""
        # check repeat times
        check_repeat_times(repeat_times)
        # check mask and get mask_o
        if is_mask_o:
            mask_o = mask
        else:
            mask_o = mask_concat(self, mask,
                                 mask_mode=mask_mode,
                                 tensor_bit_len=get_bit_len(src.dtype))
        # gen
        config = [repeat_times, _DEFAULT_STRIDE, _DEFAULT_STRIDE,
                  src_rep_stride]
        if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
            config.append(0)
            config.append(0)
        with self.new_scope():
            if mask_mode == "counter":
                orig_ctrl = set_ctrl_counter_mask(self)
            # 8 Block/repeat, 32Byte/Block
            src_extent = Expr(((repeat_times - 1)*src_rep_stride +
                               (8 - 1)*_DEFAULT_STRIDE + 1)*32)
            dst_extent = Expr(repeat_times*_DEFAULT_STRIDE*
                              get_bit_len(src.dtype)
                              // ONE_BYTE_BIT_LEN)
            # when repeat time >1 and the count of dst write element is not
            # the multi of ONE_BLK_SIZE
            dst_extent = Expr(ceil_div(dst_extent, ONE_BLK_SIZE)*ONE_BLK_SIZE)
            instr = tvm.call_extern(dst.dtype, "vcadd",
                                    dst.access_ptr("w", extent=dst_extent.get(),
                                                   offset=Expr(dst_offset).get()
                                                   ),
                                    src.access_ptr("r",
                                                   extent=src_extent.get(),
                                                   offset=Expr(src_offset).get()
                                                   ),
                                    *type_convert(config))
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)
            # reset CTRL SPR as orig_ctrl
            if mask_mode == "counter":
                reset_ctrl_counter_mask(self, orig_ctrl)

    def _first_add(self, mask_o, dst, src,  # pylint: disable=R0913
                   repeat_times, src_rep_stride):
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            elements_per_block = ONE_BLK_SIZE // DTYPE_SIZE[src.dtype]
            # in order to align the first address of dst, 32B align
            max_repeat_times = MAX_REPEAT_TIMES - elements_per_block
        else:
            max_repeat_times = MAX_REPEAT_TIMES - 1
        for_times = repeat_times // max_repeat_times
        left_repeat_times = repeat_times % max_repeat_times

        def _first_add_full_repeat(cur_index):
            src_start_index = cur_index*max_repeat_times*\
                              ((src_rep_stride*ONE_BLK_SIZE) //
                               DTYPE_SIZE[src.dtype])
            self._gen_vec_reduce_add_vcadd_part_code(
                mask_o, "normal", dst, src, max_repeat_times, src_rep_stride,
                dst_offset=cur_index*max_repeat_times,
                src_offset=src_start_index, is_mask_o=True)

        def _first_add_left_repeat():
            src_index = for_times*max_repeat_times*\
                        ((src_rep_stride*ONE_BLK_SIZE) //
                         DTYPE_SIZE[src.dtype])
            self._gen_vec_reduce_add_vcadd_part_code(
                mask_o, "normal", dst, src, left_repeat_times, src_rep_stride,
                dst_offset=for_times*max_repeat_times, src_offset=src_index,
                is_mask_o=True)

        if is_immediate_number(repeat_times):
            for index in range(for_times):
                _first_add_full_repeat(index)
            if left_repeat_times > 0:
                _first_add_left_repeat()
        else:
            with self.for_range_(0, for_times) as index:
                _first_add_full_repeat(index)
            with self.if_scope_(left_repeat_times > 0):
                _first_add_left_repeat()

    def _second_add(self, pre_elements, dst, src):
        operator_byte_size = get_bit_len(src.dtype) // ONE_BYTE_BIT_LEN
        elements_per_repeat = ONE_REP_BYTE_SIZE // operator_byte_size
        vcadd_mask = pre_elements % elements_per_repeat
        tmp_repeat_times = pre_elements // elements_per_repeat

        def _second_add_with_left_repeat():
            if is_immediate_number(tmp_repeat_times) and tmp_repeat_times == 0:
                pass
            else:
                self._gen_vec_reduce_add_vcadd_part_code(
                    elements_per_repeat, "normal", dst, src,
                    tmp_repeat_times, BLK_NUM_PER_REP)
            # in order to align the first address of dst, 32B align
            tmp_dst = self.Tensor_(dtype=dst.dtype,  # pylint: disable=E1101
                                   shape=(1,), scope=scope_ubuf, name="tmp_dst")
            self._gen_vec_reduce_add_vcadd_part_code(
                vcadd_mask, "normal", tmp_dst, src,
                MIN_REPEAT_TIMES, BLK_NUM_PER_REP,
                src_offset=elements_per_repeat*tmp_repeat_times)
            dst.set_as(tmp_dst[MIN_INDEX], dst_offset=tmp_repeat_times)

        if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
            default_repeat = 1
            self._gen_vec_reduce_add_vcadd_part_code(
                pre_elements, "counter", dst, src,
                default_repeat, BLK_NUM_PER_REP)
        else:
            if is_immediate_number(vcadd_mask):
                if vcadd_mask == 0:
                    self._gen_vec_reduce_add_vcadd_part_code(
                        elements_per_repeat, "normal", dst, src,
                        tmp_repeat_times, BLK_NUM_PER_REP)
                else:
                    _second_add_with_left_repeat()
            else:
                with self.if_scope_(vcadd_mask == MASK_VALUE_ZERO):
                    self._gen_vec_reduce_add_vcadd_part_code(
                        elements_per_repeat, "normal", dst, src,
                        tmp_repeat_times, BLK_NUM_PER_REP)
                with self.else_scope_():
                    _second_add_with_left_repeat()
        return ceil_div(pre_elements, elements_per_repeat)

    def _final_add(self, pre_elements, dst, work_tensor):
        if is_immediate_number(pre_elements):
            if pre_elements == 1:
                # move result from work_tensor to dst
                dst[MIN_INDEX].set_as(work_tensor[MIN_INDEX])
            elif pre_elements > 1:
                self._gen_vec_reduce_add_vcadd_part_code(
                    pre_elements, "normal", dst, work_tensor,
                    MIN_REPEAT_TIMES, BLK_NUM_PER_REP)
        else:
            with self.if_scope_(pre_elements == 1):
                # move result from work_tensor to dst
                dst[MIN_INDEX].set_as(work_tensor[MIN_INDEX])
            with self.else_scope_():
                self._gen_vec_reduce_add_vcadd_part_code(
                    pre_elements, "normal", dst, work_tensor,
                    MIN_REPEAT_TIMES, BLK_NUM_PER_REP)

    def _gen_vec_reduce_add_code_when_repeat_scalar(self,  # pylint: disable=R0913
                                                    mask_o, dst, src,
                                                    work_tensor, repeat_times,
                                                    src_rep_stride):
        """when repeat_times is scalar, generate vec_reduce_add code"""
        with self.if_scope_(repeat_times == 1):
            # first add
            self._gen_vec_reduce_add_vcadd_part_code(
                mask_o, "normal", dst, src, repeat_times, src_rep_stride,
                is_mask_o=True)
        with self.if_scope_(repeat_times > 1):
            # first add
            self._first_add(mask_o, work_tensor, src, repeat_times,
                            src_rep_stride)
            # second add
            second_result_num = self._second_add(repeat_times, work_tensor,
                                                 work_tensor)
            # the final add
            self._final_add(second_result_num, dst, work_tensor)

    def _gen_vec_reduce_add_code_when_repeat_imme(self,  # pylint: disable=R0913
                                                  mask_o, dst, src, work_tensor,
                                                  repeat_times, src_rep_stride):
        """when repeat_times is imme, generate vec_reduce_add code"""
        if repeat_times == 1:
            # first add
            self._gen_vec_reduce_add_vcadd_part_code(
                mask_o, "normal", dst, src, repeat_times, src_rep_stride,
                is_mask_o=True)
        else:
            # first add
            self._first_add(mask_o, work_tensor, src, repeat_times,
                            src_rep_stride)
            # second add
            second_result_num = self._second_add(repeat_times, work_tensor,
                                                 work_tensor)
            # the final add
            self._final_add(second_result_num, dst, work_tensor)

    @source_info_decorator()
    @debug.vec_reduce_add_decorator
    @vec_reduce_add_check_decorator
    def vec_reduce_add(self, mask, dst, src,  # pylint: disable=R0913
                       work_tensor, repeat_times, src_rep_stride):
        r"""
        Adds all input data.
        Description:
          Adds all input data:
          \f$dst_i = \sum_{k}^{PAR}src_{k+i*PAR} ,i\in [0,repeat-1]\f$


          Each two data pieces are added in binary tree mode.
          Assume that the source operand is 256 pieces
          of float16 data [data0, data1, data2, ..., data255], the
          computation can be completed in two repeats. The computation
          process is as follows:
          - [data0,data1,data2...data127] is the source operand of the first
          repeat. Result 01 is obtained through
          the following calculation method:
            - Add data0 and data1 to obtain data00, add data2 and data3
            to obtain data01, ..., add data124 and data125 to
            obtain data62, and add data126 and data127 to obtain data63.
            - Add data00 and data01 to obtain data000, add data02 and data03
            to obtain data001, ..., add data62 and
            data63 to obtain data031.
            - This rule applies until result01 is obtained.
          - [data128,data1,data2...data255] is the source operand of the
          second repeat. Result 02 is obtained.
          - Add result01 and result02 to obtain [data], whose destination
          operand is one float16.

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
            and src, \f$mask\in[1, 128]\f$.
            For 32-bit dst and src, \f$mask\in[1, 64]\f$. For 64-bit dst
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
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h
              and mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst and
            src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : A tensor for the start element of the destination operand
          src : A tensor for the start element of the source operand
          work_tensor : Intermediate results are stored during command
          execution to calculate the required operation space.
          Pay attention to the space size. For details, see the restrictions
          for each command.
          repeat_times : Number of iteration repeats
          src_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand
          NOTICE :
            The dst, src, and work_tensor operands have the same data type:
            - Ascend 310 AI Processor: tensors of type float16 or float32
            - Ascend 910 AI Processor: tensors of type float16 or float32
            - HiSilicon SoC (ES): tensors of type float16
            - Ascend 610 AI Processor (AI Core): tensors of type float16
            or float32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            or float32

        Returns:
          None

        Restrictions:
          - The work_tensor space requires at least repeat_times
          elements. For example, when repeat_times=120, the shape
          of work_tensor has at least 120 elements.
          - repeat_times is within the range [1, 4095]. The argument is a
          scalar of type int32, an immediate of type int,
          or an Expr of type int32.
          - src_rep_stride is within the range [0, 65535]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/int32/int64/uint16
          /uint32/uint64.
          - Note that if the calculation result overflows during the
          calculation of adding every two elements, there are
          two processing modes: return the defined maximum value or return
          inf/nan. The mode is selected by the inf/nan
          control bit. When the max-value mode is used, if the sum of float16
          data is greater than 65504, the output will
          be 65504. For example, the source operand
          is [60000, 60000, -30000, 100], 60000 + 60000 > 65504, meaning that
          the result overflows. In this case, the maximum value 65504 will be
          used as the result. Similarly,
          -30000 + 100 = -29900, 65504 - 29900 = 35604.
          - Address overlapping among src, dst, and work_tensor is not allowed.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            dst_ub = tik_instance.Tensor("float16", (32,), tik.scope_ubuf,
                                        "dst_ub")
            src_ub = tik_instance.Tensor("float16", (256,), tik.scope_ubuf,
                                        "src_ub")
            work_tensor_ub = tik_instance.Tensor("float16", (32,),
                                        tik.scope_ubuf, "work_tensor_ub")
            tik_instance.vec_reduce_add(128, dst_ub, src_ub, work_tensor_ub,
                                        2, 8)
            #Inputs:
            #src_ub=[1,1,1,,...,1]
            #Return:
            #dst_ub=[256]
        """
        # check mask and get mask_o
        mask_o = mask_concat(self, mask,
                             tensor_bit_len=get_bit_len(src.dtype))
        # check tensor overflow
        # cause tensor has at least one element, dst does not need to check.
        src_blk_stride = 1
        check_tensor_overflow((src,), mask, repeat_times, (src_blk_stride,),
                              (src_rep_stride,), ("src",))
        # check work_tensor size, when work_tensor's src and dst overlap
        work_tensor_need = Expr(repeat_times + work_tensor.offset).eval_value()
        if work_tensor_need is not None:
            work_tensor_total_size = reduce_mul(work_tensor.indice.origin_shape)
            TikCheckUtil.check_ge(
                work_tensor_total_size, work_tensor_need,
                "work_tensor tensor overflow, instruction need {} but only "
                "{}".format(work_tensor_need, work_tensor_total_size))

        # check operator address 32B aligned when core_arch is v100
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            check_address_align((work_tensor, dst, src),
                                ("work_tensor", "dst", "src"))

        # check tensor overlap
        _check_vec_reduce_add_operator_overlap(dst, src, work_tensor, mask,
                                               repeat_times, src_rep_stride)

        # gen code
        with self.context.freeze():  # pylint: disable=E1101
            if is_immediate_number(repeat_times):
                self._gen_vec_reduce_add_code_when_repeat_imme(
                    mask_o, dst, src, work_tensor, repeat_times, src_rep_stride)
            else:
                self._gen_vec_reduce_add_code_when_repeat_scalar(
                    mask_o, dst, src, work_tensor, repeat_times, src_rep_stride)

    @source_info_decorator()
    @debug.vec_all_reduce_decorator("vcmax")
    def vec_reduce_max(self, mask, dst, src, work_tensor,  # pylint: disable=R0913
                       repeat_times, src_rep_stride, cal_index=False):
        """
        Obtains the maximum value and its corresponding index position ...
        Description:
          Obtains the maximum value and its corresponding index position among
          the input data. If there are multiple
          maximum values, determine which one to be returned by referring to
          the restrictions.


          \f$\left (dst_0, dst_1\right )^T=\max src_{k+i*PAR}
          ,k\in [0,PAR-1],i\in[0,repeat-1]\f$
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
              - Value range: For 16-bit dst and src are 16 bits, mask_h
              and mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : A tensor for the start element of the destination operand
          src : A tensor for the start element of the source operand
          work_tensor : Intermediate results are stored during command
          execution to calculate the required operation space.
          Pay attention to the space size. For details, see the
          restrictions for each command.
          repeat_times : Number of iteration repeats The argument is a
          scalar of type int32, an immediate of type int,
          or an Expr of type int32. Immediate is recommended because it
          provides higher performance.
          src_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr
          of type int16/int32/int64/uint16/uint32/uint64.
          NOTICE :
            The dst, src, and work_tensor operands have the same data type:
            - Ascend 310 AI Processor: tensors of type float16 or float32
            - Ascend 910 AI Processor: tensors of type float16 or float32
            - HiSilicon SoC (ES): tensors of type float16
            - Ascend 610 AI Processor (AI Core): tensors of type float16
            or float32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            or float32

        Kwargs:
          cal_index : A bool that specifies whether to obtain the index with
          the minimum value (supported only by
          vec_reduce_max and vec_reduce_min). Defaults to False.
          The options are as follows:
            - True: Both the maximum and minimum indexes are obtained.
            - False: Only the maximum and minimum values are obtained.
            The corresponding indexes are not obtained.

        Returns:
          None

        Restrictions:
          - The argument of repeat_times is a scalar of type int32, an
          immediate of type int, or an Expr of type int32.
            - When cal_index is set to False, repeat_times
            is within the range [1, 4095].
            - When cal_index is set to True:
              - If the operand data type is int16, the maximum value of
              the operand is 32767, meaning that a maximum of
              255 iterations are supported. Therefore, repeat_times is
              within the range [1, 255].
              - If the operand data type is float16, the maximum value of the
              operand is 65504, meaning that a maximum of
              511 iterations are supported. Therefore, repeat_times is
              within the range [1, 511].
              - Similarly, if the operand data type is float32, repeat_times
              is within the range [1, 4095].
          - src_rep_stride is within the range [0, 65535]. The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/int32/int64/uint16
          /uint32/uint64.
          - The storage sequence of the dst result is: maximum value,
          corresponding index. In the result, index data is
          stored as integers. For example, if the defined data type of dst
          is float16, but that of index is uint16, an
          error occurs when the index data is read in float16 format.
          Therefore, the reinterpret_cast_to() method needs
          to be called to convert the float16 index data to corresponding
          integers.
          - Restrictions for the work_tensor space are as follows:
            - If cal_index is set to False, at least (repeat_times x 2)
            elements are required. For example, when
            repeat_times=120, the shape of work_tensor has at least 240
            elements.
            When cal_index is set to True, the space size is calculated by
            using the following formula.
            For details about examples, see Example.
            @code
                # DTYPE_SIZE indicates the data type size, in bytes.For
                # example, float16 occupies 2 bytes. elements_per_block
                # indicates the number of elements required by each block.
                elements_per_block = 32 // DTYPE_SIZE[dtype]
                # elements_per_repeat indicates the number of elements
                #required for each repeat.
                elements_per_repeat = 256 // DTYPE_SIZE[dtype]
                # Number of elements generated in the first iteration.
                it1_output_count = 2*repeat_times
                # Offset of the start position of the second iteration.ceil_div
                # is used to perform division and round up the result.
                it2_align_start = ceil_div(it1_output_count,
                                    elements_per_block)*elements_per_block
                # Number of elements generated in the second iteration.
                it2_output_count = ceil_div(it1_output_count,
                                            elements_per_repeat)*2
                # Offset of the start position of the third iteration.
                it3_align_start = ceil_div(it2_output_count,
                                    elements_per_block)*elements_per_block
                # Number of elements generated in the third iteration.
                it3_output_count = ceil_div(it2_output_count,
                                            elements_per_repeat)*2
                # Offset of the start position of the fourth iteration.
                it4_align_start = ceil_div(it3_output_count,
                                        elements_per_block)*elements_per_block
                # Number of elements generated in the fourth iteration.
                it4_output_count = ceil_div(it3_output_count,
                                        elements_per_repeat)*2
                # Finally required work_tensor size
                final_work_tensor_need_size = it2_align_start + it3_align_start
                 + it4_align_start + it4_output_count
            @endcode
            - Address overlapping between dst and work_tensor is not allowed.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            dst_ub = tik_instance.Tensor("float16", (2,), tik.scope_ubuf,
                                        "dst_ub")
            src_ub = tik_instance.Tensor("float16", (256,), tik.scope_ubuf,
                                        "src_ub")
            work_tensor_ub = tik_instance.Tensor("float16", (18,),
                                        tik.scope_ubuf, "work_tensor_ub")
            tik_instance.vec_reduce_max(128, dst_ub, src_ub, work_tensor_ub,
                                        2, 8, cal_index=True)

            #[Example 1]
            #src, work_tensor, and dst are tensors of type float16. src has
            #shape (65, 128), and repeat_times of
            #vec_reduce_max/vec_reduce_min is 65.
            #The following is an API calling example:
                tik_instance.vec_reduce_max(128, dst, src, work_tensor, 65,
                                            8, cal_index=True).
            #The space of work_tensor is calculated as follows:
                elements_per_block = 16 (elements)
                elements_per_repeat = 128 (elements)
                it1_output_count = 2*65 = 130 (elements)
                it2_align_start = ceil_div(130, 16)*16 = 144 (elements)
                it2_output_count = ceil_div(130, 128)*2 = 4 (elements)
                it3_align_start = ceil_div(4, 16)*16 = 16 (elements)
                it3_output_count = ceil_div(4, 128)*2 = 2 (elements)
            #The final maximum value and its index can be obtained after three
            #iterations. The required space of work_tensor
            #is it2_align_start + it3_align_start + it3_output_count =
            #144 + 16 + 2 = 162 (elements).

            #[Example 2]
              #src, work_tensor, and dst are tensors of type float16. src has
              #shape (65, 128). repeat_times of vec_reduce_max
              #and vec_reduce_min is a scalar with the value 65. If
              #repeat_times is a scalar or contains a scalar, four
              #iterations of calculation are required.
              #The following is an API calling example:
                  scalar = tik_instance.Scalar (init_value=65, dtype="int32")
                  tik_instance.vec_reduce_max(128, dst, src, work_tensor,
                                                scalar, 8, cal_index=True)
              #The space of work_tensor is calculated as follows:
                  elements_per_block = 16 (elements)
                  elements_per_repeat = 128 (elements)
                  it1_output_count = 2*65 = 130 (elements)
                  it2_align_start = ceil_div(130, 16)*16 = 144 (elements)
                  it2_output_count = ceil_div(130, 128)*2 = 4 (elements)
                  it3_align_start = ceil_div(4, 16)*16 = 16 (elements)
                  it3_output_count = ceil_div(4, 128)*2 = 2 (elements)
                  it4_align_start = ceil_div(2, 16)*16 = 16 (elements)
                  it4_output_count = ceil(2, 128)*2 = 2(elements)
              #In cases where repeat_times is a scalar or contains a scalar,
              # the result is obtained in the third round.
              #However, the scalar value cannot be obtained at Python
              # compilation, another round is required.
              #work_tensor = it2_align_start + it3_align_start + it4_aign_start
              #+ it4_output_count = 144 + 16 + 16 + 2 = 178 (elements)

            #[Example 3]
              #src, work_tensor, and dst are tensors of type float32. src has
              #shape (65, 64), and repeat_times of
              #vec_reduce_max/vec_reduce_min is 65.
              #The following is an API calling example:
                  tik_instance.vec_reduce_max(64, dst, src, work_tensor, 65,
                                                8, cal_index=True)
              #The space of work_tensor is calculated as follows:
                  elements_per_block = 8 (elements)
                  elements_per_repeat = 64 (elements)
                  it1_output_count = 2*65 = 130 (elements)
                  it2_align_start = ceil_div(130, 8)*8 = 136 (elements)
                  it2_output_count = ceil_div(130, 64)*2 = 6 (elements)
                  it3_align_start = ceil_div(6, 8)*8 = 8 (elements)
                  it3_output_count = ceil_div(6, 64)*2 = 2 (elements)
              #The final maximum value and its index can be obtained after
              #three iterations. The required space of work_tensor
              #is it2_align_start + it3_align_start + it3_output_count =
              # 136 + 8 + 2 = 146 (elements).

            #[Example 4]
              #src, work_tensor, and dst are float32 tensors. The shape of src
              #is (65, 64). repeat_times of vec_reduce_max
              #and vec_reduce_min is a scalar with the value 65. If
              #repeat_times is a scalar or contains a scalar,
              #four iterations of calculation are required.
              #The following is an API calling example:
                  scalar = tik_instance.Scalar (init_value=65, dtype="int32")
                  tik_instance.vec_reduce_max(64, dst, src, work_tensor,
                                                scalar, 8, cal_index=True)
              #The space of work_tensor is calculated as follows:
                  elements_per_block = 8 (elements)
                  elements_per_repeat = 64 (elements)
                  it1_output_count = 2*65 = 130 (elements)
                  it2_align_start = ceil_div(130, 8)*8 = 136 (elements)
                  it2_output_count = ceil_div(130, 64)*2 = 6 (elements)
                  it3_align_start = ceil_div(6, 8)*8 = 8 (elements)
                  it3_output_count = ceil_div(6, 64)*2 = 2 (elements)
                  it4_align_start = ceil_div(2, 8)*8 = 8 (elements)
                  it4_output_count = ceil(2, 64)*2 = 2(elements)
              #In cases where repeat_times is a scalar or contains a scalar,
              #the result is obtained in the third round.
              #However, the scalar value cannot be obtained at Python
              #compilation, another round is required.
              #work_tensor = it2_align_start + it3_align_start +
              #it4_align_start + it4_output_count = 136 + 8 + 8 + 2 = 154
        """
        self._check_vreduce_params("vcmax", dst, src, work_tensor, repeat_times,
                                   src_rep_stride, cal_index, "vec_reduce_max")
        mask_concat(self, mask,
                    tensor_bit_len=max(get_bit_len(dst.dtype),
                                       get_bit_len(src.dtype)))

        with self.context.freeze():  # pylint: disable=E1101
            if cal_index:
                if is_immediate_number(repeat_times):
                    check_dtype_overflow(mask, repeat_times, src,
                                         src_rep_stride)
                    dst_offset = Expr(dst.indice.offset).eval_value()
                    work_tensor_offset = Expr(work_tensor.indice.offset).\
                        eval_value()
                    src_offset = Expr(src.indice.offset).eval_value()
                    if is_immediate_number(src_rep_stride):
                        check_space_overflow(mask, dst, work_tensor, src,
                                             dst_offset, work_tensor_offset,
                                             src_offset, repeat_times,
                                             src_rep_stride, cal_index)
                    self._vreduce_cal_idx_imm("vcmax", mask, dst, src,
                                              work_tensor,
                                              repeat_times, src_rep_stride)
                    return
                self._vreduce_cal_idx_scalar("vcmax", mask, dst, src,
                                             work_tensor,
                                             repeat_times, src_rep_stride)
            else:
                if is_immediate_number(repeat_times):
                    dst_offset = Expr(dst.indice.offset).eval_value()
                    work_tensor_offset = Expr(work_tensor.indice.offset). \
                        eval_value()
                    src_offset = Expr(src.indice.offset).eval_value()
                    if is_immediate_number(src_rep_stride):
                        check_space_overflow(mask, dst, work_tensor, src,
                                             dst_offset, work_tensor_offset,
                                             src_offset, repeat_times,
                                             src_rep_stride, cal_index)
                    self._vreduce_no_idx_imm("vcmax", mask, dst, src,
                                             work_tensor, repeat_times,
                                             src_rep_stride)
                    return
                self._vreduce_no_idx_scalar("vcmax", mask, dst, src,
                                            work_tensor, repeat_times,
                                            src_rep_stride)

    @source_info_decorator()
    @debug.vec_all_reduce_decorator("vcmin")
    def vec_reduce_min(self, mask, dst, src, work_tensor,  # pylint: disable=R0913
                       repeat_times, src_rep_stride, cal_index=False):
        r'''
        Obtains the minimum value and its corresponding index position ...
        Description:
          Obtains the minimum value and its corresponding index position among
          the input data. If there are multiple
          minimum values, determine which one to be returned by referring to
          the restrictions.


          \f$\left (dst_0, dst_1\right )^T = \min src_{k+i*PAR}
          ,k\in [0,PAR-1],i\in[0,repeat-1]\f$

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
            The argument is an immediate of type int or a scalar of type int64
            /int32/int16.Value range: For 16-bit dst
            and src, \f$mask\in[1, 128]\f$.
            For 32-bit dst and src, \f$mask\in[1, 64]\f$. For 64-bit dst
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
             in the computation.
              - Value range: For 16-bit dst and src are 16 bits, mask_h and
              mask_l \f$\in [0, 2**64 - 1]\f$. For 32-bit dst and src,
            mask_h is 0, mask_l \f$\in [0, 2**64 - 1]\f$. For 64-bit dst
            and src, mask_h is 0, mask_l \f$\in [0, 2**32 - 1]\f$.
            - Note: mask applies to the source operand of each repeat.
          dst : A tensor for the start element of the destination operand
          src : A tensor for the start element of the source operand
          work_tensor : Intermediate results are stored during command
          execution to calculate the required operation space.
          Pay attention to the space size. For details, see the restrictions
          for each command.
          repeat_times : Number of iteration repeats The argument is a scalar
          of type int32, an immediate of type int,
          or an Expr of type int32. Immediate is recommended because it
          provides higher performance.
          src_rep_stride : Block-to-block stride between adjacent iterations
          of the source operand The argument is a
          scalar of type int16/int32/int64/uint16/uint32/uint64, an immediate
          of type int, or an Expr of type int16/int32/int64/uint16
          /uint32/uint64.
          NOTICE :
            The dst, src, and work_tensor operands have the same data type:
            - Ascend 310 AI Processor: tensors of type float16 or float32
            - Ascend 910 AI Processor: tensors of type float16 or float32
            - HiSilicon SoC (ES): tensors of type float16
            - Ascend 610 AI Processor (AI Core): tensors of type float16
            or float32
            - Ascend 610 AI Processor (Vector Core): tensors of type float16
            or float32

        Kwargs:
          cal_index : A bool that specifies whether to obtain the index with
          the minimum value (supported only by
          vec_reduce_max and vec_reduce_min). Defaults to False.
          The options are as follows:
            - True: Both the maximum and minimum indexes are obtained.
            - False: Only the maximum and minimum values are obtained.
            The corresponding indexes are not obtained.

        Returns:
          None

        Restrictions:
          - The argument of repeat_times is a scalar of type int32, an
          immediate of type int, or an Expr of type int32.
            - When cal_index is set to False, repeat_times is within
            the range [1, 4095].
            - When cal_index is set to True:
              - If the operand data type is int16, the maximum value of the
              operand is 32767, meaning that a maximum of
              255 iterations are supported. Therefore, repeat_times is
              within the range [1, 255].
              - If the operand data type is float16, the maximum value of the
              operand is 65504, meaning that a maximum of
              511 iterations are supported. Therefore, repeat_times is
              within the range [1, 511].
              - Similarly, if the operand data type is float32, repeat_times
              is within the range [1, 4095].
          - src_rep_stride is within the range [0, 65535]. The argument is
          a scalar of type int16/int32/int64/uint16/uint32/uint64,
          an immediate of type int, or an Expr of type int16/int32/int64/uint16
          /uint32/uint64.
          - The storage sequence of the dst result is: maximum value,
          corresponding index. In the result, index data is
          stored as integers. For example, if the defined data type of
          dst is float16, but that of index is uint16, an
          error occurs when the index data is read in float16 format.
          Therefore, the reinterpret_cast_to() method needs
          to be called to convert the float16 index data to
          corresponding integers.
          - Restrictions for the work_tensor space are as follows:
            - If cal_index is set to False, at least (repeat_times x 2)
            elements are required. For example, when
            repeat_times=120, the shape of work_tensor has at
            least 240 elements.
            When cal_index is set to True, the space size is calculated by
            using the following formula.
            For details about examples, see Example.
            @code
                # DTYPE_SIZE indicates the data type size, in bytes.
                #For example, float16 occupies 2 bytes. elements_per_block
                #indicates the number of elements required by each block.
                elements_per_block = 32 // DTYPE_SIZE[dtype]
                # elements_per_repeat indicates the number of elements
                #required for each repeat.
                elements_per_repeat = 256 // DTYPE_SIZE[dtype]
                # Number of elements generated in the first iteration.
                it1_output_count = 2*repeat_times
                # Offset of the start position of the second iteration.ceil_div
                # is used to perform division and round up the result.
                it2_align_start = ceil_div(it1_output_count,
                                    elements_per_block)*elements_per_block
                # Number of elements generated in the second iteration.
                it2_output_count = ceil_div(it1_output_count,
                                            elements_per_repeat)*2
                # Offset of the start position of the third iteration.
                it3_align_start = ceil_div(it2_output_count,
                                    elements_per_block)*elements_per_block
                # Number of elements generated in the third iteration.
                it3_output_count = ceil_div(it2_output_count,
                                            elements_per_repeat)*2
                # Offset of the start position of the fourth iteration.
                it4_align_start = ceil_div(it3_output_count,
                                        elements_per_block)*elements_per_block
                # Number of elements generated in the fourth iteration.
                it4_output_count = ceil_div(it3_output_count,
                                        elements_per_repeat)*2
                # Finally required work_tensor size
                final_work_tensor_need_size = it2_align_start + it3_align_start
                 + it4_align_start + it4_output_count
            @endcode
            - Address overlapping between dst and work_tensor is not allowed.

        Examples:
            from te import tik
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
            dst_ub = tik_instance.Tensor("float16", (32,), tik.scope_ubuf,
                                        "dst_ub")
            src_ub = tik_instance.Tensor("float16", (256,), tik.scope_ubuf,
                                        "src_ub")
            work_tensor_ub = tik_instance.Tensor("float16", (18,),
                                        tik.scope_ubuf, "work_tensor_ub")
            tik_instance.vec_reduce_min(128, dst_ub, src_ub, work_tensor_ub,
                                        2, 8, cal_index=True)

            #[Example 1]
            #src, work_tensor, and dst are tensors of type float16. src has
            #shape (65, 128), and repeat_times of
            #vec_reduce_max/vec_reduce_min is 65.
            #The following is an API calling example:
                tik_instance.vec_reduce_max(128, dst, src, work_tensor, 65,
                                            8, cal_index=True)
            #The space of work_tensor is calculated as follows:
                elements_per_block = 16 (elements)
                elements_per_repeat = 128 (elements)
                it1_output_count = 2*65 = 130 (elements)
                it2_align_start = ceil_div(130, 16)*16 = 144 (elements)
                it2_output_count = ceil_div(130, 128)*2 = 4 (elements)
                it3_align_start = ceil_div(4, 16)*16 = 16 (elements)
                it3_output_count = ceil_div(4, 128)*2 = 2 (elements)
            #The final maximum value and its index can be obtained after three
            #iterations. The required space of work_tensor
            #is it2_align_start + it3_align_start + it3_output_count = 144 +
            #16 + 2 = 162 (elements).

            #[Example 2]
              #src, work_tensor, and dst are tensors of type float16. src has
              #shape (65, 128). repeat_times of vec_reduce_max
              #and vec_reduce_min is a scalar with the value 65.
              #If repeat_times is a scalar or contains a scalar, four
              #iterations of calculation are required.
              #The following is an API calling example:
                  scalar = tik_instance.Scalar (init_value=65, dtype="int32")
                  tik_instance.vec_reduce_max(128, dst, src, work_tensor,
                                            scalar, 8, cal_index=True)
              #The space of work_tensor is calculated as follows:
                  elements_per_block = 16 (elements)
                  elements_per_repeat = 128 (elements)
                  it1_output_count = 2*65 = 130 (elements)
                  it2_align_start = ceil_div(130, 16)*16 = 144 (elements)
                  it2_output_count = ceil_div(130, 128)*2 = 4 (elements)
                  it3_align_start = ceil_div(4, 16)*16 = 16 (elements)
                  it3_output_count = ceil_div(4, 128)*2 = 2 (elements)
                  it4_align_start = ceil_div(2, 16)*16 = 16 (elements)
                  it4_output_count = ceil(2, 128)*2 = 2(elements)
              #In cases where repeat_times is a scalar or contains a scalar,
              #the result is obtained in the third round.
              #However, the scalar value cannot be obtained at Python
              #compilation, another round is required.
              #work_tensor = it2_align_start + it3_align_start +
              #it4_aign_start + it4_output_count = 144 + 16 + 16 + 2 = 178

            #[Example 3]
              #src, work_tensor, and dst are tensors of type float32. src has
              #shape (65, 64), and repeat_times of
              #vec_reduce_max/vec_reduce_min is 65.
              #The following is an API calling example:
                  tik_instance.vec_reduce_max(64, dst, src, work_tensor, 65,
                                            8, cal_index=True)
              #The space of work_tensor is calculated as follows:
                  elements_per_block = 8 (elements)
                  elements_per_repeat = 64 (elements)
                  it1_output_count = 2*65 = 130 (elements)
                  it2_align_start = ceil_div(130, 8)*8 = 136 (elements)
                  it2_output_count = ceil_div(130, 64)*2 = 6 (elements)
                  it3_align_start = ceil_div(6, 8)*8 = 8 (elements)
                  it3_output_count = ceil_div(6, 64)*2 = 2 (elements)
              #The final maximum value and its index can be obtained after
              #three iterations. The required space of work_tensor
              #is it2_align_start + it3_align_start + it3_output_count = 136 +
              #8 + 2 = 146 (elements).

            #[Example 4]
              #src, work_tensor, and dst are float32 tensors. The shape of
              #src is (65, 64). repeat_times of vec_reduce_max
              #and vec_reduce_min is a scalar with the value 65.
              #If repeat_times is a scalar or contains a scalar,
              #four iterations of calculation are required.
              #The following is an API calling example:
                  scalar = tik_instance.Scalar (init_value=65, dtype="int32")
                  tik_instance.vec_reduce_max(64, dst, src, work_tensor,
                                            scalar, 8, cal_index=True)
              #The space of work_tensor is calculated as follows:
                  elements_per_block = 8 (elements)
                  elements_per_repeat = 64 (elements)
                  it1_output_count = 2*65 = 130 (elements)
                  it2_align_start = ceil_div(130, 8)*8 = 136 (elements)
                  it2_output_count = ceil_div(130, 64)*2 = 6 (elements)
                  it3_align_start = ceil_div(6, 8)*8 = 8 (elements)
                  it3_output_count = ceil_div(6, 64)*2 = 2 (elements)
                  it4_align_start = ceil_div(2, 8)*8 = 8 (elements)
                  it4_output_count = ceil(2, 64)*2 = 2(elements)
              #In cases where repeat_times is a scalar or contains a scalar,
              #the result is obtained in the third round.
              #However, the scalar value cannot be obtained at Python
              #compilation, another round is required.
              #work_tensor = it2_align_start + it3_align_start +
              #it4_align_start + it4_output_count = 136 + 8 + 8 + 2 = 154
        '''
        self._check_vreduce_params("vcmin", dst, src, work_tensor, repeat_times,
                                   src_rep_stride, cal_index, "vec_reduce_min")
        # check mask
        mask_concat(self, mask, tensor_bit_len=get_bit_len(src.dtype))

        # check operator address 32B aligned when core_arch is v100
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            check_address_align((work_tensor, src),
                                ("work_tensor", "src"))
        with self.context.freeze():  # pylint: disable=E1101
            if cal_index:
                if is_immediate_number(repeat_times):
                    check_dtype_overflow(mask, repeat_times, src,
                                         src_rep_stride)
                    dst_offset = Expr(dst.indice.offset).eval_value()
                    work_tensor_offset = Expr(work_tensor.indice.offset). \
                        eval_value()
                    src_offset = Expr(src.indice.offset).eval_value()
                    if is_immediate_number(src_rep_stride):
                        check_space_overflow(mask, dst, work_tensor, src,
                                             dst_offset, work_tensor_offset,
                                             src_offset, repeat_times,
                                             src_rep_stride, cal_index)
                    self._vreduce_cal_idx_imm("vcmin", mask, dst, src,
                                              work_tensor, repeat_times,
                                              src_rep_stride)
                    return
                self._vreduce_cal_idx_scalar("vcmin", mask, dst, src,
                                             work_tensor, repeat_times,
                                             src_rep_stride)
            else:
                if is_immediate_number(repeat_times):
                    dst_offset = Expr(dst.indice.offset).eval_value()
                    work_tensor_offset = Expr(work_tensor.indice.offset). \
                        eval_value()
                    src_offset = Expr(src.indice.offset).eval_value()
                    if is_immediate_number(src_rep_stride):
                        check_space_overflow(mask, dst, work_tensor, src,
                                             dst_offset, work_tensor_offset,
                                             src_offset, repeat_times,
                                             src_rep_stride, cal_index)
                    self._vreduce_no_idx_imm("vcmin", mask, dst, src,
                                             work_tensor, repeat_times,
                                             src_rep_stride)
                    return

            self._vreduce_no_idx_scalar("vcmin", mask, dst, src,
                                        work_tensor, repeat_times,
                                        src_rep_stride)

    def _check_vreduce_params(self, name, dst, src,  # pylint: disable=R0913
                              work_tensor, repeat_times,
                              src_rep_stride, cal_index, print_name):
        """
        check the args of vreduce instructions

        Parameters
        ----------
        dst: Tensor, store results in it
        src: Tensor, the source tensor
        work_tensor: Tensor, temporary memory, for internal calculation
        repeat_times: int32, the times of instruction run
        src_rep_stride: int/Scalar/Expr, the stride between each repeat in src
        cal_index: True: get the maximum element and it's index
                   False: only get the value of maximum element
        print_name: str, "vec_reduce_max"/"vec_reduce_min"

        Returns
        -------
        None
        """

        # check tensor
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor but "
                                                   "get {}".format(type(dst)))
        TikCheckUtil.check_type_match(work_tensor, Tensor,
                                      "work_tensor should be tensor but "
                                      "get {}".format(type(work_tensor)))
        TikCheckUtil.check_type_match(
            src, Tensor, "src should be tensor but get {}".format(type(src)))

        # check scope
        TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                    "src's scope must be UB "
                                    "but get {}".format(src.scope))
        TikCheckUtil.check_equality(work_tensor.scope, scope_ubuf,
                                    "work_tensor's scope must be UB "
                                    "but get {}".format(work_tensor.scope))
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB "
                                    "but get {}".format(dst.scope))

        # dtype of dst, work_tensor and src must be same
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "dst's type should be {} but get {}"
                                    .format(src.dtype, dst.dtype))
        TikCheckUtil.check_equality(dst.dtype, work_tensor.dtype,
                                    "work_tensor's type should be {} but get {}"
                                    .format(dst.dtype, work_tensor.dtype))
        # check tensor dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's "
                                    "dtype should be equal"
                                    " to dst's dtype".format(print_name))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                            + name,
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, print_name))

        TikCheckUtil.check_type_match(repeat_times, (int, Scalar, Expr),
                                      "repeat_times should be int, Scalar or "
                                      "Expr but get {}".format(
                                          type(repeat_times)))
        TikCheckUtil.check_type_match(src_rep_stride, (int, Scalar, Expr),
                                      "src_rep_stride should be int, Scalar"
                                      "or Expr but get {}".format(
                                          type(src_rep_stride)))
        TikCheckUtil.check_type_match(cal_index, bool,
                                      "cal_index should be bool "
                                      "but get {}".
                                      format(type(cal_index)))
        check_scalar_int32(repeat_times,
                           "repeat_times should be a scalar of int32")
        check_scalar_dtype(src_rep_stride,
                           "src_rep_stride should be a scalar of int/uint")

        # if dtype is B16, current algorithm support max repeat_times is 16320
        # if dtype is B32, current algorithm support max repeat_times is 8160
        # if max repeat times over the limit, should update the algorithm
        check_vreduce_repeat_times(repeat_times, cal_index, src.dtype)

        check_integer_in_range(
            src_rep_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "src_rep_stride should be in the range of [{}, {}], "
            "input src_rep_stride: {}".format(0, MAX_REP_STRIDE_DOUBLE_BYTE - 1,
                                              src_rep_stride))

        # check operator address 32B aligned when core_arch is v100
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            check_address_align((work_tensor, src),
                                ("work_tensor", "src"))

    def _vreduce_cal_idx_scalar(self, func, mask, dst,  # pylint: disable=R0913, R0914
                                src, work_tensor, repeat_times, src_rep_stride):
        """
        the common function for vreduce instruction, repeat_times is scalar.
        this function will get both value and index.

        Parameters
        ----------
        func: function: detail instruction
        mask: int/Scalar/list, effective operation on element.
        dst: Tensor, store results in it
        src: Tensor, the source tensor
        work_tensor: Tensor, temporary memory, for internal calculation
        repeat_times: Scalar, the times of instruction run
        src_rep_stride: int/Scalar, the stride between each repeat in src

        Returns
        -------
        None
        """
        with self.if_scope(repeat_times == 1):
            self._run_vreduce_max_min(func, mask, work_tensor, src,
                                      repeat_times, src_rep_stride)
            # 0 is index of value, 1 is index of value index
            dst[0] = work_tensor[0]
            dst.set_as(work_tensor, dst_offset=1, src_offset=1)
        with self.else_scope():
            dtype_size = DTYPE_SIZE[src.dtype]
            element_num_per_rep = ONE_REP_BYTE_SIZE // dtype_size

            # iteration1
            it1_output_count = _vreduce_it1_compute(
                self, func, mask, work_tensor, src, repeat_times,
                src_rep_stride)

            # iteration2
            it2_start_pos = align_start_pos(it1_output_count, dtype_size)
            it2_output_count, it3_start_pos = self._vreduce_body_cal_scalar(
                it1_output_count, it2_start_pos, 0, element_num_per_rep, src,
                func, work_tensor, dtype_size)

            if get_bit_len(src.dtype) == 16:
                index_type = "uint16"
            else:
                index_type = "uint32"

            # iteration3
            offset_num_per_rep = ONE_BLK_SIZE // dtype_size * src_rep_stride

            with self.context.freeze():  # pylint: disable=E1101
                ex_output_count, it4_start_pos = self._vreduce_body_cal_scalar(
                    it2_output_count, it3_start_pos, it2_start_pos,
                    element_num_per_rep, src, func, work_tensor,
                    dtype_size)
                _vreduce_it4_compute(self, ex_output_count, func,
                                     work_tensor,
                                     it4_start_pos, it3_start_pos)
                dst[0].set_as(work_tensor, src_offset=it4_start_pos)
                it4_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                it4_index.set_as(work_tensor, src_offset=it4_start_pos + 1)
                it3_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                it3_index.set_as(work_tensor,
                                 src_offset=it3_start_pos + it4_index + 1)
                it2_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                it2_index.set_as(work_tensor, src_offset=it2_start_pos +
                                 element_num_per_rep*(it4_index // 2)
                                 + it3_index + 1)
                it1_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                it1_index.set_as(
                    work_tensor,
                    src_offset=
                    element_num_per_rep*(element_num_per_rep*(it4_index // 2) +
                                         it3_index) // 2 + it2_index + 1)
                pre_num = \
                    offset_num_per_rep *\
                    (element_num_per_rep*(element_num_per_rep*(it4_index // 2) +
                                          it3_index) // 2 + it2_index) // 2
                res_index = self.Scalar_(init_value=pre_num + it1_index,  # pylint: disable=E1101
                                         dtype=index_type)
                # conv index_type to dst_type
                tmp_tensor = self.Tensor(dtype=index_type,  # pylint: disable=E1101
                                         shape=(1,), name="tmp_tensor",
                                         scope=scope_ubuf)
                tmp_tensor[0].set_as(res_index)
                tmp_scalar = self.Scalar_(init_value=tmp_tensor[0],  # pylint: disable=E1101
                                          dtype=dst.dtype)
                dst.set_as(tmp_scalar, dst_offset=1)

    def _vreduce_cal_idx_imm(self, func, mask, dst,  # pylint: disable=R0913, R0914, R0915
                             src, work_tensor, repeat_times, src_rep_stride):
        """
        the common function for vreduce instruction, repeat_times is int.
        this function will get both value and index.

        Parameters
        ----------
        func: function: detail instruction
        mask: int/Scalar/list, effective operation on element.
        dst: Tensor, store results in it
        src: Tensor, the source tensor
        work_tensor: Tensor, temporary memory, for internal calculation
        repeat_times: int, the times of instruction run
        src_rep_stride: int/Scalar, the stride between each repeat in src

        Returns
        -------
        None
        """
        if repeat_times == 1:
            self._run_vreduce_max_min(func, mask, work_tensor, src,
                                      repeat_times, src_rep_stride)
            dst[0] = work_tensor[0]
            dst.set_as(work_tensor, dst_offset=1, src_offset=1)

        else:
            dtype_size = DTYPE_SIZE[src.dtype]
            element_num_per_rep = ONE_REP_BYTE_SIZE // dtype_size

            # iteration1
            it1_output_count = _vreduce_it1_compute(
                self, func, mask, work_tensor, src, repeat_times,
                src_rep_stride)

            # iteration2
            it2_start_pos = align_start_pos(it1_output_count, dtype_size)
            it2_output_count, it3_start_pos = self._vreduce_body_cal_imm(
                it1_output_count, 0, it2_start_pos, element_num_per_rep, src,
                func, work_tensor, dtype_size)

            if get_bit_len(src.dtype) == 16:
                index_type = "uint16"
            else:
                index_type = "uint32"

            # iteration3
            offset_num_per_rep = ONE_BLK_SIZE // dtype_size*src_rep_stride

            if it2_output_count == VREDUCE_PER_REP_OUTPUT:
                dst[0].set_as(work_tensor, src_offset=it2_start_pos)
                it3_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                it3_index.set_as(work_tensor, src_offset=it2_start_pos + 1)
                it2_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                it2_index.set_as(work_tensor, src_offset=it3_index + 1)
                pre_num = offset_num_per_rep*(it3_index //
                                              VREDUCE_PER_REP_OUTPUT)
                res_index = self.Scalar_(init_value=pre_num + it2_index,  # pylint: disable=E1101
                                         dtype=index_type)

                # conv index_type to dst_type
                tmp_tensor = self.Tensor(dtype=index_type,  # pylint: disable=E1101
                                         shape=(1,), name="tmp_tensor",
                                         scope=scope_ubuf)
                tmp_tensor[0].set_as(res_index)
                tmp_scalar = self.Scalar_(init_value=tmp_tensor[0],  # pylint: disable=E1101
                                          dtype=dst.dtype)
            else:
                if it2_output_count > element_num_per_rep:
                    ex_output_count, it4_start_pos = \
                        self._vreduce_body_cal_imm(
                            it2_output_count, it2_start_pos, it3_start_pos,
                            element_num_per_rep, src, func, work_tensor,
                            dtype_size)
                    _vreduce_it4_compute(self, ex_output_count, func,
                                         work_tensor,
                                         it4_start_pos, it3_start_pos)
                    dst[0].set_as(work_tensor, src_offset=it4_start_pos)
                    it4_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                    it4_index.set_as(work_tensor, src_offset=it4_start_pos + 1)
                    it3_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                    it3_index.set_as(work_tensor,
                                     src_offset=it3_start_pos + it4_index + 1)
                    it2_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                    it2_index.set_as(
                        work_tensor,
                        src_offset=
                        it2_start_pos +
                        element_num_per_rep*(it4_index //
                                             VREDUCE_PER_REP_OUTPUT) +
                        it3_index + 1)
                    it1_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                    it1_index.set_as(work_tensor, src_offset=
                                     element_num_per_rep *
                                     (element_num_per_rep *
                                      (it4_index // VREDUCE_PER_REP_OUTPUT) +
                                      it3_index) // VREDUCE_PER_REP_OUTPUT +
                                     it2_index + 1)
                    pre_num = \
                        offset_num_per_rep * \
                        (element_num_per_rep *
                         (element_num_per_rep *
                          (it4_index // VREDUCE_PER_REP_OUTPUT) +
                          it3_index) // VREDUCE_PER_REP_OUTPUT +
                         it2_index) // VREDUCE_PER_REP_OUTPUT
                else:
                    _vreduce_it4_compute(self, it2_output_count, func,
                                         work_tensor, it3_start_pos,
                                         it2_start_pos)
                    it3_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                    it3_index.set_as(work_tensor, src_offset=it3_start_pos + 1)
                    dst[0].set_as(work_tensor, src_offset=it3_start_pos)

                    it2_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                    it2_index.set_as(work_tensor,
                                     src_offset=it2_start_pos + it3_index + 1)
                    it1_index = self.Scalar_(dtype=index_type)  # pylint: disable=E1101
                    it1_index.set_as(work_tensor, src_offset=
                                     element_num_per_rep *
                                     (it3_index // VREDUCE_PER_REP_OUTPUT) +
                                     it2_index + 1)
                    pre_num = offset_num_per_rep * \
                              (element_num_per_rep *
                               (it3_index // VREDUCE_PER_REP_OUTPUT) +
                               it2_index) // VREDUCE_PER_REP_OUTPUT
                res_index = self.Scalar_(init_value=pre_num + it1_index,  # pylint: disable=E1101
                                         dtype=index_type)
                # conv index_type to dst_type
                tmp_tensor = self.Tensor(dtype=index_type, shape=(1,),  # pylint: disable=E1101
                                         name="tmp_tensor",
                                         scope=scope_ubuf)
                tmp_tensor[0].set_as(res_index)
                tmp_scalar = self.Scalar_(init_value=tmp_tensor[0],  # pylint: disable=E1101
                                          dtype=dst.dtype)
            dst.set_as(tmp_scalar, dst_offset=1)

    def _vreduce_body_cal_imm(self,  # pylint: disable=R0913, R0914
                              pre_data_count, pre_start_pos, cur_start_pos,
                              element_num_per_rep, src, func, work_tensor,
                              dtype_size):
        ex_body_repeat_times = pre_data_count // element_num_per_rep
        ex_tail_num_count = pre_data_count % element_num_per_rep
        ex_has_tail = ex_tail_num_count != 0
        ex_body_output_count = 0
        if get_bit_len(src.dtype) == 16:
            body_mask = [int("01" * 32, 2), int("01" * 32, 2)]
        else:
            body_mask = [0, int("01" * 32, 2)]
        if ex_body_repeat_times != 0:
            self._run_vreduce_max_min(func, body_mask, work_tensor, work_tensor,
                                      ex_body_repeat_times,
                                      VREDUCE_DEFAULT_SRC_REP_STRIDE,
                                      cur_start_pos, pre_start_pos)
            ex_body_output_count = VREDUCE_PER_REP_OUTPUT * ex_body_repeat_times
        ex_tail_output_count = 0
        if ex_has_tail:
            tail_mask = self.creat_mask_(ex_tail_num_count //
                                         VREDUCE_PER_REP_OUTPUT)
            self._run_vreduce_max_min(func, tail_mask, work_tensor, work_tensor,
                                      VREDUCE_MIN_REPEAT_TIMES,
                                      VREDUCE_DEFAULT_SRC_REP_STRIDE,
                                      cur_start_pos + ex_body_output_count,
                                      pre_start_pos + element_num_per_rep *
                                      ex_body_repeat_times)
            ex_tail_output_count = VREDUCE_PER_REP_OUTPUT
        output_count = ex_body_output_count + ex_tail_output_count
        next_start_pos = align_start_pos(cur_start_pos +
                                         output_count, dtype_size)
        return output_count, next_start_pos

    def _vreduce_body_cal_scalar(self,  # pylint: disable=R0913, R0914
                                 pre_data_count, cur_start_pos, pre_start_pos,
                                 element_num_per_rep, src, func, work_tensor,
                                 dtype_size):
        ex_body_repeat_times = pre_data_count // element_num_per_rep
        ex_tail_num_count = pre_data_count % element_num_per_rep
        ex_has_tail = ex_tail_num_count != 0
        ex_tail_output_count = self.Scalar_(dtype="int32",  # pylint: disable=E1101
                                            init_value=0)
        if get_bit_len(src.dtype) == 16:
            body_mask = [int("01" * 32, 2), int("01" * 32, 2)]
        else:
            body_mask = [0, int("01" * 32, 2)]
        with self.if_scope(ex_body_repeat_times != 0):
            self._run_vreduce_max_min(
                func, body_mask, work_tensor, work_tensor,
                ex_body_repeat_times, VREDUCE_DEFAULT_SRC_REP_STRIDE,
                cur_start_pos, pre_start_pos)
            ex_body_output_count = VREDUCE_PER_REP_OUTPUT * ex_body_repeat_times
        with self.if_scope(ex_has_tail):
            tail_mask = self.creat_mask_(ex_tail_num_count //
                                         VREDUCE_PER_REP_OUTPUT)
            self._run_vreduce_max_min(
                func, tail_mask, work_tensor, work_tensor,
                VREDUCE_MIN_REPEAT_TIMES, VREDUCE_DEFAULT_SRC_REP_STRIDE,
                cur_start_pos + ex_body_output_count,
                pre_start_pos + element_num_per_rep * ex_body_repeat_times)
            ex_tail_output_count.set_as(VREDUCE_PER_REP_OUTPUT)
        output_count = ex_body_output_count + ex_tail_output_count
        next_start_pos = align_start_pos(cur_start_pos +
                                         output_count, dtype_size)
        return output_count, next_start_pos

    def _second_step_no_idx_imm(self, func, dst, src,  # pylint: disable=R0913
                                cur_data, dtype_len):
        new_repeat_times = cur_data * dtype_len // ONE_REP_BYTE_SIZE
        if new_repeat_times >= 1:
            new_mask = self.creat_mask_(ONE_REP_BYTE_SIZE // dtype_len //
                                        VREDUCE_PER_REP_OUTPUT)
            # dst_rep_stride, src_block_stride set to 1,
            # src_rep_stride set to 8
            self._run_vreduce_max_min(
                func, new_mask, dst, src, new_repeat_times,
                VREDUCE_DEFAULT_SRC_REP_STRIDE)

        left_data = cur_data % (ONE_REP_BYTE_SIZE // dtype_len)

        if left_data > 0:
            new_mask = self.creat_mask_(left_data // VREDUCE_PER_REP_OUTPUT)
            # repeat_times set to 1, dst_rep_stride set to 1,
            # src_block_stride set to 1, src_rep_stride set to 8,
            self._run_vreduce_max_min(
                func, new_mask, dst, src,
                VREDUCE_MIN_REPEAT_TIMES, VREDUCE_DEFAULT_SRC_REP_STRIDE,
                new_repeat_times * VREDUCE_PER_REP_OUTPUT,
                new_repeat_times * ONE_REP_BYTE_SIZE // dtype_len)
            # have tail, new_repeat_times used to calculate output data num
            new_repeat_times += 1
        return new_repeat_times * VREDUCE_PER_REP_OUTPUT

    def _third_step_no_idx_imm(self, func, dst,  # pylint: disable=R0913
                               work_tensor, src, cur_data):
        new_mask = self.creat_mask_(cur_data // VREDUCE_PER_REP_OUTPUT)
        # repeat_times set to 1, dst_rep_stride set to 1,
        # src_block_stride set to 1, src_rep_stride set to 8,
        self._run_vreduce_max_min(func, new_mask, work_tensor, src,
                                  VREDUCE_MIN_REPEAT_TIMES,
                                  VREDUCE_DEFAULT_SRC_REP_STRIDE)
        # copy result to dst buffer
        dst[0] = work_tensor[0]

    def _second_step_no_idx_scalar(self, func, dst,  # pylint: disable=R0913
                                   src, cur_data, dtype_len):
        new_repeat_times = cur_data * dtype_len // ONE_REP_BYTE_SIZE
        tail_repeat_times = self.Scalar_(dtype="int32",  # pylint: disable=E1101
                                         init_value=0)
        with self.if_scope(new_repeat_times >= 1):
            new_mask = self.creat_mask_(ONE_REP_BYTE_SIZE // dtype_len //
                                        VREDUCE_PER_REP_OUTPUT)
            # dst_rep_stride, src_block_stride set to 1,
            # src_rep_stride set to 8
            self._run_vreduce_max_min(
                func, new_mask, dst, src, new_repeat_times,
                VREDUCE_DEFAULT_SRC_REP_STRIDE)

        left_data = cur_data % (ONE_REP_BYTE_SIZE // dtype_len)
        with self.if_scope(left_data > 0):
            new_mask = self.creat_mask_(left_data // VREDUCE_PER_REP_OUTPUT)
            # repeat_times set to 1, dst_rep_stride set to 1,
            # src_block_stride set to 1, src_rep_stride set to 8,
            self._run_vreduce_max_min(
                func, new_mask, dst, src,
                VREDUCE_MIN_REPEAT_TIMES, VREDUCE_DEFAULT_SRC_REP_STRIDE,
                new_repeat_times * VREDUCE_PER_REP_OUTPUT,
                new_repeat_times * ONE_REP_BYTE_SIZE // dtype_len)
            # have tail, new_repeat_times used to calculate output data num
            tail_repeat_times.set_as(1)
        return (new_repeat_times + tail_repeat_times) * VREDUCE_PER_REP_OUTPUT

    def _third_step_no_idx_scalar(self, func, dst,  # pylint: disable=R0913
                                  work_tensor, src, cur_data):
        new_mask = self.creat_mask_(cur_data // VREDUCE_PER_REP_OUTPUT)
        # repeat_times set to 1, dst_rep_stride set to 1,
        # src_block_stride set to 1, src_rep_stride set to 8,
        self._run_vreduce_max_min(func, new_mask, work_tensor, src,
                                  VREDUCE_MIN_REPEAT_TIMES,
                                  VREDUCE_DEFAULT_SRC_REP_STRIDE)
        # copy result to dst
        dst[0] = work_tensor[0]

    def _vreduce_no_idx_imm(self, func, mask, dst, src,  # pylint: disable=R0913
                            work_tensor, repeat_times, src_rep_stride):
        """
        the common function for vreduce instruction, repeat_times is int.
        this function will get only value.

        Parameters
        ----------
        func: function: detail instruction
        mask: int/Scalar/list, effective operation on element.
        dst: Tensor, store results in it
        src: Tensor, the source tensor
        work_tensor: Tensor, temporary memory, for internal calculation
        repeat_times: int, the times of instruction run
        src_rep_stride: int/Scalar, the stride between each repeat in src

        Returns
        -------
        None
        """
        if repeat_times == 1:
            self._run_vreduce_max_min(func, mask, work_tensor, src,
                                      repeat_times, src_rep_stride)
            dst[0] = work_tensor[0]
            return

        # first step
        dtype_len = DTYPE_SIZE[src.dtype]
        cur_data = self.run_reduce_func_(func, mask, work_tensor, src,
                                         repeat_times, src_rep_stride)
        # second step
        cur_data = self._second_step_no_idx_imm(func, work_tensor, work_tensor,
                                                cur_data, dtype_len)
        one_rep_data = ONE_REP_BYTE_SIZE // dtype_len
        # third step
        if cur_data <= one_rep_data:
            self._third_step_no_idx_imm(func, dst, work_tensor,
                                        work_tensor, cur_data)
            return

        # if new_repeat_times > 1, need to run second step again,
        # this only for float32, when first second step output 8190 elements
        cur_data = self._second_step_no_idx_imm(func, work_tensor, work_tensor,
                                                cur_data, dtype_len)
        new_repeat_times = cur_data * dtype_len // ONE_REP_BYTE_SIZE

        if new_repeat_times <= 1:
            self._third_step_no_idx_imm(func, dst, work_tensor,
                                        work_tensor, cur_data)

    def _vreduce_no_idx_scalar(self, func, mask, dst,  # pylint: disable=R0913
                               src, work_tensor,
                               repeat_times, src_rep_stride):
        """
        the common function for vreduce instruction, repeat_times is scalar.
        this function will get only value.

        Parameters
        ----------
        func: function: detail instruction
        mask: int/Scalar/list, effective operation on element.
        dst: Tensor, store results in it
        src: Tensor, the source tensor
        work_tensor: Tensor, temporary memory, for internal calculation
        repeat_times: Scalar, the times of instruction run
        src_rep_stride: int/Scalar, the stride between each repeat in src
        """
        # repeat_times is not immediate, some condition need to do as scalar
        dtype_len = DTYPE_SIZE[src.dtype]
        # first step
        cur_data = self.run_reduce_func_(func, mask, work_tensor, src,
                                         repeat_times, src_rep_stride)
        # second step
        cur_data = self._second_step_no_idx_scalar(func, work_tensor,
                                                   work_tensor, cur_data,
                                                   dtype_len)
        one_rep_data = ONE_REP_BYTE_SIZE // dtype_len

        with self.if_scope(cur_data <= one_rep_data):
            self._third_step_no_idx_scalar(func, dst, work_tensor,
                                           work_tensor, cur_data)
        with self.else_scope():
            # second second step
            cur_data = self._second_step_no_idx_scalar(func, work_tensor,
                                                       work_tensor, cur_data,
                                                       dtype_len)
            new_repeat_times = cur_data * dtype_len // ONE_REP_BYTE_SIZE

            # second third step
            with self.if_scope(new_repeat_times <= 1):
                self._third_step_no_idx_scalar(func, dst, work_tensor,
                                               work_tensor, cur_data)
