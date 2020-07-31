"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_reduce_api_.py
DESC:     provide reduce vector instructions
CREATED:  2019-08-16 18:53:42
MODIFIED: 2019-08-16 21:29:18
"""
# disabling:
# R0914: too-many-locals
# R0913: too-many-arguments
# C0302: too-many-lines
# E1101: no-member

from functools import wraps  # pylint: disable=C0302

from te import tvm
from te.platform import cce_params
from te.platform.cce_conf import intrinsic_check_support
from te.platform.cce_conf import api_check_support
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import HI3796CV300ES
from te.platform.cce_params import ASCEND_610
from te.platform.cce_params import ASCEND_620
from te.tik.common.util import DTYPE_SIZE, reduce_mul, \
    check_integer_in_range, ceil_div, get_bit_len, check_scalar_dtype, \
    check_scalar_int32, is_immediate_number
from te.tik.common.common_util import vector_tensor_overflow_check,\
    check_tensor_overflow, check_vreduce_src1_overflow, check_vector_stride, \
    check_address_overlap_vreduce, check_address_overlapping, \
    cal_extent_stride_unit_mask, vreduce_create_mask
from .. import debug
from ..api.tik_ir_builder import TikIRBuilder
from .tik_util import type_convert, concat_params, \
    change_dtype_str
from .tik_api_util import set_ctrl_counter_mask, reset_ctrl_counter_mask, \
    check_repeat_times
from ..api.tik_scalar import mask_concat, Scalar
from .tik_expr import Expr
from ..api.tik_tensor import Tensor
from .tik_api_constants import DTYPE_MAP
from .tik_params import MAX_BLK_STRIDE_DOUBLE_BYTE, \
    MAX_REP_STRIDE_DOUBLE_BYTE, MAX_STRIDE_UNIT, ONE_BYTE_BIT_LEN, \
    ONE_REP_BYTE_SIZE, PIPE_V, VECTOR_PAIR_OFFSET_LIST, \
    VECTOR_PAIR_SEGMENT_LIST, ONE_BLK_SIZE, MAX_BLK_STRIDE_SINGLE_BYTE, \
    MAX_REP_STRIDE_SINGLE_BYTE, MIN_SRC1_PATTERN, MAX_SRC1_PATTERN, \
    VREDUCE_OFFSET_LIST, VREDUCE_SEGMENT_LIST, MAXMIN_CNT_INDEX_LEN_1, \
    MAXMIN_CNT_INDEX_LEN_3, FOUR_BYTE_VALUE, TWO_BYTE_VALUE, CNT_SHIFT_POS, \
    INDEX_SHIFT_POS, TWO_IR, BLK_NUM_PER_REP, \
    MIN_REPEAT_TIMES, INSTR_DTYPE_SUPPORT_STATEMENT, \
    MAX_VREDUCE_REPEAT_TIMES, MAX_REPEAT_TIMES, VREDUCE_PER_REP_OUTPUT, \
    VREDUCE_DEFAULT_DST_BLK_STRIDE, VREDUCE_DEFAULT_SRC_BLK_STRIDE, \
    VREDUCE_DEFAULT_DST_REP_STRIDE
from ..common.tik_get_soc_name import get_soc_name
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator

_DEFAULT_NBLOCK = 1
_DEFAULT_SRC_STRIDE = 0
_DEFAULT_BLK_STRIDE = 1
_DEFAULT_STRIDE = 1
_BLOCK_LEN = 8


def vec_reduce_add_check_decorator(func):
    """bind this decorator with vec_reduce_add instructions"""
    @wraps(func)
    def wrapper(tik_instance, mask, dst, src, work_tensor,  # pylint: disable=R0913
                repeat_times, src_rep_stride):
        # check dst/src/work_tensor
        TikCheckUtil.check_type_match(
            dst, Tensor,
            "dst should be tensor, input type is:{}".format(type(dst)))
        TikCheckUtil.check_type_match(
            src, Tensor,
            "src should be tensor, input type is:{}".format(type(src)))
        TikCheckUtil.check_type_match(
            work_tensor, Tensor, "work_tensor should be tensor, "
                                 "input type is:{}".format(type(work_tensor)))
        TikCheckUtil.check_equality(
            dst.scope, scope_ubuf,
            "dst's scope must be UB, input scope is:{}".format(dst.scope))
        TikCheckUtil.check_equality(
            src.scope, scope_ubuf,
            "src's scope must be UB, input scope is:{}".format(src.scope))
        TikCheckUtil.check_equality(
            work_tensor.scope, scope_ubuf,
            "work_tensor's scope must be UB, "
            "input scope is:{}".format(work_tensor.scope))
        # check dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype"
                                    " should be equal to dst's dtype".
                                    format("vec_reduce_add"))

        TikCheckUtil.check_equality(api_check_support("tik." +
                                                      "vreduceadd",
                                                      dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "vec_reduce_add"))
        # check work_tensor dtype
        TikCheckUtil.check_equality(
            src.dtype, work_tensor.dtype,
            "work_tensor should have the same dtype as src")
        # check src_rep_stride
        TikCheckUtil.check_type_match(
            src_rep_stride, (int, Expr, Scalar),
            "src_rep_stride should be int, Expr or Scalar")
        check_scalar_dtype(
            src_rep_stride,
            "scalar_src_rep_stride should be a scalar of int/uint")
        check_integer_in_range(
            src_rep_stride, range(MAX_REP_STRIDE_DOUBLE_BYTE),
            "src_rep_stride should be in the range of [0, 65535], "
            "input value is %s" % str(src_rep_stride))
        # check repeat times
        TikCheckUtil.check_type_match(
            repeat_times, (int, Scalar, Expr),
            "repeat_times should be int, Scalar or Expr, input type of "
            "repeat_times: {}".format(type(repeat_times)))
        check_scalar_int32(repeat_times,
                           "scalar_repeat_time should be a scalar of int32")
        check_integer_in_range(
            repeat_times, range(MIN_REPEAT_TIMES, MAX_VREDUCE_REPEAT_TIMES),
            "repeat_times should be in the range of [1, {}], "
            "input repeat_times: {}".format(
                MAX_VREDUCE_REPEAT_TIMES - 1, repeat_times))
        return func(tik_instance, mask, dst, src, work_tensor,
                    repeat_times, src_rep_stride)
    return wrapper


def _cal_dst_extent_vreduce(mask_mode, repeat_times, dst, mask):
    """calculate dst extent in instruction vreduce

    Parameters
    ----------
    mask_mode: "normal" - mask normal mode
               "counter" - mask counter mode
    repeat_times : Repeated iterations times
    dst : destination operator

    Returns
    -------
    src1_extent
    """
    if mask_mode == "normal":
        dst_extent = repeat_times*ONE_REP_BYTE_SIZE
    else:
        dst_extent = mask*DTYPE_SIZE[dst.dtype]
    dst_extent = Expr(dst_extent).get()
    return dst_extent


def _cal_src1_extent_vreduce(src1_pattern, src1_rep_stride, repeat_times, mask,
                             mask_mode):
    """calculate src1 extent in instruction vreduce

    Parameters
    ----------
    src1_pattern : 6 fixed patterns for effective operation on element, or
               user defined tensor(dtype usigned int), per bit 1/0 for
               effective opration on element
    src1_rep_stride: offset of src1 operator in the same block
                     between adjacent iterations
    repeat_times : Repeated iterations times
    mask : Effective operation on element
    mask_mode: "normal" - mask normal mode
               "counter" - mask counter mode

    Returns
    -------
    src1_extent
    """
    src1_extent = 0
    if isinstance(src1_pattern, Tensor):
        if Expr(src1_rep_stride).eval_value() is None:
            src1_extent = reduce_mul(src1_pattern.indice.origin_shape)*\
                          DTYPE_SIZE[src1_pattern.dtype]
        elif Expr(src1_rep_stride).eval_value() == 0:
            src1_extent = ceil_div(
                ONE_REP_BYTE_SIZE // DTYPE_SIZE[src1_pattern.dtype],
                get_bit_len(src1_pattern.dtype))*DTYPE_SIZE[src1_pattern.dtype]
        else:
            if mask_mode == "normal":
                src1_extent = ceil_div(
                    repeat_times*ONE_REP_BYTE_SIZE //
                    DTYPE_SIZE[src1_pattern.dtype],
                    get_bit_len(src1_pattern.dtype))*\
                              DTYPE_SIZE[src1_pattern.dtype]
            else:
                src1_extent = ceil_div(mask, get_bit_len(src1_pattern.dtype))*\
                              DTYPE_SIZE[src1_pattern.dtype]
    src1_extent = Expr(src1_extent).get()
    return src1_extent


def _check_src0_overflow_vreduce(mask_mode, src0, repeat_times,  # pylint: disable=R0913
                                 src0_blk_stride,
                                 src0_rep_stride, stride_unit, mask):
    """check src0 tensor overflow in instruction vreduce

    Parameters
    ----------
    mask_mode: "normal" - mask normal mode
               "counter" - mask counter mode
    src0 : source operator
    repeat_times : Repeated iterations times
    stride_unit : address and offset unit both affect it. default = 0
    mask : Effective operation on element

    Returns
    -------
    None
    """
    if mask_mode == "normal":
        # src0
        check_tensor_overflow(
            [src0], ONE_REP_BYTE_SIZE // DTYPE_SIZE[src0.dtype],
            repeat_times, [src0_blk_stride], [src0_rep_stride], ["src0"],
            stride_unit)
    else:
        # src0
        if isinstance(mask, int):
            actual_rep_times = ceil_div(
                mask, ONE_REP_BYTE_SIZE // DTYPE_SIZE[src0.dtype])
            check_tensor_overflow([src0], mask, actual_rep_times,
                                  [src0_blk_stride], [src0_rep_stride],
                                  ["src0"], stride_unit, mask_mode)


def _check_dst_overflow_vreduce(mask_mode, src1_pattern, repeat_times, dst,
                                mask):
    """check dst tensor overflow in instruction vreduce

    Parameters
    ----------
    mask_mode: "normal" - mask normal mode
               "counter" - mask counter mode
    src1_pattern : 6 fixed patterns for effective operation on element, or
                   user defined tensor(dtype usigned int), per bit 1/0 for
                   effective opration on element
    repeat_times : Repeated iterations times
    dst : destination operator
    mask : Effective operation on element

    Returns
    -------
    None
    """
    if mask_mode == "normal":
        # dst
        if not isinstance(src1_pattern, Tensor) and \
                all(Expr(value).eval_value() is not None
                        for value in (src1_pattern, repeat_times)):
            # src1_pattern is 1~2, extract odd/even element of src0
            if src1_pattern < 3:
                dst_expected_size = repeat_times*ONE_REP_BYTE_SIZE // \
                                    DTYPE_SIZE[dst.dtype] // 2
            # src1_pattern is 3~6, extract 1/4 elements of src0
            else:
                dst_expected_size = repeat_times*ONE_REP_BYTE_SIZE // \
                                    DTYPE_SIZE[dst.dtype] // 4
            TikCheckUtil.check_ge(
                reduce_mul(dst.indice.origin_shape), dst_expected_size,
                "dst tensor overflow")
    else:
        # dst
        if not isinstance(src1_pattern, Tensor) and \
                all(Expr(value).eval_value() is not None
                        for value in (src1_pattern, mask, dst.offset)):
            # src1_pattern is 1~2, extract odd/even element of src0
            if src1_pattern < 3:
                dst_expected_size = Expr(ceil_div(mask, 2) + dst.offset) \
                    .eval_value()
            # src1_pattern is 3~6, extract 1/4 elements of src0
            else:
                dst_expected_size = Expr(ceil_div(mask, 4) + dst.offset) \
                    .eval_value()
            TikCheckUtil.check_ge(
                reduce_mul(dst.indice.origin_shape), dst_expected_size,
                "dst tensor overflow")


class TikReduceApi(TikIRBuilder):
    """Provide whole-reduce group-reduce pair-reduce api"""
    def __init__(self):
        super(TikReduceApi, self).__init__()

    def _check_reduce_func_params(self, name, dst,  # pylint: disable=R0913
                                  src, repeat_times,
                                  dst_rep_stride, src_blk_stride,
                                  src_rep_stride, stride_unit, print_name):
        """check whether reduce instruction params are legal

        Parameters
        ----------
        name : instruction name
        dst : destination tensor
        src : source tensor
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_blk_stride : offset of src operator between different block in
                         one repeat
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # check tensor
        TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        # check scope
        TikCheckUtil.check_equality(
            src.scope, scope_ubuf, "src's scope must be UB")
        TikCheckUtil.check_equality(
            dst.scope, scope_ubuf, "dst's scope must be UB")
        # check parameters type
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride(None, [dst_rep_stride, src_rep_stride],
                            None, MAX_REP_STRIDE_DOUBLE_BYTE, ["dst", "src"])
        TikCheckUtil.check_type_match(
            src_blk_stride, (int, Expr, Scalar),
            "src_blk_stride should be int, Expr or Scalar")
        check_scalar_dtype(src_blk_stride,
                           "scalar_src_blk_stride should be a scalar of int")
        check_integer_in_range(
            src_blk_stride, range(MAX_BLK_STRIDE_DOUBLE_BYTE),
            "src_blk_stride should be in the range of [0, 65535], "
            "input value is %s" % str(src_blk_stride))
        # check stride_unit
        TikCheckUtil.check_type_match(
            stride_unit, (int, Expr, Scalar),
            "stride_unit should be int, Expr or Scalar")
        check_integer_in_range(stride_unit, range(MAX_STRIDE_UNIT),
                               "stride_unit should be in the range of [0, 3], "
                               "input value is %s" % str(stride_unit))
        # check tensor dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's dtype "
                                    "should be equal"
                                    " to dst's dtype".format(print_name))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                            + name,
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, print_name))

    @staticmethod
    def _check_overflow(name, mask, dst,  # pylint: disable=R0913
                        src, repeat_times,
                        dst_rep_stride, src_blk_stride, src_rep_stride,
                        dst_offset=0, src_offset=0):
        """check tensor whether overflow

         Parameters
         ----------
         name : instruction name
         mask : Effective operation on element,
                divided into two model: Continuous and bit by bit.
         dst : destination tensor
         src : source tensor
         repeat_times : Repeated iterations times
         dst_rep_stride : offset of dst operator in the same block between
                          adjacent iterations
         src_blk_stride : offset of src operator between different block
                          in one iteration
         src_rep_stride : offset of src operator in the same block between
                          adjacent iterations
         dst_offset: dst tensor offset
         src_offset: src tensor offset

         Returns
         -------
         None
         """
        if name in ("vcmin", "vcmax"):
            block_len = 2
            vector_tensor_overflow_check(dst, mask, _DEFAULT_NBLOCK, block_len,
                                         repeat_times, _DEFAULT_SRC_STRIDE,
                                         dst_rep_stride, ori_offset=dst_offset)
        else:
            block_len = 1
            vector_tensor_overflow_check(dst, mask, _DEFAULT_NBLOCK, block_len,
                                         repeat_times, _DEFAULT_SRC_STRIDE,
                                         dst_rep_stride, ori_offset=dst_offset)
        src_bit_len = get_bit_len(src.dtype)
        parallelism = ONE_REP_BYTE_SIZE*ONE_BYTE_BIT_LEN // src_bit_len
        vector_tensor_overflow_check(src, mask,
                                     parallelism // (ONE_REP_BYTE_SIZE
                                                     // src_bit_len),
                                     ONE_REP_BYTE_SIZE // src_bit_len,
                                     repeat_times,
                                     src_blk_stride, src_rep_stride,
                                     ori_offset=src_offset)

    @source_info_decorator(depth=2)
    @debug.vec_reduce_decorator
    def _vector_whole_reduce_func(self, name,  # pylint: disable=R0913, R0914
                                  mask, dst, src, repeat_times,
                                  dst_rep_stride, src_blk_stride,
                                  src_rep_stride, stride_unit, order,
                                  maxmin_cnt_index):
        """Instruction of cross vecter operation

        Parameters
        ----------
        name : instruction name
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination tensor
        src : source tensor
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block between
                         adjacent iterations
        src_blk_stride : offset of src operator between different block
                         in one iteration
        src_rep_stride : offset of src operator in the same block between
                         adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0
        order : specify the relative position of index and value in dst
        maxmin_cnt_index : reduction of the result of this instruction,
                           contains [Scalar_maxmin, Scalar_cnt, Scalar_index]

        Returns
        -------
        None
        """
        # the arguments are too many and unused variables are used in decorator
        self._check_reduce_func_params(name, dst, src, repeat_times,
                                       dst_rep_stride, src_blk_stride,
                                       src_rep_stride, stride_unit, name)
        # mask
        mask_o = mask_concat(self, mask,
                             tensor_bit_len=max(get_bit_len(dst.dtype),
                                                get_bit_len(src.dtype)))

        # check address overlap
        if src.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_rep_stride,
                             src_blk_stride, src_rep_stride, stride_unit)):
                if name in ("vcmin", "vcmax"):
                    block_len = 2
                    check_address_overlapping(
                        name, mask, dst, src, _DEFAULT_NBLOCK,
                        ONE_REP_BYTE_SIZE // max(
                            get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                        block_len, ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                        repeat_times, _DEFAULT_BLK_STRIDE, src_blk_stride,
                        dst_rep_stride, src_rep_stride,
                        Expr(dst.offset).eval_value(),
                        Expr(src.offset).eval_value(), stride_unit)
                else:
                    block_len = 1
                    check_address_overlapping(
                        name, mask, dst, src, _DEFAULT_NBLOCK,
                        ONE_REP_BYTE_SIZE // max(
                            get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                        block_len, ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                        repeat_times, _DEFAULT_BLK_STRIDE, src_blk_stride,
                        dst_rep_stride, src_rep_stride,
                        Expr(dst.offset).eval_value(),
                        Expr(src.offset).eval_value(), stride_unit)

        # check tensor overflow
        self._check_overflow(name, mask, dst, src, repeat_times,
                             dst_rep_stride, src_blk_stride, src_rep_stride)
        # gen
        config = [repeat_times, dst_rep_stride, src_blk_stride, src_rep_stride]
        if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
            config.append(stride_unit & 0b01)
            config.append((stride_unit & 0b10) >> 1)
            if name in ("vcmax", "vcmin"):
                config.append(order)

        src_bit_len = get_bit_len(src.dtype)
        with self.new_scope():
            mem_access_param = type_convert(config)
            # 8 Block/repeat, 32Byte/Block
            src_extent = Expr(((repeat_times - 1)*src_rep_stride +
                               (8 - 1)*src_blk_stride + 1)*32)
            if name == "vcadd":
                dst_extent = Expr(repeat_times*dst_rep_stride*src_bit_len
                                  // ONE_BYTE_BIT_LEN)
            else:
                # here is vcmax and vcmin, src f16 2B aligned, dst 4B aligned
                dst_extent = Expr(repeat_times*dst_rep_stride*src_bit_len*2 //
                                  ONE_BYTE_BIT_LEN)
            # when repeat time >1 and the count of dst write element is not
            # the multi of ONE_BLK_SIZE
            dst_extent = Expr(ceil_div(dst_extent, ONE_BLK_SIZE)*ONE_BLK_SIZE)
            if src.dtype == "int16":
                instr = tvm.call_extern(dst.dtype, name,
                                        dst.reinterpret_cast_to("uint16")
                                        .access_ptr("w",
                                                    extent=dst_extent.get()),
                                        src.reinterpret_cast_to("uint16")
                                        .access_ptr("r",
                                                    extent=src_extent.get()),
                                        *mem_access_param)
            else:
                instr = tvm.call_extern(dst.dtype, name,
                                        dst.access_ptr("w",
                                                       extent=dst_extent.get()),
                                        src.access_ptr("r",
                                                       extent=src_extent.get()),
                                        *mem_access_param)
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

        # read maxmin_cnt_index
        if name in ("vcmax", "vcmin") and maxmin_cnt_index is not None:
            self._read_maxmin_cnt_index(src, maxmin_cnt_index)

    def _read_maxmin_cnt_index(self, src, maxmin_cnt_index):
        # cannot recognize context member, so disable it
        TikCheckUtil.check_in_range(
            get_soc_name(), (HI3796CV300ES, ASCEND_610, ASCEND_620),
            "{} doesn't support maxmin_cnt_index.".format(get_soc_name()))
        TikCheckUtil.check_type_match(maxmin_cnt_index, (list, tuple))
        if get_soc_name() in (ASCEND_610, ASCEND_620):
            TikCheckUtil.check_equality(
                len(maxmin_cnt_index), MAXMIN_CNT_INDEX_LEN_1,
                "maxmin_cnt_index must be a list of one Scalar")
            TikCheckUtil.check_type_match(
                maxmin_cnt_index[0], Scalar,
                "maxmin_cnt_index must be a list of Scalar")
        else:
            TikCheckUtil.check_equality(
                len(maxmin_cnt_index), MAXMIN_CNT_INDEX_LEN_3,
                "maxmin_cnt_index must be a list of three Scalar")
            for mci_scalar in maxmin_cnt_index:
                TikCheckUtil.check_type_match(
                    mci_scalar, Scalar, "maxmin_cnt_index must be"
                                        " a list of Scalar")
            TikCheckUtil.check_in_range(
                maxmin_cnt_index[1].dtype, ("int16", "int32", "int64"),
                "maxmin_cnt_index only support int16, int32, int64")
            TikCheckUtil.check_in_range(
                maxmin_cnt_index[2].dtype, ("int16", "int32", "int64"),
                "maxmin_cnt_index only support int16, int32, int64")
        TikCheckUtil.check_equality(maxmin_cnt_index[0].dtype, src.dtype)
        with self.new_scope():
            with self.context.freeze():  # pylint: disable=E1101
                maxmin_cnt_index_scalar = self.Scalar_("int64")  # pylint: disable=E1101
                self.scope_attr(cce_params.CCE_AXIS, "coproc_scope",
                                PIPE_V)
                self.emit(tvm.call_extern(
                    maxmin_cnt_index_scalar.dtype, "reg_set",
                    maxmin_cnt_index_scalar.get(),
                    tvm.call_extern(maxmin_cnt_index_scalar.dtype,
                                    "get_max_min_cnt")))
                if get_soc_name() not in (ASCEND_610, ASCEND_620):
                    maxmin_cnt_index[2].set_as(
                        (maxmin_cnt_index_scalar >> INDEX_SHIFT_POS) &
                        TWO_BYTE_VALUE)
                    maxmin_cnt_index[1].set_as(
                        (maxmin_cnt_index_scalar >> CNT_SHIFT_POS) &
                        TWO_BYTE_VALUE)
                if get_bit_len(src.dtype) == 32:
                    maxmin_cnt_index_scalar.set_as(
                        maxmin_cnt_index_scalar & FOUR_BYTE_VALUE)
                else:
                    maxmin_cnt_index_scalar.set_as(
                        maxmin_cnt_index_scalar & TWO_BYTE_VALUE)
                maxmin_cnt_index[0].set_as(
                    tvm.call_extern(maxmin_cnt_index[0].dtype,
                                    "reinterpret_cast",
                                    maxmin_cnt_index_scalar.get()))

    def vcadd(self,  # pylint: disable=R0913
              mask,
              dst,
              src,
              repeat_times,
              dst_rep_stride,
              src_blk_stride,
              src_rep_stride,
              stride_unit=0):
        """Sum for all element in a repeat.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block between
                         adjacent iterations
        src_blk_stride : offset of src operator between different block
                         in one iteration
        src_rep_stride : offset of src operator in the same block between
                         adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # 0: dst is return the max number result of the repeat
        order = 0
        self._vector_whole_reduce_func('vcadd', mask, dst, src, repeat_times,
                                       dst_rep_stride, src_blk_stride,
                                       src_rep_stride, stride_unit,
                                       order, None)

    def vcmax(self,  # pylint: disable=R0913
              mask,
              dst,
              src,
              repeat_times,
              dst_rep_stride,
              src_blk_stride,
              src_rep_stride,
              stride_unit=0,
              index_low=False,
              max_cnt_index=None):
        """Get the max value and index for all element in a repeat.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_blk_stride : offset of src operator between different
                         block in one iteration
        src_rep_stride : offset of src operator in the same block
                        between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        order = 0
        if index_low:
            # 1: dst is return the index of max number of the repeat
            order = 1
        else:
            # 0: dst is return the max number result of the repeat
            order = 0
        self._vector_whole_reduce_func('vcmax', mask, dst, src, repeat_times,
                                       dst_rep_stride, src_blk_stride,
                                       src_rep_stride, stride_unit, order,
                                       max_cnt_index)

    def vcmin(self,  # pylint: disable=R0913
              mask,
              dst,
              src,
              repeat_times,
              dst_rep_stride,
              src_blk_stride,
              src_rep_stride,
              stride_unit=0,
              index_low=False,
              min_cnt_index=None):
        """Get the min value and index for all element in a repeat.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_blk_stride : offset of src operator between different block
                         in one iteration
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        order = 0
        if index_low:
            # 1: dst is return the index of max number of the repeat
            order = 1
        else:
            # 0: dst is return the max number result of the repeat
            order = 0
        self._vector_whole_reduce_func('vcmin', mask, dst, src, repeat_times,
                                       dst_rep_stride, src_blk_stride,
                                       src_rep_stride, stride_unit, order,
                                       min_cnt_index)

    @source_info_decorator(depth=2)
    @debug.vec_reduce_group_decorator
    def _vector_group_reduce_func(self, name,  # pylint: disable=R0913, R0914
                                  mask, dst, src, repeat_times,
                                  dst_rep_stride, src_blk_stride,
                                  src_rep_stride, stride_unit):
        """Instruction of group cross vecter operation

        Parameters
        ----------
        name : instruction name
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block between
                         adjacent iterations
        src_blk_stride : offset of src operator between different block
                         in one iteration
        src_rep_stride : offset of src operator in the same block between
                         adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        self._check_reduce_func_params(name, dst, src, repeat_times,
                                       dst_rep_stride, src_blk_stride,
                                       src_rep_stride, stride_unit, name)

        # check address overlap
        if src.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_rep_stride,
                             src_blk_stride, src_rep_stride, stride_unit)):
                check_address_overlapping(
                    name, mask, dst, src, _DEFAULT_NBLOCK,
                    ONE_REP_BYTE_SIZE // max(
                        get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                    _BLOCK_LEN, ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                    repeat_times, _DEFAULT_BLK_STRIDE, src_blk_stride,
                    dst_rep_stride, src_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src.offset).eval_value(), stride_unit)

        # check tensor overflow(static)
        vector_tensor_overflow_check(dst, mask, _DEFAULT_NBLOCK, _BLOCK_LEN,
                                     repeat_times, _DEFAULT_SRC_STRIDE,
                                     dst_rep_stride)
        src_bit_len = get_bit_len(src.dtype)
        vector_tensor_overflow_check(src, mask,
                                     (ONE_REP_BYTE_SIZE*ONE_BYTE_BIT_LEN //
                                      src_bit_len) // (ONE_REP_BYTE_SIZE //
                                                       src_bit_len),
                                     ONE_REP_BYTE_SIZE // src_bit_len,
                                     repeat_times, src_blk_stride,
                                     src_rep_stride)
        # gen
        config = [repeat_times, dst_rep_stride, src_blk_stride, src_rep_stride]

        mask_o = mask_concat(self, mask,
                             tensor_bit_len=max(
                                 get_bit_len(dst.dtype),
                                 get_bit_len(src.dtype)))

        with self.new_scope():
            dst_extent = Expr(((repeat_times - 1)*dst_rep_stride + 1)*
                              src_bit_len*_BLOCK_LEN //
                              ONE_BYTE_BIT_LEN)
            #when repeat time >1 and the count of dst write element is not
            # the multi of ONE_BLK_SIZE
            dst_extent = Expr(ceil_div(dst_extent, ONE_BLK_SIZE) * ONE_BLK_SIZE)
            instr = tvm.call_extern(
                dst.dtype, name, dst.access_ptr(
                    "w", extent=dst_extent.get()),
                src.access_ptr(
                    "r", extent=Expr(((repeat_times - 1)*src_rep_stride +
                                      (_BLOCK_LEN - 1)*src_blk_stride + 1)*
                                     ONE_BLK_SIZE).get()),
                *type_convert(config))
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

    def vcgadd(self,  # pylint: disable=R0913
               mask,
               dst,
               src,
               repeat_times,
               dst_rep_stride,
               src_blk_stride,
               src_rep_stride,
               stride_unit=0):
        """Sum for all element in a block.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_blk_stride : offset of src operator between different block
                         in one iteration
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        return self._vector_group_reduce_func('vcgadd', mask, dst, src,
                                              repeat_times, dst_rep_stride,
                                              src_blk_stride, src_rep_stride,
                                              stride_unit)

    def vcgmax(self,  # pylint: disable=R0913
               mask,
               dst,
               src,
               repeat_times,
               dst_rep_stride,
               src_blk_stride,
               src_rep_stride,
               stride_unit=0):
        """Get the max value and index for all element in a block.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_blk_stride : offset of src operator between different block
                         in one iteration
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        return self._vector_group_reduce_func('vcgmax', mask, dst, src,
                                              repeat_times, dst_rep_stride,
                                              src_blk_stride, src_rep_stride,
                                              stride_unit)

    def vcgmin(self,  # pylint: disable=R0913
               mask,
               dst,
               src,
               repeat_times,
               dst_rep_stride,
               src_blk_stride,
               src_rep_stride,
               stride_unit=0):
        """Get the min value and index for all element in a block.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_blk_stride : offset of src operator between different block
                         in one iteration
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        return self._vector_group_reduce_func('vcgmin', mask, dst, src,
                                              repeat_times, dst_rep_stride,
                                              src_blk_stride, src_rep_stride,
                                              stride_unit)

    @source_info_decorator(depth=2)
    @debug.vec_reduce_wo_order_decorator
    def _vector_pair_reduce_func(self, name,  # pylint: disable=R0913, R0914
                                 mask, dst, src, repeat_times,
                                 dst_rep_stride, src_blk_stride,
                                 src_rep_stride, stride_unit, print_name=None):
        """Instruction of pair reduce vecter operation

        Parameters
        ----------
        name : instruction name
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block between
				         adjacent iterations
        src_blk_stride : offset of src operator between different block
				         in one iteration
        src_rep_stride : offset of src operator in the same block between
				         adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # the arguments are too many and unused variable is used in decorator
        if print_name is None:
            print_name = name
        self._check_reduce_func_params(name, dst, src, repeat_times,
                                       dst_rep_stride, src_blk_stride,
                                       src_rep_stride, stride_unit, print_name)
        # mask
        mask_o = mask_concat(self, mask,
                             tensor_bit_len=max(get_bit_len(dst.dtype),
                                                get_bit_len(src.dtype)))

        block_len = 64
        # check address overlap
        if src.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_rep_stride,
                             src_blk_stride, src_rep_stride, stride_unit)):
                check_address_overlapping(
                    print_name, mask, dst, src, _DEFAULT_NBLOCK,
                    ONE_REP_BYTE_SIZE // max(
                        get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                    block_len, ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                    repeat_times, _DEFAULT_BLK_STRIDE, src_blk_stride,
                    dst_rep_stride, src_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src.offset).eval_value(), stride_unit)

        # check tensor overflow(static)
        vector_tensor_overflow_check(dst, mask, _DEFAULT_NBLOCK, block_len,
                                     repeat_times, _DEFAULT_SRC_STRIDE,
                                     dst_rep_stride)
        src_bit_len = get_bit_len(src.dtype)
        parallelism = ONE_REP_BYTE_SIZE*ONE_BYTE_BIT_LEN // src_bit_len
        vector_tensor_overflow_check(src, mask,
                                     parallelism // (ONE_REP_BYTE_SIZE //
                                                     src_bit_len),
                                     ONE_REP_BYTE_SIZE // src_bit_len,
                                     repeat_times, src_blk_stride,
                                     src_rep_stride)
        # gen
        config = [dst_rep_stride, src_blk_stride, src_rep_stride, repeat_times]
        offset_list = VECTOR_PAIR_OFFSET_LIST
        segment_list = VECTOR_PAIR_SEGMENT_LIST
        args = concat_params(config, offset_list, segment_list, dtype="uint64")
        with self.new_scope():
            src_extent = Expr(((repeat_times - 1)*src_rep_stride +
                               (_BLOCK_LEN - 1)*src_blk_stride + 1)*
                              ONE_BLK_SIZE)
            # dst_extent: The result continuous 128B , see ISA
            dst_extent = Expr((repeat_times - 1)*dst_rep_stride*128 + 128)
            instr = tvm.call_extern(dst.dtype, name,
                                    dst.access_ptr("w",
                                                   extent=dst_extent.get()),
                                    src.access_ptr("r",
                                                   extent=src_extent.get()),
                                    type_convert(args))
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

    def vcpadd(self,  # pylint: disable=R0913
               mask,
               dst,
               src,
               repeat_times,
               dst_rep_stride,
               src_blk_stride,
               src_rep_stride,
               stride_unit=0):
        """do parity summation on elements in adjacent pair

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block
                         between adjacent iterations
        src_blk_stride : offset of src operator between different block
                         in one iteration
        src_rep_stride : offset of src operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        return self._vector_pair_reduce_func('vcpadd', mask, dst, src,
                                             repeat_times, dst_rep_stride,
                                             src_blk_stride, src_rep_stride,
                                             stride_unit)

    @source_info_decorator()
    @debug.vreduce_decorator
    def vreduce(self, mask, dst,  # pylint: disable=R0913, R0914, R0915
                src0, src1_pattern, repeat_times,
                src0_blk_stride, src0_rep_stride, src1_rep_stride,
                stride_unit=0, rsvd_scalar=None, mask_mode="normal"):
        """
        source vector would be reduced into shorter vector according to the
        compare masks

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src0 : source operator
        src1_pattern : 6 fixed patterns for effective operation on element, or
                       user defined tensor(dtype usigned int), per bit 1/0 for
                       effective opration on element
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
                         in one iteration
        src0_rep_stride : offset of src0 operator in the same block
                         between adjacent iterations
        src1_rep_stride: offset of src1 operator in the same block
                         between adjacent iterations
        stride_unit : address and offset unit both affect it. default = 0
        rsvd_scalar : remaining elements count, default = none
        mask_mode: "normal" - mask normal mode
                   "counter" - mask counter mode

        Returns
        -------
        None
        """
        # check dst src
        TikCheckUtil.check_type_match(
            dst, Tensor,
            "dst should be tensor, input type of dst: {}".format(type(dst)))
        TikCheckUtil.check_equality(
            dst.scope, scope_ubuf,
            "dst scope should be ub, input dst scope: {}".format(dst.scope))
        TikCheckUtil.check_type_match(
            src0, Tensor,
            "src0 should be tensor, input type of src0: {}".format(type(src0)))
        TikCheckUtil.check_equality(
            src0.scope, scope_ubuf,
            "src0 scope should be ub, input src0 scope: {}".format(src0.scope))
        # check src1_pattern
        TikCheckUtil.check_type_match(
            src1_pattern, (int, Tensor, Scalar),
            "src_pattern should be int, Tensor or Scalar")
        check_scalar_dtype(src1_pattern,
                           "scalar_src1_pattern should be a scalar of int/uint")
        check_integer_in_range(
            src1_pattern, range(MIN_SRC1_PATTERN, MAX_SRC1_PATTERN),
            "src1_pattern should be in the range of [1, 6], "
            "input src1_pattern: {}".format(src1_pattern))
        if isinstance(src1_pattern, Tensor):
            pattern = 0
            TikCheckUtil.check_equality(
                src1_pattern.scope, scope_ubuf,
                "src1_pattern scope should be ub, input src1_pattern scope: "
                "{}".format(src1_pattern.scope))
        else:
            pattern = src1_pattern
        # check stride_unit
        TikCheckUtil.check_type_match(stride_unit, int,
                                      "stride_unit should be int")
        check_integer_in_range(
            stride_unit, range(MAX_STRIDE_UNIT),
            "stride_unit should be in the range of [0, 3], "
            "input stride_unit: {}".format(stride_unit))
        # check dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src0.dtype]
        TikCheckUtil.check_equality(dst.dtype, src0.dtype,
                                    "Intrinsic {}'s src0's dtype should be "
                                    "equal to dst's dtype".format("vreduce"))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            "vreduce",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, "vreduce"))

        if isinstance(src1_pattern, Tensor):
            if dtype_str in ("f16f16", "s16s16", "u16u16"):
                TikCheckUtil.check_equality(
                    src1_pattern.dtype, "uint16",
                    "When src dtype is b16, src1 dtype should be u16, "
                    "input src1 dtype: {}".format(src1_pattern.dtype))
            else:
                TikCheckUtil.check_equality(
                    src1_pattern.dtype, "uint32",
                    "When src dtype is b32, src1 dtype should be u32, "
                    "input src1 dtype: {}".format(src1_pattern.dtype))
        # check repeat_times
        check_repeat_times(repeat_times)
        # check strides
        TikCheckUtil.check_type_match(
            src0_blk_stride, (int, Scalar, Expr),
            "src0_blk_stride should be int, Scalar or Expr, input type of "
            "src0_blk_stride: {}".format(type(src0_blk_stride)))
        check_scalar_dtype(src0_blk_stride,
                           "scalar_src0_blk_stride "
                           "should be a scalar of int/uint")
        check_integer_in_range(
            src0_blk_stride, range(MAX_BLK_STRIDE_SINGLE_BYTE),
            "src0_blk_stride should be in the range of [0, 255], "
            "input src0_blk_stride: {}".format(src0_blk_stride))
        TikCheckUtil.check_type_match(
            src0_rep_stride, (int, Scalar, Expr),
            "src0_rep_stride should be int, Scalar or Expr, input type of "
            "src0_rep_stride: {}".format(type(src0_rep_stride)))
        check_integer_in_range(
            src0_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "src0_rep_stride should be in the range of [0, 255], "
            "input src0_rep_stride: {}".format(src0_rep_stride))
        TikCheckUtil.check_type_match(
            src1_rep_stride, (int, Scalar, Expr),
            "src1_rep_stride should be int, Scalar or Expr, input type of "
            "src1_rep_stride: {}".format(type(src1_rep_stride)))
        check_integer_in_range(
            src1_rep_stride, range(MAX_REP_STRIDE_SINGLE_BYTE),
            "src1_rep_stride should be in the range of [0, 255], "
            "input src1_rep_stride: {}".format(src1_rep_stride))
        check_vector_stride(None, [src0_rep_stride, src1_rep_stride],
                            None, MAX_REP_STRIDE_SINGLE_BYTE,
                            ["src0", "src1"])
        # check mask_mode
        TikCheckUtil.check_type_match(
            mask_mode, str, "mask_mode should be str, input type of "
                            "mask_mode: {}".format(type(mask_mode)))
        TikCheckUtil.check_in_range(
            mask_mode, ("normal", "counter"),
            "mask should be 'normal' or 'counter', "
            "input mask_mode: {}".format(mask_mode))
        # check rsvd_scalar
        if rsvd_scalar is not None:
            TikCheckUtil.check_type_match(
                rsvd_scalar, Scalar,
                "rsvd_scalar should be None or Scalar, input type of "
                "rsvd_scalar: {}".format(type(rsvd_scalar)))
        # mask
        mask_o = mask_concat(self, mask, mask_mode, get_bit_len(dst.dtype))

        # check address overlap
        if src0.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, src1_pattern,
                             src0_blk_stride, src0_rep_stride)):
                check_address_overlap_vreduce(
                    dst, src0, src1_pattern, mask, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    repeat_times, src0_blk_stride, src0_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src0.offset).eval_value(), stride_unit, mask_mode)

        # check tensor overflow
        if isinstance(src1_pattern, Tensor):
            check_vreduce_src1_overflow(
                mask, mask_mode, src1_rep_stride, src1_pattern,
                reduce_mul(src1_pattern.indice.origin_shape), repeat_times)
        _check_src0_overflow_vreduce(mask_mode, src0, repeat_times,
                                     src0_blk_stride, src0_rep_stride,
                                     stride_unit, mask)
        _check_dst_overflow_vreduce(mask_mode, src1_pattern, repeat_times, dst,
                                    mask)
        # change dtype_str
        dtype_str = change_dtype_str(dst)
        # code gen
        config = [src0_blk_stride, pattern, src0_rep_stride, src1_rep_stride,
                  stride_unit, repeat_times]
        args = concat_params(config, VREDUCE_OFFSET_LIST, VREDUCE_SEGMENT_LIST)
        # cal src0 dst extent
        src0_extent = cal_extent_stride_unit_mask(
            mask, repeat_times, src0, stride_unit, src0_blk_stride,
            src0_rep_stride, mask_mode)
        dst_extent = _cal_dst_extent_vreduce(mask_mode, repeat_times, dst, mask)

        # cal src1 extent
        src1_extent = _cal_src1_extent_vreduce(src1_pattern, src1_rep_stride,
                                               repeat_times, mask, mask_mode)
        self._gen_vreduce_code(mask_mode, dst, src0, src1_pattern, rsvd_scalar,
                               dtype_str, dst_extent, src0_extent,
                               src1_extent, mask_o, args)

    def _gen_vreduce_code(self, mask_mode,  # pylint: disable=R0913
                          dst, src0, src1_pattern,
                          rsvd_scalar, dtype_str, dst_extent, src0_extent,
                          src1_extent, mask_o, args):
        # issue instruction
        with self.new_scope():
            if mask_mode == "counter":
                orig_ctrl = set_ctrl_counter_mask(self)

            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            if isinstance(src1_pattern, Tensor):
                instr = tvm.call_extern(
                    dst.dtype, "vreduce",
                    dst.reinterpret_cast_to(dtype_str).access_ptr(
                        "w", extent=dst_extent),
                    src0.reinterpret_cast_to(dtype_str).access_ptr(
                        "r", extent=src0_extent),
                    src1_pattern.reinterpret_cast_to(dtype_str).access_ptr(
                        "r", extent=src1_extent),
                    args)
                self.emit(instr)
            else:
                instr = tvm.call_extern(
                    dst.dtype, "vreduce",
                    dst.reinterpret_cast_to(dtype_str).access_ptr(
                        "w", extent=dst_extent),
                    src0.reinterpret_cast_to(dtype_str).access_ptr(
                        "r", extent=src0_extent),
                    src0.reinterpret_cast_to(dtype_str).access_ptr(
                        "r", extent=src0_extent), args)
                self.emit(instr)

            # reset CTRL SPR as orig_ctrl
            if mask_mode == "counter":
                reset_ctrl_counter_mask(self, orig_ctrl)
        # read RSVD
        if isinstance(rsvd_scalar, Scalar):
            TikCheckUtil.check_equality(
                rsvd_scalar.dtype, "uint32",
                "rsvd_scalar dtype should be uint32, "
                "input rsvd_scalar dtype: %s" % rsvd_scalar.dtype)
            with self.new_scope():
                self.emit(
                    tvm.call_extern(rsvd_scalar.dtype, "reg_set",
                                    rsvd_scalar.get(),
                                    tvm.call_extern(rsvd_scalar.dtype,
                                                    "get_rsvd_cnt")))

    def _vreduce_create_mask_scalar(self, data_len):
        # mask_len record the valid data num of low mask
        # B16: max data num of low mask is 64
        # B32: max data num of low mask is 64, all data can select by low mask
        # out data saved as: Data1Index1Data2Index2....DatanIndexn
        # so valid data is half of max data num
        mask_len = 32

        high_mask_bit = self.Scalar_(dtype='int64',  # pylint: disable=E1101
                                     name='high_mask_bit')
        low_mask_bit = self.Scalar_(dtype='int64',  # pylint: disable=E1101
                                    name='low_mask_bit')
        high_mask = self.Scalar_(dtype='uint64',  # pylint: disable=E1101
                                 name='high_mask', init_value=0)
        low_mask = self.Scalar_(dtype='uint64',  # pylint: disable=E1101
                                name='low_mask', init_value=0)
        high_mask_bit.set_as(data_len - mask_len)

        # create mask as 01010101
        with self.for_range(0, high_mask_bit):
            high_mask.set_as(high_mask << 2)
            high_mask.set_as(high_mask | 1)

        with self.if_scope(data_len >= mask_len):
            low_mask_bit.set_as(mask_len)
        with self.else_scope():
            low_mask_bit.set_as(data_len)

        with self.for_range(0, low_mask_bit):
            low_mask.set_as(low_mask << 2)
            low_mask.set_as(low_mask | 1)
        return [high_mask, low_mask]

    def creat_mask_(self, data_len):
        """
        get mask in the "0101010101" format

        Parameters
        ----------
        data_len: int, src_element // 2, src_element is the element count of
                    vreduce instruction will cover

        Returns
        -------
        the new mask
        """
        if is_immediate_number(data_len):
            return vreduce_create_mask(data_len)
        return self._vreduce_create_mask_scalar(data_len)

    def _run_vreduce_max_min(self, name, mask, dst,  # pylint: disable=R0913, R0914
                             src, repeat_times, src_rep_stride,
                             dst_offset=0, src_offset=0):
        mask_o = mask_concat(self, mask,
                             tensor_bit_len=max(get_bit_len(dst.dtype),
                                                get_bit_len(src.dtype)))
        n_block = 1
        # check address overlap
        if src.buffer == dst.buffer:
            if all(isinstance(value, int) for value in
                   (repeat_times, 1, 1, src_rep_stride, 0)):
                block_len = 2
                if isinstance(dst_offset, int) and isinstance(src_offset, int):
                    check_address_overlapping(
                        name, mask, dst, src, n_block,
                        ONE_REP_BYTE_SIZE // max(
                            get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                        block_len, ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                        repeat_times, VREDUCE_DEFAULT_DST_BLK_STRIDE,
                        VREDUCE_DEFAULT_SRC_BLK_STRIDE,
                        VREDUCE_DEFAULT_DST_REP_STRIDE, src_rep_stride,
                        Expr(dst.offset + dst_offset).eval_value(),
                        Expr(src.offset + src_offset).eval_value())

        # check tensor overflow
        self._check_overflow(name, mask, dst, src, repeat_times,
                             VREDUCE_DEFAULT_DST_REP_STRIDE,
                             VREDUCE_DEFAULT_SRC_BLK_STRIDE,
                             src_rep_stride, dst_offset, src_offset)

        # gen
        config = [repeat_times, VREDUCE_DEFAULT_DST_REP_STRIDE,
                  VREDUCE_DEFAULT_SRC_BLK_STRIDE, src_rep_stride]
        if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
            config.append(0 & 0b01)
            config.append((0 & 0b10) >> 1)
            config.append(0)

        src_bit_len = get_bit_len(src.dtype)
        with self.new_scope():
            mem_access_param = type_convert(config)
            # 8 Block/repeat, 32Byte/Block
            src_extent = Expr(((repeat_times - 1) * src_rep_stride +
                               (8 - 1) * 1 + 1) * 32)

            # here is vcmax and vcmin, src f16 2B aligned, dst 4B aligned
            dst_extent = Expr(repeat_times*src_bit_len * VREDUCE_PER_REP_OUTPUT
                              // ONE_BYTE_BIT_LEN)

            # when repeat time >1 and the count of dst write element is not
            # the multi of ONE_BLK_SIZE
            dst_extent = Expr(ceil_div(dst_extent, ONE_BLK_SIZE) * ONE_BLK_SIZE)

            dst_offset_expr = Expr(dst_offset)
            src_offset_expr = Expr(src_offset)

            if src.dtype == "int16":
                instr = tvm.call_extern(
                    dst.dtype, name,
                    dst.reinterpret_cast_to("uint16").access_ptr(
                        "w", extent=dst_extent.get(),
                        offset=dst_offset_expr.get()),
                    src.reinterpret_cast_to("uint16").access_ptr(
                        "r", extent=src_extent.get(),
                        offset=src_offset_expr.get()),
                    *mem_access_param)
            else:
                instr = tvm.call_extern(
                    dst.dtype, name,
                    dst.access_ptr("w", extent=dst_extent.get(),
                                   offset=dst_offset_expr.get()),
                    src.access_ptr("r", extent=src_extent.get(),
                                   offset=src_offset_expr.get()),
                    *mem_access_param)
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

    def run_reduce_func_(self, func, mask, dst, src,  # pylint: disable=R0913
                         repeat_times, src_rep_stride):
        """
        For vreduce, there are total 3 iterations.
        This function is the first one. We compute the data into work_tensor

        Parameters
        ----------
        func: function: detail instruction
        mask: int/Scalar/list, effective operation on element.
        dst: Tensor, store results in it
        src: Tensor, the source tensor
        repeat_times: int, the times of instruction run
        src_rep_stride: int/Scalar, the stride between each repeat in src

        Returns
        -------
        the number of the first iteration output
        """
        max_repeat_times = MAX_REPEAT_TIMES - 1
        for_range_times = repeat_times // max_repeat_times
        dtype_len = DTYPE_SIZE[src.dtype]

        # all dst_rep_stride and src_block_stride set to 1,
        # select max or min data from continuous elements of one repeat
        with self.for_range(0, for_range_times) as index:
            self._run_vreduce_max_min(
                func, mask, dst, src,
                max_repeat_times, src_rep_stride,
                index*max_repeat_times*VREDUCE_PER_REP_OUTPUT,
                index*max_repeat_times*src_rep_stride*ONE_BLK_SIZE//dtype_len)

        left_repeat_times = repeat_times % max_repeat_times

        if is_immediate_number(repeat_times):
            if left_repeat_times > 0:
                self._run_vreduce_max_min(
                    func, mask, dst, src,
                    left_repeat_times, src_rep_stride,
                    for_range_times * max_repeat_times * VREDUCE_PER_REP_OUTPUT,
                    for_range_times * max_repeat_times * src_rep_stride *
                    ONE_BLK_SIZE // dtype_len)
            return VREDUCE_PER_REP_OUTPUT*repeat_times

        with self.if_scope(left_repeat_times > 0):
            self._run_vreduce_max_min(
                func, mask, dst, src,
                left_repeat_times, src_rep_stride,
                for_range_times * max_repeat_times * VREDUCE_PER_REP_OUTPUT,
                for_range_times * max_repeat_times * src_rep_stride *
                ONE_BLK_SIZE // dtype_len)
        return VREDUCE_PER_REP_OUTPUT*repeat_times
