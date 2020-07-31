"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_vecotr_api.py
DESC:     provide vector instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
# disabling:
# R0913: too-many-arguments
# R0914: too-many-locals
# C0302: too-many-lines
# R0904: too-many-public-methods
# W0613: unused-argument
# too many lines in this file(more than 1000), so disable it.

from te import tvm  # pylint: disable=C0302
from te.platform import cce_params
from te.platform.cce_conf import intrinsic_check_support
from te.platform.cce_params import scope_ubuf
from te.platform.cce_params import HI3796CV300ES
from te.platform.cce_params import ASCEND_910
from te.platform.cce_params import ASCEND_310
from te.platform.cce_params import ASCEND_610
from te.platform.cce_params import ASCEND_620
from ..api.tik_tensor import Tensor
from .tik_expr import Expr
from .. import debug
from ..api.tik_ir_builder import TikIRBuilder
from .tik_util import type_convert, dtype_convert, insert_set_deqscale_attr, \
    concat_params
from .tik_api_util import set_ctrl_counter_mask, reset_ctrl_counter_mask, \
    check_repeat_times
from ..api.tik_scalar import mask_concat, Scalar
from .tik_params import CMPMASK_VAR, ONE_REP_BYTE_SIZE, ONE_BYTE_BIT_LEN, \
    MAX_BLK_STRIDE_DOUBLE_BYTE, MAX_BLK_STRIDE_SINGLE_BYTE, \
    MAX_REP_STRIDE_SINGLE_BYTE, PIPE_V, MAX_STRIDE_UNIT, ONE_BLK_SIZE, \
    MAX_VSEL_MODE, VSEL_MODE_TENSOR_SCALAR, VSEL_MODE_DOUBLE_TENSOR_MANY_IT, \
    VSEL_MODE_DOUBLE_TENSOR_ONE_IT, VSEL_OFFSET_LIST, VSEL_SEGMENT_LIST, \
    VEC_SCALAR_OFFSET_LIST, VEC_SCALAR_SEGMENT_LIST, ONE_IR, TWO_IR, \
    BLK_NUM_PER_REP, INSTR_DTYPE_SUPPORT_STATEMENT, MASK_LOW_IDX, MASK_HIGH_IDX
from ..common.util import check_integer_in_range, get_bit_len, \
    check_scalar_dtype, is_immediate_number, ceil_div, DTYPE_SIZE
from ..common.common_util import vector_tensor_overflow_check, \
    check_tensor_overflow, check_vector_stride, check_sel_overflow, \
    check_vconv_deqs162b8_overflow, check_sel_dst_overlap, \
    check_address_overlapping, check_vconv_deqs162b8_overlap, \
    cal_extent_stride_unit_mask, check_vshl_vshr_scalar, vector_max_offset_cal
from ..common.tik_get_soc_name import get_soc_name
from .tik_api_constants import DTYPE_MAP
from .tik_api_constants import ROUND_MODE_MAP
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator
from ..tik_lib.tik_expr import BasicExpr
from .tik_params import CONST_FIVE_THREE
from .tik_params import CONST_ONE_THREE
from .tik_params import CONST_NEG_FOUR_THREE
from .tik_params import LOG_FOUR_THREE
from .tik_params import CONST_NEG_ONE
from .tik_params import CONST_ONE
from .tik_params import CONST_HALF
from .tik_params import CONST_THREE_FOUR
from .tik_params import CONST_ONE_FIVE
from .tik_params import CONST_NEG_ONE_FOUR
from .tik_params import CONST_NEG_HALF
from .tik_params import CONST_ONE_NINE
from .tik_params import CONST_NEG_ONE_EIGHT
from .tik_params import CONST_ONE_SEVEN
from .tik_params import CONST_NEG_ONE_SIX

# round disable
_ROUND_TO_NEAREST_ENABLE = 0
_MIN_DST_BLK_STRIDE = 1
_BLOCK_LEN = 8


def _calculate_extent(repeat_times, rep_stride, block_count, blk_stride):
    """calculate extent

    Parameters
    ----------
    repeat_times : Repeated iterations times
    rep_stride : stride of operator in the same block between repeats
    block_count: block number in one repeat
    blk_stride : stride of operator between different block

    Returns
    -------
    extent
    """
    extent = Expr(((repeat_times - 1)*rep_stride +
                   (block_count - 1)*blk_stride + 1)*ONE_BLK_SIZE)
    return extent.get()


def _check_vector_binary_address_overlap(  # pylint: disable=R0913
        name, mask, dst, src0, src1, repeat_times, dst_blk_stride,
        src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
        src1_rep_stride, stride_unit, print_name):
    """check same buffer"""
    can_ovelap_instr_name = ["vadd", "vsub",
                             "vmul", "vmax", "vmin", "vor", "vand"]
    if dst.buffer == src0.buffer:
        if all(isinstance(value, int) for \
               value in (repeat_times, dst_blk_stride, dst_rep_stride,
                         src0_blk_stride, src0_rep_stride, stride_unit)):
            check_address_overlapping(
                print_name, mask, dst, src0, BLK_NUM_PER_REP,
                ONE_REP_BYTE_SIZE //
                max(get_bit_len(dst.dtype), get_bit_len(src0.dtype)),
                ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                ONE_REP_BYTE_SIZE // get_bit_len(src0.dtype),
                repeat_times, dst_blk_stride, src0_blk_stride,
                dst_rep_stride, src0_rep_stride,
                Expr(dst.offset).eval_value(),
                Expr(src0.offset).eval_value(),
                stride_unit, msg="dst and src0")

    if dst.buffer == src1.buffer:
        if all(isinstance(value, int) for \
               value in (repeat_times, dst_blk_stride, dst_rep_stride,
                         src1_blk_stride, src1_rep_stride, stride_unit)):
            if name not in can_ovelap_instr_name or \
                    dst.dtype not in ("float16", "int32", "float32") or \
                    repeat_times == 1:
                check_address_overlapping(
                    print_name, mask, dst, src1, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(dst.dtype), get_bit_len(src1.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src1.dtype),
                    repeat_times, dst_blk_stride, src1_blk_stride,
                    dst_rep_stride, src1_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src1.offset).eval_value(),
                    stride_unit, msg="dst and src1")
            else:
                _check_dst_src1_overlap_other(
                    print_name, dst, src0, src1, mask,
                    repeat_times, dst_blk_stride,
                    src0_blk_stride, src1_blk_stride, dst_rep_stride,
                    src0_rep_stride, src1_rep_stride, stride_unit)


def _check_dst_src1_overlap_other(name, dst, src0,  # pylint: disable=R0913
                                  src1, mask, repeat_times, dst_blk_stride,
                                  src0_blk_stride, src1_blk_stride,
                                  dst_rep_stride, src0_rep_stride,
                                  src1_rep_stride, stride_unit):
    """check dst src1 overlap other case"""
    # function's input params is too much, so disable them
    if Expr(dst.offset).eval_value() is not None and \
            Expr(src1.offset).eval_value() is not None:
        if Expr(dst.offset).eval_value() != \
                Expr(src1.offset).eval_value() or \
                dst_blk_stride != src1_blk_stride or \
                dst_rep_stride != 0 or src1_rep_stride != 0:
            check_address_overlapping(
                name, mask, dst, src1, BLK_NUM_PER_REP,
                ONE_REP_BYTE_SIZE //
                max(get_bit_len(dst.dtype), get_bit_len(src1.dtype)),
                ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                ONE_REP_BYTE_SIZE // get_bit_len(src1.dtype),
                repeat_times, dst_blk_stride, src1_blk_stride,
                dst_rep_stride, src1_rep_stride,
                Expr(dst.offset).eval_value(),
                Expr(src1.offset).eval_value(),
                stride_unit, msg="when dst and src1 "
                                 "are not 100% same, dst and src1")

            if src0.buffer == src1.buffer:
                if all(isinstance(value, int) for \
                       value in (src0_blk_stride,
                                 src0_rep_stride)):
                    try:
                        check_address_overlapping(
                            name, mask, src0, src1, BLK_NUM_PER_REP,
                            ONE_REP_BYTE_SIZE //
                            max(get_bit_len(src0.dtype),
                                get_bit_len(src1.dtype)),
                            ONE_REP_BYTE_SIZE // get_bit_len(src0.dtype),
                            ONE_REP_BYTE_SIZE // get_bit_len(src1.dtype),
                            repeat_times, src0_blk_stride, src1_blk_stride,
                            src0_rep_stride, src1_rep_stride,
                            Expr(src0.offset).eval_value(),
                            Expr(src1.offset).eval_value(),
                            stride_unit, msg="src0 and src1")

                        check_address_overlapping(
                            name, mask, src1, src0, BLK_NUM_PER_REP,
                            ONE_REP_BYTE_SIZE //
                            max(get_bit_len(src0.dtype),
                                get_bit_len(src1.dtype)),
                            ONE_REP_BYTE_SIZE // get_bit_len(src1.dtype),
                            ONE_REP_BYTE_SIZE // get_bit_len(src0.dtype),
                            repeat_times, src1_blk_stride, src0_blk_stride,
                            src1_rep_stride, src0_rep_stride,
                            Expr(src1.offset).eval_value(),
                            Expr(src0.offset).eval_value(),
                            stride_unit, msg="src1 and src0")
                    except (RuntimeError, SystemExit):
                        TikCheckUtil.raise_error(
                            "when repeat_times>1, "
                            "{} dst and src1 address overlap is not"
                            " support any address overlapping"
                            " between src0 and src1.".format(
                                name))


class TikVectorApi(TikIRBuilder):  # pylint: disable=R0904
    """
    Vector, Serialization, Spr Operation Api
    """
    # because vecotr gather instruction are here
    def __init__(self):
        super(TikVectorApi, self).__init__()

    @source_info_decorator()
    @debug.vadddeqrelu_decorator
    def vadddeqrelu(self,  # pylint: disable=R0913, R0914
                    mask,
                    dst,
                    deqscale,
                    src0,
                    src1,
                    repeat_times,
                    dst_blk_stride,
                    src0_blk_stride,
                    src1_blk_stride,
                    dst_rep_stride,
                    src0_rep_stride,
                    src1_rep_stride,
                    stride_unit=0):
        """vadd+vconv+vrelu

        Parameters
        ----------
        mask : Effective operation on element,
                divided into two model: Continuous and bit by bit.
        dst : destination operator
        deqscale : dst_i=relu(dequant*(dequant(src0_i+src1_i)*2^17))
        src0 : source operation
        src1 : source operation
        repeat_times: Repeated iterations times
        dst_blk_stride: offset of dst operator between different block
        src0_blk_stride: offset of src operator between different block
        src1_blk_stride: offset of src operator between different block
        dst_rep_stride: offset of dst operator in the same block between
                        two repeats
        src0_rep_stride: offset of src operator in the same block between
                         two repeats
        src1_rep_stride: offset of src operator in the same block between
                         two repeats
        stride_unit: address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # because arguments is too many and stride_unit is used in decorator
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride([dst_blk_stride, src0_blk_stride, src1_blk_stride],
                            [dst_rep_stride, src0_rep_stride, src1_rep_stride],
                            MAX_BLK_STRIDE_SINGLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE,
                            ["dst", "src0", "src1"])
        # check tensor dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src0.dtype] + DTYPE_MAP[
            src1.dtype]
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                            + "vadddeqrelu",
                                                            dtype_str), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dtype_str, "vadddeqrelu"))
        # mask
        mask_o = mask_concat(self, mask,
                             tensor_bit_len=max(get_bit_len(dst.dtype),
                                                get_bit_len(src1.dtype),
                                                get_bit_len(src0.dtype)))

        # check address overlap
        if src0.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_blk_stride, dst_rep_stride,
                             src0_blk_stride, src0_rep_stride)):
                check_address_overlapping(
                    "vadddeqrelu", mask, dst, src0, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(dst.dtype), get_bit_len(src0.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src0.dtype),
                    repeat_times, dst_blk_stride, src0_blk_stride,
                    dst_rep_stride, src0_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src0.offset).eval_value(),
                    stride_unit, msg="dst and src0")
        # check address overlap
        if src1.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_blk_stride, dst_rep_stride,
                             src1_blk_stride, src1_rep_stride)):
                check_address_overlapping(
                    "vadddeqrelu", mask, dst, src1, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(dst.dtype), get_bit_len(src1.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src1.dtype),
                    repeat_times, dst_blk_stride, src1_blk_stride,
                    dst_rep_stride, src1_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src1.offset).eval_value(),
                    stride_unit, msg="dst and src1")

        # check tensor overflow(static)
        check_tensor_overflow(
            (dst, src0, src1), mask, repeat_times,
            (dst_blk_stride, src0_blk_stride, src1_blk_stride),
            (dst_rep_stride, src0_rep_stride, src1_rep_stride),
            ("dst", "src0", "src1"))
        # gen
        config = [
            repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
            dst_rep_stride, src0_rep_stride, src1_rep_stride
        ]
        TikCheckUtil.check_type_match(
            deqscale, (int, float, Scalar),
            "deqscale only support immediate mode or Scalar")
        if isinstance(deqscale, Scalar):
            TikCheckUtil.check_equality(deqscale.dtype, "float16",
                                        "deqscale scalar must be float16")
        res_args = Expr(deqscale, dtype="float16")
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(
                tvm.call_extern("float16", "set_deqscale",
                                type_convert(res_args)),
                ONE_IR)
        with self.new_scope():
            mem_access_param = type_convert(config)
            instr = tvm.call_extern(dst.dtype, "vadddeqrelu",
                                    dst.access_ptr("w", extent=
                                                   _calculate_extent(
                                                       repeat_times,
                                                       dst_rep_stride,
                                                       _BLOCK_LEN,
                                                       dst_blk_stride)),
                                    src0.access_ptr("r", extent=
                                                    _calculate_extent(
                                                        repeat_times,
                                                        src0_rep_stride,
                                                        _BLOCK_LEN,
                                                        src0_blk_stride)),
                                    src1.access_ptr("r", extent=
                                                    _calculate_extent(
                                                        repeat_times,
                                                        src1_rep_stride,
                                                        _BLOCK_LEN,
                                                        src1_blk_stride)),
                                    *mem_access_param)
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

    def _gen_vector_single_elewise_code(self, mask,  # pylint: disable=R0913
                                        dst, src, name, config,
                                        repeat_times, dst_rep_stride,
                                        dst_blk_stride, src_rep_stride,
                                        src_blk_stride, mask_o=None):
        """generate code for vector_single_elewise_func"""
        # function's input params is too much, so disable them
        if mask_o is None:
            mask_o = mask_concat(self, mask,
                                 tensor_bit_len=max(get_bit_len(dst.dtype),
                                                    get_bit_len(src.dtype)))
        with self.new_scope():
            instr = tvm.call_extern(dst.dtype, name,
                                    dst.access_ptr("w", extent=
                                                   _calculate_extent(
                                                       repeat_times,
                                                       dst_rep_stride,
                                                       _BLOCK_LEN,
                                                       dst_blk_stride)),
                                    src.access_ptr("r", extent=
                                                   _calculate_extent(
                                                       repeat_times,
                                                       src_rep_stride,
                                                       _BLOCK_LEN,
                                                       src_blk_stride)),
                                    *type_convert(config))
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

    # X mode - 1src1dst
    @source_info_decorator(depth=2)
    @debug.vec_single_elewise_func_dec
    def _vector_single_elewise_func(self, name, mask,  # pylint: disable=R0913
                                    dst, src, repeat_times,
                                    dst_blk_stride, src_blk_stride,
                                    dst_rep_stride, src_rep_stride,
                                    stride_unit, print_name=None, mask_o=None):
        """Instruction works by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        if print_name is None:
            print_name = name
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride([dst_blk_stride, src_blk_stride],
                            [dst_rep_stride, src_rep_stride],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])
        # check stride_unit
        TikCheckUtil.check_type_match(
            stride_unit, (int, Scalar, Expr),
            "stride_unit should be int, Scalar or Expr")
        check_scalar_dtype(stride_unit,
                           "scalar_stride_unit should be a scalar of int/uint")
        check_integer_in_range(stride_unit, range(MAX_STRIDE_UNIT),
                               "stride_unit should be in the range of [0, 3]")
        # check tensor
        TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        # check scope
        TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                    "src's scope must be UB")
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB")
        # check tensor dtype
        TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                    "Intrinsic {}'s src's "
                                    "dtype should be equal to dst's dtype".
                                    format(print_name))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                            + name,
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, print_name))
        # check address overlap
        if src.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_blk_stride, dst_rep_stride,
                             src_blk_stride, src_rep_stride, stride_unit)):
                if dst_blk_stride == 0:
                    check_address_overlapping(
                        print_name, mask, dst, src, BLK_NUM_PER_REP,
                        ONE_REP_BYTE_SIZE //
                        max(get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                        ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                        ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                        repeat_times, _MIN_DST_BLK_STRIDE, src_blk_stride,
                        dst_rep_stride, src_rep_stride,
                        Expr(dst.offset).eval_value(),
                        Expr(src.offset).eval_value(), stride_unit)
                else:
                    check_address_overlapping(
                        print_name, mask, dst, src, BLK_NUM_PER_REP,
                        ONE_REP_BYTE_SIZE //
                        max(get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                        ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                        ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                        repeat_times, dst_blk_stride, src_blk_stride,
                        dst_rep_stride, src_rep_stride,
                        Expr(dst.offset).eval_value(),
                        Expr(src.offset).eval_value(), stride_unit)

        # check tensor overflow(static)
        if all(isinstance(value, int) for \
               value in (repeat_times, dst_blk_stride, dst_rep_stride,
                         src_blk_stride, src_rep_stride)):
            if dst_blk_stride == 0:
                check_tensor_overflow((dst, src), mask, repeat_times,
                                      (_MIN_DST_BLK_STRIDE, src_blk_stride),
                                      (dst_rep_stride, src_rep_stride),
                                      ("dst", "src"))
            else:
                check_tensor_overflow((dst, src), mask, repeat_times,
                                      (dst_blk_stride, src_blk_stride),
                                      (dst_rep_stride, src_rep_stride),
                                      ("dst", "src"))
        # code gen
        config = [
            repeat_times, dst_blk_stride, src_blk_stride,
            dst_rep_stride, src_rep_stride
        ]
        # fix me when pass support stride_unit param
        self._gen_vector_single_elewise_code(mask, dst, src, name, config,
                                             repeat_times, dst_rep_stride,
                                             dst_blk_stride, src_rep_stride,
                                             src_blk_stride, mask_o)

    def vrelu(self,  # pylint: disable=R0913
              mask,
              dst,
              src,
              repeat_times,
              dst_blk_stride,
              src_blk_stride,
              dst_rep_stride,
              src_rep_stride,
              stride_unit=0):
        """Do linear rectification by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_single_elewise_func('vrelu', mask, dst, src, repeat_times,
                                         dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit)

    def vexp(self,  # pylint: disable=R0913
             mask,
             dst,
             src,
             repeat_times,
             dst_blk_stride,
             src_blk_stride,
             dst_rep_stride,
             src_rep_stride,
             stride_unit=0):
        """Get natural index by single element.

        Parameters
        ----------
        mask : Effective operation on element,
               divided into two model: Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_single_elewise_func('vexp', mask, dst, src, repeat_times,
                                         dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit)

    def vln(self,  # pylint: disable=R0913
            mask,
            dst,
            src,
            repeat_times,
            dst_blk_stride,
            src_blk_stride,
            dst_rep_stride,
            src_rep_stride,
            stride_unit=0,
            name="vln"):
        """Get natural logarithm by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0
        name: instructions name

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_single_elewise_func('vln', mask, dst, src, repeat_times,
                                         dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit, print_name=name)

    def vabs(self,  # pylint: disable=R0913
             mask,
             dst,
             src,
             repeat_times,
             dst_blk_stride,
             src_blk_stride,
             dst_rep_stride,
             src_rep_stride,
             stride_unit=0):
        """Get absolute value by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_single_elewise_func('vabs', mask, dst, src, repeat_times,
                                         dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit)

    def vrec(self,  # pylint: disable=R0913
             mask,
             dst,
             src,
             repeat_times,
             dst_blk_stride,
             src_blk_stride,
             dst_rep_stride,
             src_rep_stride,
             stride_unit=0):
        """Get reciprocal by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_single_elewise_func('vrec', mask, dst, src, repeat_times,
                                         dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit)

    def _vec_rec_high_preci(self,  # pylint: disable=R0913, R0914
                            name,
                            mask,
                            dst,
                            src,
                            work_tensor,
                            repeat_times,
                            dst_rep_stride,
                            src_rep_stride,
                            tensor_split_size,
                            mask_o):
        """Get high precision reciprocal using newton method.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        work_tensor : temporary operation
        repeat_times : Repeated iterations times
        dst_rep_stride : offset of dst operator in the same block between
                        two repeats
        src_rep_stride : offset of src operator in the same block between
                        two repeats
        tensor_split_size : the size of single tensor

        Returns
        -------
        None
        """
        const_num_neg_one = -1
        const_num_two = 2

        dst_blk_stride = 1
        src_blk_stride = 1
        stride_unit = 0

        work_tensor0 = work_tensor[0:tensor_split_size]
        work_tensor1 = work_tensor[tensor_split_size:]

        self._vector_single_elewise_func(
            'vrec', mask, dst, src, repeat_times, dst_blk_stride,
            src_blk_stride, dst_rep_stride, src_rep_stride,
            stride_unit, print_name=name, mask_o=mask_o)

        # iteration 1
        # compute tmp: a*x(n-1)
        self._vector_binary_tenary_elewise_func(
            'vmul', mask, work_tensor0, src, dst, repeat_times,
            src_blk_stride, src_blk_stride, dst_blk_stride,
            src_rep_stride, src_rep_stride, dst_rep_stride,
            stride_unit, print_name=name, mask_o=mask_o)

        # compute tmp: -1*tmp
        self._vector_scalar_single_elewise_func(
            'vmuls', mask, work_tensor1, work_tensor0,
            const_num_neg_one, repeat_times, src_blk_stride,
            src_blk_stride, src_rep_stride, src_rep_stride,
            stride_unit, 0, print_name=name, mask_o=mask_o)

        # compute tmp: 2+tmp
        self._vector_scalar_single_elewise_func(
            'vadds', mask, work_tensor0, work_tensor1,
            const_num_two, repeat_times, src_blk_stride,
            src_blk_stride, src_rep_stride, src_rep_stride,
            stride_unit, 0, print_name=name, mask_o=mask_o)

        # compute x(n): tmp*x(n-1)
        self._vector_binary_tenary_elewise_func(
            'vmul', mask, work_tensor1, work_tensor0, dst,
            repeat_times, dst_blk_stride, src_blk_stride,
            dst_blk_stride, src_rep_stride, src_rep_stride,
            dst_rep_stride, stride_unit, print_name=name, mask_o=mask_o)

        # iteration 2
        # compute tmp: a*x(n-1)
        self._vector_binary_tenary_elewise_func(
            'vmul', mask, work_tensor0, src, work_tensor1, repeat_times,
            src_blk_stride, src_blk_stride, dst_blk_stride,
            src_rep_stride, src_rep_stride, src_rep_stride,
            stride_unit, print_name=name, mask_o=mask_o)

        # compute tmp: -1*tmp
        self._vector_scalar_single_elewise_func(
            'vmuls', mask, dst, work_tensor0,
            const_num_neg_one, repeat_times, src_blk_stride,
            src_blk_stride, dst_rep_stride, src_rep_stride,
            stride_unit, 0, print_name=name, mask_o=mask_o)

        # compute tmp: 2+tmp
        self._vector_scalar_single_elewise_func(
            'vadds', mask, work_tensor0, dst,
            const_num_two, repeat_times, src_blk_stride,
            src_blk_stride, src_rep_stride, dst_rep_stride,
            stride_unit, 0, print_name=name, mask_o=mask_o)

        # compute x(n): tmp*x(n-1)
        self._vector_binary_tenary_elewise_func(
            'vmul', mask, dst, work_tensor0, work_tensor1, repeat_times,
            dst_blk_stride, src_blk_stride, dst_blk_stride,
            dst_rep_stride, src_rep_stride, src_rep_stride,
            stride_unit, print_name=name, mask_o=mask_o)

    def vsqrt(self,  # pylint: disable=R0913
              mask,
              dst,
              src,
              repeat_times,
              dst_blk_stride,
              src_blk_stride,
              dst_rep_stride,
              src_rep_stride,
              stride_unit=0):
        """sqrt by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_single_elewise_func('vsqrt', mask, dst, src, repeat_times,
                                         dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit)

    def vrsqrt(self,  # pylint: disable=R0913
               mask,
               dst,
               src,
               repeat_times,
               dst_blk_stride,
               src_blk_stride,
               dst_rep_stride,
               src_rep_stride,
               stride_unit=0):
        """ vsqrt + vrec

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_single_elewise_func('vrsqrt', mask, dst, src,
                                         repeat_times, dst_blk_stride,
                                         src_blk_stride, dst_rep_stride,
                                         src_rep_stride, stride_unit)

    def _get_extend_not_list(self, mask,
                             repeat_times, src_rep_stride, dtype):
        """get work tensor extend when mask not list"""
        block_len = 32 // DTYPE_SIZE[dtype]
        if isinstance(mask, BasicExpr):  # pylint: disable=R1705
            wk_tensor_extend = self.Scalar_(  # pylint: disable=E1101
                init_value=0, dtype="uint64", name="wk_tensor_extend0")
            with self.if_scope(mask % block_len == 0):
                max_extend = (repeat_times - 1)*src_rep_stride*block_len +\
                             (mask // block_len - 1)*block_len + block_len
                wk_tensor_extend.set_as(max_extend)
            with self.else_scope():
                max_extend = (repeat_times - 1)*src_rep_stride*block_len +\
                             (mask // block_len)*block_len + mask % block_len
                max_extend = ceil_div(max_extend, block_len)*block_len
                wk_tensor_extend.set_as(max_extend)
            return wk_tensor_extend

        else:
            wk_tensor_extend = vector_max_offset_cal(
                mask, dtype, block_len, repeat_times, 1, src_rep_stride)
            wk_tensor_extend = ceil_div(wk_tensor_extend, block_len)*block_len
            return wk_tensor_extend

    def get_wk_tensor_extend(self, mask, dtype,
                             repeat_times, src_rep_stride):
        """get work tensor size single"""
        block_len = 32 // DTYPE_SIZE[dtype]
        if isinstance(mask, (tuple, list)):
            if not isinstance(    # pylint: disable=R1705
                    mask[MASK_LOW_IDX], BasicExpr):
                wk_tensor_extend = vector_max_offset_cal(
                    mask, dtype, block_len, repeat_times, 1, src_rep_stride)
                wk_tensor_extend = ceil_div(wk_tensor_extend,
                                            block_len)*block_len
                return wk_tensor_extend

            else:
                # mask_list_scalar
                wk_tensor_extend = self.Scalar_(  # pylint: disable=E1101
                    init_value=0, dtype="uint64", name="wk_tensor_extend")
                mask_len = self.Scalar_(init_value=0,  # pylint: disable=E1101
                                        dtype="uint64", name="mask_len")
                # get mask_len
                with self.if_scope(mask[MASK_HIGH_IDX] == 0):
                    with self.for_range(0, 64) as index:
                        with self.if_scope(
                                (mask[MASK_LOW_IDX] >> index) & 1 == 1):
                            mask_len.set_as(index + 1)
                with self.else_scope():
                    with self.for_range(0, 64) as index:
                        with self.if_scope(
                                (mask[MASK_HIGH_IDX] >> index) & 1 == 1):
                            mask_len.set_as(64 + index + 1)

                # get extend
                with self.if_scope(mask_len % block_len == 0):
                    max_extend = (repeat_times - 1)*src_rep_stride*block_len +\
                                 (mask_len // block_len - 1)*block_len + \
                                 block_len
                    wk_tensor_extend.set_as(max_extend)
                with self.else_scope():
                    max_extend = (repeat_times - 1)*src_rep_stride*block_len +\
                                 (mask_len // block_len)*block_len +\
                                 mask_len % block_len
                    max_extend = ceil_div(max_extend, block_len)*block_len
                    wk_tensor_extend.set_as(max_extend)
                return wk_tensor_extend
        else:
            return self._get_extend_not_list(mask, repeat_times,
                                             src_rep_stride, dtype)

    def high_rsqrt_cloud(self, name, mask,  # pylint: disable=R0913
                         dst, src, work_tensor, repeat_times,
                         dst_rep_stride, src_rep_stride,
                         tensor_split_size, mask_o):
        """cloud"""
        default_stride = 1
        work_tensor0 = work_tensor[0:tensor_split_size]
        work_tensor1 = work_tensor[tensor_split_size:]
        self._vector_single_elewise_func(
            'vsqrt', mask, work_tensor0, src, repeat_times, default_stride,
            default_stride, src_rep_stride, src_rep_stride,
            0, print_name=name, mask_o=mask_o)
        self._vec_rec_high_preci(
            name, mask, dst, work_tensor0, work_tensor1,
            repeat_times, dst_rep_stride, src_rep_stride,
            tensor_split_size, mask_o)

    def high_rsqrt_mini(self, name, mask, dst,  # pylint: disable=R0913, R0914
                        src, work_tensor, repeat_times,
                        dst_rep_stride, src_rep_stride,
                        tensor_split_size, mask_o):
        """vrsqrt instr by newton iter method"""
        default_stride = 1
        stride_unit = 0
        # calc ele
        work_tensor1 = work_tensor[0:tensor_split_size]
        work_tensor2 = work_tensor[tensor_split_size:2*tensor_split_size]
        work_tensor3 = work_tensor[2*tensor_split_size:]
        # newton init value: vrsqrt
        self._vector_single_elewise_func(
            'vrsqrt', mask, dst, src, repeat_times, default_stride,
            default_stride, dst_rep_stride, src_rep_stride,
            stride_unit, print_name=name, mask_o=mask_o)
        # mini
        iteration_times = 2
        with self.for_range(0, iteration_times):
            # method 3
            # dst maybe not allow as work_tensor
            self._vector_binary_tenary_elewise_func(
                'vmul', mask, work_tensor1, src, dst, repeat_times,
                default_stride, default_stride, default_stride,
                src_rep_stride, src_rep_stride, dst_rep_stride,
                stride_unit, print_name=name, mask_o=mask_o)
            self._vec_rec_high_preci(
                name, mask, work_tensor2, work_tensor1,
                work_tensor3, repeat_times, src_rep_stride,
                src_rep_stride, tensor_split_size, mask_o)
            self._vector_binary_tenary_elewise_func(
                'vadd', mask, work_tensor1, dst, work_tensor2,
                repeat_times, default_stride, default_stride,
                default_stride, src_rep_stride, dst_rep_stride,
                src_rep_stride, stride_unit, print_name=name, mask_o=mask_o)
            self._vector_scalar_single_elewise_func(
                'vmuls', mask, dst, work_tensor1, 0.5, repeat_times,
                default_stride, default_stride, dst_rep_stride,
                src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)

    def vnot(self,  # pylint: disable=R0913
             mask,
             dst,
             src,
             repeat_times,
             dst_blk_stride,
             src_blk_stride,
             dst_rep_stride,
             src_rep_stride,
             stride_unit=0):
        """ bitwise not.   dst = !src

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_single_elewise_func('vnot', mask, dst, src, repeat_times,
                                         dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride,
                                         stride_unit)

    @source_info_decorator(depth=2)
    @debug.vec_bin_elewise_func_dec
    def _vector_binary_tenary_elewise_func(self,  # pylint: disable=R0913, R0914
                                           name,
                                           mask,
                                           dst,
                                           src0,
                                           src1,
                                           repeat_times,
                                           dst_blk_stride,
                                           src0_blk_stride,
                                           src1_blk_stride,
                                           dst_rep_stride,
                                           src0_rep_stride,
                                           src1_rep_stride,
                                           stride_unit,
                                           store_mode=None,
                                           print_name=None,
                                           mask_o=None):
        # because arguments is too many and stride_unit is used in decorator
        if print_name is None:
            print_name = name
        # check src0, src1, dst must be tensor
        TikCheckUtil.check_type_match(
            dst, Tensor, "dst should be Tensor, input type: {}"
            .format(type(dst)))
        TikCheckUtil.check_type_match(
            src0, Tensor, "src0 should be Tensor, input type: {}"
            .format(type(src0)))
        TikCheckUtil.check_type_match(
            src1, Tensor, "src1 should be Tensor, input type: {}"
            .format(type(src1)))
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride([dst_blk_stride, src0_blk_stride, src1_blk_stride],
                            [dst_rep_stride, src0_rep_stride, src1_rep_stride],
                            MAX_BLK_STRIDE_SINGLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE,
                            ["dst", "src0", "src1"])
        # check dtype
        dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src0.dtype] + DTYPE_MAP[
            src1.dtype]
        if name in ("vmla", "vmulconv"):
            # not same dtype here!
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                                + name,
                                                                dtype_str),
                                        True,
                                        INSTR_DTYPE_SUPPORT_STATEMENT.
                                        format(dtype_str, name))
        else:
            TikCheckUtil.check_equality(dst.dtype, src0.dtype,
                                        "Intrinsic {}'s src0's "
                                        "dtype should be equal to dst's "
                                        "dtype".
                                        format(print_name))
            TikCheckUtil.check_equality(dst.dtype, src1.dtype,
                                        "Intrinsic {}'s src1's "
                                        "dtype should be equal to "
                                        "dst's dtype".
                                        format(print_name))
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                                + name,
                                                                dst.dtype),
                                        True,
                                        INSTR_DTYPE_SUPPORT_STATEMENT.
                                        format(dst.dtype, print_name))
        # check storeMode
        from .tik_params import MAX_STORE_MODE
        if isinstance(store_mode, int):
            TikCheckUtil.check_in_range(store_mode, range(MAX_STORE_MODE),
                                        "store_mode should be 0 or 1")
        # mask
        if mask_o is None:
            mask_o = mask_concat(self, mask,
                                 tensor_bit_len=max(get_bit_len(dst.dtype),
                                                    get_bit_len(src0.dtype),
                                                    get_bit_len(src1.dtype)))
        # check address overlap
        _check_vector_binary_address_overlap(
            name, mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit, print_name)

        # check tensor overflow(static)
        from .tik_params import BIT_LEN_16
        dst_bit_len = get_bit_len(dst.dtype)
        if dtype_str in ('u8s16s16vdeqs16', 's8s16s16vdeqs16'):
            dst_bit_len = BIT_LEN_16
        src0_bit_len = get_bit_len(src0.dtype)
        src1_bit_len = get_bit_len(src1.dtype)
        parallelism = ONE_REP_BYTE_SIZE*ONE_BYTE_BIT_LEN // \
                      max(dst_bit_len, src0_bit_len, src1_bit_len)
        vector_tensor_overflow_check(
            dst, mask, parallelism // (ONE_REP_BYTE_SIZE // dst_bit_len),
            ONE_REP_BYTE_SIZE // dst_bit_len, repeat_times, dst_blk_stride,
            dst_rep_stride, "dst tensor overflow.")
        vector_tensor_overflow_check(
            src0, mask, parallelism // (ONE_REP_BYTE_SIZE // src0_bit_len),
            ONE_REP_BYTE_SIZE // src0_bit_len, repeat_times, src0_blk_stride,
            src0_rep_stride, "src0 tensor overflow.")
        vector_tensor_overflow_check(
            src1, mask, parallelism // (ONE_REP_BYTE_SIZE // src1_bit_len),
            ONE_REP_BYTE_SIZE // src1_bit_len, repeat_times, src1_blk_stride,
            src1_rep_stride, "src1 tensor overflow.")
        # code gen
        config = [
            repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
            dst_rep_stride, src0_rep_stride, src1_rep_stride
        ]
        if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
            config.append(stride_unit & 0b01)
            config.append((stride_unit & 0b10) >> 1)
        if name == "vmulconv":
            name = name + "_f162" + DTYPE_MAP[dst.dtype]
        if name in ["vmadd", "vmaddrelu", "vmla"]:
            dst_acc = "rw"
        else:
            dst_acc = "w"
        with self.new_scope():
            instr = tvm.call_extern(dst.dtype, name,
                                    dst.access_ptr(dst_acc, extent=
                                                   _calculate_extent(
                                                       repeat_times,
                                                       dst_rep_stride,
                                                       _BLOCK_LEN,
                                                       dst_blk_stride)),
                                    src0.access_ptr("r", extent=
                                                    _calculate_extent(
                                                        repeat_times,
                                                        src0_rep_stride,
                                                        _BLOCK_LEN,
                                                        src0_blk_stride)),
                                    src1.access_ptr("r", extent=
                                                    _calculate_extent(
                                                        repeat_times,
                                                        src1_rep_stride,
                                                        _BLOCK_LEN,
                                                        src1_blk_stride)),
                                    *type_convert(config))
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

    def vadd(self,  # pylint: disable=R0913
             mask,
             dst,
             src0,
             src1,
             repeat_times,
             dst_blk_stride,
             src0_blk_stride,
             src1_blk_stride,
             dst_rep_stride,
             src0_rep_stride,
             src1_rep_stride,
             stride_unit=0):
        """Do add by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vadd', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vsub(self,  # pylint: disable=R0913
             mask,
             dst,
             src0,
             src1,
             repeat_times,
             dst_blk_stride,
             src0_blk_stride,
             src1_blk_stride,
             dst_rep_stride,
             src0_rep_stride,
             src1_rep_stride,
             stride_unit=0):
        """Do minus by single elements.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vsub', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vmul(self,  # pylint: disable=R0913
             mask,
             dst,
             src0,
             src1,
             repeat_times,
             dst_blk_stride,
             src0_blk_stride,
             src1_blk_stride,
             dst_rep_stride,
             src0_rep_stride,
             src1_rep_stride,
             stride_unit=0):
        """Get product by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vmul', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vdiv(self,  # pylint: disable=R0913
             mask,
             dst,
             src0,
             src1,
             repeat_times,
             dst_blk_stride,
             src0_blk_stride,
             src1_blk_stride,
             dst_rep_stride,
             src0_rep_stride,
             src1_rep_stride,
             stride_unit=0):
        """Get division by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vdiv', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vmax(self,  # pylint: disable=R0913
             mask,
             dst,
             src0,
             src1,
             repeat_times,
             dst_blk_stride,
             src0_blk_stride,
             src1_blk_stride,
             dst_rep_stride,
             src0_rep_stride,
             src1_rep_stride,
             stride_unit=0):
        """Get the max value in all elements.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vmax', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vmin(self,  # pylint: disable=R0913
             mask,
             dst,
             src0,
             src1,
             repeat_times,
             dst_blk_stride,
             src0_blk_stride,
             src1_blk_stride,
             dst_rep_stride,
             src0_rep_stride,
             src1_rep_stride,
             stride_unit=0):
        """Get the min value by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vmin', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vand(self,  # pylint: disable=R0913
             mask,
             dst,
             src0,
             src1,
             repeat_times,
             dst_blk_stride,
             src0_blk_stride,
             src1_blk_stride,
             dst_rep_stride,
             src0_rep_stride,
             src1_rep_stride,
             stride_unit=0):
        """Do bitwise and by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vand', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vor(self,  # pylint: disable=R0913
            mask,
            dst,
            src0,
            src1,
            repeat_times,
            dst_blk_stride,
            src0_blk_stride,
            src1_blk_stride,
            dst_rep_stride,
            src0_rep_stride,
            src1_rep_stride,
            stride_unit=0):
        """Do bitwise or by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vor', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vmla(self,  # pylint: disable=R0913
             mask,
             dst,
             src0,
             src1,
             repeat_times,
             dst_blk_stride,
             src0_blk_stride,
             src1_blk_stride,
             dst_rep_stride,
             src0_rep_stride,
             src1_rep_stride,
             stride_unit=0):
        """Multiply by element and accumulate. dst = src0*src1 + dst

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vmla', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vmadd(self,  # pylint: disable=R0913
              mask,
              dst,
              src0,
              src1,
              repeat_times,
              dst_blk_stride,
              src0_blk_stride,
              src1_blk_stride,
              dst_rep_stride,
              src0_rep_stride,
              src1_rep_stride,
              stride_unit=0):
        """Multiply by element and accumulate. dst = src0*dst + src1

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vmadd', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vmaddrelu(self,  # pylint: disable=R0913
                  mask,
                  dst,
                  src0,
                  src1,
                  repeat_times,
                  dst_blk_stride,
                  src0_blk_stride,
                  src1_blk_stride,
                  dst_rep_stride,
                  src0_rep_stride,
                  src1_rep_stride,
                  stride_unit=0):
        """Multiply by element and accumulate and then Linear rectification

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vmaddrelu', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vmulconv(self,  # pylint: disable=R0913
                 mask,
                 dst,
                 src0,
                 src1,
                 repeat_times,
                 dst_blk_stride,
                 src0_blk_stride,
                 src1_blk_stride,
                 dst_rep_stride,
                 src0_rep_stride,
                 src1_rep_stride,
                 stride_unit=0):
        """Multiply by element and then convert precision

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vmulconv', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    @source_info_decorator(depth=2)
    @debug.vec_scalar_single_elewise_dec
    def _vector_scalar_single_elewise_func(self,  # pylint: disable=R0913, R0914
                                           name, mask, dst, src, scalar,
                                           repeat_times, dst_blk_stride,
                                           src_blk_stride, dst_rep_stride,
                                           src_rep_stride, stride_unit,
                                           round_en,  # pylint: disable=W0613
                                           mask_mode="normal", print_name=None,
                                           mask_o=None):
        # because arguments is too many, round_en is used in decorator
        if print_name is None:
            print_name = name
        TikCheckUtil.check_type_match(dst, Tensor,
                                      "dst should be tensor, input type of "
                                      "dst: {}".format(type(dst)))
        TikCheckUtil.check_equality(
            dst.scope, scope_ubuf,
            "dst scope should be ub, input scope: {}".format(dst.scope))
        TikCheckUtil.check_type_match(src, Tensor,
                                      "src should be tensor, input type of "
                                      "src: {}".format(type(src)))
        TikCheckUtil.check_equality(
            src.scope, scope_ubuf,
            "src scope should be ub, input scope: {}".format(src.scope))
        TikCheckUtil.check_type_match(
            scalar, (int, float, Expr, Scalar),
            "scalar should be int, float, Expr or Scalar, "
            "input type of scalar: {}".format(type(scalar)))
        if name in ("vshl", "vshr"):
            check_vshl_vshr_scalar(src.dtype, scalar)
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride([dst_blk_stride, src_blk_stride],
                            [dst_rep_stride, src_rep_stride],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])
        # check stride_unit
        # v100 not support stride_unit, default value: 0
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            stride_unit = 0
        TikCheckUtil.check_type_match(
            stride_unit, int, "input stride_unit should be int, input type "
                              "of stride_unit: {}".format(type(stride_unit)))
        check_integer_in_range(
            stride_unit, range(MAX_STRIDE_UNIT),
            "stride_unit should be in the range of [0, 3], "
            "input stride_unit: {}".format(stride_unit))

        # check tensor dtype
        if isinstance(scalar, Scalar):
            TikCheckUtil.check_equality(scalar.dtype, src.dtype,
                                        "Intrinsic {}'s src's "
                                        "dtype should be equal"
                                        " to scalar's dtype".
                                        format(print_name))
            dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[
                src.dtype] + DTYPE_MAP[scalar.dtype]
        else:
            dtype_str = DTYPE_MAP[dst.dtype] + DTYPE_MAP[src.dtype]*2
        if name == "vaxpy":
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                                + name,
                                                                dtype_str),
                                        True,
                                        INSTR_DTYPE_SUPPORT_STATEMENT.
                                        format(dtype_str, name))
        else:
            TikCheckUtil.check_equality(dst.dtype, src.dtype,
                                        "Intrinsic {}'s src's "
                                        "dtype should be equal"
                                        " to dst's dtype".format(print_name))
            TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                                + name,
                                                                dst.dtype),
                                        True, INSTR_DTYPE_SUPPORT_STATEMENT.
                                        format(dst.dtype, print_name))
        # mask
        if mask_o is None:
            mask_o = mask_concat(self, mask, mask_mode,
                                 tensor_bit_len=max(get_bit_len(dst.dtype),
                                                    get_bit_len(src.dtype)))

        # check address overlap
        if dst.buffer == src.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_blk_stride, dst_rep_stride,
                             src_blk_stride, src_rep_stride, stride_unit)):
                check_address_overlapping(
                    print_name, mask, dst, src, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                    repeat_times, dst_blk_stride, src_blk_stride,
                    dst_rep_stride, src_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src.offset).eval_value(), stride_unit, mask_mode)

        # check tensor overflow(static)
        check_tensor_overflow(
            (dst, src), mask, repeat_times, (dst_blk_stride, src_blk_stride),
            (dst_rep_stride, src_rep_stride), ("dst", "src"), stride_unit,
            mask_mode)
        # cal extent
        dst_extent = cal_extent_stride_unit_mask(
            mask, repeat_times, dst, stride_unit, dst_blk_stride,
            dst_rep_stride, mask_mode)
        src_extent = cal_extent_stride_unit_mask(
            mask, repeat_times, src, stride_unit, src_blk_stride,
            src_rep_stride, mask_mode)
        # gen
        args = [repeat_times, dst_blk_stride, src_blk_stride, dst_rep_stride,
                src_rep_stride, stride_unit]
        args = [concat_params(args, VEC_SCALAR_OFFSET_LIST,
                              VEC_SCALAR_SEGMENT_LIST)]
        scalar_tmp = dtype_convert(scalar, dst.dtype)
        if name == "vshr":
            args = [repeat_times, dst_blk_stride, src_blk_stride,
                    dst_rep_stride, src_rep_stride, stride_unit & 0b01,
                    (stride_unit & 0b10) >> 1]
            args.append(round_en)
        if name == "vaxpy":
            dst_acc = dst.access_ptr("rw", extent=dst_extent)
        else:
            dst_acc = dst.access_ptr("w", extent=dst_extent)
        with self.new_scope():
            orig_ctrl = self._set_ctrl_counter_mask_counter(mask_mode)
            instr = tvm.call_extern(dst.dtype, name, dst_acc,
                                    src.access_ptr("r", extent=src_extent),
                                    scalar_tmp, *type_convert(args))
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

            # reset CTRL SPR as orig_ctrl
            self._reset_ctrl_counter_mask_counter(mask_mode, orig_ctrl)

    def _set_ctrl_counter_mask_counter(self, mask_mode):
        """
        _set_ctrl_counter_mask_counter
        mask_mode: if counter mode
        :return:
        """
        if mask_mode == "counter":
            orig_ctrl = set_ctrl_counter_mask(self)
        else:
            orig_ctrl = ""
        return orig_ctrl

    def _reset_ctrl_counter_mask_counter(self, mask_mode, orig_ctrl):
        """
        _reset_ctrl_counter_mask_counter
         mask_mode: if counter mode
         orig_ctrl: before get
        :return:
        """
        if mask_mode == "counter":
            reset_ctrl_counter_mask(self, orig_ctrl)

    def vmuls(self, mask, dst,  # pylint: disable=R0913
              src, scalar, repeat_times, dst_blk_stride,
              src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit=0,
              mask_mode="normal"):
        """Multiple each element with a scalar in a vector

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_scalar_single_elewise_func(
            'vmuls', mask, dst, src, scalar, repeat_times, dst_blk_stride,
            src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit,
            _ROUND_TO_NEAREST_ENABLE, mask_mode)

    def vadds(self, mask, dst, src,  # pylint: disable=R0913
              scalar, repeat_times, dst_blk_stride,
              src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit=0,
              mask_mode="normal"):
        """add each element with a scalar in a vector

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0
        mask_mode: mode of mask, normal/counter, default value = normal

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_scalar_single_elewise_func(
            'vadds', mask, dst, src, scalar, repeat_times, dst_blk_stride,
            src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit,
            _ROUND_TO_NEAREST_ENABLE, mask_mode)

    def vaxpy(self,  # pylint: disable=R0913
              mask,
              dst,
              src,
              scalar,
              repeat_times,
              dst_blk_stride,
              src_blk_stride,
              dst_rep_stride,
              src_rep_stride,
              stride_unit=0):
        """multiple each element with a scalar in a vector and then acculate

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_scalar_single_elewise_func('vaxpy', mask, dst, src,
                                                scalar, repeat_times,
                                                dst_blk_stride, src_blk_stride,
                                                dst_rep_stride, src_rep_stride,
                                                stride_unit,
                                                _ROUND_TO_NEAREST_ENABLE)

    def vmins(self, mask, dst, src,  # pylint: disable=R0913
              scalar, repeat_times, dst_blk_stride,
              src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit=0,
              mask_mode="normal"):
        """get min of each element in a vector with a scalar

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        scalar: source scalar
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_scalar_single_elewise_func(
            'vmins', mask, dst, src, scalar, repeat_times, dst_blk_stride,
            src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit,
            _ROUND_TO_NEAREST_ENABLE, mask_mode)

    def vmaxs(self, mask, dst,  # pylint: disable=R0913
              src, scalar, repeat_times, dst_blk_stride,
              src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit=0,
              mask_mode="normal"):
        """get max of each element in a vector with a scalar

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        scalar: source scalar
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        self._vector_scalar_single_elewise_func(
            'vmaxs', mask, dst, src, scalar, repeat_times, dst_blk_stride,
            src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit,
            _ROUND_TO_NEAREST_ENABLE, mask_mode)

    @staticmethod
    def _check_vsel_addr_overlap(  # pylint: disable=R0913
            mask, mode, dst, sel, src0, src1, repeat_times,
            dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride):
        """check address overlap for vsel instr"""
        if src0.buffer == dst.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, dst_blk_stride, dst_rep_stride,
                             src0_blk_stride, src0_rep_stride)):
                check_address_overlapping(
                    "vsel", mask, dst, src0, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(dst.dtype), get_bit_len(src0.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src0.dtype),
                    repeat_times, dst_blk_stride, src0_blk_stride,
                    dst_rep_stride, src0_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src0.offset).eval_value(), msg="dst and src0")

        if mode in (VSEL_MODE_DOUBLE_TENSOR_ONE_IT,
                    VSEL_MODE_DOUBLE_TENSOR_MANY_IT):
            # check address overlap
            if src1.buffer == dst.buffer:
                if all(isinstance(value, int) for \
                       value in (repeat_times, dst_blk_stride, dst_rep_stride,
                                 src1_blk_stride, src1_rep_stride)):
                    check_address_overlapping(
                        "vsel", mask, dst, src1, BLK_NUM_PER_REP,
                        ONE_REP_BYTE_SIZE //
                        max(get_bit_len(dst.dtype), get_bit_len(src1.dtype)),
                        ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                        ONE_REP_BYTE_SIZE // get_bit_len(src1.dtype),
                        repeat_times, dst_blk_stride, src1_blk_stride,
                        dst_rep_stride, src1_rep_stride,
                        Expr(dst.offset).eval_value(),
                        Expr(src1.offset).eval_value(), msg="dst and src1")
        if mode in (VSEL_MODE_TENSOR_SCALAR, VSEL_MODE_DOUBLE_TENSOR_MANY_IT):
            # check dst sel addressoverlap
            if dst.buffer == sel.buffer:
                if all(isinstance(value, int) for \
                       value in (repeat_times, dst_blk_stride, dst_rep_stride)):
                    check_sel_dst_overlap(
                        dst, src0, sel, mask,
                        Expr(dst.offset).eval_value(),
                        Expr(sel.offset).eval_value(),
                        repeat_times, dst_blk_stride, dst_rep_stride)


    ### X mode -vsel
    @source_info_decorator()
    @debug.vsel_decorator
    def vsel(self, mask,  # pylint: disable=R0913, R0914
             mode, dst, sel, src0, src1, repeat_times,
             dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride=0,
             src0_rep_stride=0, src1_rep_stride=0, name="vsel", mask_o=None):
        """
        mode    |   dst |   sel           | src0    |   src1                |
        -------------------------------------------------------------------
        0       |   dst |   CMPMASK_VAR   | src0    |   src1                |
        1       |   dst |   Tensor        | src0    |   src1(Scalar/Imme)   |
        2       |   dst |   Tensor        | src0    |   src1                |
        """
        # because arguments is too many
        # check mode
        TikCheckUtil.check_type_match(mode, int,
                                      "mode should be int.")
        check_integer_in_range(mode, range(MAX_VSEL_MODE),
                               "mode should be in the range of [0, 2]")
        if get_soc_name() in (ASCEND_310, ASCEND_910):
            TikCheckUtil.check_equality(mode, 0, "v100 doesn't support mode"
                                                 " {}.".format(mode))
        # check repeat_times
        check_repeat_times(repeat_times)
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        TikCheckUtil.check_type_match(src0, Tensor, "src0 should be tensor")
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB")
        TikCheckUtil.check_equality(src0.scope, scope_ubuf,
                                    "src0's scope must be UB")
        if mode == VSEL_MODE_TENSOR_SCALAR:
            TikCheckUtil.check_type_match(
                src1, (int, float, Scalar),
                "when mode is 1, src1 should be int, float or Scalar")
        else:
            TikCheckUtil.check_type_match(src1, Tensor, "when mode isn't 1, "
                                                        "src1 should be tensor")
            TikCheckUtil.check_equality(src1.scope, scope_ubuf,
                                        "src1's scope must be UB")
        # check dtype
        self._check_dtype_str_vsel(dst, src0, src1, name)
        # check strides
        check_vector_stride([dst_blk_stride, src0_blk_stride, src1_blk_stride],
                            [dst_rep_stride, src0_rep_stride, src1_rep_stride],
                            MAX_BLK_STRIDE_SINGLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src0", "src1"])
        # check tensor overflow(static)
        if mode in (VSEL_MODE_DOUBLE_TENSOR_ONE_IT,
                    VSEL_MODE_DOUBLE_TENSOR_MANY_IT):
            check_tensor_overflow(
                (dst, src0, src1), mask, repeat_times,
                (dst_blk_stride, src0_blk_stride, src1_blk_stride),
                (dst_rep_stride, src0_rep_stride, src1_rep_stride),
                ("dst", "src0", "src1"))
        else:
            check_tensor_overflow(
                (dst, src0), mask, repeat_times,
                (dst_blk_stride, src0_blk_stride),
                (dst_rep_stride, src0_rep_stride),
                ("dst", "src0"))
        # check sel
        if mode in (VSEL_MODE_TENSOR_SCALAR, VSEL_MODE_DOUBLE_TENSOR_MANY_IT):
            TikCheckUtil.check_type_match(
                sel, Tensor, "when mode is 1 or 2, sel should be tensor")
            TikCheckUtil.check_equality(sel.scope, scope_ubuf,
                                        "sel's scope must be UB")
            TikCheckUtil.check_in_range(
                sel.dtype, ("uint8", "uint16", "uint32", "uint64"),
                "when mode is 1 or 2, sel should be uint8, uint16, uint32 or "
                "uint64")
            # check sel overflow
            check_sel_overflow(dst, src0, sel, mask, repeat_times)
        else:
            TikCheckUtil.check_is(sel, CMPMASK_VAR,
                                  "Please assure sel is cmpmask.")
        # mask
        if mask_o is None:
            mask_o = mask_concat(self, mask,
                                 tensor_bit_len=max(get_bit_len(dst.dtype),
                                                    get_bit_len(src0.dtype)))
        # check address overlap
        self._check_vsel_addr_overlap(
            mask, mode, dst, sel, src0, src1, repeat_times,
            dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride,
            src0_rep_stride, src1_rep_stride)

        # gen
        config = [dst_blk_stride, src0_blk_stride, src1_blk_stride,
                  dst_rep_stride, src0_rep_stride, src1_rep_stride, mode,
                  repeat_times]
        args = concat_params(config, VSEL_OFFSET_LIST, VSEL_SEGMENT_LIST)
        # set cmpmask
        from .tik_api_util import set_vsel_cmpmask
        set_vsel_cmpmask(self, mode, src0, src1, sel)

        with self.new_scope():
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o),
                      ONE_IR)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            mem_access_param = type_convert(args)
            if mode == 1:
                # ISA 7.3
                instr = tvm.call_extern(
                    dst.dtype, "vsel",
                    dst.access_ptr(
                        "w", extent=_calculate_extent(
                            repeat_times, dst_rep_stride, _BLOCK_LEN,
                            dst_blk_stride)),
                    src0.access_ptr(
                        "r", extent=_calculate_extent(
                            repeat_times, src0_rep_stride, _BLOCK_LEN,
                            src0_blk_stride)),
                    sel.reinterpret_cast_to(src0.dtype).access_ptr("r"),
                    mem_access_param)
            else:
                # ISA 6.4 only support mode = 0
                instr = tvm.call_extern(
                    dst.dtype, "vsel",
                    dst.access_ptr("w", extent=_calculate_extent(
                        repeat_times, dst_rep_stride, _BLOCK_LEN,
                        dst_blk_stride)),
                    src0.access_ptr("r", extent=_calculate_extent(
                        repeat_times, src0_rep_stride, _BLOCK_LEN,
                        src0_blk_stride)),
                    src1.access_ptr("r", extent=_calculate_extent(
                        repeat_times, src1_rep_stride, _BLOCK_LEN,
                        src1_blk_stride)), mem_access_param)
            self.emit(instr, ONE_IR)

    def _check_dtype_str_vsel(self, dst, src0, src1, name):
        """ check dtype of instruction vsel

        Parameters
        ----------
        dst : destination operator
        src : source operation

        Returns
        -------
        None
        """
        TikCheckUtil.check_equality(dst.dtype, src0.dtype,
                                    "Intrinsic {}'s src0's "
                                    "dtype should be equal "
                                    "to dst's dtype".format(name))
        if isinstance(src1, (Scalar, Tensor)):
            TikCheckUtil.check_equality(dst.dtype, src1.dtype,
                                        "Intrinsic {}'s src1's "
                                        "dtype should be equal "
                                        "to dst's dtype".format(name))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_vsel",
                                                            dst.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(dst.dtype, name))

    @source_info_decorator()
    @debug.vconv_decorator
    def vconv(self, mask, round_mode,  # pylint: disable=R0913, R0914
              dst, src, repeat_times, dst_blk_stride,
              src_blk_stride, dst_rep_stride, src_rep_stride, deqscale=None,
              ldst_high_half=False, stride_unit=0, name="vconv", mask_o=None):
        """Accurate numerical conversion between integers
            and floating point numbers.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        round_mode : 'none', 'round', 'floor', 'ceil'/'ceilling', 'away-zero',
            'to-zero', 'odd'
        dst : destination operator
        src : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        deqscale : default None

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        # check ldst_high_half
        TikCheckUtil.check_type_match(ldst_high_half, bool,
                                      "ldst_high_half should be bool type.")
        # check tensor
        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB")
        TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                    "src's scope must be UB")
        # check repeat
        check_repeat_times(repeat_times)
        # check dtype & round_mode
        TikCheckUtil.check_in_range(round_mode, ROUND_MODE_MAP,
                                    "round_mode: {} is not supported".format(
                                        round_mode))

        dtype_str = DTYPE_MAP[src.dtype] + "2" + DTYPE_MAP[dst.dtype]
        total_dtype_str = dtype_str + ROUND_MODE_MAP[round_mode]
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_"
                                                            + "vconv",
                                                            total_dtype_str),
                                    True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(total_dtype_str, name))
        # check deqscale
        if dtype_str == "s322f16":
            dtype_str = "deq"
            TikCheckUtil.check_type_match(
                deqscale, (float, Scalar),
                "Please specify your deqscale: immediate/Scalar(float16) "
                "for deq-mode.")
        elif dtype_str in ["s162u8", "s162s8"]:
            TikCheckUtil.check_type_match(
                deqscale, (int, Tensor, Scalar),
                "Please specify your deqscale: Tensor(uint64) for vdeq8-mode, "
                "immediate/Scalar(uint64) for deq8-mode.")
            if isinstance(deqscale, Tensor):
                dtype_str = "vdeqs162b8"
                TikCheckUtil.check_equality(deqscale.dtype, "uint64",
                                            "deqscale should be uint64.")
            else:
                dtype_str = "deqs162b8"
                if isinstance(deqscale, Scalar):
                    TikCheckUtil.check_equality(deqscale.dtype, "uint64",
                                                "deqscale should be uint64.")
        else:
            TikCheckUtil.check_is(deqscale, None,
                                  "deqscale should be None for current conv")
        # check strides
        check_vector_stride([dst_blk_stride, src_blk_stride],
                            [dst_rep_stride, src_rep_stride],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["dst", "src"])
        if mask_o is None:
            mask_o = mask_concat(self, mask,
                                 tensor_bit_len=max(get_bit_len(dst.dtype),
                                                    get_bit_len(src.dtype)))
        if "deqs162b8" in dtype_str:
            # check address overlap
            check_vconv_deqs162b8_overlap(
                src, dst, deqscale, mask, repeat_times, ldst_high_half,
                [dst_blk_stride, src_blk_stride],
                [dst_rep_stride, src_rep_stride], stride_unit)

            # check tensor overflow(static)
            check_vconv_deqs162b8_overflow(
                src, dst, deqscale, mask, repeat_times, ldst_high_half,
                [dst_blk_stride, src_blk_stride],
                [dst_rep_stride, src_rep_stride], stride_unit)
        else:
            # check address overlap
            if src.buffer == dst.buffer:
                if all(isinstance(value, int) for \
                       value in (repeat_times, dst_blk_stride, dst_rep_stride,
                                 src_blk_stride, src_rep_stride, stride_unit)):
                    check_address_overlapping(
                        name, mask, dst, src, BLK_NUM_PER_REP,
                        ONE_REP_BYTE_SIZE //
                        max(get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                        ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                        ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                        repeat_times, dst_blk_stride, src_blk_stride,
                        dst_rep_stride, src_rep_stride,
                        Expr(dst.offset).eval_value(),
                        Expr(src.offset).eval_value(), stride_unit)
            check_tensor_overflow(
                (dst, src), mask, repeat_times,
                (dst_blk_stride, src_blk_stride),
                (dst_rep_stride, src_rep_stride), ("dst", "src"))
        # deqscale
        if "deq" in dtype_str:
            insert_set_deqscale_attr(self, deqscale, dtype_str, dst.dtype)

        # code gen
        self._gen_vconv_code(dst, src, dtype_str, mask_o,
                             ROUND_MODE_MAP[round_mode], repeat_times,
                             dst_blk_stride, src_blk_stride, dst_rep_stride,
                             src_rep_stride, ldst_high_half, stride_unit)

    @source_info_decorator()
    @debug.mov_cmpmask_to_tensor_decorator
    def mov_cmpmask_to_tensor(self, dst, src_cmpmask):
        """move the data from vcmp/scatter_vcmp to dst tensor

        Parameters
        ----------
        dst : target tensor, uint64
        src_cmpmask : the result of vcmp/scatter_vcmp

        Returns
        -------
        None
        """
        TikCheckUtil.check_is(src_cmpmask, CMPMASK_VAR, "src_cmpmask is error")

        TikCheckUtil.check_type_match(dst, Tensor, "dst should be tensor")
        TikCheckUtil.check_in_range(
            dst.dtype, ("uint8", "uint16", "uint32", "uint64"),
            "dst's dtype should be uint8, uint16, uint32 or uint64")
        # check tensor scope
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope must be UB")
        # extent: 2 uint64, 8 Bytes/uint64
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(
                tvm.call_extern("uint64", "get_cmpmask",
                                dst.access_ptr("w",
                                               extent=Expr(2*8).get())),
                ONE_IR)

    @source_info_decorator()
    def mov_tensor_to_cmpmask(self, src):
        """move the data from src tensor to dst_cmpmask

        Parameters
        ----------
        src : 2 continuous uint64 data

        Returns
        -------
        CMPMASK_VAR target var for vsel/scatter_vsel operation
        """
        return self.mov_tensor_to_cmpmask_(src)

    @debug.mov_tensor_to_cmpmask_decorator
    def mov_tensor_to_cmpmask_(self, src):
        """move the data from src tensor to dst_cmpmask
        note: use this function to call mov_tensor_to_cmpmask inside!!
        """
        # check src type
        TikCheckUtil.check_type_match(src, Tensor, "src should be tensor")
        # check tensor scope
        TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                    "src's scope must be UB")
        # extent: 2 uint64, 8 Bytes/uint64
        with self.new_scope():
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(
                tvm.call_extern("uint64", "set_cmpmask",
                                src.access_ptr("r", extent=Expr(2*8).get())),
                ONE_IR)
        return CMPMASK_VAR

    def apply_for_new_alloc(self,  # pylint: disable=R0913
                            dtype,
                            shape,
                            scope=cce_params.scope_ubuf,
                            name='tmp_buf',
                            buffer_reuse_id=None):
        """alloc buffer

        Parameters
        ----------
        dtype : tensor's dtype
        shape : tensor's shape
        scope : gm or ubuf
        name : tensor's name
        buffer_reuse_id : tensor index for reuse/no_reuse

        Returns
        -------
        buffer
        """
        # function's input params is too much, so disable them
        buf_var = self.allocate(dtype, shape, name=name, scope=scope,
                                buffer_reuse_id=buffer_reuse_id)
        tmp_buffer = tvm.decl_buffer(shape,
                                     buf_var.dtype,
                                     name=name,
                                     scope=scope,
                                     data=buf_var)
        return tmp_buffer

    def _gen_vconv_code(self, dst,  # pylint: disable=R0913, R0914
                        src, dtype_str, mask_o,
                        round_mode, repeat_times, dst_blk_stride,
                        src_blk_stride, dst_rep_stride, src_rep_stride,
                        ldst_high_half, stride_unit):
        # function's input params is too much, so disable them
        args = [dtype_convert(repeat_times, "int64"), dst_blk_stride,
                src_blk_stride, dst_rep_stride, src_rep_stride]
        if get_soc_name() in (HI3796CV300ES, ASCEND_610, ASCEND_620):
            args.append(stride_unit & 0b01)
            args.append((stride_unit & 0b10) >> 1)
        if dtype_str in ["deqs162b8", "vdeqs162b8"]:
            if ldst_high_half:
                args.append(1)
            else:
                args.append(0)
        with self.new_scope():
            instr = tvm.call_extern(
                dst.dtype, "vconv_" + dtype_str + round_mode,
                dst.access_ptr("w", extent=_calculate_extent(
                    repeat_times, dst_rep_stride, _BLOCK_LEN, dst_blk_stride)),
                src.access_ptr("r", extent=_calculate_extent(
                    repeat_times, src_rep_stride, _BLOCK_LEN, src_blk_stride)),
                *type_convert(args))
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            self.emit(instr, TWO_IR)

    def vshl(self, mask, dst, src, scalar, # pylint: disable=R0913
             repeat_times, dst_blk_stride, src_blk_stride, dst_rep_stride,
             src_rep_stride, stride_unit=0, mask_mode="normal"):
        """logic shift left for each element in the source vector, the shift
        left distance is indicated by scalar, scalar must be less than or equal
        to16 or 32 for type = b16/b32

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        scalar: source scalar, indicating shift left distance
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0
        mask_mode: mode of mask, normal/counter, default value = normal

        Returns
        -------
        None
        """
        self._vector_scalar_single_elewise_func(
            'vshl', mask, dst, src, scalar, repeat_times, dst_blk_stride,
            src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit,
            1, mask_mode)

    def vshr(self, mask, dst, src, scalar, repeat_times, # pylint: disable=R0913
             dst_blk_stride, src_blk_stride, dst_rep_stride, src_rep_stride,
             stride_unit=0, round_en=False, mask_mode="normal"):
        """logic shift right for type=u32/u16 or arithmetic shift right for
        type=s32/s16 for each element in the source vector. the shift right
        distance is indicated by scalar, scalar must be less than or equal to
        32/16 for b32/b16

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src : source operation
        scalar: source scalar, indicating shift right distance
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block in
                         one repeat
        src_blk_stride : offset of src operator between different block in
                         one repeat
        dst_rep_stride : offset of dst operator in the same block between
                         two repeats
        src_rep_stride : offset of src operator in the same block between
                         two repeats
        stride_unit : address and offset unit both affect it. default = 0
        round_en: 1 means rounding is applied during arithmetic shift right
        mask_mode: mode of mask, normal/counter, default value = normal

        Returns
        -------
        None
        """
        TikCheckUtil.check_type_match(round_en, bool,
                                      "round_en should be bool, input type: %s"
                                      % str(type(round_en)))
        self._vector_scalar_single_elewise_func(
            'vshr', mask, dst, src, scalar, repeat_times, dst_blk_stride,
            src_blk_stride, dst_rep_stride, src_rep_stride, stride_unit,
            int(round_en), mask_mode)

    def vaddrelu(self, mask, dst, src0, src1, # pylint: disable=R0913
                 repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
                 dst_rep_stride, src0_rep_stride, src1_rep_stride,
                 stride_unit=0):
        """Do addrelu by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vaddrelu', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def vsubrelu(self, mask, dst, src0, src1,  # pylint: disable=R0913
                 repeat_times, dst_blk_stride, src0_blk_stride, src1_blk_stride,
                 dst_rep_stride, src0_rep_stride, src1_rep_stride,
                 stride_unit=0):
        """Do subrelu by single element.

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
            Continuous and bit by bit.
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        dst_blk_stride : offset of dst operator between different block
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        dst_rep_stride : offset of dst operator in the same block between
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between
        stride_unit : address and offset unit both affect it. default = 0

        Returns
        -------
        None
        """
        # function's input params is too much, so disable them
        return self._vector_binary_tenary_elewise_func(
            'vsubrelu', mask, dst, src0, src1, repeat_times, dst_blk_stride,
            src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride,
            src1_rep_stride, stride_unit)

    def _fp162fp32_func_compute(self, func,  # pylint: disable=R0913, R0914
                                name, mask, dst, src, work_tensor,
                                repeat_times, dst_rep_stride, src_rep_stride,
                                fp32_rep_stride, tmp_tensor_size):
        src_blk_stride = 1
        if work_tensor.dtype == "float16":
            tmp_src_tensor = \
                work_tensor[0:tmp_tensor_size].reinterpret_cast_to("float32")
            tmp_dst_tensor = \
                work_tensor[tmp_tensor_size:
                            tmp_tensor_size*2].reinterpret_cast_to("float32")
            tmp_work_tensor = \
                work_tensor[tmp_tensor_size*2:].reinterpret_cast_to("float32")
            tmp_tensor_size = tmp_tensor_size * \
                              DTYPE_SIZE[work_tensor.dtype] // \
                              DTYPE_SIZE["float32"]
        else:
            tmp_src_tensor = work_tensor[0:tmp_tensor_size]
            tmp_dst_tensor = work_tensor[tmp_tensor_size:tmp_tensor_size*2]
            tmp_work_tensor = work_tensor[tmp_tensor_size*2:]
        mask_o = mask_concat(self, mask, tensor_bit_len=32)
        self.vconv(mask, "none", tmp_src_tensor,
                   src, repeat_times, src_blk_stride, src_blk_stride,
                   fp32_rep_stride, src_rep_stride, name=name, mask_o=mask_o)

        func(name, mask, tmp_dst_tensor, tmp_src_tensor,
             tmp_work_tensor, repeat_times, fp32_rep_stride,
             fp32_rep_stride, tmp_tensor_size, mask_o)

        self.vconv(mask, "none", dst, tmp_dst_tensor,
                   repeat_times, src_blk_stride, src_blk_stride,
                   dst_rep_stride, fp32_rep_stride, name=name, mask_o=mask_o)

    def _get_src_extend(self, mask,  # pylint: disable=R0913
                        repeat_times, src_rep_stride,
                        work_tensor, size_factor):
        defualt_src_dtype = "float16"
        src_data_size = self.get_wk_tensor_extend(mask, defualt_src_dtype,
                                                  repeat_times, src_rep_stride)
        if work_tensor.dtype == "float16":
            tmp_tensor_size = src_data_size * 2
        else:
            tmp_tensor_size = src_data_size
        needed_tensor_size = tmp_tensor_size * size_factor
        work_tensor_size = work_tensor.size
        if is_immediate_number((needed_tensor_size, work_tensor_size)):
            TikCheckUtil.check_ge(work_tensor_size, needed_tensor_size,
                                  "Input work tensor size(%d) must be more "
                                  "than needed size(%d)" %
                                  (work_tensor_size, needed_tensor_size))
        return tmp_tensor_size

    @staticmethod
    def _get_wk_tensor_stride_imm(src_rep_stride):
        if src_rep_stride <= 4:
            fp32_rep_stride_1 = src_rep_stride*2
        else:
            fp32_rep_stride_1 = 8
        return fp32_rep_stride_1

    def _get_wk_tensor_stride_scalar(self, src_rep_stride):
        fp32_rep_stride = self.Scalar_(    # pylint: disable=E1101
            init_value=0, dtype="int32")
        with self.if_scope(src_rep_stride < 4):
            fp32_rep_stride.set_as(src_rep_stride*2)
        with self.else_scope():
            fp32_rep_stride.set_as(8)
        return fp32_rep_stride

    def _fp162fp32_func_mask_list(self,  # pylint: disable=R0913, R0914
                                  func, name, mask, dst, src,
                                  work_tensor, repeat_times,
                                  dst_rep_stride, src_rep_stride,
                                  fp32_rep_stride, size_factor):
        default_start_offset = 64
        if isinstance(mask[MASK_LOW_IDX], BasicExpr):
            low_mask = self.Scalar_(  # pylint: disable=E1101
                init_value=0, dtype="uint64", name="low_mask")
            high_mask = self.Scalar_(  # pylint: disable=E1101
                init_value=0, dtype="uint64", name="high_mask")
            low_mask.set_as(mask[MASK_HIGH_IDX])
            with self.if_scope(low_mask > 0):
                tmp_tensor_size = self._get_src_extend(
                    [high_mask, mask[MASK_HIGH_IDX]], repeat_times,
                    src_rep_stride, work_tensor, size_factor)
                self._fp162fp32_func_compute(
                    func, name, [high_mask, low_mask],
                    dst[default_start_offset:],
                    src[default_start_offset:], work_tensor, repeat_times,
                    dst_rep_stride, src_rep_stride,
                    fp32_rep_stride, tmp_tensor_size)

            low_mask.set_as(mask[MASK_LOW_IDX])
            tmp_tensor_size = self._get_src_extend(
                [high_mask, mask[MASK_LOW_IDX]], repeat_times, src_rep_stride,
                work_tensor, size_factor)
            self._fp162fp32_func_compute(
                func, name, [high_mask, low_mask], dst, src, work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride,
                fp32_rep_stride, tmp_tensor_size)

        else:
            low_mask = mask[MASK_LOW_IDX]
            high_mask = mask[MASK_HIGH_IDX]

            if high_mask > 0:
                tmp_tensor_size = self._get_src_extend(
                    [0, high_mask], repeat_times,
                    src_rep_stride, work_tensor, size_factor)
                self._fp162fp32_func_compute(
                    func, name, [0, high_mask], dst[default_start_offset:],
                    src[default_start_offset:], work_tensor,
                    repeat_times, dst_rep_stride, src_rep_stride,
                    fp32_rep_stride, tmp_tensor_size)

            tmp_tensor_size = self._get_src_extend(
                [0, low_mask], repeat_times,
                src_rep_stride, work_tensor, size_factor)

            self._fp162fp32_func_compute(
                func, name, [0, low_mask], dst, src, work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride,
                fp32_rep_stride, tmp_tensor_size)

    def _fp162fp32_func_mask_scalar(self,  # pylint: disable=R0913, R0914
                                    func, name, mask, dst, src, work_tensor,
                                    repeat_times, dst_rep_stride,
                                    src_rep_stride, fp32_rep_stride,
                                    size_factor):
        default_start_offset = 64
        with self.if_scope(mask <= 64):
            tmp_tensor_size = self._get_src_extend(
                mask, repeat_times, src_rep_stride, work_tensor, size_factor)
            self._fp162fp32_func_compute(
                func, name, mask, dst, src, work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride,
                fp32_rep_stride, tmp_tensor_size)
        with self.else_scope():
            low_mask = self.Scalar_(  # pylint: disable=E1101
                init_value=64, dtype="uint64", name="low_mask")
            high_mask = self.Scalar_(  # pylint: disable=E1101
                init_value=0, dtype="uint64", name="high_mask")
            high_mask.set_as(mask - 64)
            tmp_tensor_size = self._get_src_extend(
                high_mask, repeat_times,
                src_rep_stride, work_tensor, size_factor)
            self._fp162fp32_func_compute(
                func, name, high_mask, dst[default_start_offset:],
                src[default_start_offset:], work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride,
                fp32_rep_stride, tmp_tensor_size)

            tmp_tensor_size = self._get_src_extend(
                low_mask, repeat_times,
                src_rep_stride, work_tensor, size_factor)
            self._fp162fp32_func_compute(
                func, name, low_mask, dst, src, work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride,
                fp32_rep_stride, tmp_tensor_size)

    def _fp162fp32_func_mask_imm(self,  # pylint: disable=R0913, R0914
                                 func, name, mask, dst, src, work_tensor,
                                 repeat_times, dst_rep_stride,
                                 src_rep_stride, fp32_rep_stride, size_factor):
        default_start_offset = 64
        if mask < 64:
            tmp_tensor_size = self._get_src_extend(
                mask, repeat_times, src_rep_stride, work_tensor, size_factor)

            self._fp162fp32_func_compute(
                func, name, mask, dst, src, work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride,
                fp32_rep_stride, tmp_tensor_size)
        else:
            low_mask = 64
            high_mask = mask - 64
            if high_mask > 0:
                tmp_tensor_size = self._get_src_extend(high_mask,
                                                       repeat_times,
                                                       src_rep_stride,
                                                       work_tensor,
                                                       size_factor)
                self._fp162fp32_func_compute(
                    func, name, high_mask, dst[default_start_offset:],
                    src[default_start_offset:], work_tensor,
                    repeat_times, dst_rep_stride, src_rep_stride,
                    fp32_rep_stride, tmp_tensor_size)

            tmp_tensor_size = self._get_src_extend(low_mask, repeat_times,
                                                   src_rep_stride,
                                                   work_tensor, size_factor)
            self._fp162fp32_func_compute(
                func, name, low_mask, dst, src, work_tensor,
                repeat_times, dst_rep_stride, src_rep_stride,
                fp32_rep_stride, tmp_tensor_size)

    def _fp162fp32_high_preci_func(self,  # pylint: disable=R0913, R0914
                                   func, name, mask, dst, src, work_tensor,
                                   repeat_times, dst_rep_stride,
                                   src_rep_stride, size_factor):
        if is_immediate_number(src_rep_stride):
            fp32_rep_stride = self._get_wk_tensor_stride_imm(src_rep_stride)
        else:
            fp32_rep_stride = self._get_wk_tensor_stride_scalar(src_rep_stride)

        if isinstance(mask, (list, tuple)):
            self._fp162fp32_func_mask_list(func, name, mask, dst,
                                           src, work_tensor,
                                           repeat_times, dst_rep_stride,
                                           src_rep_stride, fp32_rep_stride,
                                           size_factor)
        elif is_immediate_number(mask):
            self._fp162fp32_func_mask_imm(func, name, mask, dst,
                                          src, work_tensor,
                                          repeat_times, dst_rep_stride,
                                          src_rep_stride, fp32_rep_stride,
                                          size_factor)
        elif isinstance(mask, BasicExpr):
            self._fp162fp32_func_mask_scalar(
                func, name, mask, dst, src, work_tensor, repeat_times,
                dst_rep_stride, src_rep_stride, fp32_rep_stride, size_factor)

    def vln_calculate_by_taylor(self,  # pylint: disable=R0913, R0914, R0915
                                name, mask, dst,
                                src, work_tensor,
                                repeat_times, dst_blk_stride,
                                src_blk_stride, dst_rep_stride,
                                src_rep_stride):
        """
        calculate ln(raw_tensor), use taylor expansion to calculate log
        """
        src_data_size = self.get_wk_tensor_extend(mask, src.dtype,
                                                  repeat_times, src_rep_stride)
        src_offset = src_rep_stride * 16  # need align with 32B
        mask_o = mask_concat(self, mask, tensor_bit_len=get_bit_len(src.dtype))
        stride_unit = 0

        def _vadds_vmul(dst_data, src_data, tmp_work_tensor, scalar_value):
            self._vector_scalar_single_elewise_func(
                'vadds', mask, tmp_work_tensor, dst_data, scalar_value,
                repeat_times, src_blk_stride, src_blk_stride, src_rep_stride,
                src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)
            self._vector_binary_tenary_elewise_func(
                'vmul', mask, dst_data, src_data, tmp_work_tensor,
                repeat_times, src_blk_stride, src_blk_stride, src_blk_stride,
                src_rep_stride, src_rep_stride, src_rep_stride,
                stride_unit, print_name=name, mask_o=mask_o)

        def _taylor_compute_five(dst_data, src_data, tmp_work_tensor):
            self._vector_scalar_single_elewise_func(
                'vmuls', mask, dst_data, src_data, CONST_ONE_FIVE,
                repeat_times, src_blk_stride, src_blk_stride, src_rep_stride,
                src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)

            _vadds_vmul(dst_data, src_data,
                        tmp_work_tensor, CONST_NEG_ONE_FOUR)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_ONE_THREE)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_NEG_HALF)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_ONE)

        def _taylor_compute_nine(dst_data, src_data, tmp_work_tensor):
            self._vector_scalar_single_elewise_func(
                'vmuls', mask, dst_data, src_data, CONST_ONE_NINE,
                repeat_times, src_blk_stride, src_blk_stride, src_rep_stride,
                src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)

            _vadds_vmul(dst_data, src_data, tmp_work_tensor,
                        CONST_NEG_ONE_EIGHT)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_ONE_SEVEN)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_NEG_ONE_SIX)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_ONE_FIVE)
            _vadds_vmul(dst_data, src_data,
                        tmp_work_tensor, CONST_NEG_ONE_FOUR)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_ONE_THREE)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_NEG_HALF)
            _vadds_vmul(dst_data, src_data, tmp_work_tensor, CONST_ONE)

        def _ln_compute_block_lt_5_3_gt_1(dst_data, src_data, tmp_work_tensor):
            # CONST_NEG_ONE + CONST_NEG_ONE * CONST_ONE_THREE
            self._vector_scalar_single_elewise_func(
                'vadds', mask, tmp_work_tensor, src_data, CONST_NEG_FOUR_THREE,
                repeat_times, src_blk_stride, src_blk_stride, src_rep_stride,
                src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)
            self._vector_scalar_single_elewise_func(
                'vmuls', mask, tmp_work_tensor[src_data_size:src_data_size*2],
                tmp_work_tensor, CONST_THREE_FOUR, repeat_times,
                src_blk_stride, src_blk_stride, src_rep_stride, src_rep_stride,
                stride_unit, 0, print_name=name, mask_o=mask_o)
            self._vector_scalar_elewise_func(  # pylint: disable=E1101
                'vector_dup', mask,
                tmp_work_tensor[src_data_size*2:src_data_size*3],
                CONST_ONE_THREE, repeat_times,
                src_blk_stride, src_rep_stride,
                stride_unit, print_name=name, mask_o=mask_o)
            tmp_tensor = \
                tmp_work_tensor[src_data_size*3:
                                src_data_size*3 +
                                repeat_times*8].reinterpret_cast_to("uint64")
            self._vector_scalar_single_elewise_func(
                'vadds', mask, tmp_work_tensor, src_data, CONST_NEG_ONE,
                repeat_times, src_blk_stride, src_blk_stride, src_rep_stride,
                src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)

            with self.for_range(0, repeat_times) as index:
                cmpmask = self._vcmp_elewise_func(  # pylint: disable=E1101
                    'vcmp_ge', mask, tmp_work_tensor[index*src_offset],
                    tmp_work_tensor[src_data_size*2 + index*src_offset],
                    src_blk_stride, src_blk_stride,
                    print_name=name, mask_o=mask_o)
                self.mov_cmpmask_to_tensor(tmp_tensor[index*2], cmpmask)
                self.vsel(mask, 0, dst_data[index*src_offset], cmpmask,
                          tmp_work_tensor[src_data_size + index*src_offset],
                          tmp_work_tensor[index*src_offset], 1, src_blk_stride,
                          src_blk_stride, src_blk_stride,
                          name=name, mask_o=mask_o)

            _taylor_compute_five(
                tmp_work_tensor[0:src_data_size], dst_data,
                tmp_work_tensor[src_data_size*2:src_data_size*3])

            # phase3: add log(4/3)
            self._vector_scalar_single_elewise_func(
                'vadds', mask, tmp_work_tensor[src_data_size:src_data_size*2],
                tmp_work_tensor[0:src_data_size], LOG_FOUR_THREE,
                repeat_times, src_blk_stride, src_blk_stride, src_rep_stride,
                src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)

            with self.for_range(0, repeat_times) as index:
                cmpmask = self.mov_tensor_to_cmpmask(tmp_tensor[index * 2])
                self.vsel(mask, 0, dst_data[index * src_offset],
                          cmpmask,
                          tmp_work_tensor[src_data_size + index * src_offset],
                          tmp_work_tensor[index * src_offset], 1,
                          src_blk_stride, src_blk_stride,
                          src_blk_stride, name=name, mask_o=mask_o)

        def _ln_compute_block_gt_5_3(dst_data, tmp_dst_data, src_data,
                                     tmp_work_tensor):
            """
            when src_data > 5/3, use vlog directly
            Parameters
            ----------
            src_data: input tensor that we want to calculate log

            Returns
            -------
            res : return of log

            """
            # if src_data > 5/3, use vlog

            self._vector_single_elewise_func(
                'vln', mask, tmp_work_tensor[src_data_size:src_data_size*2],
                src_data, repeat_times, src_blk_stride, src_blk_stride,
                src_rep_stride, src_rep_stride,
                stride_unit, print_name=name, mask_o=mask_o)
            self._vector_scalar_elewise_func(  # pylint: disable=E1101
                'vector_dup', mask,
                tmp_work_tensor[0:src_data_size],
                CONST_FIVE_THREE, repeat_times,
                src_blk_stride, src_rep_stride,
                stride_unit, print_name=name, mask_o=mask_o)

            with self.for_range(0, repeat_times) as index:
                cmpmask = self._vcmp_elewise_func(  # pylint: disable=E1101
                    'vcmp_ge', mask, src_data[index*src_offset],
                    tmp_work_tensor[index*src_offset],
                    src_blk_stride, src_blk_stride,
                    print_name=name, mask_o=mask_o)
                self.vsel(mask, 0, dst_data[index*src_offset], cmpmask,
                          tmp_work_tensor[src_data_size+index*src_offset],
                          tmp_dst_data[index*src_offset], 1,
                          src_blk_stride, src_blk_stride,
                          src_blk_stride, name=name, mask_o=mask_o)

        def _ln_compute_block_gt_half_lt_1(dst_data, tmp_dst_data, src_data,
                                           tmp_work_tensor):
            self._vector_scalar_single_elewise_func(
                'vadds', mask, tmp_work_tensor, src_data, CONST_NEG_ONE,
                repeat_times, src_blk_stride, src_blk_stride, src_rep_stride,
                src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)
            _taylor_compute_nine(
                tmp_work_tensor[src_data_size:src_data_size*2],
                tmp_work_tensor,
                tmp_work_tensor[src_data_size*2:src_data_size*3])
            self._vector_scalar_elewise_func(  # pylint: disable=E1101
                'vector_dup', mask, tmp_work_tensor, CONST_ONE,
                repeat_times, src_blk_stride, src_rep_stride,
                stride_unit, print_name=name, mask_o=mask_o)

            with self.for_range(0, repeat_times) as index:
                cmpmask = self._vcmp_elewise_func(  # pylint: disable=E1101
                    'vcmp_le', mask, src_data[index*src_offset],
                    tmp_work_tensor[index*src_offset],
                    src_blk_stride, src_blk_stride,
                    print_name=name, mask_o=None)

                self.vsel(mask, 0, dst_data[index*src_offset], cmpmask,
                          tmp_work_tensor[src_data_size+index*src_offset],
                          tmp_dst_data[index*src_offset],
                          1, src_blk_stride, src_blk_stride,
                          src_blk_stride, name=name, mask_o=mask_o)

        def _ln_compute_block_lt_half(dst_data, tmp_dst_data, src_data,
                                      tmp_work_tensor):
            self._fp162fp32_high_preci_func(
                self._vec_rec_high_preci, name, mask,
                tmp_work_tensor[0:src_data_size], src_data,
                tmp_work_tensor[src_data_size:],
                repeat_times, src_rep_stride, src_rep_stride, 4)

            _ln_compute_block_gt_5_3(
                tmp_work_tensor[src_data_size*3:src_data_size*4], tmp_dst_data,
                tmp_work_tensor[0:src_data_size],
                tmp_work_tensor[src_data_size:src_data_size*3])

            self._vector_scalar_single_elewise_func(
                'vmuls', mask, tmp_work_tensor[src_data_size:src_data_size*2],
                tmp_work_tensor[src_data_size*3:src_data_size*4],
                CONST_NEG_ONE, repeat_times, src_blk_stride,
                src_blk_stride, src_rep_stride, src_rep_stride,
                stride_unit, 0, print_name=name, mask_o=mask_o)

            self._vector_scalar_elewise_func(  # pylint: disable=E1101
                'vector_dup', mask, tmp_work_tensor, CONST_HALF,
                repeat_times, src_blk_stride, src_rep_stride,
                stride_unit, print_name=name, mask_o=mask_o)

            dst_offset = dst_rep_stride*16
            with self.for_range(0, repeat_times) as index:
                cmpmask = self._vcmp_elewise_func(  # pylint: disable=E1101
                    'vcmp_le', mask, src_data[index*src_offset],
                    tmp_work_tensor[index*src_offset],
                    src_blk_stride, src_blk_stride,
                    print_name=name, mask_o=mask_o)
                self.vsel(mask, 0, dst_data[index*dst_offset], cmpmask,
                          tmp_work_tensor[src_data_size+index*src_offset],
                          tmp_dst_data[index*src_offset], 1, dst_blk_stride,
                          src_blk_stride, src_blk_stride,
                          name=name, mask_o=mask_o)

        _ln_compute_block_lt_5_3_gt_1(
            work_tensor[:src_data_size], src,
            work_tensor[src_data_size:src_data_size*5])
        _ln_compute_block_gt_5_3(
            work_tensor[src_data_size*4:src_data_size*5],
            work_tensor[:src_data_size], src,
            work_tensor[src_data_size:src_data_size*3])
        _ln_compute_block_gt_half_lt_1(
            work_tensor[0:src_data_size],
            work_tensor[src_data_size*4:src_data_size*5], src,
            work_tensor[src_data_size:src_data_size*4])
        _ln_compute_block_lt_half(dst, work_tensor[:src_data_size], src,
                                  work_tensor[src_data_size:])

    def _expm1_taylor(self, name, mask, dst,  # pylint: disable=R0913, R0914
                      src, work_tensor, repeat_times,
                      dst_rep_stride, src_rep_stride, tmp_tensor_size, mask_o):
        stride_unit = 0
        wk_tensor1 = work_tensor[:tmp_tensor_size]
        wk_tensor2 = work_tensor[tmp_tensor_size:2*tmp_tensor_size]

        self._vector_scalar_single_elewise_func(
            'vadds', mask, dst, src, 0, repeat_times, 1, 1, dst_rep_stride,
            src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)

        taylor_param = 1 / 2
        self._vector_scalar_single_elewise_func(
            'vadds', mask, wk_tensor1, src, 0,
            repeat_times, 1, 1, src_rep_stride,
            src_rep_stride, stride_unit, 0, print_name=name, mask_o=mask_o)
        self._vector_binary_tenary_elewise_func(
            'vmul', mask, wk_tensor2, wk_tensor1, src, repeat_times,
            1, 1, 1, src_rep_stride, src_rep_stride, src_rep_stride,
            stride_unit, print_name=name, mask_o=mask_o)
        self._vector_scalar_single_elewise_func(
            'vmuls', mask, wk_tensor1, wk_tensor2, taylor_param,
            repeat_times, 1, 1, src_rep_stride, src_rep_stride,
            stride_unit, 0, print_name=name, mask_o=mask_o)
        self._vector_binary_tenary_elewise_func(
            'vadd', mask, wk_tensor2, wk_tensor1, dst, repeat_times,
            1, 1, 1, src_rep_stride, src_rep_stride, dst_rep_stride,
            stride_unit, print_name=name, mask_o=mask_o)

        for index in range(3, 8):
            taylor_param = 1 / index
            if index % 2 == 1:
                self._vector_binary_tenary_elewise_func(
                    'vmul', mask, dst, wk_tensor1, src, repeat_times, 1, 1,
                    1, dst_rep_stride, src_rep_stride, src_rep_stride,
                    stride_unit, print_name=name, mask_o=mask_o)
                self._vector_scalar_single_elewise_func(
                    'vmuls', mask, wk_tensor1, dst, taylor_param,
                    repeat_times, 1, 1, src_rep_stride, dst_rep_stride,
                    stride_unit, 0, print_name=name, mask_o=mask_o)
                self._vector_binary_tenary_elewise_func(
                    'vadd', mask, dst, wk_tensor2, wk_tensor1, repeat_times,
                    1, 1, 1, dst_rep_stride, src_rep_stride, src_rep_stride,
                    stride_unit, print_name=name, mask_o=mask_o)
            else:
                self._vector_binary_tenary_elewise_func(
                    'vmul', mask, wk_tensor2, wk_tensor1,
                    src, repeat_times, 1, 1,
                    1, src_rep_stride, src_rep_stride, src_rep_stride,
                    stride_unit, print_name=name, mask_o=mask_o)
                self._vector_scalar_single_elewise_func(
                    'vmuls', mask, wk_tensor1, wk_tensor2, taylor_param,
                    repeat_times, 1, 1, src_rep_stride, src_rep_stride,
                    stride_unit, 0, print_name=name, mask_o=mask_o)
                self._vector_binary_tenary_elewise_func(
                    'vadd', mask, wk_tensor2, dst, wk_tensor1, repeat_times,
                    1, 1, 1, src_rep_stride, dst_rep_stride, src_rep_stride,
                    stride_unit, print_name=name, mask_o=mask_o)

    @debug.vec_sel_decorator
    def vec_sel_(self, mask, mode, dst,  # pylint: disable=R0913, R0914
                 sel, src0, src1, repeat_times,
                 dst_rep_stride=0, src0_rep_stride=0,
                 src1_rep_stride=0, instr_name=None, mask_o=None):
        """vec_sel instr for inner"""
        default_blk_stride = 1
        if instr_name is None:
            instr_name = "vec_sel"
        # check mode
        TikCheckUtil.check_type_match(mode, int,
                                      "mode should be int.")
        TikCheckUtil.check_type_match(
            sel, Tensor, "sel should be tensor")
        TikCheckUtil.check_equality(sel.scope, scope_ubuf,
                                    "sel's scope must be UB")
        TikCheckUtil.check_in_range(
            sel.dtype, ("uint8", "uint16", "uint32", "uint64"),
            "sel dtype should be uint8, uint16, uint32 or uint64,"
            " input: %s" % sel.dtype)
        with self.context.freeze():  # pylint: disable=E1101
            if mode == 0:
                # change sel to cmpmask
                sel_cmpmask = self.mov_tensor_to_cmpmask(sel)
                return self.vsel(
                    mask, mode, dst, sel_cmpmask, src0, src1, repeat_times,
                    default_blk_stride, default_blk_stride, default_blk_stride,
                    dst_rep_stride, src0_rep_stride, src1_rep_stride,
                    instr_name, mask_o)
            return self.vsel(
                mask, mode, dst, sel, src0, src1, repeat_times,
                default_blk_stride, default_blk_stride,
                default_blk_stride, dst_rep_stride, src0_rep_stride,
                src1_rep_stride, instr_name, mask_o)
