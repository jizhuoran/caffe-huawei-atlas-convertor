"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_scatter_vecotr_api.py
DESC:     provide scatter vector instructions
CREATED:  2019-08-12 18:53:42
MODIFIED: 2019-08-12 21:29:18
"""
# disabling:
# R0913: too-many-arguments

from te import tvm
from te.platform import cce_params
from te.platform.cce_conf import intrinsic_check_support
from te.platform.cce_params import scope_ubuf
from .. import debug
from .tik_expr import Expr
from ..api.tik_ir_builder import TikIRBuilder
from .tik_util import type_convert, dtype_convert, concat_params
from ..common.common_util import check_vector_stride, check_tensor_overflow, \
    check_address_overlapping
from ..common.util import DTYPE_SIZE, get_bit_len, reduce_mul, ceil_div
from .tik_params import CMPMASK_VAR, MAX_BLK_STRIDE_DOUBLE_BYTE, \
    MAX_BLK_STRIDE_SINGLE_BYTE, PIPE_V, MAX_REP_STRIDE_SINGLE_BYTE, \
    ONE_REP_BYTE_SIZE, DEFAULT_STRIDE, ONE_IR, TWO_IR, BLK_NUM_PER_REP, \
    ONE_BLK_SIZE, VCMPVS_OFFSET_LIST, VCMPVS_SEGMENT_LIST,\
    INSTR_DTYPE_SUPPORT_STATEMENT
from ..api.tik_scalar import mask_concat, Scalar
from ..api.tik_tensor import Tensor
from .tik_check_util import TikCheckUtil
from .tik_api_util import check_repeat_times
from .tik_source_info import source_info_decorator

_VCMP_REP_STRIDE = 0

def _check_vcmpvs_dst_tensor_overflow(dst, src, repeat_times):
    """check vcmpvs dst tensor overflow"""
    dst_ele_expected = Expr(repeat_times * ONE_REP_BYTE_SIZE //
                            DTYPE_SIZE[src.dtype] // get_bit_len(dst.dtype) +
                            dst.offset).eval_value()
    dst_ele_actual = reduce_mul(dst.indice.origin_shape)
    TikCheckUtil.check_le(
        dst_ele_expected, dst_ele_actual,
        "dst tensor overflow, expected dst shape: {}, actual dst shape: "
        "{}".format(dst_ele_expected, dst_ele_actual))

class TikCompareApi(TikIRBuilder):
    """Provide compare api"""
    def __init__(self):
        super(TikCompareApi, self).__init__()

    @source_info_decorator(depth=2)
    @debug.vcmp_decorator
    def _vcmp_elewise_func(self, name, mask,  # pylint: disable=R0913
                           src0, src1, src0_blk_stride,
                           src1_blk_stride, print_name=None, mask_o=None):
        """vcmp instruction, compare src once

        Parameters
        ----------
        name : instruction name
        mask : Effective operation on element, divided into two model:
                Continuous and bit by bit.
        src0 : source operation
        src1 : source operation
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block

        Returns
        -------
        CMPMASK
        """
        if print_name is None:
            print_name = name
        # check tensor
        TikCheckUtil.check_type_match(src0, Tensor,
                                      "src0's dtype should be tensor")
        TikCheckUtil.check_type_match(src1, Tensor,
                                      "src1's dtype should be tensor")

        TikCheckUtil.check_equality(src0.dtype, src1.dtype,
                                    "Intrinsic {}'s src0's dtype should be "
                                    "equal to src1's dtype".format(print_name))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_vcmp",
                                                            src0.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(src0.dtype, print_name))

        # check blk_stride type
        check_vector_stride([src0_blk_stride, src1_blk_stride], None,
                            MAX_BLK_STRIDE_DOUBLE_BYTE, None,
                            ["src0", "src1"])
        # check scope
        TikCheckUtil.check_equality(src0.scope, scope_ubuf,
                                    "src's scope should be UB")
        TikCheckUtil.check_equality(src1.scope, scope_ubuf,
                                    "src's scope should be UB")
        # mask
        if mask_o is None:
            mask_o = mask_concat(self, mask,
                                 tensor_bit_len=max(get_bit_len(src0.dtype),
                                                    get_bit_len(src1.dtype)))
        # check tensor overflow
        repeat_times = 1
        check_tensor_overflow((src0, src1), mask, repeat_times,
                              (src0_blk_stride, src1_blk_stride),
                              (_VCMP_REP_STRIDE, _VCMP_REP_STRIDE),
                              ("src0", "src1"))

        # code gen
        # default 1 repeat time for vcmp
        strides = [DEFAULT_STRIDE, src0_blk_stride, src1_blk_stride,
                   DEFAULT_STRIDE, DEFAULT_STRIDE, DEFAULT_STRIDE]
        with self.new_scope():
            mem_access_param = type_convert([repeat_times] + strides)
            if src0.dtype == "int16":
                instr = tvm.call_extern(
                    src0.dtype, name,
                    src0.reinterpret_cast_to("uint16").access_ptr("w"),
                    src1.reinterpret_cast_to("uint16").access_ptr("r"),
                    *mem_access_param)
            else:
                instr = tvm.call_extern(src0.dtype, name,
                                        src0.access_ptr("w"),
                                        src1.access_ptr("r"),
                                        *mem_access_param)
            self.emit(tvm.call_extern("int64", "set_vector_mask", *mask_o))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            # one ir is set_vector_mask, one is call_extern
            self.emit(instr, TWO_IR)
        return CMPMASK_VAR

    def vcmp_lt(self, mask, src0,  # pylint: disable=R0913
                src1, src0_stride, src1_stride):
        """src0 < src1

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
                Continuous and bit by bit.
        src0 : source operation
        src1 : source operation
        src0_stride : offset of src operator between different block
        src1_stride : offset of src operator between different block

        Returns
        -------
        CMPMASK
        """
        # function's input params is too much, so disable them
        return self._vcmp_elewise_func('vcmp_lt', mask, src0, src1,
                                       src0_stride, src1_stride)

    def vcmp_gt(self, mask, src0,  # pylint: disable=R0913
                src1, src0_stride, src1_stride):
        """src0 > src1

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
                Continuous and bit by bit.
        src0 : source operation
        src1 : source operation
        src0_stride : offset of src operator between different block
        src1_stride : offset of src operator between different block

        Returns
        -------
        CMPMASK
        """
        # function's input params is too much, so disable them
        return self._vcmp_elewise_func('vcmp_gt', mask, src0, src1,
                                       src0_stride, src1_stride)

    def vcmp_ge(self, mask, src0,  # pylint: disable=R0913
                src1, src0_stride, src1_stride):
        """src0 >= src1

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
                Continuous and bit by bit.
        src0 : source operation
        src1 : source operation
        src0_stride : offset of src operator between different block
        src1_stride : offset of src operator between different block

        Returns
        -------
        CMPMASK
        """
        # function's input params is too much, so disable them
        return self._vcmp_elewise_func('vcmp_ge', mask, src0, src1,
                                       src0_stride, src1_stride)

    def vcmp_eq(self, mask, src0,  # pylint: disable=R0913
                src1, src0_stride, src1_stride):
        """src0 = src1

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
                Continuous and bit by bit.
        src0 : source operation
        src1 : source operation
        src0_stride : offset of src operator between different block
        src1_stride : offset of src operator between different block

        Returns
        -------
        CMPMASK
        """
        # function's input params is too much, so disable them
        return self._vcmp_elewise_func('vcmp_eq', mask, src0, src1,
                                       src0_stride, src1_stride)

    def vcmp_ne(self, mask, src0,  # pylint: disable=R0913
                src1, src0_stride, src1_stride):
        """src0 != src1

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
                Continuous and bit by bit.
        src0 : source operation
        src1 : source operation
        src0_stride : offset of src operator between different block
        src1_stride : offset of src operator between different block

        Returns
        -------
        CMPMASK
        """
        # function's input params is too much, so disable them
        return self._vcmp_elewise_func('vcmp_ne', mask, src0, src1,
                                       src0_stride, src1_stride)

    def vcmp_le(self, mask, src0,  # pylint: disable=R0913
                src1, src0_stride, src1_stride):
        """src0 < src1

        Parameters
        ----------
        mask : Effective operation on element, divided into two model:
                Continuous and bit by bit.
        src0 : source operation
        src1 : source operation
        src0_stride : offset of src operator between different block
        src1_stride : offset of src operator between different block

        Returns
        -------
        CMPMASK
        """
        # function's input params is too much, so disable them
        return self._vcmp_elewise_func('vcmp_le', mask, src0, src1,
                                       src0_stride, src1_stride)

    @source_info_decorator(depth=2)
    @debug.vcmpv_decorator
    def _vcmpv_elewise_func(self, name, dst,  # pylint: disable=R0913, R0914
                            src0, src1, repeat_times,
                            src0_blk_stride, src1_blk_stride, src0_rep_stride,
                            src1_rep_stride, print_name=None):
        """vcmpv instruction, compare by element

        Parameters
        ----------
        name : instruction name
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between

        Returns
        -------
        None
        """
        if print_name is None:
            print_name = name
        # check parameter repeat_times
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride([src0_blk_stride, src1_blk_stride],
                            [src0_rep_stride, src1_rep_stride],
                            MAX_BLK_STRIDE_SINGLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["src0", "src1"])
        # check dtype
        TikCheckUtil.check_in_range(
            dst.dtype, ["uint64", "uint32", "uint16", "uint8"],
            "dst should be uint64, uint32, uint16 or uint8")
        TikCheckUtil.check_equality(src0.dtype, src1.dtype,
                                    "Intrinsic {}'s src0's dtype should"
                                    " be equal to src1's dtype".
                                    format(print_name))

        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            name,
                                                            src0.dtype), True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(src0.dtype, print_name))

        # check scope
        TikCheckUtil.check_equality(src0.scope, scope_ubuf,
                                    "src's scope should be UB")
        TikCheckUtil.check_equality(src1.scope, scope_ubuf,
                                    "src's scope should be UB")
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst's scope should be UB")

        default_blk_stride = 1
        default_rep_stride = 8
        # check address overlap
        if dst.buffer == src0.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, src0_blk_stride, src0_rep_stride)):
                check_address_overlapping(
                    print_name, ONE_REP_BYTE_SIZE // max(
                        DTYPE_SIZE[dst.dtype], DTYPE_SIZE[src0.dtype]),
                    dst, src0, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE //
                    max(get_bit_len(dst.dtype), get_bit_len(src0.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src0.dtype),
                    repeat_times, default_blk_stride, src0_blk_stride,
                    default_rep_stride, src0_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src0.offset).eval_value(), msg="dst and src0")

        if dst.buffer == src1.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, src1_blk_stride, src1_rep_stride)):
                check_address_overlapping(
                    print_name, ONE_REP_BYTE_SIZE // max(
                        DTYPE_SIZE[dst.dtype], DTYPE_SIZE[src0.dtype]),
                    dst, src1, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE // max(
                        get_bit_len(dst.dtype), get_bit_len(src0.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src1.dtype),
                    repeat_times, default_blk_stride, src1_blk_stride,
                    default_rep_stride, src1_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src1.offset).eval_value(), msg="dst and src1")

        # check tensor overflow(static)
        check_tensor_overflow(
            [src0, src1], ONE_REP_BYTE_SIZE // DTYPE_SIZE[src0.dtype],
            repeat_times, [src0_blk_stride, src1_blk_stride],
            [src0_rep_stride, src1_rep_stride], ["src0", "src1"])

        # gen
        default_stride = 0
        config = [
            repeat_times, default_stride, src0_blk_stride, src1_blk_stride,
            default_stride, src0_rep_stride, src1_rep_stride,
        ]
        with self.new_scope():
            instr = tvm.call_extern(
                dst.dtype, name,
                dst.access_ptr("w", cast_dtype="uint8"), src0.access_ptr("r"),
                src1.access_ptr("r"), *type_convert(config))
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            # 1 ir is call_extern
            self.emit(instr, ONE_IR)

    def vcmpv_lt(self, dst, src0,  # pylint: disable=R0913
                 src1, repeat_times, src0_blk_stride,
                 src1_blk_stride, src0_rep_stride, src1_rep_stride):
        """src0 < src1  compare by element

        Parameters
        ----------
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        # function's input params is too much, so disable them
        return self._vcmpv_elewise_func('vcmpv_lt', dst, src0, src1,
                                        repeat_times, src0_blk_stride,
                                        src1_blk_stride, src0_rep_stride,
                                        src1_rep_stride)

    def vcmpv_gt(self, dst, src0,  # pylint: disable=R0913
                 src1, repeat_times, src0_blk_stride,
                 src1_blk_stride, src0_rep_stride, src1_rep_stride):
        """src0 > src1  compare by element

        Parameters
        ----------
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        # function's input params is too much, so disable them
        return self._vcmpv_elewise_func('vcmpv_gt', dst, src0, src1,
                                        repeat_times, src0_blk_stride,
                                        src1_blk_stride, src0_rep_stride,
                                        src1_rep_stride)

    def vcmpv_ge(self, dst, src0,  # pylint: disable=R0913
                 src1, repeat_times, src0_blk_stride,
                 src1_blk_stride, src0_rep_stride, src1_rep_stride):
        """src0 >= src1  compare by element

        Parameters
        ----------
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        # function's input params is too much, so disable them
        return self._vcmpv_elewise_func('vcmpv_ge', dst, src0, src1,
                                        repeat_times, src0_blk_stride,
                                        src1_blk_stride, src0_rep_stride,
                                        src1_rep_stride)

    def vcmpv_eq(self, dst, src0,  # pylint: disable=R0913
                 src1, repeat_times, src0_blk_stride,
                 src1_blk_stride, src0_rep_stride, src1_rep_stride):
        """src0 = src1  compare by element

        Parameters
        ----------
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        # function's input params is too much, so disable them
        return self._vcmpv_elewise_func('vcmpv_eq', dst, src0, src1,
                                        repeat_times, src0_blk_stride,
                                        src1_blk_stride, src0_rep_stride,
                                        src1_rep_stride)

    def vcmpv_ne(self, dst, src0,  # pylint: disable=R0913
                 src1, repeat_times, src0_blk_stride,
                 src1_blk_stride, src0_rep_stride, src1_rep_stride):
        """src0 != src1  compare by element

        Parameters
        ----------
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        # function's input params is too much, so disable them
        return self._vcmpv_elewise_func('vcmpv_ne', dst, src0, src1,
                                        repeat_times, src0_blk_stride,
                                        src1_blk_stride, src0_rep_stride,
                                        src1_rep_stride)

    def vcmpv_le(self, dst, src0,  # pylint: disable=R0913
                 src1, repeat_times, src0_blk_stride,
                 src1_blk_stride, src0_rep_stride, src1_rep_stride):
        """src0 <= src1  compare by element

        Parameters
        ----------
        dst : destination operator
        src0 : source operation
        src1 : source operation
        repeat_times : Repeated iterations times
        src0_blk_stride : offset of src operator between different block
        src1_blk_stride : offset of src operator between different block
        src0_rep_stride : offset of src operator in the same block between
        src1_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        # function's input params is too much, so disable them
        return self._vcmpv_elewise_func('vcmpv_le', dst, src0, src1,
                                        repeat_times, src0_blk_stride,
                                        src1_blk_stride, src0_rep_stride,
                                        src1_rep_stride)

    @source_info_decorator(depth=2)
    @debug.vcmpvs_elewise_func_decorator
    def _vcmpvs_elewise_func(self, name, dst,  # pylint: disable=R0913
                             src, scalar, repeat_times,
                             src_blk_stride, src_rep_stride):
        """vcmpvs instruction, compare by element

        Parameters
        ----------
        name : instruction name
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        None
        """
        # check dst src
        TikCheckUtil.check_type_match(
            dst, Tensor,
            "dst should be tensor, input type of dst: {}".format(type(dst)))
        TikCheckUtil.check_equality(dst.scope, scope_ubuf,
                                    "dst scope should be ub, "
                                    "input scope: {}".format(dst.scope))
        TikCheckUtil.check_type_match(
            src, Tensor,
            "src should be tensor, input type of src: {}".format(type(src)))
        TikCheckUtil.check_equality(src.scope, scope_ubuf,
                                    "src scope should be ub, "
                                    "input scope: {}".format(src.scope))
        # check repeat
        check_repeat_times(repeat_times)
        # check strides
        check_vector_stride([src_blk_stride], [src_rep_stride],
                            MAX_BLK_STRIDE_DOUBLE_BYTE,
                            MAX_REP_STRIDE_SINGLE_BYTE, ["src"])
        # check dtype
        dtype_list = ("uint64", "uint32", "uint16", "uint8")
        TikCheckUtil.check_in_range(
            dst.dtype, dtype_list, "dst dtype should be unsigned int, "
                                   "input dst dtype: {}".format(dst.dtype))
        if isinstance(scalar, Scalar):
            TikCheckUtil.check_equality(src.dtype, scalar.dtype,
                                        "Intrinsic {}'s src's dtype should"
                                        " be equal to scalar's dtype".
                                        format(name))
        TikCheckUtil.check_equality(intrinsic_check_support("Intrinsic_" +
                                                            name, src.dtype),
                                    True,
                                    INSTR_DTYPE_SUPPORT_STATEMENT.
                                    format(src.dtype, name))
        default_blk_stride = 1
        default_rep_stride = 8
        # check address overlap
        if dst.buffer == src.buffer:
            if all(isinstance(value, int) for \
                   value in (repeat_times, src_blk_stride, src_rep_stride)):
                check_address_overlapping(
                    name, ONE_REP_BYTE_SIZE // max(
                        DTYPE_SIZE[dst.dtype], DTYPE_SIZE[src.dtype]),
                    dst, src, BLK_NUM_PER_REP,
                    ONE_REP_BYTE_SIZE // max(
                        get_bit_len(dst.dtype), get_bit_len(src.dtype)),
                    ONE_REP_BYTE_SIZE // get_bit_len(dst.dtype),
                    ONE_REP_BYTE_SIZE // get_bit_len(src.dtype),
                    repeat_times, default_blk_stride, src_blk_stride,
                    default_rep_stride, src_rep_stride,
                    Expr(dst.offset).eval_value(),
                    Expr(src.offset).eval_value())

        # check src tensor overflow(static)
        check_tensor_overflow((src,), ONE_REP_BYTE_SIZE//DTYPE_SIZE[src.dtype],
                              repeat_times, (src_blk_stride,),
                              (src_rep_stride,), ("src",))
        # check dst tensor overflow
        if all(Expr(value).eval_value() is not None for value
               in (repeat_times, dst.offset)):
            _check_vcmpvs_dst_tensor_overflow(dst, src, repeat_times)

        self._gen_vcmpvs_elewise_func_code(name, scalar, src, dst,
                                           src_blk_stride, src_rep_stride,
                                           repeat_times)

    def _gen_vcmpvs_elewise_func_code(self, name,  # pylint: disable=R0913
                                      scalar, src, dst,
                                      src_blk_stride, src_rep_stride,
                                      repeat_times):
        """gen vcmpvs code

        Parameters
        ----------
        name : instruction name
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        None
        """
        # gen
        config = [src_blk_stride, src_rep_stride, repeat_times]
        args = concat_params(config, VCMPVS_OFFSET_LIST, VCMPVS_SEGMENT_LIST)
        scalar_tmp = dtype_convert(scalar, src.dtype)
        # cal extent
        src_extent = Expr(((repeat_times - 1)*src_rep_stride +
                           (BLK_NUM_PER_REP - 1)*src_blk_stride + 1)*
                          ONE_BLK_SIZE).get()
        dst_extent = repeat_times*ONE_REP_BYTE_SIZE//DTYPE_SIZE[src.dtype]//\
                     get_bit_len(dst.dtype)*DTYPE_SIZE[dst.dtype]
        # 32B aligned
        dst_extent = Expr(ceil_div(dst_extent, ONE_BLK_SIZE)*ONE_BLK_SIZE).get()
        with self.new_scope():
            instr = tvm.call_extern(
                dst.dtype, name,
                dst.reinterpret_cast_to("uint8").access_ptr("w",
                                                            extent=dst_extent),
                src.access_ptr("r", extent=src_extent), scalar_tmp, args)
            self.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            # 1 ir is call_extern
            self.emit(instr, ONE_IR)

    def vcmpvs_lt(self, dst, src, scalar,  # pylint: disable=R0913
                  repeat_times, src_blk_stride, src_rep_stride):
        """src < scalar  compare by element

        Parameters
        ----------
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        return self._vcmpvs_elewise_func('vcmpvs_lt', dst,
                                         src, scalar, repeat_times,
                                         src_blk_stride, src_rep_stride)

    def vcmpvs_gt(self, dst, src, scalar,  # pylint: disable=R0913
                  repeat_times, src_blk_stride, src_rep_stride):
        """src > scalar  compare by element

        Parameters
        ----------
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        return self._vcmpvs_elewise_func('vcmpvs_gt', dst,
                                         src, scalar, repeat_times,
                                         src_blk_stride, src_rep_stride)

    def vcmpvs_ge(self, dst, src, scalar,  # pylint: disable=R0913
                  repeat_times, src_blk_stride, src_rep_stride):
        """src >= scalar  compare by element

        Parameters
        ----------
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        return self._vcmpvs_elewise_func('vcmpvs_ge', dst,
                                         src, scalar, repeat_times,
                                         src_blk_stride, src_rep_stride)

    def vcmpvs_eq(self, dst, src, scalar,  # pylint: disable=R0913
                  repeat_times, src_blk_stride, src_rep_stride):
        """src == scalar  compare by element

        Parameters
        ----------
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        return self._vcmpvs_elewise_func('vcmpvs_eq', dst,
                                         src, scalar, repeat_times,
                                         src_blk_stride, src_rep_stride)

    def vcmpvs_ne(self, dst, src, scalar,  # pylint: disable=R0913
                  repeat_times, src_blk_stride, src_rep_stride):
        """src != scalar  compare by element

        Parameters
        ----------
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        return self._vcmpvs_elewise_func('vcmpvs_ne', dst,
                                         src, scalar, repeat_times,
                                         src_blk_stride, src_rep_stride)

    def vcmpvs_le(self, dst, src, scalar,  # pylint: disable=R0913
                  repeat_times, src_blk_stride, src_rep_stride):
        """src <= scalar  compare by element

        Parameters
        ----------
        dst : destination operator
        src : source operation
        scalar : source scalar operation
        repeat_times : Repeated iterations times
        src_blk_stride : offset of src operator between different block
        src_rep_stride : offset of src operator in the same block between

        Returns
        -------
        byte 1 mean true  0 mean false
        """
        return self._vcmpvs_elewise_func('vcmpvs_le', dst,
                                         src, scalar, repeat_times,
                                         src_blk_stride, src_rep_stride)
