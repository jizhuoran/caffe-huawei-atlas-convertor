"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     tik_util.py
DESC:     util file for tik
CREATED:  2019-04-18 18:53:42
MODIFIED: 2019-07-19 17:59:30
"""
# disabling:
# W0622: redefined-builtin
# W0601: global-variable-undefined
# C0103: invalid-name
# R0913: too-many-arguments
# C0302: too-many-lines
from __future__ import print_function

from te import tvm

from te.platform import cce_params
from ..common.util import get_bit_len, is_immediate_number
from .tik_params import SIX_STRIDE_OFFSET_LIST, \
    SIX_STRIDE_SEGMENT_LIST, FOUR_STRIDE_OFFSET_LIST, \
    FOUR_STRIDE_SEGMENT_LIST, THREE_STRIDE_OFFSET_LIST, \
    THREE_STRIDE_SEGMENT_LIST, TWO_STRIDE_OFFSET_LIST, TWO_STRIDE_SEGMENT_LIST,\
    ONE_IR, PIPE_S, PIPE_V, DEQSCALE_46BIT_MASK, DEQSCALE_46BIT_SHIFT_POS
from .tik_params import ONE_BLK_SIZE
from .tik_check_util import TikCheckUtil
from .tik_source_info import source_info_decorator


_CONST_TRUE = 1
_CONST_FALSE = 0


@source_info_decorator()
def all(*args):  # pylint: disable=W0622
    """
     judging whether the all input argument result are true
     For example: a and b and c and ...

     Parameters
     ----------
     args :input argument

     Returns
     -------
     return : bool
             all result
     """
    args_new = []
    from .tik_expr import Expr
    for i in args:
        if isinstance(i, bool):
            args_new.append(tvm.const(_CONST_TRUE if i else _CONST_FALSE,
                                      "uint1"))
        elif isinstance(i, Expr):
            args_new.append(i.get())
        else:
            TikCheckUtil.raise_error("only bool expression is support for all")
    return Expr(tvm.all(*args_new))


@source_info_decorator()
def any(*args):  # pylint: disable=W0622
    """
     judging whether any result are true
     For example: a or b or c or ...

     Parameters
     ----------
     args :input argument

     Returns
     -------
     return : bool
             or result
     """
    args_new = []
    from .tik_expr import Expr
    for i in args:
        if isinstance(i, bool):
            args_new.append(tvm.const(_CONST_TRUE if i else _CONST_FALSE,
                                      "uint1"))
        elif isinstance(i, Expr):
            args_new.append(i.get())
        else:
            TikCheckUtil.raise_error("only bool expression is support for any")
    return Expr(tvm.any(*args_new))


def list_cmp(lhl, rhl):
    """
     two list compare

     Parameters
     ----------
     lhl : compare list
     rhl : be compared list

     Returns
     -------
     return : bool
         compare result
     """
    if len(lhl) != len(rhl):
        return False
    for data in zip(lhl, rhl):
        if data[0] != data[1]:
            return False
    return True


def type_convert(data_list, dtype=None):
    """
    data type convert

    Parameters
    ----------
    data_list : data list
    dtype: convert data type

    Returns
    -------
    return : data type
    """
    from .tik_expr import Expr
    if isinstance(data_list, (list, tuple)):
        data_new = []
        for i in data_list:
            if dtype is not None:
                data_new.append(Expr(i, dtype).get())
            else:
                data_new.append(Expr(i).get())
        return data_new
    if dtype is not None:
        return Expr(data_list, dtype).get()
    return Expr(data_list).get()


def non_stmt_judge(s_value):

    """
    judging whether is Empty Procedure

    Parameters
    ----------
    s_value : string value

    Returns
    -------
    return : bool
            judging result

    """
    from te.tvm import stmt, expr
    return (isinstance(s_value, stmt.Evaluate) and
            isinstance(s_value.value, expr.IntImm) and
            (s_value.value.value == 0))


def warning_info(value_str):
    """
    print warning information

    Parameters
    ----------
    value_str : print warming string

    Returns
    -------
    return : warning information
    """
    print("WARNING:%s" % value_str)


def dtype_convert(value, dtype):
    """Get target's scope

    Parameters
    ----------
    name : str, The scope name

    Returns
    -------
    str : the key of scope
    """
    ret = type_convert(value)
    return ret.astype(dtype)


def concat_params(params, offset_list=None, segment_list=None, dtype="int64"):
    """
    concat params
    :param params:
    :param offset_list:
    :param segment_list:
    :return: convert(Expr)
    """
    # disable it because it's to support python3
    from .tik_expr import Expr
    if offset_list is None:
        offset_list = [1]*len(params)
    if segment_list is None:
        segment_list = [1]*len(params)
    TikCheckUtil.check_equality(len(params), len(offset_list))
    TikCheckUtil.check_equality(len(params), len(segment_list))
    py_value = Expr(0, dtype)
    tvm_value = Expr(0, dtype)
    for (value, offset, segment) in zip(params, offset_list, segment_list):
        TikCheckUtil.check_type_match(offset, int)
        TikCheckUtil.check_type_match(segment, int)
        offset_v = Expr(offset, dtype)
        if isinstance(value, (float, int)):
            py_value |= Expr((value & (2**segment - 1)), dtype) << offset_v
        else:
            tvm_value |= (Expr(value).astype(dtype) &
                          (2**segment - 1)) << Expr(offset, dtype)
    return type_convert(Expr(py_value, dtype) | tvm_value)


def vector_config_encoding(repeat_times, strides):
    """[dst_blk_stride, src0_blk_stride, src1_blk_stride,
    dst_rep_stride, src0_rep_stride, src1_rep_stride]"""
    # disable it because it's to support python3
    from .tik_params import FOUR_STRIDE, SIX_STRIDE, THREE_STRIDE, \
        TWO_STRIDE, MAX_RET
    # [dst_blk_stride, src0_blk_stride, src1_blk_stride,
    #     dst_rep_stride, src0_rep_stride, src1_rep_stride]
    if len(strides) == SIX_STRIDE:
        args = list(strides) + [repeat_times]
        offset_list = SIX_STRIDE_OFFSET_LIST
        segment_list = SIX_STRIDE_SEGMENT_LIST
        ret = concat_params(args, offset_list, segment_list)
    # [dst_blk_stride, src_blk_stride, dst_rep_stride, src_rep_stride]
    elif len(strides) == FOUR_STRIDE:
        args = list(strides) + [repeat_times]
        offset_list = FOUR_STRIDE_OFFSET_LIST
        segment_list = FOUR_STRIDE_SEGMENT_LIST
        ret = concat_params(args, offset_list, segment_list)
    # [dst_rep_stride, src_blk_stride, src_rep_stride]/[dst_rep_stride,
    #       src0_rep_stride, src1_rep_stride]
    elif len(strides) == THREE_STRIDE:
        args = list(strides) + [repeat_times]
        offset_list = THREE_STRIDE_OFFSET_LIST
        segment_list = THREE_STRIDE_SEGMENT_LIST
        ret = concat_params(args, offset_list, segment_list)
    # [dst_rep_stride, src_rep_stride]
    elif len(strides) == TWO_STRIDE:
        args = list(strides) + [repeat_times]
        offset_list = TWO_STRIDE_OFFSET_LIST
        segment_list = TWO_STRIDE_SEGMENT_LIST
        ret = concat_params(args, offset_list, segment_list)
    else:
        TikCheckUtil.raise_error("only support vector")

    if isinstance(ret, int):
        TikCheckUtil.check_in_range(
            ret, range(MAX_RET),
            "result should be in the range of [0, 2**64 - 1]")
    return ret


def vector_encode_args(self, repeat_times, strides):
    """
    vector encode args
    """
    debug_mode = bool(
        hasattr(self, "debug_mode") and getattr(self, "debug_mode"))
    if debug_mode:
        return [repeat_times] + list(strides)
    return [vector_config_encoding(repeat_times, strides)]


def insert_set_deqscale_attr(tik_instance, deqscale, dtype_str, dst_dtype):
    """
    insert a set deqscale attr in IR
    :param tik_instance: tik object
    :param deqscale:
    :param dtype_str: string
    :return: None
    """
    from .tik_expr import Expr
    if dtype_str == "deq":
        with tik_instance.new_scope():
            tik_instance.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            tik_instance.emit(
                tvm.call_extern("float16", "set_deqscale",
                                type_convert(Expr(deqscale, dtype="float16"))),
                ONE_IR)
    elif dtype_str == "deqs162b8":
        deq_46 = (deqscale & DEQSCALE_46BIT_MASK) >> DEQSCALE_46BIT_SHIFT_POS
        if dst_dtype == "int8" and is_immediate_number(deqscale):
            # check deqscale[46]: 1
            TikCheckUtil.check_equality(
                deq_46, 1, "deqscale[46] bit should be 1 "
                           "when converting int16 to int8")
        elif dst_dtype == "uint8" and is_immediate_number(deqscale):
            # check deqscale[46]: 0
            TikCheckUtil.check_equality(
                deq_46, 0, "deqscale[46] bit should be 0 "
                           "when converting int16 to uint8")
        with tik_instance.new_scope():
            tik_instance.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
            tik_instance.emit(tvm.call_extern(
                "int64", "set_deqscale",
                type_convert(Expr(deqscale, dtype="uint64"))), ONE_IR)
    elif dtype_str == "vdeqs162b8":
        with tik_instance.context.freeze():
            scale_addr = tik_instance.Scalar_("int64")
            set_tensor_addr_to_scalar(tik_instance, scale_addr, deqscale)
            with tik_instance.new_scope():
                tik_instance.scope_attr(cce_params.CCE_AXIS,
                                        "coproc_scope", PIPE_V)
                tik_instance.emit(tvm.call_extern(
                    "int64", "set_deqscale",
                    tvm.call_extern("int64", "reinterpret_cast",
                                    scale_addr.get())), ONE_IR)


def need_check_out_of_scope(tik_instance):
    """
    if this isn't debug or building cce,tensor or scalar state variable
    should be checked in the scope.
    :param tik_instance:
    :return bool
    True: tensor or scalar should be checked
    False: tensor or scalar shouldn't be checked
    """
    return not tik_instance.is_debugging and not tik_instance.is_building_cce


def change_dtype_str(dst):
    """change dtype_str based on dtype of dst

    Parameters
    ----------
    dst : destination tensor

    Returns
    -------
    dtype_str
    """
    bit_len = get_bit_len(dst.dtype)
    if bit_len == 8:
        dtype_str = "uint8"
    elif bit_len == 16:
        dtype_str = "uint16"
    else:
        dtype_str = "uint32"

    return dtype_str


def emit_scatter_instr(tik_instance, total_ir_num, intrin_block):
    """emit scatter instruction

    Parameters
    ----------
    tik_instance: tik object
    total_ir_num: total ir number
    instrin_block: instruction stmt block

    Returns
    -------
    orig_ctrl: original value of CTRL
    """
    tik_instance.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_V)
    tik_instance.scope_attr(cce_params.CCE_AXIS, "mem_access_scope",
                            tvm.call_extern("int64", "__dummy__", "VA_reg_set"))
    total_ir_num += ONE_IR
    tik_instance.emit(intrin_block, total_ir_num)


def set_tensor_addr_to_scalar(tik_instance, scale_addr, tensor):
    """set tensor addr to scalar, for set_deqscale

    Parameters
    ----------
    scale_addr: int64 scalar, src of set_deqscale
    tensor: input tensor

    Returns
    -------
    scale_addr: scalar stored tensor addr
    """
    with tik_instance.new_scope():
        tik_instance.scope_attr(cce_params.CCE_AXIS, "coproc_scope", PIPE_S)
        tmp_const = tvm.const(ONE_BLK_SIZE, "int64")
        tmp_cast = tvm.expr.Cast(
            "int64", tvm.call_extern("handle", "", tensor.access_ptr("r")))
        scale_addr.set_as(tmp_cast // tmp_const)
