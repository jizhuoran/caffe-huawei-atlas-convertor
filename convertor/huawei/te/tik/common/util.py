"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     util.py
DESC:     provide common function
CREATED:  2019-08-13 11:33:42
MODIFIED: 2019-08-13 11:33:42
"""

import sys

from te import tvm
from te.platform.cce_params import scope_cbuf, scope_ubuf, \
    scope_ca, scope_cb, scope_cc
from ..tik_lib.tik_params import MASK_VALUE_ZERO, ONE_BLK_SIZE, VA0_INDEX, \
    MASK_LEN_128, VNCHWCONV_LIST_LEN, ONE_REP_BYTE_SIZE, BLK_NUM_PER_REP, \
    MASK_HIGH_IDX, MASK_LEN_FULL_MODE, MIN_INDEX, MAX_INDEX, MASK_LOW_IDX, \
    CONST_MASK_VALUE, MAX_LOW_MASK_LEN, PYTHON_VERSION3, PYTHON_VERSION_IDX, \
    MAX_MASK_HALF, MIN_MASK_HALF, BIT_LEN_32
from ..tik_lib.tik_check_util import TikCheckUtil
_BLK_LEN_16 = 16


DTYPE_SIZE = {
    'uint8': 1,
    'int8': 1,
    'uint16': 2,
    'int16': 2,
    'uint32': 4,
    'int32': 4,
    'float16': 2,
    'float32': 4,
    'int64': 8,
    'uint64': 8,
    'float64': 8
}


DTYPE_FOR_INPUT_SCALAR = {"uint8": 0, "int8": 1, "uint16": 2, "int16": 3, "uint32": 4,
                          "int32": 5, "uint64": 6,
                          "int64": 7, "float16": 8, "float32": 9}

DTYPE_INT_VALUE = {"uint8": (0, 255), "int8": (-128, 127),
                   "uint16": (0, 65535), "int16": (-32768, 32767),
                   "uint32": (0, 4294967295), "int32": (-2147483648, 2147483647),
                   "uint64": (0, 18446744073709551615),
                   "int64": (-9223372036854775808, 9223372036854775807)
                   }

def check_scope(buffer_scope):
    """
    check the scope.
    :param buffer_scope:
    :return: None;
    """
    scope_list = [scope_cbuf, scope_ubuf, scope_ca, scope_cb, scope_cc]
    TikCheckUtil.check_in_range(buffer_scope, scope_list, "scope out of Tensor Scope")

def reduce_mul(shape):
    """
    calculate shape list value
    Apply a function of two arguments cumulatively to the items of a sequence,
    from left to right, so as to reduce the sequence to a single value.
    For example, reduce(lambda x, y: x*y, [1, 2, 3, 4, 5]) calculates
    ((((1*2)*3)*4)*5).  If initial is present, it is placed before the items
    of the sequence in the calculation, and serves as a default when the
    sequence is empty.
    Parameters
    ----------
    shape : TVM shape

    Returns
    -------
    return : int
        reduce multiplication
    """
    # pylint: disable=W0622
    # disable it because it's to support python3
    from functools import reduce
    return reduce(lambda i, j: i*j, shape)


def ceil_div(a_value, b_value):
    """
    ceil division
    Parameters
    ----------
    a_value :operator
    b_value :division value

    Returns
    -------
    return:computational results
    """
    return (a_value + b_value - 1) // b_value


def get_bit_len(dtype):
    """
    calculate bits of dtype of TVM
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : int
        bit length of dtype.
    """
    index = 0
    for i in dtype:
        if i.isdigit():
            break
        index += 1
    if dtype[index:].isdigit():
        return int(dtype[index:])
    return TikCheckUtil.raise_error("get bits of dtype failed")


def check_integer_in_range(var, var_range, msg="Variable out of range"):
    """
    check the integer if in the range
    :param var:
    :param var_range:
    :param msg: exception message
    :return: None
    """
    if isinstance(var, int):
        TikCheckUtil.check_in_range(var, var_range, msg)


def check_imme_mask_full_mode(mask, tensor_bit_len):
    """check immediate_number mask value with full mode

    Parameters
    ----------
    mask : Effective operation on element, divided into two model:
           Continuous and bit by bit.
    tensor_bit_len : bit length of operation tensor's dtype

    Returns
    -------
    None
    """
    # for immediate list mask
    for data in mask:
        TikCheckUtil.check_type_match(data, int,
                                      "mask list value should be int, "
                                      "input type: {}".format(type(data)))
        TikCheckUtil.check_ge(
            data, MIN_MASK_HALF,
            "mask value should be in the range of [0, 2**64-1], "
            "input mask: {}".format(data))
        TikCheckUtil.check_le(
            data, MAX_MASK_HALF,
            "mask value should be in the range of [0, 2**64-1], "
            "input mask:{}".format(data))
    # mask can not be all zero
    TikCheckUtil.check_not_equality(
        sum(mask), 0, "mask list value can not be [0, 0]")
    # b32, mask_h should be 0
    if tensor_bit_len == BIT_LEN_32:
        TikCheckUtil.check_equality(
            mask[0], MASK_VALUE_ZERO,
            "mask_h should be 0 for b32 tensor, input mask_h: "
            "{}".format(mask[0]))


def _tensor_list_overflow_check(tensor_list, name,  # pylint: disable=R0913
                                mask_len, repeat_times,
                                rep_stride, valid_num_per_block,
                                store_high_half=False):
    """ check scatter vector instruction, input tensor_list tensor overflow
    for api module and debug module

    Parameters
    ----------
    tensor_list: input parameters, dst_list or src_list, length = 8
    name: name of tensor_list, for printing check RuntimeError message
    mask_len: in last repeat, length between lowest digit and top effective
              digit of mask
    repeat_times: numbers of iterations of this instruction
    rep_stride: stride of blocks between two repeats
    valid_num_per_block: actual valid numbers of each block

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    from ..tik_lib.tik_expr import Expr
    # blk_num: number of blks whose mask_len_per_blk is blk_len
    blk_num = ceil_div(mask_len, valid_num_per_block)
    # mask_left: mask_len of last processed blk
    mask_left = mask_len % valid_num_per_block
    if mask_left == MASK_VALUE_ZERO:
        mask_left = valid_num_per_block
    # elements of 1 blk
    blk_len = ONE_BLK_SIZE // DTYPE_SIZE[tensor_list[VA0_INDEX].dtype]
    TikCheckUtil.check_in_range(
        blk_len, (valid_num_per_block, 2*valid_num_per_block),
        "please check tensor list dtype!")
    for index in range(blk_num):
        if repeat_times == 1:
            if blk_len != valid_num_per_block and store_high_half:
                expected_ele = rep_stride*blk_len + blk_len + \
                               tensor_list[index].offset
            else:
                expected_ele = rep_stride*blk_len + valid_num_per_block + \
                               tensor_list[index].offset
        else:
            if blk_len != valid_num_per_block and store_high_half:
                expected_ele = (repeat_times - 1)*rep_stride*blk_len + \
                               blk_len + tensor_list[index].offset
            else:
                expected_ele = (repeat_times - 1)*rep_stride*blk_len + \
                               valid_num_per_block + tensor_list[index].offset

        # for last processed blk, mask_len_per_blk not blk_len but mask_left
        if index == blk_num - 1:
            expected_ele = expected_ele - valid_num_per_block + mask_left

        actual_ele = reduce_mul(tensor_list[index].indice.origin_shape)

        expected_ele = Expr(expected_ele).eval_value()
        actual_ele = Expr(actual_ele).eval_value()
        if expected_ele is not None and actual_ele is not None:
            TikCheckUtil.check_ge(
                actual_ele, expected_ele,
                "{}[{}] tensor overflow, expected elements: {}, actual "
                "elements: {}".format(name, index, expected_ele, actual_ele))


def check_vnchwconv_overflow(tensor, name_list,  # pylint: disable=R0913
                             repeat_times, rep_stride_list,
                             store_high_half_list, dtype_str):
    """ check scatter vector instruction, input tensor_list tensor overflow
    for api module and debug module, for vnchwconv b8

    Parameters
    ----------
    tensor: list of tensor_list
    name: name of tensor_list, for printing check RuntimeError message
    repeat_times: numbers of iterations of this instruction
    rep_stride_list: list of stride of blocks between two repeats
    store_high_half_list: list of store_high_half
    dtype_str: b8, b16 or b32

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    from ..tik_lib.tik_expr import Expr
    repeat_times = Expr(repeat_times).eval_value()
    if repeat_times is None:
        return
    for tensor_list, name, rep_stride, store_high_half in \
            zip(tensor, name_list, rep_stride_list, store_high_half_list):
        if dtype_str == 'b8':
            _tensor_list_overflow_check(
                tensor_list, name, MASK_LEN_128, repeat_times, rep_stride,
                _BLK_LEN_16, store_high_half)
        else:
            _tensor_list_overflow_check(
                tensor_list, name, ONE_BLK_SIZE*VNCHWCONV_LIST_LEN //
                DTYPE_SIZE[tensor_list[VA0_INDEX].dtype], repeat_times,
                rep_stride,
                ONE_BLK_SIZE // DTYPE_SIZE[tensor_list[VA0_INDEX].dtype],
                store_high_half)


def check_scatter_vector_overflow(tensor, name_list,  # pylint: disable=R0913
                                  mask, repeat_times,
                                  rep_stride_list, store_high_half=False,
                                  mask_mode="normal"):
    """ check scatter vector instruction, input tensor_list tensor overflow
    for api module and debug module

    Parameters
    ----------
    tensor: list of tensor_list
    name: name of tensor_list, for printing check RuntimeError message
    mask: Effective operation on element
    repeat_times: numbers of iterations of this instruction
    rep_stride_list: list of stride of blocks between two repeats
    mask_mode: mask mode, normal or counter

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    bit_len = []
    for tensor_in in tensor:
        tensor_bit_len = get_bit_len(tensor_in[VA0_INDEX].dtype)
        bit_len.append(tensor_bit_len)
    # valid elements of 1 blk
    valid_num_per_block = ONE_REP_BYTE_SIZE // max(bit_len)
    TikCheckUtil.check_equality(
        len(tensor), len(name_list),
        "length of input params tensor and name_list should be equal!")
    TikCheckUtil.check_equality(
        len(tensor), len(rep_stride_list),
        "length of input params tensor and rep_stride_list should be equal!")
    for tensor_list, name, rep_stride in \
            zip(tensor, name_list, rep_stride_list):
        _check_tensor_list_overflow(tensor_list, name, mask, repeat_times,
                                    rep_stride, valid_num_per_block,
                                    store_high_half=store_high_half,
                                    mask_mode=mask_mode)

def _check_tensor_list_overflow(tensor_list, name,  # pylint: disable=R0913
                                mask, repeat_times,
                                rep_stride, valid_num_per_block,
                                store_high_half=False, mask_mode="normal"):
    """ check scatter vector instruction, input tensor_list tensor overflow
    for api module and debug module

    Parameters
    ----------
    tensor_list: input parameters, dst_list or src_list, length = 8
    name: name of tensor_list, for printing check RuntimeError message
    mask: Effective operation on element
    repeat_times: numbers of iterations of this instruction
    rep_stride: stride of blocks between two repeats
    mask_mode: mask mode, normal or counter

    Returns
    -------
    None
    """
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    from ..tik_lib.tik_expr import Expr
    if mask_mode == "counter":
        if Expr(mask).eval_value() is None:
            return
        rep_len = valid_num_per_block*BLK_NUM_PER_REP
        repeat_times = ceil_div(mask, rep_len)
        mask = mask % rep_len
        if mask == 0:
            mask = rep_len

    repeat_times = Expr(repeat_times).eval_value()
    if repeat_times is None:
        return

    if not isinstance(mask, (list, tuple)):
        mask = [mask]
    for value in mask:
        if Expr(value).eval_value() is None:
            return
    if len(mask) == 1:
        mask_len = mask[MASK_HIGH_IDX]
    else:
        mask_len, _ = get_mask_len(mask)
    TikCheckUtil.check_ge(
        len(tensor_list)*valid_num_per_block, mask_len,
        "Please check number of {} for current mask.".format(name))

    _tensor_list_overflow_check(tensor_list, name, mask_len, repeat_times,
                                rep_stride, valid_num_per_block,
                                store_high_half=store_high_half)


def check_scatter_dict_for_overlap(src_dict, dst_dict, name, msg):
    """check src_dict and dst_dict"""
    for buffer in src_dict.keys():
        if buffer in dst_dict.keys():
            for interval_src in src_dict[buffer]:
                for interval_dst in dst_dict[buffer]:
                    if max(interval_src[0], interval_dst[0]) < \
                            min(interval_src[1], interval_dst[1]):
                        TikCheckUtil.raise_error(
                            "{} {} not support partially "
                            "address overlap, only support repeat_time=1"
                            " fully overlapping.".format(name, msg))


def get_mask_len(mask):
    """
    get mask len when in others situation
    :param mask:
    :return: mask len
    """
    # mask length must greater than 1
    TikCheckUtil.check_equality(
        len(mask), MASK_LEN_FULL_MODE, "length of mask should be 2")
    mask_len = 0
    if mask[MASK_HIGH_IDX] == 0:
        for index in range(MIN_INDEX, MAX_INDEX):
            if not mask[MASK_LOW_IDX] & (CONST_MASK_VALUE >> index) == 0:
                mask_len = MAX_LOW_MASK_LEN - index
                break
    else:
        for index in range(MIN_INDEX, MAX_INDEX):
            if not mask[MASK_HIGH_IDX] & (CONST_MASK_VALUE >> index) == 0:
                mask_len = MAX_LOW_MASK_LEN - index + MAX_INDEX
                break
    return mask_len, index


def check_scalar_dtype(var, msg="Scalar dtype should be int"):
    """
    check the Scalar dtype if int
    :param var :
    :param msg : exception message
    :return: None
    """
    if is_basic_expr(var):
        TikCheckUtil.check_in_range("int", var.dtype, msg)


def check_scalar_dtype_float(var, msg="Scalar dtype should be float"):
    """
    check the Scalar dtype if int
    :param var:
    :param msg: exception message
    :return: None
    """
    if is_basic_expr(var):
        TikCheckUtil.check_in_range("float", var.dtype, msg)


def check_scalar_int32(var, msg="Scalar dtype must be int32"):
    """
    check the Scalar dtype if int32
    :param var:
    :param msg: exception message
    :return: None
    """
    if is_basic_expr(var):
        TikCheckUtil.check_equality(var.dtype, "int32", msg)


def instance_judge(data_list, type_list):
    """
    judging data list type

    Parameters
    ----------
    data_list : data list
    type_list : type list

    Returns
    -------
    return : bool
         instance judge result
    """
    if not isinstance(type_list, (tuple, list)):
        type_list = [type_list]
    if not isinstance(data_list, (tuple, list)):
        data_list = [data_list]
    for data in data_list:
        if not isinstance(data, tuple(type_list)):
            return False
    return True


def tvm_immediate_number(data_list):
    """
    judging whether tvm immediate number is in tvm data type
    Parameters
    ----------
    data_list : data list

    Returns
    -------
    return : bool
         tvm immediate number judge result
    """
    type_list = [tvm.expr.IntImm, tvm.expr.FloatImm, tvm.expr.UIntImm]
    return instance_judge(data_list, type_list)


def is_immediate_number(data_list):
    """
    judging python immediate number

    Parameters
    ----------
    data_list : data list

    Returns
    -------
    return : bool
         python immediate number judge result
    """
    # pylint: disable=W0601, C0103, W0622
    # disable it because it's to support python3
    global long
    if sys.version_info[PYTHON_VERSION_IDX] >= PYTHON_VERSION3:
        long = int
    return instance_judge(data_list, (float, int, long))


def tik_reg_type(data_list):
    """
    judging judging register data type

    Parameters
    ----------
    data_list : data list

    Returns
    -------
    return : bool
         tik_register number judge result
    """
    from ..tik_lib.tik_expr import BasicExpr
    type_list = [tvm.expr.Load, tvm.expr.BinaryOpExpr, BasicExpr]
    return instance_judge(data_list, type_list)


def is_basic_expr(data_list):
    """
    judging BasicExpr data type

    Parameters
    ----------
    data_list : data list

    Returns
    -------
    return : bool
         BasicExpr data type judge result
    """
    from ..tik_lib.tik_expr import BasicExpr
    type_list = [BasicExpr]
    return instance_judge(data_list, type_list)


def tvm_var_type(data_list):
    """
    judging value data type

    Parameters
    ----------
    data_list : data list

    Returns
    -------
    return : bool
         tvm var type judge result
    """
    type_list = [tvm.expr.Load, tvm.expr.BinaryOpExpr,
                 tvm.expr.Var, tvm.expr.Cast]
    return instance_judge(data_list, type_list)


def get_check_feed_dict(feed_dict, input_list_tensor,  # pylint: disable=R0913
                        input_list_var, build_list_tensor,
                        build_cce_input_names, build_cce_input_tensor_names,
                        build_cce_input_var_names):
    """get and check input feed_dict"""
    input_feed_dict_names = " ".join(list(feed_dict.keys()))
    feed_dict_tensor = {}
    feed_dict_var = {}
    for key, value in feed_dict.items():
        if isinstance(value, (int, float)):
            feed_dict_var[key] = value
        else:
            feed_dict_tensor[key] = value

    feed_dict_tensor_names = " ".join(list(feed_dict_tensor.keys()))
    feed_dict_var_names = " ".join(list(feed_dict_var.keys()))
    if build_cce_input_names in (" ", ""):
        build_cce_input_names = "None"
    if input_feed_dict_names == "":
        input_feed_dict_names = "None"
    if build_cce_input_tensor_names == "":
        build_cce_input_tensor_names = "None"
    if build_cce_input_var_names == "":
        build_cce_input_var_names = "None"
    if feed_dict_tensor_names == "":
        feed_dict_tensor_names = "None"
    if feed_dict_var_names == "":
        feed_dict_var_names = "None"
    TikCheckUtil.check_equality(
        build_list_tensor, set(feed_dict.keys()),
        "BuildCCE input list is " + build_cce_input_names +
        ", but feed_dict list is " + input_feed_dict_names)
    TikCheckUtil.check_equality(
        set(input_list_var), set(feed_dict_var.keys()),
        "BuildCCE InputScalar list is " + build_cce_input_var_names +
        ", but feed_dict InputScalar list is " + feed_dict_var_names)
    TikCheckUtil.check_equality(
        set(input_list_tensor), set(feed_dict_tensor.keys()),
        "BuildCCE Tensor list is " + build_cce_input_tensor_names +
        ", but feed_dict Tensor list is " + feed_dict_tensor_names)
    return feed_dict_tensor, feed_dict_var


class TikUtil():
    """Provide some common util function"""

    @staticmethod
    def to_list(var):
        """
        if var is not list, convert to list
        :param var:
        :return: list object
        """
        var1 = var
        if not isinstance(var, (list, tuple)):
            var1 = [var1]
        return var1

    @staticmethod
    def to_int(var):
        """
        convert var to int
        :param var:
        :return: 1 or 0
        """
        if var:
            return 1
        return 0

    @staticmethod
    def get_storage_scope(name):
        """get the storage scope name

        Parameters
        ----------
        name : input

        Returns
        -------
        the result name
        """
        tmp = name.split(".")
        if tmp[0].lower() == "global":
            return "OUT"
        if tmp[1].count('UB'):
            return "UB"
        return tmp[1]
