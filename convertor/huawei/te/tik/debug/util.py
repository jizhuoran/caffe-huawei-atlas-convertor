"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
FILE:     util.py
DESC:     debug util
CREATED:  2019-7-04 20:12:13
MODIFIED: 2019-7-24 14:04:45
"""
# disabling:
# E1101: no-member

import re
import numpy as np

from te.tvm._ffi.base import TVMError
from te.tik.common.util import TikUtil, DTYPE_SIZE, ceil_div, \
    get_bit_len, get_mask_len, check_scatter_dict_for_overlap
from te.tik.tik_lib.tik_params import ONE_BYTE_BIT_LEN, ONE_REP_BYTE_SIZE, \
    BLK_NUM_PER_REP, VA0_INDEX, ONE_BLK_SIZE, BIT_LEN_8, MASK_VALUE_128, \
    VSEL_MODE_TENSOR_SCALAR, VSEL_BLK_PARALLEL_BIT
from te.tik.tik_lib.tik_check_util import TikCheckUtil

_VEC_DATA_TYPE_ENCODING = {
    'uint8': 0b000, 'uint16': 0b001,
    'uint32': 0b010, 'int8': 0b011,
    'int32': 0b100, 'float16': 0b101,
    'fmix': 0b110, 'float32': 0b111,
    'int16': 0b001
}
_MIN_MASK = 1
_MAX_MASK_N = 129
_MAX_MASK_64 = 65
_MASK_VALUE_ZERO = 0
_MAX_MASK0_VALUE = 2**64 - 1
_MASK1_SHIFT_POS = 64
_MAX_COUNTER_MASK = 2**32
_BIT_LEN_16 = 16
_BIT_LEN_32 = 32


def tvm_value_to_int(value):
    """convert value to int

    Parameters
    ----------
    value : to be converted variable

    Returns
    -------
    value in int type
    """
    from te import tvm
    if isinstance(value, (tvm.expr.IntImm, tvm.expr.FloatImm,
                          tvm.expr.StringImm, tvm.expr.UIntImm)):
        return value.value
    if isinstance(value, (tvm.container.Array, list, tuple)):
        tmp = []
        for tmp_value in value:
            tmp.append(tvm_value_to_int(tmp_value))
        return tmp
    return TikCheckUtil.raise_error("convert value is illegal")


def get_const_shape(shape, context):
    """compute the shape

    Parameters
    ----------
    shape : to be computed
    context : context

    Returns
    -------
    the result
    """
    ret_shape = [context.evaluate_expr(s) for s in shape]
    return ret_shape


def get_flatten_idx(indice, context):
    """compute the index

    Parameters
    ----------
    indice : to be computed
    context : context

    Returns
    -------
    the result
    """
    indices = indice.indice
    const_shape = get_const_shape(indice.origin_shape, context)
    acc = 1
    idx = 0
    for index, shape_i in reversed(list(zip(indices, const_shape))):
        if isinstance(index, slice):
            this_dim_id = context.evaluate_expr(index.start)
        else:
            this_dim_id = index

        idx += this_dim_id*acc
        acc *= shape_i
    return context.evaluate_expr(idx)


def get_np_dtype_by_name(typename):
    """get the type of numpy by name

    Parameters
    ----------
    typename : numpy type name

    Returns
    -------
    np: numpy
    """
    return getattr(np, typename)


def cvt_float_to_uint(dtype, value):
    """convert float to uint"""
    np_dt = getattr(np, dtype)
    d_t = np.dtype(dtype)
    bit_size = d_t.itemsize*ONE_BYTE_BIT_LEN
    target_np = getattr(np, 'uint' + str(bit_size))
    return np_dt(value).view(target_np).item()


_TYPE_REGEX = re.compile('([a-z]*)([0-9]*)')


def get_dtype_bit_width(dtype):
    """get the dtype's bit width

    Returns
    -------
    bit_width: dtype bit width
    """
    _, bit_width = _TYPE_REGEX.search(dtype).groups()
    return bit_width


def reinterpret_type(src_type, dst_type, value):
    """convert value from src_type to dst_type"""
    if src_type == dst_type:
        return value
    _, src_bit_len = _TYPE_REGEX.search(src_type).groups()
    dst_type_name, _ = _TYPE_REGEX.search(dst_type).groups()
    tmp_type_name = dst_type_name + src_bit_len
    # reinterpret to same size than cast to type
    tmp_np_type = getattr(np, tmp_type_name)
    dst_np_type = getattr(np, dst_type)
    src_np_type = getattr(np, src_type)
    return src_np_type(value).view(tmp_np_type).astype(dst_np_type).item()


def get_dtype_size(dtype):
    """get the dtype's size

    Returns
    -------
    dtype size
    """
    return DTYPE_SIZE[dtype]


def copy_tensor_to_model(context, env, tensor, align=512, check_align=False,
                         require_xt=True, access_mode='r', offset=0):
    """copy tensor to pv model

    Parameters
    ----------
    context : class Context
    env: temp environment
    tensor: tik tensor
    align: align bit len
    check_align: bool
                 is check align
    require_xt: bool
                is require xt reg
    access_mode: buffer read or write mode
    offset: tensor offset

    Returns
    -------
    info of buffer
    """
    # pylint: disable=R0913
    return env.copy_tensor_to_model(context, tensor, align, check_align,
                                    require_xt, access_mode, offset)


def copy_tensor_to_model_get_addr(context, env,
                                  tensor, align, access_mode):
    """copy tensor to model to get address"""
    _, addr, _, _ = env.copy_tensor_to_model(context,
                                             tensor, align,
                                             True, False, access_mode)
    dtype_size_ = get_dtype_size(tensor.dtype)
    flatten_idx = get_flatten_idx(tensor.indice, context)
    addr += flatten_idx*dtype_size_
    return addr


def set_vector_mask(mask_n, context, mask_mode="normal",
                    tensor_bit_len=_BIT_LEN_16):
    """set vector mask"""
    model = context.model
    from ..tik_lib.tik_expr import BasicExpr
    if isinstance(mask_n, BasicExpr):
    # if not isinstance(mask_n, int):
        mask_n = context.evaluate_expr(mask_n)

    # mask counter mode
    if mask_mode == "counter":
        TikCheckUtil.check_in_range(
            mask_n, range(_MIN_MASK, _MAX_COUNTER_MASK),
            "In counter_mode, mask value should be in the range of "
            "[1, 2**32-1], input mask: {}".format(mask_n))
        # low 64 bit
        mask_0 = mask_n & _MAX_MASK0_VALUE
        # high 64 bit
        mask_1 = 0

        model.write_spr('MASK0', mask_0)
        model.write_spr('MASK1', mask_1)
    # mask normal mode
    else:
        if isinstance(mask_n, int):
            # check mask
            if tensor_bit_len == _BIT_LEN_16:
                TikCheckUtil.check_in_range(
                    mask_n, range(_MIN_MASK, _MAX_MASK_N),
                    "mask value should be in the range of [1, 128] for b16 "
                    "tensor, input mask: {}".format(mask_n))
            # b32, [1, 64]
            else:
                TikCheckUtil.check_in_range(
                    mask_n, range(_MIN_MASK, _MAX_MASK_64),
                    "mask value should be in the range of [1, 64] for b32 "
                    "tensor, input mask: {}".format(mask_n))
            mask_value = 2**mask_n - 1
            # low 64 bit
            mask_0 = mask_value & _MAX_MASK0_VALUE
            # high 64 bit
            mask_1 = mask_value >> _MASK1_SHIFT_POS

            model.write_spr('MASK0', mask_0)
            model.write_spr('MASK1', mask_1)
        elif isinstance(mask_n, (list, tuple)):
            TikCheckUtil.check_equality(len(mask_n), 2)
            mask_tmp = []
            is_mask_s64 = False
            for i, mask_value in enumerate(mask_n):
                if isinstance(mask_value, BasicExpr):
                # if not isinstance(mask_n, int):
                    if mask_value.dtype == 'int64':
                        is_mask_s64 = True
                    mask_value = context.evaluate_expr(mask_value)
                # when mask_value.dtype is s64, we don't check the range of
                # mask_value for using it to express [0, 2**64-1] in u64
                if not is_mask_s64:
                    TikCheckUtil.check_in_range(
                        mask_value, range(_MAX_MASK0_VALUE + 1),
                        "mask value should be in the range of [0, 2**64-1], "
                        "input mask: {}".format(mask_value))
                model.write_spr('MASK' + str(1 - i), mask_value)
                mask_tmp.append(mask_value)
            # mask can not be all zero, also check s64
            TikCheckUtil.check_not_equality(
                mask_tmp, [_MASK_VALUE_ZERO, _MASK_VALUE_ZERO],
                "mask list value can not be [0, 0]")
            # b32, mask_h should be 0
            if tensor_bit_len == _BIT_LEN_32:
                TikCheckUtil.check_equality(
                    mask_tmp[0], _MASK_VALUE_ZERO,
                    "mask_h should be 0 for b32 tensor, input mask_h: "
                    "{}".format(mask_tmp[0]))


def make_tvm_imm(type_name, value):
    """create the immediately number for tvm

    Parameters
    ----------
    type_name : data type
    value: data value

    Returns
    -------
    data value
    """
    try:
        if type_name.startswith('float'):
            from te.tvm.make import FloatImm
            return FloatImm(type_name, float(value))
        if type_name.startswith('int'):
            from te.tvm.make import IntImm
            return IntImm(type_name, int(value))
        if type_name.startswith('uint'):
            from te.tvm.make import UIntImm
            return UIntImm(type_name, int(value))
    except (TypeError, ValueError):
        return value
    return TikCheckUtil.raise_error(
        'unsupported dtype %s for value %s' % (type_name, str(value)))


def safe_get_value(var):
    """get value"""
    # pylint: disable=E1101
    # tvm.is_large_unit, tvm.large_unit_div and tvm.large_unit_rem are C++ API
    try:
        value = var.value
    except (AttributeError, TVMError):
        from te import tvm
        if isinstance(var, tvm.expr.UIntImm) and tvm.is_large_uint(var):
            half = tvm.large_uint_div(var)
            rem = tvm.large_uint_rem(var)
            # half is 1/2 of var, so multiple 2
            value = half*2 + rem
        else:
            TikCheckUtil.raise_error("can not get value safely!")
    return value


def _check_debug_scatter_mask_len(context, bit_len, mask, mask_mode="normal"):
    """check mask len for scatter address overlap"""
    if not isinstance(mask, (list, tuple)):
        mask = [mask]
    if len(mask) == 1 or mask_mode == "counter":
        mask_len = mask[0]
    else:
        mask_len, _ = get_mask_len(mask)

    mask_len = context.evaluate_expr(mask_len)
    if isinstance(mask_len, int):
        if mask_len % (ONE_BYTE_BIT_LEN*ONE_REP_BYTE_SIZE // bit_len) == 0:
            return True
        if bit_len == BIT_LEN_8 and mask_len % MASK_VALUE_128 == 0:
            return True
    return False


def _get_debug_block_begin_end(context, tensor, blk_len,
                               valid_num_per_block,
                               rep_stride, time, store_high_half=False):
    """get tensor a block's begin and end of each repeat"""
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    if blk_len != valid_num_per_block and store_high_half:
        begin = context.evaluate_expr(
            tensor.offset + time*rep_stride*blk_len)*DTYPE_SIZE[tensor.dtype]
        end = context.evaluate_expr(begin + blk_len*DTYPE_SIZE[tensor.dtype])
    else:
        begin = context.evaluate_expr(
            tensor.offset + time*rep_stride*blk_len)*DTYPE_SIZE[tensor.dtype]
        end = context.evaluate_expr(
            begin + valid_num_per_block*DTYPE_SIZE[tensor.dtype])

    return begin, end


def _get_debug_src_dst_buffer_dict(
        context, dst_list, src_list, dst_rep_stride,
        src_rep_stride, dst_dict, src_dict, valid_num_per_block, time=1,
        repeat_times=1, store_high_half=False, src_store_high_half=None):
    """get src_list, dst_list buffer dict info"""
    # pylint: disable=R0913, R0914
    # function's input params is too much, so disable them
    dst_blk_len = ONE_BLK_SIZE // DTYPE_SIZE[dst_list[VA0_INDEX].dtype]
    src_blk_len = ONE_BLK_SIZE // DTYPE_SIZE[src_list[VA0_INDEX].dtype]
    for dst in dst_list:
        begin, end = _get_debug_block_begin_end(
            context, dst, dst_blk_len, valid_num_per_block,
            dst_rep_stride, time, store_high_half)
        if isinstance(end, int):
            if dst.buffer not in dst_dict.keys():
                dst_dict[dst.buffer] = []
                dst_dict[dst.buffer].append([begin, end])
            else:
                dst_dict[dst.buffer].append([begin, end])
    if repeat_times > 1:
        time += 1
    for src in src_list:
        if src_store_high_half is not None:
            begin, end = _get_debug_block_begin_end(
                context, src, src_blk_len, valid_num_per_block,
                src_rep_stride, time, src_store_high_half)
        else:
            begin, end = _get_debug_block_begin_end(
                context, src, src_blk_len, valid_num_per_block,
                src_rep_stride, time, store_high_half)
        if isinstance(end, int):
            if src.buffer not in src_dict.keys():
                src_dict[src.buffer] = []
                src_dict[src.buffer].append([begin, end])
            else:
                src_dict[src.buffer].append([begin, end])
    return dst_dict, src_dict


def _get_debug_same_bffer_num(context, dst_list, src_list):
    """get number of same buffer"""
    same_buffer_num = 0
    for src in src_list:
        if context.evaluate_expr(src.offset) is not None:
            for dst in dst_list:
                if context.evaluate_expr(dst.offset) is not None:
                    if src.buffer == dst.buffer and \
                            context.evaluate_expr(src.offset) == \
                            context.evaluate_expr(dst.offset):
                        same_buffer_num += 1
    return same_buffer_num


def _check_debug_tensor_offset(context, dst_list, src_list):
    """get tensor's offset is None"""
    for dst, src in zip(dst_list, src_list):
        dst_offset = context.evaluate_expr(dst.offset)
        src_offset = context.evaluate_expr(src.offset)
        if dst_offset is None or src_offset is None:
            return False
    return True


def debug_check_scatter_overlap(
        context, mask, dst_list, src_list, repeat_times,
        dst_rep_stride, src_rep_stride,
        mask_mode="normal", store_high_half=False,
        src_store_high_half=None,
        name="scatter instr", msg="dst_list and src_list"):
    """check scatter instr address overlapping"""
    # pylint: disable=R0913, R0914
    # function's input params is too much, so disable them
    if not _check_debug_scatter_mask_len(
            context,
            max(get_bit_len(dst_list[VA0_INDEX].dtype),
                get_bit_len(src_list[VA0_INDEX].dtype)),
            mask, mask_mode):
        return
    dst_valid_num_per_block = ONE_REP_BYTE_SIZE // \
                              get_bit_len(dst_list[VA0_INDEX].dtype)
    src_valid_num_per_block = ONE_REP_BYTE_SIZE // \
                              get_bit_len(src_list[VA0_INDEX].dtype)
    valid_num_per_block = min(dst_valid_num_per_block, src_valid_num_per_block)
    if mask_mode == "counter":
        rep_len = valid_num_per_block*BLK_NUM_PER_REP
        repeat_times = ceil_div(mask, rep_len)

    repeat_times = context.evaluate_expr(repeat_times)
    if repeat_times is None:
        return
    # check offset
    if not _check_debug_tensor_offset(context, dst_list, src_list):
        return
    dst_dict = {}
    src_dict = {}
    if repeat_times == 1:
        if dst_rep_stride == src_rep_stride:
            # check 100% same
            same_buffer_num = _get_debug_same_bffer_num(context,
                                                        dst_list, src_list)
            if same_buffer_num == len(src_list)*len(dst_list):
                # each VA block same, allow overlap, check end. return
                return
        dst_dict, src_dict = _get_debug_src_dst_buffer_dict(
            context, dst_list, src_list, dst_rep_stride,
            src_rep_stride, dst_dict, src_dict,
            valid_num_per_block,
            store_high_half=store_high_half,
            src_store_high_half=src_store_high_half)
        # check dict for overlap
        check_scatter_dict_for_overlap(src_dict, dst_dict, name, msg)
    else:
        for time in range(repeat_times - 1):
            dst_dict = {}
            src_dict = {}
            dst_dict, src_dict = _get_debug_src_dst_buffer_dict(
                context, dst_list, src_list, dst_rep_stride,
                src_rep_stride, dst_dict, src_dict,
                valid_num_per_block, time, repeat_times,
                store_high_half=store_high_half,
                src_store_high_half=src_store_high_half)

            # check dict for overlap
            check_scatter_dict_for_overlap(src_dict, dst_dict, name, msg)


def _get_debug_vsel_sel_extent(sel, mode, time, bit_len):
    """get  vsel sel tensor extent"""
    if mode == VSEL_MODE_TENSOR_SCALAR:
        sel_tmp = sel[VA0_INDEX]
    else:
        sel_tmp = sel
    sel_bit_len = get_bit_len(sel_tmp.dtype)
    sel_num_each_time = VSEL_BLK_PARALLEL_BIT // bit_len // sel_bit_len
    begin = (time+1)*sel_num_each_time + sel_tmp.offset
    end = begin + sel_num_each_time
    return begin, end, sel_tmp


def _get_debug_vsel_dst_extent_list(context, dst_list, dst_rep_stride,
                                    valid_num_per_block, repeat_times, time=0):
    """get  vsel dst_list extent"""
    # pylint: disable=R0913
    # function's input params is too much, so disable them
    # elements of 1 blk
    dst_dict = {}
    dst_blk_len = ONE_BLK_SIZE // DTYPE_SIZE[dst_list[VA0_INDEX].dtype]
    if repeat_times == 1:
        for dst in dst_list:
            begin, end = _get_debug_block_begin_end(
                context, dst, dst_blk_len, valid_num_per_block,
                dst_rep_stride, repeat_times)
            if isinstance(end, int):
                if dst.buffer not in dst_dict.keys():
                    dst_dict[dst.buffer] = []
                    dst_dict[dst.buffer].append([begin, end])
                else:
                    dst_dict[dst.buffer].append([begin, end])
    else:
        for dst in dst_list:
            begin, end = _get_debug_block_begin_end(
                context, dst, dst_blk_len, valid_num_per_block,
                dst_rep_stride, time)
            if isinstance(end, int):
                if dst.buffer not in dst_dict.keys():
                    dst_dict[dst.buffer] = []
                    dst_dict[dst.buffer].append([begin, end])
                else:
                    dst_dict[dst.buffer].append([begin, end])
    return dst_dict


def _check_debug_sel_dst_begin_end(context,
                                   dst_dict, sel_begin, sel_end, sel_tmp):
    """check sel dst begin end overlap"""
    sel_begin = context.evaluate_expr(sel_begin)
    sel_end = context.evaluate_expr(sel_end)
    if isinstance(sel_end, int):
        for buffer in dst_dict.keys():
            if buffer == sel_tmp.buffer:
                for interval_dst in dst_dict[buffer]:
                    if max(sel_begin, interval_dst[0]) < \
                            min(sel_end, interval_dst[1]):
                        TikCheckUtil.raise_error(
                            "scater_vsel dst_list and sel"
                            " not support address overlapping")


def check_db_vsel_dst_sel_overlap(context, dst_list, sel, src0_list,
                                  mask, mode, repeat_times, dst_rep_stride):
    """check vsel dst_list and sel address overlap"""
    # pylint: disable=R0913, R0914
    # function's input params is too much, so disable them
    bit_len = get_bit_len(max(dst_list[VA0_INDEX].dtype,
                              src0_list[VA0_INDEX].dtype))
    mask_list = TikUtil.to_list(mask)
    if not isinstance(repeat_times, int):
        return
    # continous
    if len(mask_list) == 1:
        mask_len = mask_list[0]
    # decrete
    else:
        mask_len, _ = get_mask_len(mask_list)

    if isinstance(mask_len, int):
        if mask_len % (ONE_BYTE_BIT_LEN*ONE_REP_BYTE_SIZE // bit_len) != 0:
            return

    if repeat_times == 1:
        sel_begin, sel_end, sel_tmp = _get_debug_vsel_sel_extent(
            sel, mode, repeat_times-1, bit_len)
        dst_dict = _get_debug_vsel_dst_extent_list(
            context, dst_list, dst_rep_stride,
            ONE_REP_BYTE_SIZE // bit_len, repeat_times)
        _check_debug_sel_dst_begin_end(
            context, dst_dict, sel_begin, sel_end, sel_tmp)
    else:
        for time in range(repeat_times - 1):
            sel_begin, sel_end, sel_tmp = _get_debug_vsel_sel_extent(
                sel, mode, time, bit_len)
            dst_dict = _get_debug_vsel_dst_extent_list(
                context, dst_list, dst_rep_stride,
                ONE_REP_BYTE_SIZE // bit_len, repeat_times, time)
            _check_debug_sel_dst_begin_end(
                context, dst_dict, sel_begin, sel_end, sel_tmp)
