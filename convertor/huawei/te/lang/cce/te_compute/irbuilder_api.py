#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Ir Build kernel Api
"""
from __future__ import absolute_import

from te import tvm


# the max value int 64bit, 2**64 - 1
MAX_VALUE_UINT64 = 18446744073709551615

def get_loopnum_and_masklist(data_len, align_len):
    """get_loopnum_and_masklist

    Parameters
    ----------
    data_len: int
        data_len
    align_len: int
        align_len

    Returns
    -------
    resut_list: list
        [loop num, tail flag, [align_mask0, align_mask1]]
    """
    mask0 = 0
    mask1 = 0
    is_need_mask = 0
    loop = data_len // align_len
    _tmp = data_len % align_len
    if _tmp != 0:
        is_need_mask = 1
        if _tmp < 64:
            for _ in range(_tmp):
                mask0 = mask0*2 + 1
            mask1 = 0
        else:
            mask0 = int((MAX_VALUE_UINT64 // 2)*2 + 1)
            for _ in range(_tmp - 64):
                mask1 = mask1*2 + 1
    align_mask0 = tvm.const(mask0, "uint64")
    align_mask1 = tvm.const(mask1, "uint64")

    return loop, is_need_mask, [align_mask0, align_mask1]


def ib_new_alloc(ir_builder, dtype, shape, name, scope):
    """decl new buffer

    Parameters
    ----------
    ir_builder : tvm.ir_builder
        Developer API of IR node builder make function.
    dtype : string
        buffer date type.
    shape : list of int
        buffer shape.
    name : string
        buffer name.
    scope : string
        buffer memory scope.
    Returns
    -------
    buffer : tvm.schedule.Buffer
        Symbolic data buffer.
    """
    buf_var = ir_builder.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)

    return new_buffer


def kernel_two_to_one_common_fuc(_ib, addr_info_list, data_info_list, fuc_type):
    """kernel for to do 2 input and one out

    Parameters
    ----------
    _ib: tvm.ir_builder.IRBuilder
        Developer API of IR node builder make function.
    addr_info_list: list
        the addr info for input and output,
        [[out_ub,out_offset],[input1_ub, input1_offset],[input1_ub, input1_offset]]
    data_info_list: list
        [data_len, data_align],data_align = vector align length
    fuc_type: str
        cce cmd str, ex:vmul/vadd/vsub

    Returns
    -------
    None
    """
    _data_len, _align_len = data_info_list
    _addr_list = [addr_list[0] for addr_list in addr_info_list]
    _offset_list = [offset_list[1] for offset_list in addr_info_list]

    mask_paras = get_loopnum_and_masklist(_data_len, _align_len)
    if mask_paras[0] != 0:
        _ib.emit(tvm.call_extern(_addr_list[0].dtype, fuc_type,
                                 _addr_list[0].access_ptr("rw",
                                                          offset=_offset_list[0]),
                                 _addr_list[1].access_ptr("r",
                                                          offset=_offset_list[1]),
                                 _addr_list[2].access_ptr("r",
                                                          offset=_offset_list[2]),
                                 mask_paras[0], 1, 1, 1, 8, 8, 8))
    if mask_paras[1] == 1:
        _offset_list_mask = [offset + _align_len*mask_paras[0] for offset in _offset_list]
        _ib.emit(tvm.call_extern(
            "uint64", 'set_vector_mask', mask_paras[2][1], mask_paras[2][0]))
        _ib.emit(tvm.call_extern(_addr_list[0].dtype, fuc_type,
                                 _addr_list[0].access_ptr("rw", offset=_offset_list_mask[0]),
                                 _addr_list[1].access_ptr("r", offset=_offset_list_mask[1]),
                                 _addr_list[2].access_ptr("r", offset=_offset_list_mask[2]),
                                 1, 1, 1, 1, 8, 8, 8))
        _ib.emit(tvm.call_extern(
            "uint64", 'set_vector_mask', tvm.const(-1, "uint64"), tvm.const(-1, "uint64")))


def kernel_one_to_one_common_fuc(_ib, addr_info_list, data_info_list, fuc_type):
    """kernel for to do 1 input and one out

    Parameters
    ----------
    _ib: tvm.ir_builder.IRBuilder
        Developer API of IR node builder make function.
    addr_info_list: list
        the addr info for input and output,
        [[out_ub,out_offset],[input1_ub, input1_offset]]
    data_info_list: list
        [data_len, data_align],data_align = vector align length
    fuc_type: str
        cce cmd str, ex:"vrec"

    Returns
    -------
    None
    """
    _data_len, _align_len = data_info_list
    _addr_list = [addr_list[0] for addr_list in addr_info_list]
    _offset_list = [offset_list[1] for offset_list in addr_info_list]

    mask_paras = get_loopnum_and_masklist(_data_len, _align_len)
    if mask_paras[0] != 0:
        _ib.emit(tvm.call_extern(_addr_list[0].dtype, fuc_type,
                                 _addr_list[0].access_ptr("rw",
                                                          offset=_offset_list[0]),
                                 _addr_list[1].access_ptr("r",
                                                          offset=_offset_list[1]),
                                 mask_paras[0], 1, 1, 8, 8))
    if mask_paras[1] == 1:
        _offset_list_mask = [offset + _align_len*mask_paras[0] for offset in _offset_list]
        _ib.emit(tvm.call_extern(
            "uint64", 'set_vector_mask', mask_paras[2][1], mask_paras[2][0]))
        _ib.emit(tvm.call_extern(_addr_list[0].dtype, fuc_type,
                                 _addr_list[0].access_ptr("rw", offset=_offset_list_mask[0]),
                                 _addr_list[1].access_ptr("r", offset=_offset_list_mask[1]),
                                 1, 1, 1, 8, 8))
        _ib.emit(tvm.call_extern(
            "uint64", 'set_vector_mask', tvm.const(-1, "uint64"), tvm.const(-1, "uint64")))


def kernel_scalar_to_one_fuc(_ib, addr_info_list, data_info_list, fuc_info_list):
    """kernel for to do 1 input and 1 scaler and one out

    Parameters
    ----------
    _ib: tvm.ir_builder.IRBuilder
        Developer API of IR node builder make function.
    addr_info_list: list
        the addr info for input and output,
        [[out_ub,out_offset],[input1_ub, input1_offset]]
    data_info_list: list
        [data_len, data_align],data_align = vector align length
    fuc_info_list: list
        [cce cmd str,scalar], ex:[vadds, 1]

    Returns
    -------
    None
    """
    _data_len, _align_len = data_info_list
    _addr_list = [addr_list[0] for addr_list in addr_info_list]
    _offset_list = [offset_list[1] for offset_list in addr_info_list]
    _op_fuc, _op_value = fuc_info_list

    if isinstance(_op_value, (int, float)):
        _value = tvm.const(_op_value, _addr_list[0].dtype)
    else:
        _value = _op_value
    mask_paras = get_loopnum_and_masklist(_data_len, _align_len)
    if mask_paras[0] != 0:
        _ib.emit(tvm.call_extern(_addr_list[0].dtype, _op_fuc,
                                 _addr_list[0].access_ptr("rw",
                                                          offset=_offset_list[0]),
                                 _addr_list[1].access_ptr("r",
                                                          offset=_offset_list[1]),
                                 _value,
                                 mask_paras[0], 1, 1, 8, 8))
    if mask_paras[1] == 1:
        _offset_list_mask = [offset + _align_len*mask_paras[0] for offset in _offset_list]
        _ib.emit(tvm.call_extern(
            "uint64", 'set_vector_mask', mask_paras[2][1], mask_paras[2][0]))
        _ib.emit(tvm.call_extern(_addr_list[0].dtype, _op_fuc,
                                 _addr_list[0].access_ptr("rw", offset=_offset_list_mask[0]),
                                 _addr_list[1].access_ptr("r", offset=_offset_list_mask[1]),
                                 _value,
                                 1, 1, 1, 8, 8))
        _ib.emit(tvm.call_extern(
            "uint64", 'set_vector_mask', tvm.const(-1, "uint64"), tvm.const(-1, "uint64")))


def kernel_cp_fuc(_ib, addr_info_list, data_info_list, fuc_type):
    """kernel for cce fun copy_xx_to_xx

    Parameters
    ----------
    _ib: tvm.ir_builder.IRBuilder
        Developer API of IR node builder make function.
    addr_info_list: list
        the addr info for input and output,
        [[out_ub,out_offset],[input_ub, input_offset]]
    data_info_list: list
        [data_len, data_align],data_align = one block align length
    fuc_type: str
        cce cmd str,scalar, ex:copy_xx_to_xx

    Returns
    -------
    None
    """
    _src_addr, _src_offset = addr_info_list[1]
    _des_addr, _des_offset = addr_info_list[0]
    _data_len, _align_len = data_info_list

    mask_paras = get_loopnum_and_masklist(_data_len, _align_len)
    if mask_paras[0] != -1:
        if fuc_type not in ("copy_gm_to_cbuf",):
            _ib.emit(tvm.call_extern(_des_addr.dtype, fuc_type,
                                     _des_addr.access_ptr("rw", offset=_des_offset),
                                     _src_addr.access_ptr("r", offset=_src_offset),
                                     0, 1, mask_paras[0] + mask_paras[1], 0, 0))
        else:
            _ib.emit(tvm.call_extern(_des_addr.dtype, fuc_type,
                                     _des_addr.access_ptr("rw", offset=_des_offset),
                                     _src_addr.access_ptr("r", offset=_src_offset),
                                     0, 1, mask_paras[0] + mask_paras[1], 0, 0, 0))

# pylint: disable=too-many-locals
def kernel_cast_to_fuc(_ib, addr_info_list, data_info_list, fuc_type):
    """kernel for cce fun f162s32

    Parameters
    ----------
    _ib: tvm.ir_builder.IRBuilder
        Developer API of IR node builder make function.
    addr_info_list: list
        the addr info for input and output,
        [[out_ub,out_offset],[input_ub, input_offset]]
    data_info_list: list
        [data_len, data_align],data_align = one block align length
    fuc_type: str
        cce cmd str,scalar, ex:vconv_f162s32r

    Returns
    -------
    None
    """
    _data_len, _align_len = data_info_list
    _addr_list = [addr_list[0] for addr_list in addr_info_list]
    _offset_list = [offset_list[1] for offset_list in addr_info_list]

    if fuc_type in ("vconv_u82f16", "vconv_s82f16",):
        stride0, stride1 = 4, 2
    elif fuc_type in ("vconv_f162f32", "vconv_f162s32r", "vconv_f162s32f"):
        stride0, stride1 = 8, 4
    elif fuc_type in ("vconv_deq", "vconv_f322f16",):
        stride0, stride1 = 4, 8
    elif fuc_type in ("vconv_f322s32r", "vconv_s322f32", "vconv_f322s32f"):
        stride0, stride1 = 8, 8

    mask_paras = get_loopnum_and_masklist(_data_len, _align_len)

    def _ib_fun(_offset_list, repeat_count, is_tail=False):
        if is_tail is False:
            repeat, des_stride, src_stride = repeat_count, stride0, stride1
        else:
            repeat, des_stride, src_stride = 1, 8, 8
        des_stride, src_stride = stride0, stride1
        if is_tail:
            _ib.emit(tvm.call_extern(
                "uint64", 'set_vector_mask', mask_paras[2][1], mask_paras[2][0]))
        _ib.emit(tvm.call_extern(_addr_list[0].dtype, fuc_type,
                                 _addr_list[0].access_ptr("rw", offset=_offset_list[0]),
                                 _addr_list[1].access_ptr("r", offset=_offset_list[1]),
                                 repeat, 1, 1, des_stride, src_stride))
        if is_tail:
            _ib.emit(tvm.call_extern(
                "uint64", 'set_vector_mask', tvm.const(-1, "uint64"), tvm.const(-1, "uint64")))

    if mask_paras[0] != 0:
        loop_round = (mask_paras[0] + 255 - 1) // 255
        loop_round_left = mask_paras[0] % 255
        with _ib.for_range(0, loop_round, name="loop_round") as loop_rd_i:
            _loop_offset_list = [offset + _align_len * 255 * loop_rd_i for
                                 offset in _offset_list]
            with _ib.if_scope(loop_rd_i != loop_round - 1):
                _ib_fun(_loop_offset_list, 255)
            with _ib.else_scope():
                _ib_fun(_loop_offset_list, loop_round_left)

    if mask_paras[1] == 1:
        _offset_list_mask = [offset + _align_len*mask_paras[0] for offset in _offset_list]
        _ib_fun(_offset_list_mask, 1, is_tail=True)


def kernel_vector_dup_fuc(_ib, des_info_list, dup_value, data_info_list):
    """broadcast one value to ub

    Parameters
    ----------
    _ib: tvm.ir_builder.IRBuilder
        Developer API of IR node builder make function.
    des_info_list: list
        the addr info for output,
        [[out_ub,out_offset]]
    dup_value: tvm.expr.Load or int or float
        the value will be broatcast
    data_info_list: list
        [data_len, data_align],data_align = 8 block align length

    Returns
    -------
    None
    """
    _des_addr, _des_offset = des_info_list
    _data_len, _align_len = data_info_list

    if isinstance(dup_value, (tvm.expr.Load,)):
        broadcase_value = dup_value
    else:
        broadcase_value = tvm.const(dup_value, _des_addr.dtype)

    mask_paras = get_loopnum_and_masklist(_data_len, _align_len)
    if mask_paras[0] != 0:
        _ib.emit(tvm.call_extern(_des_addr.dtype, 'vector_dup',
                                 _des_addr.access_ptr("rw", offset=_des_offset),
                                 broadcase_value, mask_paras[0],
                                 1, 1, 8, 8))
    if mask_paras[1] == 1:
        _ib.emit(tvm.call_extern(
            "uint64", 'set_vector_mask', mask_paras[2][1], mask_paras[2][0]))
        _ib.emit(tvm.call_extern(_des_addr.dtype, 'vector_dup',
                                 _des_addr.access_ptr("rw",
                                                      offset=_des_offset
                                                      + mask_paras[0]*_align_len),
                                 broadcase_value,
                                 1, 1, 1, 8, 8))
        _ib.emit(tvm.call_extern(
            "uint64", 'set_vector_mask', tvm.const(-1, "uint64"), tvm.const(-1, "uint64")))
