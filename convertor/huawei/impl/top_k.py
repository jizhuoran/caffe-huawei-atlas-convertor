#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=too-many-lines
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

topk
"""

from te import platform as tbe_platform
from te import tvm
from te.platform.cce_build import build_config

from topi.cce import util

SHAPE_SIZE_LIMIT = 1 << 30
FP16_MINIMUM = -65504


# pylint: disable=too-many-instance-attributes,too-many-public-methods
# pylint: disable=attribute-defined-outside-init,old-style-class
class GlobalVar:
    """GlobalVar Class Defination"""
    def __init__(self):
        """"
        __init__
        """
        self.data_gm = None
        self.data_gm_out = None
        self.indices_gm = None
        self.indices_gm_out = None
        self.indices_ub = None
        self.indices_out_fp16_ub = None
        self.indices_out_int32_ub = None
        self.data_tail_block_ub = None
        self.indices_tail_block_ub = None
        self.region_k_ub = None
        self.region_k2_ub = None
        self.data_ub = None
        self.region_ub = None
        self.region_sorted_ub = None
        self.reg_min_number = None
        self.reg_addr = None
        self.reg_addr_buffer = None

    def set_data_gm(self, data_gm):
        """"
        set_data_gm
        """
        self.data_gm = data_gm

    def get_data_gm(self):
        """"
        get_data_gm
        """
        return self.data_gm

    def set_data_gm_out(self, data_gm_out):
        """"
        set_data_gm_out
        """
        self.data_gm_out = data_gm_out

    def get_data_gm_out(self):
        """"
        data_gm_out
        """
        return self.data_gm_out

    def set_indices_gm(self, indices_gm):
        """"
        set_indices_gm
        """
        self.indices_gm = indices_gm

    def get_indices_gm(self):
        """"
        get_indices_gm
        """
        return self.indices_gm

    def set_indices_gm_out(self, indices_gm_out):
        """"
        set_indices_gm_out
        """
        self.indices_gm_out = indices_gm_out

    def get_indices_gm_out(self):
        """"
        get_indices_gm_out
        """
        return self.indices_gm_out

    def set_indices_ub(self, indices_ub):
        """"
        set_indices_ub
        """
        self.indices_ub = indices_ub

    def get_indices_ub(self):
        """"
        get_indices_ub
        """
        return self.indices_ub

    def set_indices_out_fp16_ub(self, indices_out_fp16_ub):
        """"
        set_indices_out_fp16_ub
        """
        self.indices_out_fp16_ub = indices_out_fp16_ub

    def get_indices_out_fp16_ub(self):
        """"
        get_indices_out_fp16_ub
        """
        return self.indices_out_fp16_ub

    def set_indices_out_int32_ub(self, indices_out_int32_ub):
        """"
        set_indices_out_int32_ub
        """
        self.indices_out_int32_ub = indices_out_int32_ub

    def get_indices_out_int32_ub(self):
        """"
        get_indices_out_int32_ub
        """
        return self.indices_out_int32_ub

    def set_data_tail_block_ub(self, data_tail_block_ub):
        """"
        set_data_tail_block_ub
        """
        self.data_tail_block_ub = data_tail_block_ub

    def get_data_tail_block_ub(self):
        """"
        get_data_tail_block_ub
        """
        return self.data_tail_block_ub

    def set_indices_tail_block_ub(self, indices_tail_block_ub):
        """"
        set_indices_tail_block_ub
        """
        self.indices_tail_block_ub = indices_tail_block_ub

    def get_indices_tail_block_ub(self):
        """"
        get_indices_tail_block_ub
        """
        return self.indices_tail_block_ub

    def set_region_k2_ub(self, region_k2_ub):
        """"
        set_region_k2_ub
        """
        self.region_k2_ub = region_k2_ub

    def get_region_k2_ub(self):
        """"
        get_region_k2_ub
        """
        return self.region_k2_ub

    def set_data_ub(self, data_ub):
        """"
        set_data_ub
        """
        self.data_ub = data_ub

    def get_data_ub(self):
        """"
        get_data_ub
        """
        return self.data_ub

    def set_region_ub(self, region_ub):
        """"
        set_region_ub
        """
        self.region_ub = region_ub

    def get_region_ub(self):
        """"
        get_region_ub
        """
        return self.region_ub

    def set_region_sorted_ub(self, region_sorted_ub):
        """"
        set_region_sorted_ub
        """
        self.region_sorted_ub = region_sorted_ub

    def get_region_sorted_ub(self):
        """"
        get_region_sorted_ub
        """
        return self.region_sorted_ub

    def set_region_k_ub(self, region_k_ub):
        """"
        set_region_k_ub
        """
        self.region_k_ub = region_k_ub

    def get_region_k_ub(self):
        """"
        get_region_k_ub
        """
        return self.region_k_ub

    def set_reg_min_number(self, reg_min_number):
        """"
        set_reg_min_number
        """
        self.reg_min_number = reg_min_number

    def get_reg_min_number(self):
        """"
        get_reg_min_number
        """
        return self.reg_min_number

    def set_reg_addr(self, reg_addr):
        """"
        set_reg_addr
        """
        self.reg_addr = reg_addr

    def get_reg_addr(self):
        """"
        get_reg_addr
        """
        return self.reg_addr

    def set_reg_addr_buffer(self, reg_addr_buffer):
        """"
        set_reg_addr_buffer
        """
        self.reg_addr_buffer = reg_addr_buffer

    def get_reg_addr_buffer(self):
        """"
        get_reg_addr_buffer
        """
        return self.reg_addr_buffer

    def set_indices_out_final_ub(self, indices_out_final_ub):
        """
        set_indices_out_final_ub
        """
        self.indices_out_final_ub = indices_out_final_ub

    def get_indices_out_final_ub(self):
        """
        get_indices_out_final_ub
        """
        return self.indices_out_final_ub

    def set_offset_ub(self, offset_ub):
        """
        set_offset_ub
        """
        self.offset_ub = offset_ub

    def get_offset_ub(self):
        """
        get_offset_ub
        """
        return self.offset_ub

    def set_offset_fp16_ub(self, offset_fp16_ub):
        """
        set_offset_fp16_ub
        """
        self.offset_fp16_ub = offset_fp16_ub

    def get_offset_fp16_ub(self):
        """
        get_offset_fp16_ub
        """
        return self.offset_fp16_ub

    def set_offset_int32_ub(self, offset_int32_ub):
        """
        set_offset_int32_ub
        """
        self.offset_int32_ub = offset_int32_ub

    def get_offset_int32_ub(self):
        """
        get_offset_int32_ub
        """
        return self.offset_int32_ub

    def set_offset_gm(self, offset_gm):
        """
        set_offset_gm
        """
        self.offset_gm = offset_gm

    def get_offset_gm(self):
        """
        get_offset_gm
        """
        return self.offset_gm


GLOBAL_VAR = GlobalVar()


def _new_alloc(tvm_ir, dtype, shape, name, scope):
    """
    alloc memory for decl new buffer
    """
    buf_var = tvm_ir.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = tvm.decl_buffer(shape,
                                 buf_var.dtype,
                                 name=name,
                                 scope=scope,
                                 data=buf_var)
    return new_buffer


def set_mask(length):
    """
    calculate MASK in cce

    Parameters
    ----------
    length : int
        calculate length

    Returns
    -------
    mask : tuple of int
        low and high bit of mask.
    """
    length = int(length)
    mask1 = 2**max(length - 64, 0) - 1
    mask2 = 2**min(length, 64) - 1
    return mask1, mask2


def set_mask_insn(tvm_ir, type_, bits=128):
    """
    set_mask_insn
    """
    mask1, mask2 = set_mask(bits)
    tvm_ir.emit(
        tvm.call_extern(type_, 'set_vector_mask',
                        tvm.const(mask1, dtype='uint64'),
                        tvm.const(mask2, dtype='uint64')))


# pylint: disable=too-many-arguments
def conv_fp162s32(tvm_ir, s32ub, s32ub_offset, fp16ub, fp16ub_offset, num):
    """
    fp16 to int32
    """
    repeat = (num) // 64
    remain = (num) % 64
    if repeat > 0:
        tvm_ir.emit(
            tvm.call_extern('int32', 'vconv_f162s32f',
                            s32ub.access_ptr('w', offset=s32ub_offset),
                            fp16ub.access_ptr('r', offset=fp16ub_offset),
                            repeat, 1, 1, 8, 4))
    if remain > 0:
        set_mask_insn(tvm_ir, 'int32', remain)
        tvm_ir.emit(
            tvm.call_extern(
                'int32', 'vconv_f162s32f',
                s32ub.access_ptr('w', offset=s32ub_offset + repeat * 64),
                fp16ub.access_ptr('r', offset=fp16ub_offset + repeat * 64), 1,
                1, 1, 8, 4))
        set_mask_insn(tvm_ir, 'int32', 128)


# pylint: disable=too-many-arguments
def emit_copy_gm_to_ubuf(tvm_ir,
                         dtype,
                         dst,
                         src,
                         nburst,
                         burstlen,
                         srcstride,
                         dststride,
                         dst_offset=0,
                         src_offset=0):
    """
    emit_copy_gm_to_ubuf
    """
    tvm_ir.emit(
        tvm.call_extern(dtype, 'copy_gm_to_ubuf',
                        dst.access_ptr('w', offset=dst_offset),
                        src.access_ptr('r', offset=src_offset), 0, nburst,
                        burstlen, srcstride, dststride))


# pylint: disable=too-many-arguments
def emit_copy_ubuf_to_gm(tvm_ir,
                         dtype,
                         dst,
                         src,
                         nburst,
                         burstlen,
                         srcstride,
                         dststride,
                         dst_offset=0,
                         src_offset=0):
    """
    emit_copy_ubuf_to_gm
    """
    tvm_ir.emit(
        tvm.call_extern(dtype, 'copy_ubuf_to_gm',
                        dst.access_ptr('w', offset=dst_offset),
                        src.access_ptr('r', offset=src_offset), 0, nburst,
                        burstlen, srcstride, dststride))


def emit_vmuls(tvm_ir, dst, src, cnt):
    """
    emit_vmuls
    """
    repeat_255 = cnt // 128
    repeat_remain = cnt % 128
    times = (repeat_255 + 254) // 255
    if repeat_255 > 0:
        with tvm_ir.for_range(0, times, name='vmuls_i0') as i:
            times_len = tvm.min(repeat_255 - i * 255, 255)
            tvm_ir.emit(
                tvm.call_extern('float16', 'vmuls', dst.access_ptr('w',
                                offset=i * 128 * 255),
                                src.access_ptr('r', offset=i * 128 * 255),
                                tvm.const(-1), times_len, 1, 1, 8, 8))
    if repeat_remain > 0:
        set_mask_insn(tvm_ir, 'int32', repeat_remain)
        tvm_ir.emit(
            tvm.call_extern('float16', 'vmuls', dst.access_ptr('w',
                            offset=repeat_255 * 128),
                            src.access_ptr('r', offset=repeat_255 * 128),
                            tvm.const(-1), 1, 1, 1, 8, 8))
        set_mask_insn(tvm_ir, 'int32', 128)


# pylint: disable=too-many-arguments
def emit_vconcat(tvm_ir, dst, src, mode, cnt, dst_offset=0, src_offset=0):
    """
    emit_vconcat
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16
    with tvm_ir.if_scope(repeat_255 > 0):
        with tvm_ir.for_range(0, repeat_255, name='vconcat_i0') as i:
            config = tvm.const((255 << 56) + (mode << 16), dtype='uint64')
            tvm_ir.emit(
                tvm.call_extern(
                    'float16', 'vconcat',
                    dst.access_ptr('w', offset=dst_offset + i * 255 * 16 * 8),
                    src.access_ptr('r', offset=src_offset + i * 255 * 16),
                    config))
    if repeat_remain > 0:
        config = tvm.const((repeat_remain << 56) + (mode << 16),
                           dtype='uint64')
        tvm_ir.emit(
            tvm.call_extern(
                'float16', 'vconcat',
                dst.access_ptr('w',
                               offset=dst_offset + 255 * 16 * 8 * repeat_255),
                src.access_ptr('r', offset=src_offset + 255 * 16 * repeat_255),
                config))


# pylint: disable=too-many-arguments
def emit_vextract(tvm_ir, dst, src, mode, cnt, dst_offset=0, src_offset=0):
    """
    emit_vextract
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16
    if repeat_255 > 0:
        with tvm_ir.for_range(0, repeat_255, name='i0') as i:
            config = tvm.const((255 << 56) + (mode << 16), dtype='uint64')
            tvm_ir.emit(
                tvm.call_extern(
                    'float16', 'vextract',
                    dst.access_ptr('w', offset=dst_offset + i * 255 * 16),
                    src.access_ptr('r', offset=src_offset + i * 255 * 16 * 8),
                    config))
    if repeat_remain > 0:
        config = tvm.const((repeat_remain << 56) + (mode << 16),
                           dtype='uint64')
        tvm_ir.emit(
            tvm.call_extern(
                'float16', 'vextract',
                dst.access_ptr('w', offset=dst_offset + 255 * 16 * repeat_255),
                src.access_ptr('r',
                               offset=src_offset + 255 * 16 * 8 * repeat_255),
                config))


# pylint: disable=too-many-arguments
def emit_vbitsort(tvm_ir, dst, src, cnt, dst_offset=0, src_offset=0):
    """
    emit_vbitsort
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16
    if repeat_255 > 0:
        with tvm_ir.for_range(0, repeat_255, name='i0') as i:
            config = tvm.const((255 << 56), dtype='uint64')
            tvm_ir.emit(
                tvm.call_extern(
                    'float16', 'vbitsort',
                    dst.access_ptr('w', offset=dst_offset + i * 255 * 16 * 8),
                    src.access_ptr('r', offset=src_offset + i * 255 * 16 * 8),
                    config))
    if repeat_remain > 0:
        config = tvm.const((repeat_remain << 56), dtype='uint64')
        tvm_ir.emit(
            tvm.call_extern(
                'float16', 'vbitsort',
                dst.access_ptr('w',
                               offset=dst_offset + 255 * 16 * 8 * repeat_255),
                src.access_ptr('r',
                               offset=src_offset + 255 * 16 * 8 * repeat_255),
                config))


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements
def _merge_recur(tvm_ir,
                 src_ub,
                 dst_ub,
                 reg_addr,
                 reg_addr_buffer,
                 last_dim,
                 total_region_list,
                 level,
                 region_offset=0):
    """
    _merge_recur
    merge multi sorted region proposal list to one sorted region proposal list
    """
    #vmrgsort4 can merger at most 4 sorted region list
    loops = total_region_list // 4
    remain = total_region_list % 4

    merge_n0 = 16 * (4**(level - 1))
    merge_n1 = merge_n0
    merge_n2 = merge_n0
    merge_n3 = merge_n0
    merge_repeat = loops
    need_tail_process = False
    if loops > 0 and remain == 0:
        if merge_n0 * 4 * loops > last_dim:
            merge_repeat = loops - 1
            n012 = merge_n0 + merge_n1 + merge_n2
            merge_left = last_dim - ((merge_n0 * 4 * (loops - 1)) + n012)
            need_tail_process = True
    if merge_repeat > 0:
        config = tvm.const((15 << 60) | (merge_n0 << 8) | (merge_n1 << 20) |
                           (merge_n2 << 32) | (merge_n3 << 44) | merge_repeat,
                           dtype='uint64')
        reg_addr[0] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern('handle', '',
                            src_ub.access_ptr('r', offset=region_offset)))
        reg_addr[1] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r', offset=region_offset + merge_n0 * 8)))
        reg_addr[2] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r',
                                  offset=region_offset + merge_n0 * 8 +
                                  merge_n1 * 8)))
        reg_addr[3] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r',
                                  offset=region_offset + merge_n0 * 8 +
                                  merge_n1 * 8 + merge_n2 * 8)))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w'),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[0])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=1),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[1])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=2),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[2])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=3),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[3])))

        tvm_ir.emit(
            tvm.call_extern('float16', 'vmrgsort4',
                            dst_ub.access_ptr('w', offset=region_offset),
                            reg_addr_buffer.access_ptr('r'), config))

    if need_tail_process:
        tail_offset = 4 * merge_n0 * merge_repeat * 8
        config = tvm.const((15 << 60) | (merge_n0 << 8) | (merge_n1 << 20) |
                           (merge_n2 << 32) | (merge_left << 44) | 1,
                           dtype='uint64')
        reg_addr[0] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r', offset=region_offset + tail_offset)))
        reg_addr[1] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r',
                                  offset=region_offset + tail_offset +
                                  merge_n0 * 8)))
        addr2_offset = region_offset + tail_offset + merge_n0 * 8 + merge_n1 * 8
        reg_addr[2] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern('handle', '',
                            src_ub.access_ptr('r', offset=addr2_offset)))
        addr3_offset = region_offset + tail_offset
        addr3_offset = addr3_offset + merge_n0 * 8 + merge_n1 * 8 + merge_n2 * 8
        reg_addr[3] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern('handle', '',
                            src_ub.access_ptr('r', offset=addr3_offset)))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w'),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[0])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=1),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[1])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=2),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[2])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=3),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[3])))

        tvm_ir.emit(
            tvm.call_extern(
                'float16', 'vmrgsort4',
                dst_ub.access_ptr('w', offset=region_offset + tail_offset),
                reg_addr_buffer.access_ptr('r'), config))

    if loops > 0:
        offset = 4 * loops * 16 * (4**(level - 1))
    else:
        offset = 0

    if remain == 3:
        merge_n0 = 16 * (4**(level - 1))
        merge_n1 = merge_n0
        merge_n2 = last_dim - (offset + merge_n0 + merge_n1)
        config = tvm.const((7 << 60) | (merge_n0 << 8) | (merge_n1 << 20) |
                           (merge_n2 << 32) | 1,
                           dtype='uint64')
        reg_addr[0] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r', offset=region_offset + offset * 8)))
        reg_addr[1] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r',
                                  offset=region_offset + offset * 8 +
                                  merge_n0 * 8)))
        reg_addr[2] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r',
                                  offset=region_offset + offset * 8 +
                                  merge_n0 * 8 + merge_n1 * 8)))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w'),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[0])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=1),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[1])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=2),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[2])))

        tvm_ir.emit(
            tvm.call_extern(
                'float16', 'vmrgsort4',
                dst_ub.access_ptr('w', offset=region_offset + offset * 8),
                reg_addr_buffer.access_ptr('r'), config))
    elif remain == 2:
        merge_n0 = 16 * (4**(level - 1))
        merge_n1 = last_dim - (offset + merge_n0)
        config = tvm.const((3 << 60) | (merge_n0 << 8) | (merge_n1 << 20) | 1,
                           dtype='uint64')
        reg_addr[0] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r', offset=region_offset + offset * 8)))
        reg_addr[1] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern(
                'handle', '',
                src_ub.access_ptr('r',
                                  offset=region_offset + offset * 8 +
                                  merge_n0 * 8)))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w'),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[0])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=1),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[1])))
        tvm_ir.emit(
            tvm.call_extern(
                'float16', 'vmrgsort4',
                dst_ub.access_ptr('w', offset=region_offset + offset * 8),
                reg_addr_buffer.access_ptr('r'), config))
    elif remain == 1:
        merge_n0 = last_dim - offset
        num_blocks_write = (merge_n0 * 16 + 31) // 32
        tvm_ir.emit(
            tvm.call_extern(
                'float16', 'copy_ubuf_to_ubuf',
                dst_ub.access_ptr('w', offset=region_offset + offset * 8),
                src_ub.access_ptr('r', offset=region_offset + offset * 8), 0,
                1, num_blocks_write, 0, 0))

    next_total_region_list = (total_region_list + 3) // 4

    if next_total_region_list <= 1:
        return dst_ub
    return _merge_recur(tvm_ir, dst_ub, src_ub, reg_addr, reg_addr_buffer,
                        last_dim, next_total_region_list, level + 1,
                        region_offset)


def merge_region(tvm_ir, dst, src, rows, cols):
    """
    merge_region
    """
    reg_addr = GLOBAL_VAR.get_reg_addr()
    reg_addr_buffer = GLOBAL_VAR.get_reg_addr_buffer()
    cols_padding = ((cols + 15) // 16) * 16
    with tvm_ir.for_range(0, rows, name='merge_i0') as i:
        result_ub = _merge_recur(tvm_ir,
                                 src,
                                 dst,
                                 reg_addr,
                                 reg_addr_buffer,
                                 cols, (cols + 15) // 16,
                                 1,
                                 region_offset=i * cols_padding * 8)
    return result_ub


def sort_region(tvm_ir, dst, src, rows, cols):
    """
    sort_region
    """
    emit_vbitsort(tvm_ir, dst, src, cnt=rows * cols)
    if cols > 16:
        result_ub = merge_region(tvm_ir,
                                 dst=src,
                                 src=dst,
                                 rows=rows,
                                 cols=cols)
    else:
        result_ub = dst
    return result_ub


def copy_gm_to_ubuf(tvm_ir, dst, src, num_rows, cols, col_start, gm_offset):
    """
    copy_gm_to_ubuf copy data from gm to ubuf
    """
    cols_padding = ((cols + 15) // 16) * 16
    burstlen = (cols * 2 + 31) // 32
    with tvm_ir.for_range(0, num_rows, name='gm2ub_i0') as i:
        emit_copy_gm_to_ubuf(tvm_ir,
                             'float16',
                             dst,
                             src,
                             1,
                             burstlen,
                             0,
                             0,
                             dst_offset=cols_padding * i,
                             src_offset=cols * i + col_start + gm_offset)


def copy_gm_to_ubuf_func(tvm_ir, dst, src, num_rows, cols, col_start,
                         gm_offset, largest):
    """
    copy_gm_to_ubuf copy data from gm to ubuf
    """
    cols_padding = ((cols + 15) // 16) * 16
    burstlen = (cols * 2 + 31) // 32
    reg_min_number = GLOBAL_VAR.get_reg_min_number()
    with tvm_ir.for_range(0, num_rows, name='gm2ub_i0') as i:
        emit_copy_gm_to_ubuf(tvm_ir,
                             'float16',
                             dst,
                             src,
                             1,
                             burstlen,
                             0,
                             0,
                             dst_offset=cols_padding * i,
                             src_offset=cols * i + col_start + gm_offset)
    if not largest:
        emit_vmuls(tvm_ir, dst, dst, cnt=num_rows * cols_padding)
    with tvm_ir.for_range(0, num_rows, name='gm2ub_i0') as i:
        for j in range(cols_padding - cols):
            tvm_ir.emit(
                tvm.call_extern(
                    'float16', 'reg_mov',
                    dst.access_ptr('w', offset=cols_padding * i + cols + j),
                    tvm.call_extern('float16', 'reg', reg_min_number[0])))


def copy_ubuf_to_gm(tvm_ir,
                    dtype,
                    dst,
                    src,
                    num_rows,
                    cols_padding,
                    k,
                    tail_block_ub,
                    gm_offset=0,
                    multi_core=False):
    """
    copy_ubuf_to_gm
    """
    if dtype == 'float16':
        burstlen = (k * 2 + 31) // 32
        blocklen = 16
    elif dtype == 'int32':
        burstlen = (k * 4 + 31) // 32
        blocklen = 8

    with tvm_ir.for_range(0, num_rows - 1, name='ub2gmi0') as i:
        emit_copy_ubuf_to_gm(tvm_ir,
                             dtype,
                             dst,
                             src,
                             1,
                             burstlen,
                             0,
                             0,
                             dst_offset=k * i + gm_offset,
                             src_offset=cols_padding * i)
    if multi_core and k > 16:
        emit_copy_ubuf_to_gm(tvm_ir,
                             dtype,
                             dst,
                             src,
                             1,
                             burstlen - 1,
                             0,
                             0,
                             dst_offset=k * (num_rows - 1) + gm_offset,
                             src_offset=cols_padding * (num_rows - 1))
        for i in range(blocklen):
            tvm_ir.emit(
                tvm.call_extern(
                    dtype, 'reg_mov', tail_block_ub.access_ptr('w', offset=i),
                    src.access_ptr('r',
                                   offset=cols_padding * (num_rows - 1) + k -
                                   blocklen + i)))
        emit_copy_ubuf_to_gm(tvm_ir,
                             dtype,
                             dst,
                             tail_block_ub,
                             1,
                             1,
                             0,
                             0,
                             dst_offset=k * (num_rows - 1) + gm_offset + k -
                             blocklen,
                             src_offset=0)

    else:
        emit_copy_ubuf_to_gm(tvm_ir,
                             dtype,
                             dst,
                             src,
                             1,
                             burstlen,
                             0,
                             0,
                             dst_offset=k * (num_rows - 1) + gm_offset,
                             src_offset=cols_padding * (num_rows - 1))


def merge_two_sorted_region(tvm_ir, dst, src_region_k, src_region_sorted,
                            len_region_k, len_region_sorted):
    """
    merge_two_sorted_region
    """
    reg_addr = GLOBAL_VAR.get_reg_addr()
    reg_addr_buffer = GLOBAL_VAR.get_reg_addr_buffer()
    if len_region_k < 4096:
        merge_n0 = len_region_k
        merge_n1 = len_region_sorted
        config = tvm.const((3 << 60) | (merge_n0 << 8) | (merge_n1 << 20) | 1,
                           dtype='uint64')
        reg_addr[0] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern('handle', '', src_region_k.access_ptr('r',
                                                                  offset=0)))
        reg_addr[1] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern('handle', '',
                            src_region_sorted.access_ptr('r', offset=0)))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w'),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[0])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=1),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[1])))
        tvm_ir.emit(
            tvm.call_extern('float16', 'vmrgsort4',
                            dst.access_ptr('w', offset=0),
                            reg_addr_buffer.access_ptr('r'), config))

    elif len_region_k >= 4096:
        merge_n0 = 2048
        merge_n1 = 2048
        merge_n2 = len_region_sorted
        config = tvm.const((7 << 60) | (merge_n0 << 8) | (merge_n1 << 20) |
                           (merge_n2 << 32) | 1,
                           dtype='uint64')
        reg_addr[0] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern('handle', '', src_region_k.access_ptr('r')))
        reg_addr[1] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern('handle', '',
                            src_region_k.access_ptr('r', offset=(2048) * 8)))
        reg_addr[2] = tvm.expr.Cast(
            'uint64',
            tvm.call_extern('handle', '', src_region_sorted.access_ptr('r')))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w'),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[0])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=1),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[1])))
        tvm_ir.emit(
            tvm.call_extern(
                'uint64', 'reg_mov', reg_addr_buffer.access_ptr('w', offset=2),
                tvm.call_extern(reg_addr.dtype, 'reg', reg_addr[2])))
        tvm_ir.emit(
            tvm.call_extern('float16', 'vmrgsort4', dst.access_ptr('w'),
                            reg_addr_buffer.access_ptr('r'), config))


def copy_region(tvm_ir, dst, src, num, dst_offset=0):
    """
    copy_region
    """
    burstlen = (num * 2 * 8 + 31) // 32
    tvm_ir.emit(
        tvm.call_extern('float16', 'copy_ubuf_to_ubuf',
                        dst.access_ptr('w', offset=dst_offset),
                        src.access_ptr('r'), 0, 1, burstlen, 0, 0))


def _add(tvm_ir, dst, src1, src2, rows, cols_padding):
    # process 256B data per repeat for vsub
    vadd_len = 64
    repeat = (rows * cols_padding) // vadd_len
    remain = (rows * cols_padding) % vadd_len
    if repeat > 0:
        tvm_ir.emit(
            tvm.call_extern('int32', 'vadd', dst.access_ptr('w'),
                            src1.access_ptr('r'), src2.access_ptr('r'), repeat,
                            1, 1, 1, 8, 8, 8))

    if remain > 0:
        set_mask_insn(tvm_ir, 'int32', remain)
        tvm_ir.emit(
            tvm.call_extern('int32', 'vadd',
                            dst.access_ptr('w', offset=repeat * vadd_len),
                            src1.access_ptr('r', offset=repeat * vadd_len),
                            src2.access_ptr('r', offset=repeat * vadd_len), 1,
                            1, 1, 1, 8, 8, 8))
        set_mask_insn(tvm_ir, 'int32', 128)


def topk_rows(tvm_ir, row_start_in_core, rows, cols, k, core_rows_start,
              multi_core, largest):
    """
    topk_rows do topk action muilti rows
    """
    data_gm = GLOBAL_VAR.get_data_gm()
    data_gm_out = GLOBAL_VAR.get_data_gm_out()
    indices_gm = GLOBAL_VAR.get_indices_gm()
    indices_gm_out = GLOBAL_VAR.get_indices_gm_out()
    indices_ub = GLOBAL_VAR.get_indices_ub()
    indices_out_fp16_ub = GLOBAL_VAR.get_indices_out_fp16_ub()
    indices_out_int32_ub = GLOBAL_VAR.get_indices_out_int32_ub()
    data_ub = GLOBAL_VAR.get_data_ub()
    region_ub = GLOBAL_VAR.get_region_ub()
    region_sorted_ub = GLOBAL_VAR.get_region_sorted_ub()
    data_tail_block_ub = GLOBAL_VAR.get_data_tail_block_ub()
    indices_tail_block_ub = GLOBAL_VAR.get_indices_tail_block_ub()
    offset_ub = GLOBAL_VAR.get_offset_ub()
    offset_fp16_ub = GLOBAL_VAR.get_offset_fp16_ub()
    offset_int32_ub = GLOBAL_VAR.get_offset_int32_ub()
    indices_out_final_ub = GLOBAL_VAR.get_indices_out_final_ub()
    cols_padding = ((cols + 15) // 16) * 16
    copy_gm_to_ubuf_func(tvm_ir,
                         data_ub,
                         data_gm,
                         num_rows=rows,
                         cols=cols,
                         col_start=0,
                         gm_offset=row_start_in_core * cols +
                         core_rows_start * cols, largest=largest)
    copy_gm_to_ubuf(tvm_ir,
                    indices_ub,
                    indices_gm,
                    num_rows=1,
                    cols=cols,
                    col_start=0,
                    gm_offset=0)
    copy_gm_to_ubuf(tvm_ir,
                    offset_ub,
                    indices_gm,
                    num_rows=1,
                    cols=cols,
                    col_start=cols,
                    gm_offset=0)
    emit_vconcat(tvm_ir, region_ub, data_ub, mode=4, cnt=rows * cols_padding)
    with tvm_ir.for_range(0, rows, name='i0') as i:
        emit_vconcat(tvm_ir,
                     region_ub,
                     indices_ub,
                     mode=1,
                     cnt=cols_padding,
                     dst_offset=i * cols_padding * 8,
                     src_offset=0)
        emit_vconcat(tvm_ir,
                     region_ub,
                     offset_ub,
                     mode=2,
                     cnt=cols_padding,
                     dst_offset=i * cols_padding * 8,
                     src_offset=0)
    result_ub = sort_region(tvm_ir, region_sorted_ub, region_ub, rows,
                            cols_padding)
    emit_vextract(tvm_ir, data_ub, result_ub, mode=4, cnt=rows * cols_padding)
    with tvm_ir.for_range(0, rows, name='i0') as i:
        emit_vextract(tvm_ir,
                      indices_out_fp16_ub,
                      result_ub,
                      mode=1,
                      cnt=cols_padding,
                      dst_offset=i * cols_padding,
                      src_offset=i * cols_padding * 8)
        emit_vextract(tvm_ir,
                      offset_fp16_ub,
                      result_ub,
                      mode=2,
                      cnt=cols_padding,
                      dst_offset=i * cols_padding,
                      src_offset=i * cols_padding * 8)
    if not largest:
        emit_vmuls(tvm_ir, data_ub, data_ub, cnt=rows * cols_padding)
    with tvm_ir.for_range(0, rows, name='i0') as i:
        conv_fp162s32(tvm_ir, indices_out_int32_ub, i * cols_padding,
                      indices_out_fp16_ub, i * cols_padding, cols_padding)
        conv_fp162s32(tvm_ir, offset_int32_ub, i * cols_padding,
                      offset_fp16_ub, i * cols_padding, cols_padding)
    _add(tvm_ir, indices_out_final_ub, indices_out_int32_ub, offset_int32_ub,
         rows, cols_padding)
    copy_ubuf_to_gm(tvm_ir,
                    'float16',
                    data_gm_out,
                    data_ub,
                    rows,
                    cols_padding,
                    k,
                    tail_block_ub=data_tail_block_ub,
                    gm_offset=row_start_in_core * k + core_rows_start * k,
                    multi_core=multi_core)
    copy_ubuf_to_gm(tvm_ir,
                    'int32',
                    indices_gm_out,
                    indices_out_final_ub,
                    rows,
                    cols_padding,
                    k,
                    tail_block_ub=indices_tail_block_ub,
                    gm_offset=row_start_in_core * k + core_rows_start * k,
                    multi_core=multi_core)


def topk_a_row_by_part(tvm_ir, row_start_in_core, cols, k, core_rows_start,
                       multi_core, largest):
    """
    topk_a_row_by_part
    """
    data_gm = GLOBAL_VAR.get_data_gm()
    data_ub = GLOBAL_VAR.get_data_ub()
    data_gm_out = GLOBAL_VAR.get_data_gm_out()
    indices_gm = GLOBAL_VAR.get_indices_gm()
    indices_gm_out = GLOBAL_VAR.get_indices_gm_out()
    indices_ub = GLOBAL_VAR.get_indices_ub()
    indices_out_int32_ub = GLOBAL_VAR.get_indices_out_int32_ub()
    region_ub = GLOBAL_VAR.get_region_ub()
    region_sorted_ub = GLOBAL_VAR.get_region_sorted_ub()
    region_k_ub = GLOBAL_VAR.get_region_k_ub()
    region_k2_ub = GLOBAL_VAR.get_region_k2_ub()
    data_tail_block_ub = GLOBAL_VAR.get_data_tail_block_ub()
    indices_tail_block_ub = GLOBAL_VAR.get_indices_tail_block_ub()
    #last dim is two long, so by partition, each part process 1024 elements

    offset_ub = GLOBAL_VAR.get_offset_ub()
    offset_int32_ub = GLOBAL_VAR.get_offset_int32_ub()
    indices_out_final_ub = GLOBAL_VAR.get_indices_out_final_ub()

    cols_per_part = 1024
    k_padding = ((k + 15) // 16) * 16
    cols_padding = ((cols + 15) // 16) * 16
    part_cnt = (cols + cols_per_part - 1) // cols_per_part
    last_part_cols = cols - ((part_cnt - 1) * cols_per_part)
    last_part_cols_padding = ((last_part_cols + 15) // 16) * 16

    gm_offset = row_start_in_core * cols + core_rows_start * cols

    copy_gm_to_ubuf_func(tvm_ir,
                         data_ub,
                         data_gm,
                         num_rows=1,
                         cols=cols_per_part,
                         col_start=0,
                         gm_offset=gm_offset, largest=largest)
    copy_gm_to_ubuf(tvm_ir,
                    indices_ub,
                    indices_gm,
                    num_rows=1,
                    cols=cols_per_part,
                    col_start=0,
                    gm_offset=0)
    copy_gm_to_ubuf(tvm_ir,
                    offset_ub,
                    indices_gm,
                    num_rows=1,
                    cols=cols_per_part,
                    col_start=cols,
                    gm_offset=0)
    emit_vconcat(tvm_ir, region_ub, data_ub, mode=4, cnt=cols_per_part)
    emit_vconcat(tvm_ir, region_ub, indices_ub, mode=1, cnt=cols_per_part)
    emit_vconcat(tvm_ir, region_ub, offset_ub, mode=2, cnt=cols_per_part)
    result_ub = sort_region(tvm_ir, region_sorted_ub, region_ub, 1,
                            cols_per_part)
    copy_region(tvm_ir, dst=region_k_ub, src=result_ub, num=cols_per_part)

    with tvm_ir.for_range(0, part_cnt - 2, name='topk_i0') as i:
        copy_gm_to_ubuf_func(tvm_ir,
                             data_ub,
                             data_gm,
                             num_rows=1,
                             cols=cols_per_part,
                             col_start=cols_per_part * (i + 1),
                             gm_offset=gm_offset, largest=largest)
        copy_gm_to_ubuf(tvm_ir,
                        indices_ub,
                        indices_gm,
                        num_rows=1,
                        cols=cols_per_part,
                        col_start=cols_per_part * (i + 1),
                        gm_offset=0)
        copy_gm_to_ubuf(tvm_ir,
                        offset_ub,
                        indices_gm,
                        num_rows=1,
                        cols=cols_per_part,
                        col_start=cols + cols_per_part * (i + 1),
                        gm_offset=0)
        emit_vconcat(tvm_ir, region_ub, data_ub, mode=4, cnt=cols_per_part)
        emit_vconcat(tvm_ir, region_ub, indices_ub, mode=1, cnt=cols_per_part)
        emit_vconcat(tvm_ir, region_ub, offset_ub, mode=2, cnt=cols_per_part)
        result_ub = sort_region(tvm_ir, region_sorted_ub, region_ub, 1,
                                cols_per_part)

        with tvm_ir.if_scope(i == 0):
            merge_two_sorted_region(tvm_ir,
                                    dst=region_k2_ub,
                                    src_region_k=region_k_ub,
                                    src_region_sorted=result_ub,
                                    len_region_k=cols_per_part,
                                    len_region_sorted=cols_per_part)
            copy_region(tvm_ir,
                        dst=region_k_ub,
                        src=region_k2_ub,
                        num=cols_per_part * 2)
        with tvm_ir.if_scope(i == 1):
            merge_two_sorted_region(tvm_ir,
                                    dst=region_k2_ub,
                                    src_region_k=region_k_ub,
                                    src_region_sorted=result_ub,
                                    len_region_k=cols_per_part * 2,
                                    len_region_sorted=cols_per_part)
            copy_region(tvm_ir,
                        dst=region_k_ub,
                        src=region_k2_ub,
                        num=cols_per_part * 3)
        with tvm_ir.if_scope(i == 2):
            merge_two_sorted_region(tvm_ir,
                                    dst=region_k2_ub,
                                    src_region_k=region_k_ub,
                                    src_region_sorted=result_ub,
                                    len_region_k=cols_per_part * 3,
                                    len_region_sorted=cols_per_part)
            copy_region(tvm_ir,
                        dst=region_k_ub,
                        src=region_k2_ub,
                        num=cols_per_part * 4)
        with tvm_ir.if_scope(i >= 3):
            merge_two_sorted_region(tvm_ir,
                                    dst=region_k2_ub,
                                    src_region_k=region_k_ub,
                                    src_region_sorted=result_ub,
                                    len_region_k=cols_per_part * 4,
                                    len_region_sorted=cols_per_part)

            copy_region(tvm_ir,
                        dst=region_k_ub,
                        src=region_k2_ub,
                        num=cols_per_part * 5)

    copy_gm_to_ubuf_func(tvm_ir,
                         data_ub,
                         data_gm,
                         num_rows=1,
                         cols=last_part_cols,
                         col_start=(part_cnt - 1) * cols_per_part,
                         gm_offset=gm_offset, largest=largest)
    copy_gm_to_ubuf(tvm_ir,
                    indices_ub,
                    indices_gm,
                    num_rows=1,
                    cols=last_part_cols,
                    col_start=(part_cnt - 1) * cols_per_part,
                    gm_offset=0)
    copy_gm_to_ubuf(tvm_ir,
                    offset_ub,
                    indices_gm,
                    num_rows=1,
                    cols=last_part_cols,
                    col_start=cols + (part_cnt - 1) * cols_per_part,
                    gm_offset=0)
    emit_vconcat(tvm_ir,
                 region_ub,
                 data_ub,
                 mode=4,
                 cnt=last_part_cols_padding)
    emit_vconcat(tvm_ir,
                 region_ub,
                 indices_ub,
                 mode=1,
                 cnt=last_part_cols_padding)
    emit_vconcat(tvm_ir,
                 region_ub,
                 offset_ub,
                 mode=2,
                 cnt=last_part_cols_padding)
    result_ub = sort_region(tvm_ir, region_sorted_ub, region_ub, 1,
                            last_part_cols_padding)
    merge_two_sorted_region(tvm_ir,
                            dst=region_k2_ub,
                            src_region_k=region_k_ub,
                            src_region_sorted=result_ub,
                            len_region_k=4096,
                            len_region_sorted=last_part_cols_padding)

    emit_vextract(tvm_ir, region_k_ub, region_k2_ub, mode=4, cnt=k_padding)
    emit_vextract(tvm_ir,
                  region_k_ub,
                  region_k2_ub,
                  mode=1,
                  cnt=k_padding,
                  dst_offset=k_padding)
    emit_vextract(tvm_ir,
                  region_k_ub,
                  region_k2_ub,
                  mode=2,
                  cnt=k_padding,
                  dst_offset=k_padding * 2)
    if not largest:
        emit_vmuls(tvm_ir, region_k_ub, region_k_ub, cnt=k_padding)
    conv_fp162s32(tvm_ir, indices_out_int32_ub, 0, region_k_ub, k_padding,
                  k_padding)
    conv_fp162s32(tvm_ir, offset_int32_ub, 0, region_k_ub, k_padding * 2,
                  k_padding)

    _add(tvm_ir, indices_out_final_ub, indices_out_int32_ub, offset_int32_ub,
         1, k_padding)

    copy_ubuf_to_gm(tvm_ir,
                    'float16',
                    data_gm_out,
                    region_k_ub,
                    num_rows=1,
                    cols_padding=cols_padding,
                    k=k,
                    tail_block_ub=data_tail_block_ub,
                    gm_offset=row_start_in_core * k + core_rows_start * k,
                    multi_core=multi_core)
    copy_ubuf_to_gm(tvm_ir,
                    'int32',
                    indices_gm_out,
                    indices_out_final_ub,
                    1,
                    cols_padding,
                    k,
                    tail_block_ub=indices_tail_block_ub,
                    gm_offset=row_start_in_core * k + core_rows_start * k,
                    multi_core=multi_core)


def tiling(rows, cols):
    """
    Funcion for tiling
    """
    ret = []  # rows for each core

    #num_cores = cce_conf.CceProductParams().getParams("Device_core_num")
    num_cores = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    if rows < num_cores:
        for i in range(rows):
            ret.append(1)
        return ret, rows, 1
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    # 1.2. data and indices tail block 3. vmrgsort4 addr block
    ub_bytes = ub_size_bytes - 32 - 32 - 32
    cols_padding = ((cols + 15) // 16) * 16

    remain = rows % num_cores
    for i in range(num_cores):
        ret.append(rows // num_cores)
    for i in range(remain):
        ret[i] = ret[i] + 1

    if cols <= 4096:
        ub_bytes = ub_bytes - cols_padding * 4  # indices_ub
        row_bytes = cols_padding * 50
        batch = ub_bytes // row_bytes
    else:
        batch = 1
    turning = remain
    if remain == 0:
        turning = num_cores

    return ret, turning, batch


# pylint: disable=locally-disabled,too-many-locals
def _kernel_ir(ins, outs, k, largest):
    """
    Funtion for common process in top_k op
    """
    tvm_ir = tvm.ir_builder.create()
    input_a = ins[0]
    indices = ins[1]
    output = outs[0]
    indices_out = outs[1]
    shape = list(input_a.shape)
    cols = int(shape[-1])
    cols_padding = ((cols + 15) // 16) * 16
    rows = 1
    for i in range(len(shape) - 1):
        rows = rows * int(shape[i])
    multi_core = True
    rows_cores, turn, batch = tiling(rows, cols)
    if k < 16:
        rows_cores = [rows]
        turn = 1
        multi_core = False

    if len(rows_cores) <= 1:
        multi_core = False

    if cols > 4096:
        cols_per_part = 1024
        data_ub = _new_alloc(tvm_ir,
                             'float16', (cols_per_part, ),
                             name='data_ub',
                             scope=tbe_platform.scope_ubuf)
        indices_ub = _new_alloc(tvm_ir,
                                'float16', (cols_per_part, ),
                                name='indices_ub',
                                scope=tbe_platform.scope_ubuf)
        indices_out_fp16_ub = indices_ub
        indices_out_int32_ub = _new_alloc(tvm_ir,
                                          'int32', (1, 4096),
                                          name='indices_out_int32_ub',
                                          scope=tbe_platform.scope_ubuf)
        indices_out_final_ub = _new_alloc(tvm_ir,
                                          'int32', (1, 4096),
                                          name='indices_out_final_ub',
                                          scope=tbe_platform.scope_ubuf)
        offset_ub = _new_alloc(tvm_ir,
                               'float16', (cols_per_part, ),
                               name='offset_ub',
                               scope=tbe_platform.scope_ubuf)
        offset_fp16_ub = offset_ub
        offset_int32_ub = _new_alloc(tvm_ir,
                                     'int32', (1, 4096),
                                     name='offset_int32_ub',
                                     scope=tbe_platform.scope_ubuf)

        region_ub = _new_alloc(tvm_ir,
                               'float16', (1, 1024 * 8),
                               name='region_ub',
                               scope=tbe_platform.scope_ubuf)
        region_sorted_ub = _new_alloc(tvm_ir,
                                      'float16', (1, 1024 * 8),
                                      name='region_sorted_ub',
                                      scope=tbe_platform.scope_ubuf)
        region_k_ub = _new_alloc(tvm_ir,
                                 'float16', (1, 5120 * 8),
                                 name='region_k_ub',
                                 scope=tbe_platform.scope_ubuf)
        region_k2_ub = _new_alloc(tvm_ir,
                                  'float16', (1, 5120 * 8),
                                  name='region_k2_ub',
                                  scope=tbe_platform.scope_ubuf)
    else:
        data_ub = _new_alloc(tvm_ir,
                             'float16', (batch, cols_padding),
                             name='data_ub',
                             scope=tbe_platform.scope_ubuf)
        indices_ub = _new_alloc(tvm_ir,
                                'float16', (cols_padding, ),
                                name='indices_ub',
                                scope=tbe_platform.scope_ubuf)
        indices_out_fp16_ub = _new_alloc(tvm_ir,
                                         'float16', (batch, cols_padding),
                                         name='indices_out_fp16_ub',
                                         scope=tbe_platform.scope_ubuf)
        indices_out_int32_ub = _new_alloc(tvm_ir,
                                          'int32', (batch, cols_padding),
                                          name='indices_out_int32_ub',
                                          scope=tbe_platform.scope_ubuf)

        indices_out_final_ub = _new_alloc(tvm_ir,
                                          'int32', (batch, cols_padding),
                                          name='indices_out_final_ub',
                                          scope=tbe_platform.scope_ubuf)

        offset_ub = _new_alloc(tvm_ir,
                               'float16', (cols_padding, ),
                               name='offset_ub',
                               scope=tbe_platform.scope_ubuf)
        offset_fp16_ub = _new_alloc(tvm_ir,
                                    'float16', (batch, cols_padding),
                                    name='offset_fp16_ub',
                                    scope=tbe_platform.scope_ubuf)
        offset_int32_ub = _new_alloc(tvm_ir,
                                     'int32', (batch, cols_padding),
                                     name='offset_int32_ub',
                                     scope=tbe_platform.scope_ubuf)

        region_ub = _new_alloc(tvm_ir,
                               'float16', (batch, cols_padding * 8),
                               name='region_ub',
                               scope=tbe_platform.scope_ubuf)
        region_sorted_ub = _new_alloc(tvm_ir,
                                      'float16', (batch, cols_padding * 8),
                                      name='region_sorted_ub',
                                      scope=tbe_platform.scope_ubuf)
        region_k_ub = None
        region_k2_ub = None

    data_tail_block_ub = _new_alloc(tvm_ir,
                                    'float16', (16, ),
                                    name='data_tail_block_ub',
                                    scope=tbe_platform.scope_ubuf)
    indices_tail_block_ub = _new_alloc(tvm_ir,
                                       'int32', (8, ),
                                       name='indices_tail_block_ub',
                                       scope=tbe_platform.scope_ubuf)
    reg_min_number = tvm_ir.allocate('float16', (1, ),
                                     scope=tbe_platform.scope_reg,
                                     name='reg_min_number')
    reg_min_number[0] = tvm.const(FP16_MINIMUM, dtype='float16')
    reg_addr = tvm_ir.allocate('uint64', [4],
                               scope=tbe_platform.scope_reg,
                               name='reg_addr')
    reg_addr_buffer = _new_alloc(tvm_ir,
                                 'uint64', [4],
                                 name='reg_addr_buf',
                                 scope=tbe_platform.scope_ubuf)

    GLOBAL_VAR.set_data_gm_out(output)
    GLOBAL_VAR.set_data_ub(data_ub)
    GLOBAL_VAR.set_region_ub(region_ub)
    GLOBAL_VAR.set_region_sorted_ub(region_sorted_ub)
    GLOBAL_VAR.set_region_k_ub(region_k_ub)
    GLOBAL_VAR.set_reg_min_number(reg_min_number)
    GLOBAL_VAR.set_reg_addr(reg_addr)
    GLOBAL_VAR.set_reg_addr_buffer(reg_addr_buffer)
    GLOBAL_VAR.set_indices_ub(indices_ub)
    GLOBAL_VAR.set_indices_out_fp16_ub(indices_out_fp16_ub)
    GLOBAL_VAR.set_indices_out_int32_ub(indices_out_int32_ub)
    GLOBAL_VAR.set_indices_gm_out(indices_out)
    GLOBAL_VAR.set_data_gm(input_a)
    GLOBAL_VAR.set_indices_gm(indices)
    GLOBAL_VAR.set_region_k2_ub(region_k2_ub)
    GLOBAL_VAR.set_data_tail_block_ub(data_tail_block_ub)
    GLOBAL_VAR.set_indices_tail_block_ub(indices_tail_block_ub)
    GLOBAL_VAR.set_offset_ub(offset_ub)
    GLOBAL_VAR.set_offset_fp16_ub(offset_fp16_ub)
    GLOBAL_VAR.set_offset_int32_ub(offset_int32_ub)
    GLOBAL_VAR.set_indices_out_final_ub(indices_out_final_ub)

    blocks = len(rows_cores)
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ir.scope_attr(block_index, "thread_extent", blocks)
    rows_per_core1 = rows_cores[0]
    rows_per_core2 = rows_cores[0] - 1

    loops1 = rows_per_core1 // batch
    loops2 = rows_per_core2 // batch

    remain1 = rows_per_core1 % batch
    remain2 = rows_per_core2 % batch

    with tvm_ir.if_scope(block_index.var < turn):
        core_rows_start = rows_per_core1 * block_index
        if cols > 4096:
            with tvm_ir.for_range(0, loops1, name='i0') as i:
                topk_a_row_by_part(tvm_ir,
                                   row_start_in_core=i,
                                   cols=cols,
                                   k=k,
                                   core_rows_start=core_rows_start,
                                   multi_core=multi_core, largest=largest)
        else:
            with tvm_ir.for_range(0, loops1, name='i0') as i:
                topk_rows(tvm_ir,
                          row_start_in_core=i * batch,
                          rows=batch,
                          cols=cols,
                          k=k,
                          core_rows_start=core_rows_start,
                          multi_core=multi_core, largest=largest)
            if remain1 > 0:
                topk_rows(tvm_ir,
                          row_start_in_core=loops1 * batch,
                          rows=remain1,
                          cols=cols,
                          k=k,
                          core_rows_start=core_rows_start,
                          multi_core=multi_core, largest=largest)

    with tvm_ir.if_scope(block_index.var >= turn):
        core_rows_start = (rows_per_core1 *
                           turn) + (rows_per_core2) * (block_index.var - turn)
        if cols > 4096:
            with tvm_ir.for_range(0, loops2, name='i0') as i:
                topk_a_row_by_part(tvm_ir,
                                   row_start_in_core=i,
                                   cols=cols,
                                   k=k,
                                   core_rows_start=core_rows_start,
                                   multi_core=multi_core, largest=largest)
        else:
            with tvm_ir.for_range(0, loops2, name='i0') as i:
                topk_rows(tvm_ir,
                          row_start_in_core=i * batch,
                          rows=batch,
                          cols=cols,
                          k=k,
                          core_rows_start=core_rows_start,
                          multi_core=multi_core, largest=largest)
            if remain2 > 0:
                topk_rows(tvm_ir,
                          row_start_in_core=loops2 * batch,
                          rows=remain2,
                          cols=cols,
                          k=k,
                          core_rows_start=core_rows_start,
                          multi_core=multi_core, largest=largest)

    return tvm_ir.get()


# pylint: disable=redefined-builtin,unused-argument
def check_supported(input_tensor,
                    indices_tensor,
                    out_tensor,
                    out_indices_tensor,
                    k,
                    sorted=True,
                    dim=-1,
                    largest=True,
                    kernel_name='top_k'):
    """
    check whether ai_core is supported
    """
    if sorted is not True:
        return False
    shape = input_tensor.get("shape")
    #max int fp16 can represent
    if shape[-1] > 65500:
        return False
    #ub size limitation
    if k > 4096:
        return False
    return True


# pylint: disable=redefined-builtin,unused-argument
@util.check_input_type(dict, dict, dict, dict, int, bool, int, bool, str)
def top_k(input_tensor,
          indices_tensor,
          out_tensor,
          out_indices_tensor,
          k,
          sorted=True,
          dim=-1,
          largest=True,
          kernel_name='top_k'):
    """
    Select top K elements from  last dimension
    Parameters
    ----------
    input_tensor: dict.
        Shape and dtype of input to be reversed.
    indices_tensor: dict
    out_tensor: dict.
        Shape and dtype of output.
    out_indices_tensor: dict
    k: int.
        Number of largest elements to be select
    sorted : bool
        if is sorted
    kernel_name : str
        cce kernel name, default value is "cce_topk"
    Returns
    -------
    None
    """

    shape = input_tensor.get("shape")
    dtype = input_tensor.get("dtype")
    indices_shape = indices_tensor.get("shape")
    indices_dtype = indices_tensor.get("dtype")
    out_shape = out_tensor.get("shape")
    out_dtype = out_tensor.get("dtype")
    out_indices_dtype = out_indices_tensor.get("dtype")
    check_list = ("float16", )
    util.check_dtype_rule(dtype, check_list)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    util.check_dtype_rule(out_dtype, check_list)
    util.check_dtype_rule(out_indices_dtype, ("int32", "uint32"))

    shape_dim = len(shape)
    out_shape_dim = len(out_shape)

    if shape_dim != out_shape_dim:
        raise RuntimeError("input tensor and output tensort dim not equal.")
    for i in range(shape_dim - 1):
        if shape[i] != out_shape[i]:
            raise RuntimeError("output tensor dimension not valid.")
    if out_shape[-1] != k:
        raise RuntimeError("output tensor last dim must equal to k")

    if k < 1 or k > shape[-1]:
        raise RuntimeError("K must in ( 1,shape[-1] ]")
    if k > 4096:
        raise RuntimeError("k cannot bigger than 4096")
    data_input = tvm.placeholder(shape, dtype=dtype.lower(), name='data_a')
    indices = tvm.placeholder(indices_shape,
                              dtype=indices_dtype.lower(),
                              name='indices')
    data_buf = tvm.decl_buffer(out_shape, dtype='float16')
    indices_buf = tvm.decl_buffer(out_shape, dtype='int32')
    res, indices_out = tvm.extern([shape, indices_shape],
                                  [data_input, indices],
                                  lambda ins, outs:
                                  _kernel_ir(ins, outs, k, largest),
                                  name='output',
                                  dtype=dtype.lower(),
                                  out_buffers=[data_buf, indices_buf])
    sch = tvm.create_schedule([res.op, indices_out.op])
    with build_config:
        tvm.build(sch, [data_input, indices, res, indices_out],
                  'cce', name=kernel_name)
