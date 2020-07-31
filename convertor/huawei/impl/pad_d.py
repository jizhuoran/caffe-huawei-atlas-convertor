#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

pad
"""
# pylint: disable=locally-disabled,too-many-lines
import json
import os

import te.platform.cce_params as cce_params
from te import platform as tbe_platform
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util
from impl.pad_align_reorder_ub import pad_align


def _ceil_div(value, block):
    """
    Integrate the input value by block.
    """
    return (value + block - 1) // block


def _ceil_fill(value, block):
    """
    Fill the input value by block.
    """
    return _ceil_div(value, block) * block


def _prod(values):
    """
    Prod the input values by multiply.
    """
    res = 1
    for value in values:
        res *= value

    return res


def _get_output_shape(input_shape, paddings):
    """
    Derive the shape size of the output.
    """
    output_shape = [0 for _ in range(len(input_shape))]
    for i, _ in enumerate(zip(input_shape, paddings, output_shape)):
        output_shape[i] = paddings[i][0] + input_shape[i] + paddings[i][1]

    return output_shape


# pylint: disable=locally-disabled,too-many-arguments,too-many-branches,
# pylint: disable=locally-disabled,too-many-statements,too-many-locals
# pylint: disable=locally-disabled,unused-variable,too-many-return-statements
# pylint: disable=unused-argument,consider-using-in,import-outside-toplevel
def _do_cast(params, input_tensor, gm_cast, shape, dtype_src, dtype_dst):
    # vconv count of elements every time
    ele_cnt = 128
    vconv_group = 255
    vconv_ele = vconv_group * ele_cnt
    if dtype_src == "int8" and dtype_dst == "float16":
        vconv_insn = "vconv_s82f16"
        src_stride = 4
        dst_stride = 8
    elif dtype_src == "uint8" and dtype_dst == "float16":
        vconv_insn = "vconv_u82f16"
        src_stride = 4
        dst_stride = 8
    elif dtype_src == "float16" and dtype_dst == "int8":
        vconv_insn = "vconv_f162s8"
        src_stride = 8
        dst_stride = 4
    else:
        vconv_insn = "vconv_f162u8"
        src_stride = 8
        dst_stride = 4

    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype_src) // 8
    dtype_dst_size = tbe_platform.cce_intrin.get_bit_len(dtype_dst) // 8
    ub_len = tbe_platform.cce_conf.get_soc_spec(
        tbe_platform.cce_conf.UB_SIZE) // (dtype_size + dtype_dst_size)
    block_ele = ub_len // ele_cnt
    ub_block_ele = block_ele * ele_cnt
    shape_len = _prod(shape[:])
    num_split = shape_len // ele_cnt
    num_mod = shape_len % ele_cnt
    if num_mod > 0:
        num_count = num_split + 1
    else:
        num_count = num_split
    num_cycle_index = num_count // block_ele
    num_cycle_mod = num_count % block_ele
    if num_cycle_mod > 0:
        num_cycle = num_cycle_index + 1
    else:
        num_cycle = num_cycle_index
    cp_align_len_src = cce_params.BLOCK_REDUCE_INT8 // dtype_size
    cp_align_len_fp16 = cce_params.BLOCK_REDUCE_INT8 // dtype_dst_size

    if num_cycle > 1:
        src_ubuf = _apply_for_new_alloc(params.ib_, dtype_src, ub_block_ele,
                                        cp_align_len_src, tbe_platform.scope_ubuf)
        dst_ubuf = _apply_for_new_alloc(params.ib_, dtype_dst, ub_block_ele,
                                        cp_align_len_fp16, tbe_platform.scope_ubuf)
    else:
        src_ubuf = _apply_for_new_alloc(params.ib_, dtype_src,
                                        num_count * ele_cnt, cp_align_len_src,
                                        tbe_platform.scope_ubuf)
        dst_ubuf = _apply_for_new_alloc(params.ib_, dtype_dst,
                                        num_count * ele_cnt, cp_align_len_fp16,
                                        tbe_platform.scope_ubuf)

    with params.ib_.for_range(0, num_cycle, name="i") as i:
        with params.ib_.if_scope(i < num_cycle - 1):
            params.ib_.emit(
                tvm.call_extern(
                    dtype_src, "copy_gm_to_ubuf", src_ubuf.access_ptr("w"),
                    input_tensor.access_ptr('r', offset=i * ub_block_ele),
                    0, 1,
                    _ceil_div(ub_block_ele, cp_align_len_src), 0, 0))
            ub_vconv_cycle_index = ub_block_ele // vconv_ele
            ub_vconv_cycle = ub_vconv_cycle_index + 1

            with params.ib_.for_range(0, ub_vconv_cycle, name="k") as k:
                with params.ib_.if_scope(k < ub_vconv_cycle - 1):
                    params.ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        params.uint64_all_one,
                                        params.uint64_all_one))
                    params.ib_.emit(
                        tvm.call_extern(
                            dtype_dst, vconv_insn,
                            dst_ubuf.access_ptr("w", offset=k * vconv_ele),
                            src_ubuf.access_ptr('r', offset=k * vconv_ele),
                            vconv_group, 1, 1, dst_stride, src_stride))
                with params.ib_.else_scope():
                    ub_vconv_mod = ub_block_ele - \
                                   (ub_vconv_cycle - 1) * vconv_ele
                    ub_vconv_mod_repeat_more = ub_vconv_mod // ele_cnt
                    ub_src = src_stride
                    ub_dst = dst_stride
                    ub_stride = 1

                    if ub_vconv_mod_repeat_more > 0:
                        params.ib_.emit(
                            tvm.call_extern("uint64", 'set_vector_mask',
                                            params.uint64_all_one,
                                            params.uint64_all_one))
                        params.ib_.emit(
                            tvm.call_extern(dtype_dst, vconv_insn,
                                            dst_ubuf.access_ptr(
                                                "w", offset=(ub_vconv_cycle -
                                                             1) * vconv_ele),
                                            src_ubuf.access_ptr(
                                                'r', offset=(ub_vconv_cycle -
                                                             1) * vconv_ele),
                                            ub_vconv_mod_repeat_more,
                                            ub_stride, ub_stride,
                                            ub_dst, ub_src))

            params.ib_.emit(
                tvm.call_extern(
                    dtype_dst, "copy_ubuf_to_gm",
                    gm_cast.access_ptr("w", offset=i * ub_block_ele),
                    dst_ubuf.access_ptr('r', offset=0), 0, 1,
                    _ceil_div(ub_block_ele, cp_align_len_fp16), 0, 0))

        with params.ib_.else_scope():
            shape_mod = shape_len - (num_cycle - 1) * ub_block_ele
            params.ib_.emit(
                tvm.call_extern(
                    dtype_src, "copy_gm_to_ubuf", src_ubuf.access_ptr("w"),
                    input_tensor.access_ptr(
                        'r', offset=(num_cycle - 1) * ub_block_ele), 0, 1,
                    _ceil_div(shape_mod, cp_align_len_src), 0, 0))
            vconv_cycle_index = shape_mod // vconv_ele  # 255*128=32640
            vconv_cycle_mod = shape_mod % vconv_ele
            if vconv_cycle_mod > 0:
                vconv_cycle = vconv_cycle_index + 1
            else:
                vconv_cycle = vconv_cycle_index

            with params.ib_.for_range(0, vconv_cycle, name="j") as j:
                with params.ib_.if_scope(j < vconv_cycle - 1):
                    params.ib_.emit(
                        tvm.call_extern("uint64", 'set_vector_mask',
                                        params.uint64_all_one,
                                        params.uint64_all_one))
                    params.ib_.emit(
                        tvm.call_extern(
                            dtype_dst, vconv_insn,
                            dst_ubuf.access_ptr("w", offset=j * vconv_ele),
                            src_ubuf.access_ptr('r', offset=j * vconv_ele),
                            vconv_group, 1, 1, dst_stride, src_stride))
                with params.ib_.else_scope():
                    vconv_mod = shape_mod - (vconv_cycle - 1) * vconv_ele
                    vconv_mod_repeat_more = vconv_mod // ele_cnt
                    if vconv_mod_repeat_more > 1:
                        src = src_stride
                        dst = dst_stride
                    else:
                        src = 0
                        dst = 0
                    if vconv_mod == 1:
                        stride = 0
                    else:
                        stride = 1
                    vconv_mod_repeat_one = vconv_mod - \
                                           (vconv_mod_repeat_more * ele_cnt)
                    if vconv_mod_repeat_more > 0:
                        params.ib_.emit(
                            tvm.call_extern("uint64", 'set_vector_mask',
                                            params.uint64_all_one,
                                            params.uint64_all_one))
                        params.ib_.emit(
                            tvm.call_extern(
                                dtype_dst, vconv_insn,
                                dst_ubuf.access_ptr(
                                    "w", offset=(vconv_cycle - 1) * vconv_ele),
                                src_ubuf.access_ptr(
                                    'r', offset=(vconv_cycle - 1) * vconv_ele),
                                vconv_mod_repeat_more, stride, stride, dst,
                                src))
                    if vconv_mod_repeat_one > 0:
                        if vconv_mod_repeat_one <= 64:
                            mask = 0
                            for _ in range(vconv_mod_repeat_one):
                                mask = mask * 2 + 1
                            params.ib_.emit(
                                tvm.call_extern("uint64", 'set_vector_mask', 0,
                                                mask))
                        else:
                            offset = vconv_mod_repeat_one - 64
                            mask = 0
                            for _ in range(offset):
                                mask = mask * 2 + 1
                            params.ib_.emit(
                                tvm.call_extern("uint64", 'set_vector_mask',
                                                mask, params.uint64_all_one))

                        if vconv_mod_repeat_one == 1:
                            stride_one = 0
                        else:
                            stride_one = 1
                        params.ib_.emit(
                            tvm.call_extern(dtype_dst, vconv_insn,
                                            dst_ubuf.access_ptr(
                                                "w",
                                                offset=(vconv_cycle - 1) *
                                                vconv_ele +
                                                vconv_mod_repeat_more *
                                                ele_cnt),
                                            src_ubuf.access_ptr(
                                                'r',
                                                offset=(vconv_cycle - 1) *
                                                vconv_ele +
                                                vconv_mod_repeat_more *
                                                ele_cnt),
                                            1, stride_one, stride_one, 0, 0))

                        params.ib_.emit(
                            tvm.call_extern("uint64", 'set_vector_mask',
                                            params.uint64_all_one,
                                            params.uint64_all_one))

            params.ib_.emit(
                tvm.call_extern(
                    dtype_dst, "copy_ubuf_to_gm",
                    gm_cast.access_ptr(
                        "w", offset=(num_cycle - 1) * ub_block_ele),
                    dst_ubuf.access_ptr('r', offset=0), 0, 1,
                    _ceil_div(shape_mod, cp_align_len_fp16), 0, 0))


def _apply_for_new_alloc(ib_, dtype, buf_len, align_size,
                         scope=tbe_platform.scope_ubuf):
    """
    Request caching space for the calculation process.
    """
    shape = (_ceil_fill(buf_len, align_size),)
    buf_var = ib_.allocate(dtype, shape, name="tmp_buf", scope=scope)
    new_buffer = tvm.decl_buffer(
        shape,
        buf_var.dtype,
        name="tmp_buf",
        scope=tbe_platform.scope_ubuf,
        data=buf_var)

    return new_buffer


def _do_vector_dump(ubuf, ubuf_offset, dup_len, constant_values, params):
    """
    Description vector dump operation.
    """

    def _dump(data_len, cycle_offset):
        """
        Emit instruction 'vector_dup'.
        """
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'vector_dup',
                ubuf.access_ptr("rw", offset=ubuf_offset + cycle_offset),
                constant_values, _ceil_div(data_len,
                                           params.vec_align_len), 1, 1, 8, 8))

    dump_buffer_max_len = params.uint8_max_value * params.vec_align_len
    num_cycle = dup_len // dump_buffer_max_len

    params.ib_.emit(tvm.call_extern("uint64", 'set_vector_mask',
                                    tvm.const(-1, 'uint64'),
                                    tvm.const(-1, 'uint64')))

    with params.ib_.for_range(0, num_cycle, for_type="serial", name="i") as i:
        _dump(dump_buffer_max_len, i * dump_buffer_max_len)
    tail_len = dup_len % dump_buffer_max_len

    with params.ib_.if_scope(tail_len > 0):
        _dump(tail_len, num_cycle * dump_buffer_max_len)


def _do_dump_to_gm(bufs, ubuf_len, padding_len, params, multi_core_top=False):
    """
    Moving data from gm to ub
    """
    ubuf, gm_ = bufs

    def _ubuf_gm_align_ok(data_len, cycle_offset):
        """
        Emit instruction 'copy_ubuf_to_gm' in an aligned scene.
        """
        with params.ib_.if_scope(data_len >= params.cp_align_len):
            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_ubuf_to_gm',
                    gm_.buf.access_ptr("rw", offset=gm_.offset + cycle_offset),
                    ubuf.buf.access_ptr("r", offset=ubuf.offset), 0, 1,
                    data_len // params.cp_align_len, 0, 0))

    def _ubuf_gm_align_fail(data_len, cycle_offset):
        """
        Emit instruction 'copy_ubuf_to_gm' in an unaligned scene.
        """
        align_ok_len = (data_len // params.cp_align_len) * params.cp_align_len
        with params.ib_.if_scope(align_ok_len > 0):
            _ubuf_gm_align_ok(align_ok_len, cycle_offset)

        gm_actual_offset = gm_.offset + cycle_offset + data_len - \
                           params.cp_align_len
        with params.ib_.if_scope(data_len - align_ok_len > 0):
            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_ubuf_to_gm',
                    gm_.buf.access_ptr("rw", offset=gm_actual_offset),
                    ubuf.buf.access_ptr("r", offset=ubuf.offset), 0, 1, 1, 0,
                    0))

    def _ubuf_gm(data_len, cycle_offset):
        """
        Emit instruction 'copy_ubuf_to_gm'.
        """
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_ubuf_to_gm',
                gm_.buf.access_ptr("rw", offset=gm_.offset + cycle_offset),
                ubuf.buf.access_ptr("r", offset=ubuf.offset), 0, 1,
                _ceil_div(data_len, params.cp_align_len), 0, 0))

    num_cycle = padding_len // ubuf_len
    with params.ib_.for_range(0, num_cycle, for_type="serial", name="i") as i:
        _ubuf_gm_align_ok(ubuf_len, i * ubuf_len)

    tail_len = padding_len % ubuf_len
    with params.ib_.if_scope(tail_len > 0):
        if multi_core_top:
            _ubuf_gm_align_fail(tail_len, num_cycle * ubuf_len)
        else:
            _ubuf_gm(tail_len, num_cycle * ubuf_len)


def _do_padding(padding_len, out_ubuf, gm_out_buf, params,
                multi_core_top=False):
    """
    Pad a tensor according to the paddings you specify.
    """
    if padding_len <= 0:
        return gm_out_buf.offset, out_ubuf.offset

    if out_ubuf.buf == 0:
        dump_buf = _apply_for_new_alloc(params.ib_, params.dtype,
                                        params.unified_buffer_len,
                                        params.cp_align_len,
                                        tbe_platform.scope_ubuf)
        dump_buf_len = min(params.unified_buffer_len,
                           _ceil_fill(padding_len, params.cp_align_len))

        _do_vector_dump(dump_buf, 0, dump_buf_len, params.constant_values,
                        params)
        _do_dump_to_gm((PadBuf(dump_buf, 0), gm_out_buf),
                       dump_buf_len,
                       padding_len,
                       params,
                       multi_core_top=multi_core_top)
        return gm_out_buf.offset + padding_len, out_ubuf.offset
    else:
        align_offset, mask = params.get_align_mask(out_ubuf.offset)

        params.ib_.emit(
            tvm.call_extern("uint64", 'set_vector_mask', params.uint64_all_one,
                            mask))

        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'vector_dup',
                out_ubuf.buf.access_ptr(
                    "rw", offset=out_ubuf.offset - align_offset),
                params.constant_values, 1, 1, 1, 8, 8))

        params.ib_.emit(
            tvm.call_extern("uint64", 'set_vector_mask', params.uint64_all_one,
                            params.uint64_all_one))

        # 0 < align_offset < params.cp_align_len, so add params.cp_align_len
        remaining_length = padding_len + params.cp_align_len - \
                           params.vec_align_len

        if padding_len - params.vec_align_len > 0:
            _do_vector_dump(
                out_ubuf.buf,
                out_ubuf.offset - align_offset + params.vec_align_len,
                remaining_length, params.constant_values, params)
        elif remaining_length > 0:
            with params.ib_.if_scope(
                padding_len + align_offset - params.vec_align_len > 0):
                _do_vector_dump(
                    out_ubuf.buf,
                    out_ubuf.offset - align_offset + params.vec_align_len,
                    remaining_length, params.constant_values, params)

        return gm_out_buf.offset, out_ubuf.offset + padding_len


def _mode_large_last_axis(data_len,
                          gm_out_buf,
                          gm_in_buf,
                          params,
                          multi_core_top=False):
    """
    Repeat the pad operation when the last axis is too big to finish
    pad at one time.
    """
    tmp_buf = _apply_for_new_alloc(params.ib_, params.dtype,
                                   params.unified_buffer_len,
                                   params.cp_align_len,
                                   tbe_platform.scope_ubuf)

    def _gm_ubuf_gm_align_ok(copy_len, cycle_offset):
        """
        Emit instruction 'copy_gm_to_ubuf' in an aligned scene.
        """
        with params.ib_.if_scope(copy_len >= params.cp_align_len):
            len_burst = copy_len // params.cp_align_len
            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_gm_to_ubuf',
                    tmp_buf.access_ptr("rw", offset=0),
                    gm_in_buf.buf.access_ptr(
                        "r", offset=gm_in_buf.offset + cycle_offset), 0, 1,
                    len_burst, 0, 0))

            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_ubuf_to_gm',
                    gm_out_buf.buf.access_ptr(
                        "rw", offset=gm_out_buf.offset + cycle_offset),
                    tmp_buf.access_ptr("r", offset=0), 0, 1, len_burst, 0, 0))

    def _gm_ubuf_gm_align_fail(copy_len, cycle_offset):
        """
        Emit instruction 'copy_gm_to_ubuf' in an unaligned scene.
        """
        align_ok_len = (copy_len // params.cp_align_len) * params.cp_align_len
        with params.ib_.if_scope(align_ok_len > 0):
            _gm_ubuf_gm_align_ok(align_ok_len, cycle_offset)

        with params.ib_.if_scope(copy_len - align_ok_len > 0):
            ex_offset = cycle_offset + copy_len - params.cp_align_len
            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_gm_to_ubuf',
                    tmp_buf.access_ptr("rw", offset=0),
                    gm_in_buf.buf.access_ptr(
                        "r", offset=gm_in_buf.offset + ex_offset), 0, 1, 1, 0,
                    0))
            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_ubuf_to_gm',
                    gm_out_buf.buf.access_ptr(
                        "rw", offset=gm_out_buf.offset + ex_offset),
                    tmp_buf.access_ptr("r", offset=0), 0, 1, 1, 0, 0))

    def _gm_ubuf_gm(copy_len, cycle_offset):
        """
        Emit instruction 'copy_gm_to_ubuf'.
        """
        len_burst = _ceil_div(copy_len, params.cp_align_len)
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_gm_to_ubuf',
                tmp_buf.access_ptr("rw", offset=0),
                gm_in_buf.buf.access_ptr(
                    "r", offset=gm_in_buf.offset + cycle_offset), 0, 1,
                len_burst, 0, 0))

        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_ubuf_to_gm',
                gm_out_buf.buf.access_ptr(
                    "rw", offset=gm_out_buf.offset + cycle_offset),
                tmp_buf.access_ptr("r", offset=0), 0, 1, len_burst, 0, 0))

    num_cycle = data_len // params.unified_buffer_len
    with params.ib_.for_range(0, num_cycle, for_type="serial", name="i") as i:
        _gm_ubuf_gm_align_ok(params.unified_buffer_len,
                             i * params.unified_buffer_len)

    tail_len = data_len % params.unified_buffer_len
    with params.ib_.if_scope(tail_len > 0):
        if multi_core_top:
            _gm_ubuf_gm_align_fail(tail_len,
                                   num_cycle * params.unified_buffer_len)
        else:
            _gm_ubuf_gm(tail_len, num_cycle * params.unified_buffer_len)


def _mask_copy(out_ubuf, data_buf, data_offset, data_len, params):
    """
    Achieve data copy indirectly by adding operation and setting
    mask for unaligned data.
    """
    align_offset, mask = params.get_align_mask(out_ubuf.offset)

    if params.dtype == "float16":
        dup_len = 128
    else:
        dup_len = 64
    dup_buf = _apply_for_new_alloc(params.ib_, params.dtype, dup_len,
                                   params.cp_align_len,
                                   tbe_platform.scope_ubuf)

    params.ib_.emit(
        tvm.call_extern(params.dtype, 'vector_dup',
                        dup_buf.access_ptr("rw", offset=0), 0, 1, 1, 1, 8, 8))

    params.ib_.emit(
        tvm.call_extern("uint64", 'set_vector_mask', params.uint64_all_one,
                        mask))

    params.ib_.emit(
        tvm.call_extern(
            params.dtype, 'vadd',
            out_ubuf.buf.access_ptr("w", offset=out_ubuf.offset -
                                    align_offset),
            data_buf.access_ptr("r", offset=data_offset),
            dup_buf.access_ptr("r", offset=0), 1, 1, 1, 1, 8, 8, 8))

    params.ib_.emit(
        tvm.call_extern("uint64", 'set_vector_mask', params.uint64_all_one,
                        params.uint64_all_one))

    def _do_copy(copy_len):
        """
        Emit instruction 'copy_ubuf_to_ubuf'.
        """
        params.ib_.emit(
            tvm.call_extern(params.dtype, 'copy_ubuf_to_ubuf',
                            out_ubuf.buf.access_ptr(
                                "rw", offset=out_ubuf.offset - align_offset +
                                params.vec_align_len),
                            data_buf.access_ptr(
                                "r", offset=data_offset +
                                params.vec_align_len), 0, 1,
                            _ceil_div(copy_len + params.cp_align_len -
                                      params.vec_align_len + align_offset,
                                      params.cp_align_len), 0, 0))

    if data_len - params.vec_align_len > 0:
        _do_copy(data_len)
    elif data_len + params.cp_align_len - params.vec_align_len > 0:
        with params.ib_.if_scope(
            data_len + align_offset - params.vec_align_len > 0):
            _do_copy(data_len)


def _mode_align_dst_src_fail(out_ubuf, gm_in_buf, data_len, gm_align, params):
    """
    In a scene where source data is not aligned and target data is aligned
    """

    align_offset, _ = params.get_align_mask(out_ubuf.offset)
    tmp_buf = _apply_for_new_alloc(params.ib_, params.dtype,
                                   data_len + params.cp_align_len,
                                   params.cp_align_len,
                                   tbe_platform.scope_ubuf)

    params.ib_.emit(
        tvm.call_extern(
            params.dtype,
            'copy_gm_to_ubuf',
            tmp_buf.access_ptr("rw", offset=0),
            gm_in_buf.buf.access_ptr("r", offset=gm_in_buf.offset),
            0,
            1,
            _ceil_div(data_len + params.cp_align_len, params.cp_align_len),
            # clear warning, 0 < align_offset < params.cp_align_len
            0,
            0))

    params.ib_.emit(
        tvm.call_extern(
            params.dtype,
            'copy_ubuf_to_gm',
            gm_align.access_ptr("rw", offset=align_offset),
            tmp_buf.access_ptr("r", offset=0),
            0,
            1,
            _ceil_div(data_len + params.cp_align_len, params.cp_align_len),
            # clear warning, 0 < align_offset < params.cp_align_len
            0,
            0))

    params.ib_.emit(
        tvm.call_extern(
            params.dtype,
            'copy_gm_to_ubuf',
            tmp_buf.access_ptr("rw", offset=0),
            gm_align.access_ptr("r", offset=0),
            0,
            1,
            _ceil_div(data_len + params.cp_align_len, params.cp_align_len),
            # clear warning, 0 < align_offset < params.cp_align_len
            0,
            0))

    _mask_copy(out_ubuf, tmp_buf, 0, data_len, params)


def _data_copy(axis, bufs, gm_align, params, multi_core_top=False):
    """
    Achieving data moving in different scenarios
    """
    out_buf, data_buf, gm_out_buf, gm_in_buf = bufs
    one_block_size = _prod(params.out_shape[axis + 1:])
    data_axis_end_len = params.in_shape[axis] * one_block_size

    if params.copy_mode == params.enum_copy_mode['large_last_axis']:
        # When the last dimension is large, the data is processed in a
        # sub-section
        _mode_large_last_axis(
            data_axis_end_len,
            gm_out_buf,
            gm_in_buf,
            params,
            multi_core_top=multi_core_top)
    elif params.copy_mode == params.enum_copy_mode['align_ok']:
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_ubuf_to_ubuf',
                out_buf.buf.access_ptr("rw", offset=out_buf.offset),
                data_buf.buf.access_ptr("r", offset=data_buf.offset), 0, 1,
                _ceil_div(params.in_shape[axis] * one_block_size,
                          params.cp_align_len), 0, 0))
    elif params.copy_mode == params.enum_copy_mode['align_dst_fail']:
        # When outbuf offset does not meet 32 byte alignment,
        # but after offset the inbuf and outbuf can meet both
        align_offset, _ = params.get_align_mask(out_buf.offset)
        _mask_copy(out_buf, data_buf.buf, data_buf.offset - align_offset,
                   data_axis_end_len, params)
    elif params.copy_mode == params.enum_copy_mode['align_dst_src_fail']:
        # When outbuf offset does not meet 32 byte alignment,
        # but after offset the inbuf and outbuf can not meet both
        _mode_align_dst_src_fail(out_buf, gm_in_buf, data_axis_end_len,
                                 gm_align, params)


# pylint: disable=locally-disabled,too-many-instance-attributes
class PadParams:
    """
    Define the parameters for pad operation.

    Interfaces
    ----------
    get_align_mask: Provide access to masks of computations.
    get_data_copy: Provide access to align mode.
    get_copy_mode: Provide access to data moving mode.
    multi_core_get_copy_mode: Data moving mode in multi-cores scenario.
    """

    def __init__(self, ib_, key_args, dtype):
        """
        Initialization method
        """
        self.enum_copy_mode = {
            'align_ok': 1,
            'align_dst_fail': 2,
            'align_dst_src_fail': 3,
            'large_last_axis': 4
        }

        self.ib_ = ib_
        self.in_shape, self.paddings, self.constant_values = key_args
        self.dtype = dtype
        self.out_shape = _get_output_shape(self.in_shape, self.paddings)
        self.copy_mode = self.enum_copy_mode['align_ok']
        # Convert byts to Bytes
        self.type_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
        self.cp_align_len = cce_params.BLOCK_REDUCE_INT8 // self.type_size
        self.unified_buffer_len = \
            (tbe_platform.cce_conf.get_soc_spec(
                tbe_platform.cce_conf.UB_SIZE) - 256) // self.type_size
        self.vec_align_len = \
            cce_params.VECTOR_INST_BLOCK_WIDTH // self.type_size
        # Maximum number of uint8
        self.uint8_max_value = 255
        # Number corresponding to 64-bit masks when they are all 1: 2**64 -1
        self.uint64_all_one = 18446744073709551615
        self.mask = ib_.allocate(
            "uint64", (1,), name="mask", scope=tbe_platform.scope_reg)
        self.align_offset = \
            ib_.allocate("int32", (2,), name="align_offset",
                         scope=tbe_platform.scope_reg)
        self.device_core_num = \
            tbe_platform.cce_conf.get_soc_spec(
                tbe_platform.cce_conf.CORE_NUM)
        self.block = tvm.thread_axis("blockIdx.x")
        self.ib_.scope_attr(self.block, "thread_extent", self.device_core_num)
        self.in_multi_core_mode = False

    def get_align_mask(self, offset):
        """
        Computing mask required for different operations
        """
        if isinstance(offset, int):
            align_offset = offset % self.cp_align_len
            mask = self.uint64_all_one - (2 ** align_offset - 1)
            return align_offset, mask

        self.align_offset[0] = 1
        self.align_offset[0] = \
            offset % (self.cp_align_len * self.align_offset[0])
        self.mask[0] = \
            tvm.const(self.uint64_all_one // 2,
                      "uint64") * tvm.const(2, "uint64") + \
            tvm.const(1, "uint64")
        with self.ib_.for_range(
            0, self.align_offset[0], for_type="serial", name="i"):
            self.mask[0] = self.mask[0] * tvm.const(2, "uint64")

        return self.align_offset[0], self.mask[0]

    def get_data_copy(self, in_shape, paddings, axis, buf_args):
        """
        Setting up different types of data alignment patterns
        """
        out_ubuf, out_ubuf_offset, data_offset = buf_args
        padding_head, padding_tail = paddings[axis]

        if out_ubuf:
            if in_shape[axis] % self.cp_align_len == 0 and \
                    padding_head % self.cp_align_len == 0 and \
                    padding_tail % self.cp_align_len == 0:
                if self.copy_mode != self.enum_copy_mode['align_dst_fail'] and\
                        self.copy_mode != \
                        self.enum_copy_mode['align_dst_src_fail']:
                    self.copy_mode = self.enum_copy_mode['align_ok']
            else:
                if self.copy_mode != self.enum_copy_mode['align_dst_src_fail']:
                    self.copy_mode = self.enum_copy_mode['align_dst_fail']

                align_offset = out_ubuf_offset % self.cp_align_len
                data_offset_align = data_offset - align_offset

                if data_offset_align % self.cp_align_len != 0:
                    self.copy_mode = self.enum_copy_mode['align_dst_src_fail']
        else:
            self.copy_mode = self.enum_copy_mode['large_last_axis']

    def get_copy_mode(self, in_shape, paddings, axis, buf_args):
        """
        Setting up different types of data moving patterns
        """
        out_ubuf, out_ubuf_offset, data_offset = buf_args
        in_data_len = _prod(in_shape[axis:])
        out_data_len = _prod(_get_output_shape(in_shape, paddings)[axis:])
        create_data_buf = \
            (out_ubuf is False and _ceil_fill(in_data_len, self.cp_align_len) +
             _ceil_fill(out_data_len,
                        self.cp_align_len) < self.unified_buffer_len)

        if create_data_buf:
            out_ubuf = True
        one_block_size = _prod(_get_output_shape(in_shape,
                                                 paddings)[axis + 1:])

        if out_ubuf:
            out_ubuf_offset = \
                out_ubuf_offset + paddings[axis][0] * one_block_size
        in_one_block_size = _prod(in_shape[axis + 1:])

        if axis + 1 >= len(in_shape):
            self.get_data_copy(in_shape, paddings, axis,
                               (out_ubuf, out_ubuf_offset, data_offset))
        else:
            for i in range(in_shape[axis]):
                if out_ubuf:
                    out_ubuf_offset = out_ubuf_offset + i * one_block_size
                    data_offset = data_offset + i * in_one_block_size
                self.get_copy_mode(in_shape, paddings, axis + 1,
                                   (out_ubuf, out_ubuf_offset, data_offset))

        return self.copy_mode

    def multi_core_get_copy_mode(self, in_shape, paddings, axis, buf_args):
        """
        Setting up different types of data moving patterns
        in multi-cores scenario
        """
        out_ubuf, out_ubuf_offset, data_offset = buf_args
        self.in_multi_core_mode = True
        if axis + 1 >= len(in_shape):
            self.copy_mode = self.enum_copy_mode['large_last_axis']
        else:
            self.get_copy_mode(in_shape, paddings, axis + 1,
                               (out_ubuf, out_ubuf_offset, data_offset))

        return self.copy_mode


class PadBuf:
    """
    Define buffers and offsets involved in operations

    Interfaces
    ----------
    get_buf: Provide access to buffer of the operation.
    get_offset: Provide access to offset in the buffer.
    """

    def __init__(self, buf, offset):
        """
        Initialization method
        """
        self.buf = buf
        self.offset = offset

    def get_buf(self):
        """
        Get the buffer of the operation
        """
        return self.buf

    def get_offset(self):
        """
        Get the offset of the data in the buffer
        """
        return self.offset


def _pad_wc_axis(axis, bufs, params, n_align, n_padding):
    out_buf, data_buf, gm_out_buf, gm_in_buf = bufs
    data_len = _prod(params.in_shape[axis:])
    out_data_len = _prod(params.out_shape[axis:])
    c_axis = params.in_shape[axis+1]
    pad_head = n_align + n_padding
    pad_tail = params.paddings[axis][1] * c_axis
    if params.dtype == "float16":
        dup_len = 128
    else:
        dup_len = 64

    def _get_masks(data_len, out_data_len):
        circle_mask = []
        calc_num = data_len
        total_num = out_data_len
        nums_per_repeat = dup_len

        # init circle mask
        circle_mask.extend([0] * pad_head)
        circle_mask.extend([1] * calc_num)
        circle_mask.extend([0] * pad_tail)
        circle_mask *= (total_num // len(circle_mask))
        tail = nums_per_repeat - total_num % nums_per_repeat
        if tail:
            circle_mask.extend([0] * tail)
        masks = []
        for i in range(0, len(circle_mask), nums_per_repeat):
            mask = circle_mask[i:i+nums_per_repeat]
            mask = ''.join([str(b) for b in mask])[::-1]
            masks.append(mask)
        return masks

    def _get_mask_and_repeat(masks):
        from itertools import groupby
        mask = 0xffffffffffffffff
        mask_len = 64
        for i, group in groupby(masks):
            value = int(i, 2)
            low = value & mask
            high = (value >> mask_len) & mask
            yield high, low, len(list(group))

    def _data_copy_wc():
        tmp_buf = _apply_for_new_alloc(params.ib_, params.dtype,
                                       pad_head + out_data_len +
                                       params.cp_align_len,
                                       params.cp_align_len, tbe_platform.scope_ubuf)

        params.ib_.emit(
            tvm.call_extern(
                params.dtype,
                'copy_gm_to_ubuf',
                tmp_buf.access_ptr("rw", offset=pad_head),
                gm_in_buf.buf.access_ptr("r", offset=gm_in_buf.offset),
                0,
                1,
                _ceil_div(data_len, params.cp_align_len),
                # clear warning, 0 < align_offset < params.cp_align_len
                0,
                0))

        dup_buf = _apply_for_new_alloc(params.ib_, params.dtype, dup_len,
                                       params.cp_align_len, tbe_platform.scope_ubuf)
        params.ib_.emit(
            tvm.call_extern(params.dtype, 'vector_dup',
                            dup_buf.access_ptr("rw",
                                               offset=0), 0, 1, 1, 1, 8, 8))

        masks = _get_masks(data_len, out_data_len + n_align)
        intrin_param = _get_mask_and_repeat(masks)
        n_repeats = 0
        for hi_mask, lo_mask, repeat in intrin_param:
            if hi_mask == 0 and lo_mask == 0:
                n_repeats += repeat
                continue
            params.ib_.emit(
                tvm.call_extern("uint64", 'set_vector_mask',
                                tvm.const(hi_mask, 'uint64'),
                                tvm.const(lo_mask, 'uint64')))

            while repeat > 0:
                if repeat > 255:
                    vec_repeat = 255
                else:
                    vec_repeat = repeat
                repeat -= vec_repeat
                params.ib_.emit(
                    tvm.call_extern(
                        params.dtype, 'vadd',
                        out_buf.buf.access_ptr("w", offset=dup_len *
                                               n_repeats),
                        tmp_buf.access_ptr("r", offset=dup_len * n_repeats),
                        dup_buf.access_ptr('r', offset=0),
                        vec_repeat, 1, 1, 1, 8, 8, 0))
                n_repeats += vec_repeat

        burst_len = _ceil_div(n_align + n_padding, 32 // params.type_size)
        if burst_len > 0:
            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_ubuf_to_gm',
                    gm_out_buf.buf.access_ptr("rw", offset=gm_out_buf.offset),
                    out_buf.buf.access_ptr("r", offset=0),
                    0, 1,
                    burst_len, 0, 0))

    _data_copy_wc()


# pylint: disable=locally-disabled,too-many-locals
def _pad_recursive_fun(axis, bufs, gm_align, params, multi_core_top=False):
    """
    Achieve pad operations with different shapes norms by recursively way.
    """
    out_buf, data_buf, gm_out_buf, gm_in_buf = bufs
    in_data_len = _prod(params.in_shape[axis:])
    out_data_len = _prod(params.out_shape[axis:])

    create_data_buf = \
        (out_buf.buf == 0 and _ceil_fill(in_data_len, params.cp_align_len) +
         _ceil_fill(out_data_len,
                    params.cp_align_len) < params.unified_buffer_len)

    # fuse the last to axis if paddings for the last-axis is zero
    fuse_wc_axis = create_data_buf and \
                   axis + 2 == len(params.in_shape) and \
                   params.in_shape[axis + 1] == params.out_shape[axis + 1] and\
                   params.paddings[axis + 1][0] == 0 and \
                   params.paddings[axis + 1][1] == 0

    # the fisrt out_data_len is for vecdup(zero),
    # the second out_data_len plus padding is the finnal output in UB.
    # so, the total size should less than UB
    fuse_wc_axis = fuse_wc_axis and \
                   (_ceil_fill(out_data_len, params.cp_align_len)*2 +
                    _ceil_fill(out_data_len - in_data_len,
                               params.cp_align_len)) <\
                   params.unified_buffer_len

    if create_data_buf:
        if params.copy_mode == params.enum_copy_mode['align_dst_src_fail']:
            out_buf.buf = \
                _apply_for_new_alloc(params.ib_, params.dtype,
                                     out_data_len, params.cp_align_len,
                                     tbe_platform.scope_ubuf)
        else:
            out_buf.buf = \
                _apply_for_new_alloc(params.ib_, params.dtype, out_data_len,
                                     params.cp_align_len,
                                     tbe_platform.scope_ubuf)

            if not fuse_wc_axis:

                data_buf.buf = \
                    _apply_for_new_alloc(params.ib_, params.dtype,
                                         in_data_len, params.cp_align_len,
                                         tbe_platform.scope_ubuf)
                params.ib_.emit(
                    tvm.call_extern(
                        params.dtype, 'copy_gm_to_ubuf',
                        data_buf.buf.access_ptr("rw", offset=0),
                        gm_in_buf.buf.access_ptr("r", offset=gm_in_buf.offset),
                        0, 1,
                        _ceil_div(in_data_len, params.cp_align_len), 0, 0))

    one_block_size = _prod(params.out_shape[axis + 1:])
    in_one_block_size = _prod(params.in_shape[axis + 1:])

    n_align = 0
    n_padding = 0
    if not fuse_wc_axis:
        gm_out_buf.offset, out_buf.offset = \
            _do_padding(params.paddings[axis][0] * one_block_size,
                        out_buf, gm_out_buf, params)
    else:
        c_axis = params.in_shape[axis+1]
        n_padding = params.paddings[axis][0] * c_axis
        factor = 32 // params.type_size
        if n_padding % factor != 0:
            n_align = factor - n_padding % factor

    if fuse_wc_axis:

        tvm.call_extern("uint64", 'set_vector_mask',
                        tvm.const(-1, 'uint64'), tvm.const(-1, 'uint64'))

        dup_len = out_data_len + n_align
        _do_vector_dump(out_buf.buf, out_buf.offset, dup_len,
                        params.constant_values, params)

        _pad_wc_axis(
            axis, (out_buf, data_buf, gm_out_buf, gm_in_buf),
            params, n_align, n_padding)

        gm_out_buf.offset += n_padding

    elif axis + 1 >= len(params.in_shape):
        _data_copy(
            axis, (out_buf, data_buf, gm_out_buf, gm_in_buf),
            gm_align,
            params,
            multi_core_top=multi_core_top)
    else:
        with params.ib_.for_range(
            0, params.in_shape[axis],
            for_type="serial", name="i") as i:
            if out_buf.buf != 0:
                out_ubuf_offset_tmp = out_buf.offset + i * one_block_size
                data_buf.offset = data_buf.offset + i * in_one_block_size
            else:
                out_ubuf_offset_tmp = out_buf.offset

            _pad_recursive_fun(
                axis + 1,
                (PadBuf(out_buf.buf, out_ubuf_offset_tmp), data_buf,
                 PadBuf(gm_out_buf.buf, gm_out_buf.offset + i *
                        one_block_size),
                 PadBuf(gm_in_buf.buf,
                        gm_in_buf.offset + i * in_one_block_size)), gm_align,
                params)

    if out_buf.buf != 0:
        out_buf.offset = out_buf.offset + \
                         params.in_shape[axis] * one_block_size
    if out_buf.buf == 0:
        gm_out_buf.offset = gm_out_buf.offset + \
                            params.in_shape[axis] * one_block_size

    if not fuse_wc_axis:
        gm_out_buf.offset, out_buf.offset = \
            _do_padding(params.paddings[axis][1] * one_block_size,
                        out_buf, gm_out_buf, params,
                        multi_core_top=multi_core_top)

    if create_data_buf:
        _save_out_buf(out_buf.buf, gm_out_buf,
                      (params.paddings[axis][1] * one_block_size, in_data_len,
                       out_data_len), params, fuse_wc_axis, n_align, n_padding)


def _save_out_buf(out_ubuf, gm_out_buf, data_lens, params, fuse_wc_axis=False,
                  n_align=0, n_padding=0):
    """
    Moving data from ubuf to external storage(gm)
    """
    tail_pad_len, in_data_len, out_data_len = data_lens
    if fuse_wc_axis:
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_ubuf_to_gm',
                gm_out_buf.buf.access_ptr("rw", offset=gm_out_buf.offset),
                out_ubuf.access_ptr("r", offset=n_align + n_padding), 0, 1,
                _ceil_div(out_data_len - n_padding, params.cp_align_len),
                0, 0))
    elif params.in_multi_core_mode:
        align_tail = out_data_len % params.cp_align_len
        if align_tail != 0:
            # Avoid multi-core data coverage
            if tail_pad_len > align_tail:
                actual_offset = gm_out_buf.offset + out_data_len - \
                                params.cp_align_len
                params.ib_.emit(
                    tvm.call_extern(
                        params.dtype,
                        'copy_ubuf_to_gm',
                        gm_out_buf.buf.access_ptr("rw", offset=actual_offset),
                        out_ubuf.access_ptr(
                            "r", offset=out_data_len - align_tail),
                        # The rest of the block must be constant_values.
                        0,
                        1,
                        1,
                        0,
                        0))
            else:
                actual_offset = gm_out_buf.offset + \
                                params.cp_align_len - align_tail
                params.ib_.emit(
                    tvm.call_extern(
                        params.dtype, 'copy_ubuf_to_gm',
                        gm_out_buf.buf.access_ptr("rw", offset=actual_offset),
                        out_ubuf.access_ptr(
                            "r", offset=out_data_len - align_tail), 0, 1, 1, 0,
                        0))

                params.ib_.emit(
                    tvm.call_extern(
                        params.dtype, 'copy_gm_to_ubuf',
                        out_ubuf.access_ptr(
                            "rw", offset=out_data_len - align_tail),
                        gm_out_buf.buf.access_ptr(
                            "r", offset=gm_out_buf.offset), 0, 1,
                        _ceil_div(in_data_len, params.cp_align_len), 0, 0))

                actual_offset = gm_out_buf.offset + \
                                out_data_len - params.cp_align_len
                params.ib_.emit(
                    tvm.call_extern(
                        params.dtype,
                        'copy_ubuf_to_gm',
                        gm_out_buf.buf.access_ptr("rw", offset=actual_offset),
                        out_ubuf.access_ptr(
                            "r", offset=out_data_len - align_tail),
                        # The rest of the block must be constant_values.
                        0,
                        1,
                        1,
                        0,
                        0))

        if out_data_len > params.cp_align_len:
            params.ib_.emit(
                tvm.call_extern(
                    params.dtype, 'copy_ubuf_to_gm',
                    gm_out_buf.buf.access_ptr("rw", offset=gm_out_buf.offset),
                    out_ubuf.access_ptr("r", offset=0), 0, 1,
                    out_data_len // params.cp_align_len, 0, 0))
    else:
        params.ib_.emit(
            tvm.call_extern(
                params.dtype, 'copy_ubuf_to_gm',
                gm_out_buf.buf.access_ptr("rw", offset=gm_out_buf.offset),
                out_ubuf.access_ptr("r", offset=0), 0, 1,
                _ceil_div(out_data_len, params.cp_align_len), 0, 0))


def _multi_core_do_padding(padding_len, constant_values, gm_out_buf,
                           gm_out_offset, params):
    """
    Pad a tensor according to the paddings you specify in multi-core scenario.
    """
    if padding_len <= 0:
        return

    dump_buf = _apply_for_new_alloc(params.ib_, params.dtype,
                                    params.unified_buffer_len,
                                    params.cp_align_len,
                                    tbe_platform.scope_ubuf)
    block_size_1 = _ceil_div(padding_len, params.device_core_num)
    block_size = _ceil_fill(block_size_1, params.cp_align_len)

    dump_buf_len = min(params.unified_buffer_len, block_size)
    _do_vector_dump(dump_buf, 0, dump_buf_len, constant_values, params)

    if padding_len // block_size == 0:
        with params.ib_.if_scope(params.block.var == 0):
            _do_dump_to_gm(
                (PadBuf(dump_buf, 0),
                 PadBuf(gm_out_buf,
                        gm_out_offset + params.block.var * block_size)),
                dump_buf_len, padding_len, params)
    else:
        with params.ib_.if_scope((params.block.var + 1) * block_size
                                 <= padding_len):
            _do_dump_to_gm(
                (PadBuf(dump_buf, 0),
                 PadBuf(gm_out_buf,
                        gm_out_offset + params.block.var * block_size)),
                dump_buf_len, block_size, params)

        if padding_len % block_size != 0:
            with params.ib_.else_scope():
                with params.ib_.if_scope(
                    padding_len - params.block.var * block_size > 0):
                    _do_dump_to_gm(
                        (PadBuf(dump_buf, 0),
                         PadBuf(gm_out_buf,
                                gm_out_offset + params.block.var *
                                block_size)),
                        dump_buf_len,
                        padding_len % block_size,
                        params,
                        multi_core_top=True)


# pylint: disable=locally-disabled,too-many-locals
def _pad_multi_core_fun(axis, bufs, gm_align, params):
    """
    The method of realizing pad operation in multi-core scenario.
    """
    out_buf, data_buf, gm_out_buf, gm_in_buf = bufs
    one_block_size = _prod(params.out_shape[axis + 1:])

    _multi_core_do_padding(params.paddings[axis][0] * one_block_size,
                           params.constant_values, gm_out_buf.buf,
                           gm_out_buf.offset, params)

    gm_out_buf.offset = \
        gm_out_buf.offset + params.paddings[axis][0] * one_block_size
    in_one_block_size = _prod(params.in_shape[axis + 1:])

    if axis + 1 >= len(params.in_shape):
        one_block_size = _prod(params.out_shape[axis + 1:])
        data_axis_end_len = params.in_shape[axis] * one_block_size
        block_size_1 = _ceil_div(data_axis_end_len, params.device_core_num)
        block_size = _ceil_fill(block_size_1, params.cp_align_len)

        with params.ib_.if_scope((params.block.var + 1) * block_size
                                 <= data_axis_end_len):
            _mode_large_last_axis(
                block_size,
                PadBuf(gm_out_buf.buf,
                       gm_out_buf.offset + params.block.var * block_size),
                PadBuf(gm_in_buf.buf,
                       gm_in_buf.offset + params.block.var * block_size),
                params)

        if data_axis_end_len % block_size != 0:
            with params.ib_.else_scope():
                with params.ib_.if_scope(
                    data_axis_end_len - params.block.var * block_size > 0):
                    _mode_large_last_axis(
                        data_axis_end_len % block_size,
                        PadBuf(
                            gm_out_buf.buf,
                            gm_out_buf.offset + params.block.var * block_size),
                        PadBuf(gm_in_buf.buf, gm_in_buf.offset +
                               params.block.var * block_size),
                        params,
                        multi_core_top=True)
    else:
        repeat = _ceil_div(params.in_shape[axis], params.device_core_num)

        with params.ib_.for_range(0, repeat, for_type="serial", name="i") as i:
            j = params.block.var * repeat + i

            with params.ib_.if_scope(j < params.in_shape[axis]):
                _pad_recursive_fun(
                    axis + 1,
                    (out_buf, data_buf,
                     PadBuf(gm_out_buf.buf,
                            gm_out_buf.offset + j * one_block_size),
                     PadBuf(gm_in_buf.buf,
                            gm_in_buf.offset + j * in_one_block_size)),
                    gm_align,
                    params,
                    multi_core_top=True)

    gm_out_buf.offset = gm_out_buf.offset + \
                        params.in_shape[axis] * one_block_size
    _multi_core_do_padding(params.paddings[axis][1] * one_block_size,
                           params.constant_values, gm_out_buf.buf,
                           gm_out_buf.offset, params)


# pylint: disable=locally-disabled,too-many-arguments
def _intrin_factor(input_tensor, output_res, gm_cast, gm_align, output_tensor,
                   args):
    """
    Implement split logic and select multi-core or single-core operations
    based on the results of segmentation
    """
    in_shape, paddings, constant_values = args
    ib_ = tvm.ir_builder.create()

    dtype_src = input_tensor.dtype
    if dtype_src == "int8" or dtype_src == "uint8":
        dtype_now = "float16"
    else:
        dtype_now = dtype_src

    params = PadParams(
        ib_, (in_shape, paddings, tvm.const(constant_values, dtype_now)),
        dtype_now)

    if dtype_src == "int8" or dtype_src == "uint8":
        _do_cast(params, input_tensor, gm_cast, in_shape, dtype_src, dtype_now)
        input_cast = gm_cast
        output_mid = output_tensor
    else:
        input_cast = input_tensor
        output_mid = output_res

    tmp_mode = params.get_copy_mode(in_shape, paddings, 0, (False, 0, 0))

    if len(in_shape) == 1 and _prod(in_shape[0:]) >= params.cp_align_len and \
            (paddings[0][1] >= params.cp_align_len or
             paddings[0][1] == 0) and (tmp_mode !=
                                       params.enum_copy_mode[
                                           'align_dst_src_fail']):

        params.copy_mode = params.enum_copy_mode['align_ok']
        params.multi_core_get_copy_mode(in_shape, paddings, 0, (False, 0, 0))
        _pad_multi_core_fun(0, (PadBuf(0, 0), PadBuf(0, 0), PadBuf(
            output_mid, 0), PadBuf(input_cast, 0)), gm_align, params)

    elif _prod(in_shape[1:]) >= params.cp_align_len and \
            (paddings[1][1] * _prod(params.out_shape[2:]) >=
             params.cp_align_len
             or paddings[1][1] == 0) \
            and (tmp_mode != params.enum_copy_mode['align_dst_src_fail']):

        params.copy_mode = params.enum_copy_mode['align_ok']
        params.multi_core_get_copy_mode(in_shape, paddings, 0, (False, 0, 0))
        _pad_multi_core_fun(0, (PadBuf(0, 0), PadBuf(0, 0), PadBuf(
            output_mid, 0), PadBuf(input_cast, 0)), gm_align, params)
    else:
        params.copy_mode = params.enum_copy_mode['align_ok']
        params.get_copy_mode(in_shape, paddings, 0, (False, 0, 0))
        with params.ib_.if_scope(params.block.var == 0):
            _pad_recursive_fun(0, (PadBuf(0, 0), PadBuf(
                0, 0), PadBuf(output_mid, 0), PadBuf(input_cast, 0)), gm_align,
                               params)

    if dtype_src == "int8" or dtype_src == "uint8":
        _do_cast(params, output_mid, output_res, params.out_shape, dtype_now,
                 dtype_src)

    return ib_.get()


def _write_code(wkspace_dict, fname):
    """
    write workspaces to json file

    """
    fname = os.path.realpath(fname)
    if fname.startswith(os.getcwd()):
        if os.path.exists(fname):
            with open(fname, "r") as f_var:
                load_dict = json.load(f_var)
            load_dict.update(wkspace_dict)
            with open(fname, "w") as f_var:
                json.dump(
                    load_dict,
                    f_var,
                    sort_keys=True,
                    indent=4,
                    separators=(',', ':'))


def _set_mask(length):
    """
    calculate MASK in cce

    """
    length = int(length)
    mask1 = 2**max(length - 64, 0) - 1
    mask2 = 2**min(length, 64) - 1
    return mask1, mask2


def _set_mask_insn(tvm_ir, type_, bits=128):
    """
    set_mask_insn
    """
    mask1, mask2 = _set_mask(bits)
    tvm_ir.emit(tvm.call_extern(type_, 'set_vector_mask',
                                tvm.const(mask1, dtype='uint64'),
                                tvm.const(mask2, dtype='uint64')))


def _zero_ub(tvm_ir, buf_addr, buf_len, dtype):
    """
    clean a ub memory
    """
    dup_len = 128
    if dtype in ('float32', 'int32'):
        dup_len = 64
    if dtype in ('int8', 'uint8'):
        dup_len = 256
    repeat = buf_len // dup_len
    remain = buf_len % dup_len
    if repeat > 255:
        repeat_255 = repeat // 255
        with tvm_ir.for_range(0, repeat_255, name='i0') as i:
            tvm_ir.emit(
                tvm.call_extern(
                    dtype, 'vector_dup',
                    buf_addr.access_ptr('w', offset=i*255*dup_len),
                    0, 255, 1, 1, 8, 8))
        if repeat % 255 > 0:
            tvm_ir.emit(
                tvm.call_extern(
                    dtype, 'vector_dup',
                    buf_addr.access_ptr('w',
                                        offset=repeat_255*255*dup_len),
                    0, repeat % 255, 1, 1, 8, 8))

    else:
        tvm_ir.emit(
            tvm.call_extern(dtype, 'vector_dup', buf_addr.access_ptr('w'), 0,
                            repeat, 1, 1, 8, 8))

    if remain > 0:
        _set_mask_insn(tvm_ir, dtype, remain)
        tvm_ir.emit(tvm.call_extern(dtype, 'vector_dup',
                                    buf_addr.access_ptr(
                                        'w',
                                        offset=repeat*dup_len),
                                    0, 1, 1, 1, 8, 8))
        _set_mask_insn(tvm_ir, dtype, 128)


def _zero_ub_tail(tvm_ir, buf_addr, buf_offset, zero_len, dtype):
    """
    clean a ub memory from end-point
    """
    block_len = 16
    if dtype in ('float32', 'int32'):
        block_len = 8
    mask1 = 0
    mask2 = (2**(block_len) - 1) - (2**(block_len - zero_len) - 1)
    tvm_ir.emit(tvm.call_extern(dtype, 'set_vector_mask',
                                tvm.const(mask1, dtype='uint64'),
                                tvm.const(mask2, dtype='uint64')))

    tvm_ir.emit(tvm.call_extern(dtype, 'vector_dup',
                                buf_addr.access_ptr('w', offset=buf_offset), 0,
                                1, 1, 1, 8, 8))
    _set_mask_insn(tvm_ir, dtype, 128)


# pylint: disable=too-many-arguments
def _emit_copy_ubuf_to_gm(tvm_ir, dtype, dst, src, nburst, burstlen, srcstride,
                          dststride, dst_offset=0, src_offset=0):
    """
    emit_copy_ubuf_to_gm
    """
    tvm_ir.emit(tvm.call_extern(dtype, 'copy_ubuf_to_gm',
                                dst.access_ptr('w', offset=dst_offset),
                                src.access_ptr('r', offset=src_offset), 0,
                                nburst, burstlen, srcstride, dststride))


# pylint: disable=too-many-arguments
def _emit_copy_ubuf_to_gm_safely(tvm_ir, dtype, dst, src, count, tail_ub,
                                 dst_offset=0,
                                 src_offset=0):
    """
    emit_copy_ubuf_to_gm: count must equal or large than block_len
    """
    block_len = 16
    if dtype in ('float32', 'int32'):
        block_len = 8
    elif dtype in ('int8', 'uint8'):
        block_len = 32
    num_block = count // block_len
    tvm_ir.emit(tvm.call_extern(dtype, 'copy_ubuf_to_gm',
                                dst.access_ptr('w', offset=dst_offset),
                                src.access_ptr('r', offset=src_offset), 0,
                                1, num_block, 0, 0))
    if count % block_len != 0:
        for i in range(block_len):
            tvm_ir.emit(
                tvm.call_extern(
                    dtype, 'reg_mov', tail_ub.access_ptr('w', offset=i),
                    src.access_ptr('r',
                                   offset=src_offset + count - block_len + i)))
        tvm_ir.emit(tvm.call_extern(dtype, 'copy_ubuf_to_gm',
                                    dst.access_ptr(
                                        'w',
                                        offset=dst_offset + count - block_len),
                                    tail_ub.access_ptr('r'), 0, 1, 1, 0, 0))


# pylint: disable=too-many-arguments
def _emit_copy_gm_to_ubuf(tvm_ir, dtype, dst, src, nburst, burstlen, srcstride,
                          dststride, dst_offset=0, src_offset=0):
    """
    emit_copy_gm_to_ubuf
    """
    tvm_ir.emit(tvm.call_extern(dtype, 'copy_gm_to_ubuf',
                                dst.access_ptr('w', offset=dst_offset),
                                src.access_ptr('r', offset=src_offset), 0,
                                nburst, burstlen, srcstride, dststride))


def _save_padding_bottom(tvm_ir, tail_block_ub, dst, src, input_x, paddings,
                         block_index, batch_rows):
    """
    copy ubuf to gm, for multi-core process, need tail process
    """
    dtype = input_x.dtype
    shape = list(input_x.shape)
    cols = int(shape[-1])
    rows = int(shape[-2])
    rows_padding = int(shape[-2]) + paddings[-2][0] + paddings[-2][1]
    rows_padding_up = paddings[-2][0]
    rows_padding_bottom = paddings[-2][1]
    cols_padding = int(shape[-1]) + paddings[-1][0] + paddings[-1][1]
    cols_padding_left = int(paddings[-1][0])
    cols_padding_right = int(paddings[-1][1])
    block_offset = block_index*rows_padding*cols_padding
    block_len = 16
    if dtype in ('float32', 'int32'):
        block_len = 8
    elif dtype in ('int8', 'uint8'):
        block_len = 32
    cols_align = ((cols_padding + block_len - 1) // block_len)*block_len
    burstlen = (cols + cols_padding_right + block_len - 1) // block_len

    with tvm_ir.for_range(0, rows_padding_bottom - 1, name='i0') as i:
        bottom_offset = (i + rows_padding_up + rows)*cols_padding
        _emit_copy_ubuf_to_gm(
            tvm_ir, dtype, dst, src, 1, burstlen, 0, 0,
            dst_offset=block_offset + bottom_offset + cols_padding_left,
            src_offset=(i + batch_rows)*cols_align)

    last_nburst = burstlen - 1
    if last_nburst < 1:
        last_nburst = 1
    _emit_copy_ubuf_to_gm(
        tvm_ir, dtype, dst, src, 1, last_nburst, 0, 0,
        block_offset + (rows_padding - 1)*cols_padding + cols_padding_left,
        (rows_padding_bottom + batch_rows - 1)*cols_align)

    if burstlen > 1:
        for i in range(block_len):
            src_offset = (rows_padding_bottom + batch_rows - 1)*cols_align + \
                         cols + cols_padding_right - block_len + i
            tvm_ir.emit(
                tvm.call_extern(
                    dtype, 'reg_mov', tail_block_ub.access_ptr('w', offset=i),
                    src.access_ptr('r', offset=src_offset)))
        gm_offset = block_offset + rows_padding*cols_padding - block_len
        _emit_copy_ubuf_to_gm(tvm_ir, dtype, dst, tail_block_ub, 1, 1, 0, 0,
                              gm_offset)


def _get_batch_rows(rows_padding_bottom, cols_align, dtype):
    """
    return maximum num rows ub can load at once
    """
    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    padding_bytes = rows_padding_bottom*cols_align*dtype_size
    tempub_bytes = cols_align*dtype_size
    batch_rows = (ub_size_bytes - 32 - padding_bytes - tempub_bytes) // \
                 (cols_align*dtype_size)
    return batch_rows


def _load_and_save(tvm_ir, input_x, output_y, ub_addr, block_index, row_start,
                   num_rows, paddings):
    """
    load data from gm to ubuf and save to gm
    """
    dtype = input_x.dtype
    shape = list(input_x.shape)
    out_shape = output_y.shape
    block_len = 16
    if dtype in ('float32', 'int32'):
        block_len = 8
    elif dtype in ('int8', 'uint8'):
        block_len = 32
    rows = int(shape[-2])
    cols = int(shape[-1])
    rows_padding = int(out_shape[-2])
    cols_padding = int(out_shape[-1])
    rows_padding_up = int(paddings[-2][0])
    cols_padding_left = int(paddings[-1][0])
    cols_padding_right = int(paddings[-1][1])
    cols_align = ((cols_padding + block_len - 1) // block_len)*block_len
    block_offset = block_index*rows_padding*cols_padding
    # load data from gm to ubuf, every row is 32B aligned in ubuf
    zero_len = block_len - cols % block_len
    with tvm_ir.for_range(0, num_rows, name='i0') as i:
        burstlen = (cols + block_len - 1) // block_len
        _emit_copy_gm_to_ubuf(tvm_ir, dtype, ub_addr, input_x, 1,
                              burstlen,
                              0, 0,
                              i*cols_align,
                              block_index*rows*cols + (i + row_start)*cols)

        # last dim may not 32B aligned, so the data in block tail is not zero
        if cols % block_len != 0:
            _zero_ub_tail(tvm_ir, ub_addr,
                          i*cols_align + cols - (cols % block_len),
                          zero_len, dtype)

    burstlen = (cols + cols_padding_right + block_len - 1) // block_len
    with tvm_ir.for_range(0, num_rows, name='i0') as i:
        bottom_offset = (i + row_start + rows_padding_up)*cols_padding
        _emit_copy_ubuf_to_gm(
            tvm_ir, dtype, output_y, ub_addr, 1, burstlen, 0, 0,
            block_offset + bottom_offset + cols_padding_left,
            i*cols_align)
    return


def _pad_for_n_c_hw(ins, outs, paddings):
    """
    pad for (N1,N2,...,N3,H,W) shape
    N1~N3 axis do not has padding, H and W can have padding
    """
    tvm_ir = tvm.ir_builder.create()
    input_data = ins[0]
    output = outs[0]
    dtype = input_data.dtype
    shape = list(input_data.shape)
    out_shape = output.shape
    block_len = 16
    if dtype in ('float32', 'int32'):
        block_len = 8
    elif dtype in ('int8', 'uint8'):
        block_len = 32
    non_pad_nums = 1
    for i in range(len(shape) - 2):
        non_pad_nums *= int(shape[i])
    rows = int(shape[-2])
    cols = int(shape[-1])
    rows_padding = int(out_shape[-2])
    cols_padding = int(out_shape[-1])
    rows_padding_up = int(paddings[-2][0])
    rows_padding_bottom = int(paddings[-2][1])
    cols_padding_left = int(paddings[-1][0])
    cols_padding_right = int(paddings[-1][1])
    cols_align = ((cols_padding + block_len - 1) // block_len)*block_len

    batch_row = _get_batch_rows(rows_padding_bottom, cols_align, dtype)

    buflen = (rows_padding_bottom + batch_row)*cols_align
    out_ub = _apply_for_new_alloc(tvm_ir, dtype, buflen, block_len,
                                  scope=tbe_platform.scope_ubuf)
    temp_ub = _apply_for_new_alloc(tvm_ir, dtype, cols_align, block_len,
                                   scope=tbe_platform.scope_ubuf)
    tail_block_ub = _apply_for_new_alloc(tvm_ir, dtype, block_len, block_len,
                                         scope=tbe_platform.scope_ubuf)
    shape_temp_ub = (_ceil_fill(cols_align, block_len))
    _zero_ub(tvm_ir, temp_ub, shape_temp_ub, dtype)

    def mov_fun(block_offset, block_index):
        """
        function of move
        """
        # if last dim has left padding, zero the gm memeory at first
        if cols_padding_left > 0:
            num_block = (cols_padding_left + block_len - 1) // block_len
            with tvm_ir.for_range(0, rows_padding, name='i0') as i:
                _emit_copy_ubuf_to_gm(tvm_ir, dtype, output, temp_ub, 1,
                                      num_block, 0, 0,
                                      dst_offset=block_offset + i*cols_padding)

        # set padding up zero
        if rows_padding_up > 0:
            num_block = (cols + cols_padding_right + block_len - 1) //\
                        block_len
            with tvm_ir.for_range(0, rows_padding_up, name='i0') as i:
                _emit_copy_ubuf_to_gm(
                    tvm_ir, dtype, output, temp_ub, 1, num_block, 0, 0,
                    dst_offset=block_offset + i*cols_padding +
                    cols_padding_left)

        num_batch = rows // batch_row
        rows_remain = rows % batch_row

        with tvm_ir.for_range(0, num_batch, name='i0') as i:
            _load_and_save(tvm_ir, input_data, output, out_ub, block_index,
                           i*batch_row, batch_row, paddings)
        if rows_remain > 0:
            _load_and_save(tvm_ir, input_data, output, out_ub, block_index,
                           num_batch*batch_row, rows_remain, paddings)

        # save padding bottom
        if rows_padding_bottom > 0:
            _save_padding_bottom(tvm_ir, tail_block_ub, output, out_ub,
                                 input_data, paddings, block_index, batch_row)
    if cols_padding < block_len or non_pad_nums > 65536 or \
            rows_padding_bottom == 0:
        _zero_ub(tvm_ir, out_ub, buflen, dtype)
        with tvm_ir.for_range(0, non_pad_nums, name='i0') as index:
            _zero_ub(tvm_ir, temp_ub, cols_align, dtype)
            block_offset = index*rows_padding*cols_padding
            mov_fun(block_offset, index)
    else:
        blocks = non_pad_nums
        block_index = tvm.thread_axis("blockIdx.x")
        tvm_ir.scope_attr(block_index, "thread_extent", blocks)
        block_offset = block_index*rows_padding*cols_padding

        _zero_ub(tvm_ir, out_ub, buflen, dtype)
        mov_fun(block_offset, block_index)

    return tvm_ir.get()


def _pad_for_n_hw_c(ins, outs, paddings):
    """
    pad for (N1,N2,...,N3,H,W) shape
    N1~N3 axis do not has padding, H and W can have padding
    """
    tvm_ir = tvm.ir_builder.create()
    input_data = ins[0]
    output = outs[0]
    dtype = input_data.dtype
    shape = list(input_data.shape)
    out_shape = output.shape
    block_len = 16
    if dtype in ('float32', 'int32'):
        block_len = 8
    elif dtype in ('int8', 'uint8'):
        block_len = 32
    num = 1
    for i in range(len(shape) - 3):
        num = num*shape[i]

    ori_w = int(shape[-2])
    ori_h = int(shape[-3])
    w_padding = int(out_shape[-2])
    h_padding = int(out_shape[-3])
    w_padding_before = int(paddings[-2][0])
    w_padding_after = int(paddings[-2][1])
    h_padding_before = int(paddings[-3][0])
    h_padding_after = int(paddings[-3][1])
    out_c = int(out_shape[-1])
    c_align = ((out_c + block_len - 1) // block_len)*block_len

    buflen = (w_padding*c_align)
    out_ub = _apply_for_new_alloc(tvm_ir, dtype, buflen, block_len,
                                  scope=tbe_platform.scope_ubuf)
    tail_block_ub = _apply_for_new_alloc(tvm_ir, dtype, block_len, block_len,
                                         scope=tbe_platform.scope_ubuf)
    blocks = num
    block_index = tvm.thread_axis("blockIdx.x")
    tvm_ir.scope_attr(block_index, "thread_extent", blocks)
    block_offset_out = block_index*h_padding*w_padding*out_c
    block_offset_in = block_index*ori_h*ori_w*out_c

    # zero a ub buff with after-padding size
    _zero_ub(tvm_ir, out_ub, buflen, dtype)

    num_block = (w_padding*out_c + block_len - 1) // block_len
    if h_padding_before > 0:
        with tvm_ir.for_range(0, h_padding_before) as i:
            _emit_copy_ubuf_to_gm(tvm_ir, dtype, output, out_ub, 1, num_block,
                                  0, 0,
                                  dst_offset=block_offset_out +
                                  i*w_padding*out_c)

    data_burst_len = (ori_w*out_c + block_len - 1) // block_len
    zero_burst_len_before = (w_padding_before*out_c + block_len - 1) //\
                            block_len
    zero_burst_len_after = (w_padding_after*out_c + block_len - 1) //\
                           block_len

    with tvm_ir.for_range(0, ori_h, name='i0') as i:
        padding_h_offset = (h_padding_before + i)*w_padding*out_c
        _emit_copy_gm_to_ubuf(tvm_ir, dtype, out_ub, input_data, 1,
                              data_burst_len, 0, 0,
                              src_offset=block_offset_in + i*ori_w*out_c)
        if w_padding_before > 0:
            _emit_copy_ubuf_to_gm(
                tvm_ir, dtype, output, out_ub, 1, zero_burst_len_before, 0, 0,
                dst_offset=block_offset_out + padding_h_offset,
                src_offset=ori_w*c_align)
        _emit_copy_ubuf_to_gm(
            tvm_ir, dtype, output, out_ub, 1, data_burst_len, 0, 0,
            dst_offset=block_offset_out + padding_h_offset +
            w_padding_before*out_c)
        if w_padding_after > 0:
            _emit_copy_ubuf_to_gm(
                tvm_ir, dtype, output, out_ub, 1, zero_burst_len_after, 0, 0,
                dst_offset=block_offset_out + padding_h_offset +
                (ori_w + w_padding_before)*out_c,
                src_offset=ori_w*c_align)

    _zero_ub(tvm_ir, out_ub, buflen, dtype)
    h_offset = (h_padding_before + ori_h)*w_padding*out_c
    if h_padding_after > 1:
        with tvm_ir.for_range(0, h_padding_after - 1) as i:
            _emit_copy_ubuf_to_gm(
                tvm_ir, dtype, output, out_ub, 1, num_block,
                0, 0,
                dst_offset=block_offset_out + h_offset + i*w_padding*out_c)

    count = w_padding*out_c
    h2_offset = (h_padding_before + ori_h + h_padding_after - 1) *\
                w_padding * out_c
    _emit_copy_ubuf_to_gm_safely(
        tvm_ir, dtype, output, out_ub, count, tail_block_ub,
        dst_offset=block_offset_out + h2_offset)

    return tvm_ir.get()


def _check_align(shape, paddings, dtype, model):

    in_paddings= paddings.copy()
    in_shape = shape.copy()
    ou_shape = _get_output_shape(in_shape, in_paddings)
    axis = len(in_shape) - 1

    # only support fp16 and fp32
    if dtype == "float16":
        num_bit = 2
    else:
        num_bit = 4
    while axis >= 0:
        in_num = in_paddings[axis][model] * _prod(ou_shape[axis+1:])
        if in_num > 0:
            if in_num * num_bit % 32 != 0 or \
                    _prod(in_shape[axis:]) * num_bit % 32 != 0:
                return False
            else:
                return True
            break
        axis -= 1

    return True


def _check_optimization_nhwc(input_x, paddings):
    """
    check whether the shape and padding mode are able to optimize
    """
    shape = list(input_x.get("shape"))
    dtype = input_x.get("dtype")
    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    if len(shape) <= 2:
        return False
    h_padding_after = paddings[-3][1]
    w_padding = shape[-2] + paddings[-2][0] + paddings[-2][1]
    ori_c = shape[-1]
    block_len = 16
    if dtype in ('float32', 'int32'):
        block_len = 8
    elif dtype in ('int8', 'uint8'):
        block_len = 32
    c_align = ((ori_c + block_len - 1) // block_len) * block_len

    if h_padding_after <= 0:
        return False
    if paddings[-1][0] != 0 or paddings[-1][1] != 0:
        return False

    if h_padding_after*w_padding*ori_c*dtype_size < 32:
        return False
    if dtype in ('uint8', 'int8'):
        return False
    if w_padding*c_align*dtype_size + 32 > ub_size_bytes:
        return False
    num = 1
    # max core num cannot exceed 65535
    for i in range(len(shape) - 3):
        num = num*shape[i]
        if num > 65535:
            return False
        if paddings[i][0] != 0:
            return False
        if paddings[i][1] != 0:
            return False
    if paddings[-1][0] == 0 and paddings[-1][1] == 0 and\
            paddings[-2][0] == 0 and paddings[-2][1] == 0 and\
            shape[-1] * shape[-2] * dtype_size < 32:
        return False

    # when -1 dim and -2 dim are too large to save in ub
    # choose True
    ou_shape = _get_output_shape(shape, paddings)
    if dtype == "float16":
        ub_maxsize = tbe_platform. \
                         cce_conf. \
                         get_soc_spec(tbe_platform.
                                      cce_conf.UB_SIZE) // 32 * 16
    else:
        ub_maxsize = tbe_platform. \
                         cce_conf. \
                         get_soc_spec(tbe_platform.
                                      cce_conf.UB_SIZE) // 32 * 8
    if _prod(shape[-2:]) + _prod(ou_shape[-2:]) > ub_maxsize:
        return True

    # when shape is 32B align, choose False firstly
    if _check_align(shape, paddings, dtype, 0) and \
            _check_align(shape, paddings, dtype, 1):
        return False

    return True


def _check_optimization_nchw(input_x, paddings):
    """
    check whether the shape and padding mode are able to optimize
    """
    shape = list(input_x.get("shape"))
    dtype = input_x.get("dtype")
    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
    ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    if len(shape) < 2:
        return False
    rows_padding_bottom = paddings[-2][1]
    cols_padding = shape[-1] + paddings[-1][0] + paddings[-1][1]

    if paddings[-1][0] == 0 and paddings[-1][1] == 0:
        return False

    if dtype in ('uint8', 'int8'):
        return False
    if (rows_padding_bottom + 1)*cols_padding*dtype_size + 32 > ub_size_bytes:
        return False
    num = 1
    # max core num cannot exceed 65535
    for i in range(len(shape) - 2):
        num = num*shape[i]
        if paddings[i][0] != 0:
            return False
        if paddings[i][1] != 0:
            return False

    if _check_align(shape, paddings, dtype, 0) and \
            _check_align(shape, paddings, dtype, 1):
        return False

    return True


def fused_not_padding_axis(shape, paddings, padd_axis):
    """
    real_input_shape: fuse axis which not in padd_axis
    real_padd_axis: new pad axis after fused
    real_paddings: new padding after fused
    """
    real_input_shape = []
    real_padd_axis = []
    real_paddings = []
    fused_axis_value = 1
    new_axis = 0
    for index, value in enumerate(shape):
        if index in padd_axis:
            if fused_axis_value != 1:
                real_input_shape.append(fused_axis_value)
                real_paddings.append([0, 0])
                new_axis += 1
            real_input_shape.append(value)
            real_padd_axis.append(new_axis)
            real_paddings.append(paddings[index])
            new_axis += 1
            fused_axis_value = 1
        else:
            fused_axis_value *= value
    if fused_axis_value != 1:
        real_input_shape.append(fused_axis_value)
        real_paddings.append([0, 0])

    return real_input_shape, real_paddings, real_padd_axis


def _pattern_align(shape, paddings, dtype):

    # not 32B align
    in_shape = shape.copy()
    in_paddings = paddings.copy()
    ou_shape = _get_output_shape(in_shape, in_paddings)

    if dtype not in ['float16', 'float32']:
        return False, in_shape, in_paddings

    # not padding, dtype is fp16 or fp32, in this branch
    if ou_shape == in_shape:
        return True, in_shape, in_paddings

    if not (_check_align(in_shape, in_paddings, dtype, 0) and
            _check_align(in_shape, in_paddings, dtype, 1)):
        return False, in_shape, in_paddings

    if dtype == "float16":
        ub_maxsize = tbe_platform.\
                         cce_conf.\
                         get_soc_spec(tbe_platform.
                                      cce_conf.UB_SIZE) // 32 * 16
    else:
        ub_maxsize = tbe_platform.\
                         cce_conf.\
                         get_soc_spec(tbe_platform.
                                      cce_conf.UB_SIZE) // 32 * 8

    if len(in_shape) > 2:
        padd_axis = []
        for index, value in enumerate(paddings):
            if value not in [[0, 0], (0, 0)]:
                padd_axis.append(index)

        # fused [0, 0] and [0, 0]
        # eg: [11,11,11], [[1,1],[0,0],[0,0]] -> [11,121], [[1,1],[0,0]]
        in_shape, in_paddings, \
        padd_axis = fused_not_padding_axis(in_shape, in_paddings, padd_axis)
        ou_shape = _get_output_shape(in_shape, in_paddings)

    if len(in_shape) < 2:
        return False, in_shape, in_paddings

    if ou_shape[-1] + in_shape[-1] > ub_maxsize:
        return False, in_shape, in_paddings

    return True, in_shape, in_paddings


# pylint: disable=locally-disabled,too-many-arguments,too-many-branches,
# pylint: disable=locally-disabled,too-many-statements
@util.check_input_type(dict, dict, (list, tuple), str, (float, int), str)
def pad_d(input_x, output_x, paddings, kernel_name="pad_d"):
    """ calculating pad tensor by paddings parameters

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        shape and dtype of output
    paddings: list or tuple.
        For each dimension D of input, paddings[D, 0] indicates how many
        values to add
        before the contents of tensor in that dimension, and paddings[D, 1]
        indicates
        how many values to add after the contents of tensor in that dimension.
    kernel_name : str
        cce kernel name, default value is "pad_d"

    Returns
    -------
    None.
    """
    shape = list(input_x.get("shape"))
    paddings = list(paddings)
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)

    if len(paddings) is not len(shape):
        raise RuntimeError(
            "Paddings and shape are not the same length.")
    for padding in paddings:
        if len(padding) != 2:
            raise RuntimeError("Paddings's shape is not in the form of (n,2)")
        if (not isinstance(padding[0], int)) or (not isinstance(
                padding[1], int)):
            raise RuntimeError("Paddings only suppot int")

    check_list_dtype = ("float16", "float32", "int32", "int8", "uint8")
    dtype = input_x.get("dtype").lower()
    util.check_dtype_rule(dtype, check_list_dtype)
    util.check_kernel_name(kernel_name)
    data = tvm.placeholder(shape, name="data", dtype=dtype)
    if dtype == "int8" or dtype == "uint8":
        dtype_now = "float16"
    else:
        dtype_now = dtype
    dtype_size = tbe_platform.cce_intrin.get_bit_len(dtype_now) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // dtype_size
    one_core_align = _ceil_fill(shape[len(shape) - 1] + cp_align_len,
                                cp_align_len)

    if _check_optimization_nchw(input_x, paddings):
        res = tvm.extern([_get_output_shape(shape, paddings)], [data],
                         lambda ins, outs: _pad_for_n_c_hw(ins, outs,
                                                           paddings),
                         name="res", dtype=dtype)
        sch = tvm.create_schedule(res.op)
        build_list = [data, res]
        with build_config:
            tvm.build(sch, build_list, "cce", name=kernel_name)

    elif _check_optimization_nhwc(input_x, paddings):
        res = tvm.extern([_get_output_shape(shape, paddings)], [data],
                         lambda ins, outs: _pad_for_n_hw_c(ins, outs,
                                                           paddings),
                         name="res", dtype=dtype)
        sch = tvm.create_schedule(res.op)
        build_list = [data, res]
        with build_config:
            tvm.build(sch, build_list, "cce", name=kernel_name)

    else:
        in_shape = []
        in_paddings = []
        for index, value in enumerate(shape):
            if value != 1 or (paddings[index] != [0, 0]
                              and paddings[index] != (0, 0)):
                in_shape.append(value)
                in_paddings.append(paddings[index])

        option, new_shape, new_paddings = _pattern_align(in_shape, in_paddings, dtype)

        if option:
            res = pad_align(new_shape, new_paddings, dtype, kernel_name)
            return res
        else:
            res = \
                tvm.extern(
                    [_get_output_shape(in_shape, in_paddings), shape, [one_core_align],
                     _get_output_shape(in_shape, in_paddings)], [data],
                    lambda ins, outs: _intrin_factor(ins[0], outs[0], outs[1],
                                                     outs[2], outs[3],
                                                     (in_shape, in_paddings,
                                                      0)),
                    name="res", dtype=[dtype, dtype_now, dtype_now, dtype_now])

            sch = tvm.create_schedule(res[0].op)
            build_list = [data, res[0], res[1], res[2], res[3]]

            with build_config:
                tvm.build(sch, build_list, "cce", name=kernel_name)

    if dtype == "int8" or dtype == "uint8":
        size_align = one_core_align * dtype_size + 32
        in_shape_size = _prod(shape[:])
        size_in_cast = in_shape_size * dtype_size + 32
        size_out_cast = \
            _prod(_get_output_shape(shape, paddings)) * dtype_size + 32
        total_size = [size_in_cast, size_align, size_out_cast]
        num_workspace = 3
        workspace_dict = \
            {"workspace": {"num": num_workspace, "size": total_size}}
        _write_code(workspace_dict, "kernel_meta/" + kernel_name + ".json")
    else:
        size_align = one_core_align * dtype_size + 32
        total_size = [size_align]
        num_workspace = 1
        workspace_dict = \
            {"workspace": {"num": num_workspace, "size": total_size}}
        _write_code(workspace_dict, "kernel_meta/" + kernel_name + ".json")
