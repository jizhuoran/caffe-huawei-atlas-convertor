#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

embed
"""

from te import tvm
import te.platform.cce_params as param
import te.platform.cce_intrin as intrin
from te.platform import cce_util as cce_util

def allocate_UB(ib, dtype, size, name):
    buf_var = ib.allocate(dtype, (size,), name, scope="local.UB")
    return tvm.decl_buffer((size,), dtype, name, scope="local.UB", data=buf_var)


def emit_copy_gm_ubuf(ib, cmd, dtype, size, dst, dst_offset, src, src_offset, sync_id):
    """
    emit copy_gm_to_ubuf, or emit copy_ubuf_to_gm.

    ib : instance of ir_builder

    cmd : string
        commond type, copy_gm_to_ubuf or copy_ubuf_to_gm
    """
    nBurst = size.value // (2**16*32)
    tail = size.value % (2**16*32)
    burst_ele_num = 256 // intrin.get_bit_len(dtype)
    sid = 0
    if cmd == "copy_gm_to_ubuf":
        sid = cce_util.get_dma_sid("Sid_copy_gm_to_ubuf")

    with ib.new_scope():
        ib.scope_attr(param.CCE_AXIS, "coproc_scope", sync_id)
        if nBurst:
            ib.emit(tvm.call_extern(
                dtype, cmd,
                dst.access_ptr("w", offset=dst_offset),
                src.access_ptr("rw", offset=src_offset),
                sid, nBurst, 0xFFFF, 0, 0))
        if tail > 0:
            dst_offset = dst_offset + nBurst*(2**16*32)
            src_offset = src_offset + nBurst*(2**16*32)
            lenBurst = (tail + burst_ele_num - 1) // burst_ele_num
            ib.emit(tvm.call_extern(
                dtype, cmd,
                dst.access_ptr("w", offset=dst_offset),
                src.access_ptr("rw", offset=src_offset),
                sid, 1, lenBurst, 0, 0))


def emit_vadd(ib, dtype, size, dst, src0, src1):
    """
    emit vadd cmd.

    ib : instance of ir_builder

    """
    dst_stridem0 = 1
    src0_stridem0 = 1
    src1_stridem0 = 1
    dst_stridem1 = 8
    src0_stridem1 = 8
    src1_stridem1 = 8
    total_pipe_line = 11
    base_pipe_line = 2

    local_total_len = size.value

    repeat_times = local_total_len // 128
    remain_len = local_total_len - repeat_times*128
    if repeat_times > 0:
        local_repeat_times = repeat_times
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 2)
            repeat_dst_offset = 0
            repeat_src_offset = 0
            while local_repeat_times > 0:
                if local_repeat_times > 255:
                    tmp_repeat_times = 255
                else:
                    tmp_repeat_times = local_repeat_times
                ib.emit(tvm.call_extern(
                    dtype, "vadd",
                    dst.access_ptr("w", offset=repeat_dst_offset),
                    src0.access_ptr("r", offset=repeat_src_offset),
                    src1.access_ptr("r", offset=repeat_src_offset),
                    tmp_repeat_times,
                    dst_stridem0, src0_stridem0, src1_stridem0,
                    dst_stridem1, src0_stridem1, src1_stridem1))
                repeat_dst_offset += 2*tmp_repeat_times
                repeat_src_offset += 2*tmp_repeat_times
                local_repeat_times -= 255
    if remain_len > 0:
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", base_pipe_line + total_pipe_line*1)
            mask1, mask2 = intrin.set_mask(remain_len)
            ib.emit(tvm.call_extern(
                dtype, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", base_pipe_line + total_pipe_line*2)
            ib.emit(tvm.call_extern(
                dtype, "vadd",
                dst.access_ptr("w", offset=repeat_times*128),
                src0.access_ptr("r", offset=repeat_times*128),
                src1.access_ptr("r", offset=repeat_times*128),
                1,
                dst_stridem0, src0_stridem0, src1_stridem0,
                dst_stridem1, src0_stridem1, src1_stridem1))
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", base_pipe_line + total_pipe_line*3)
            mask1, mask2 = intrin.set_mask(128)
            ib.emit(tvm.call_extern(
                dtype, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))


def emit_vconv_f162s32r(ib, dtype, size, dst, src):
    """
    emit vadd cmd.

    ib : instance of ir_builder

    """
    dst_stridem0 = 1
    src_stridem0 = 1
    dst_stridem1 = 8
    src_stridem1 = 8
    total_pipe_line = 11
    base_pipe_line = 2

    local_total_len = size.value

    repeat_times = local_total_len // 128
    remain_len = local_total_len - repeat_times*128
    if repeat_times > 0:
        local_repeat_times = repeat_times
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", base_pipe_line + total_pipe_line*4)
            repeat_dst_offset = 0
            repeat_src_offset = 0
            while local_repeat_times > 0:
                if local_repeat_times > 255:
                    tmp_repeat_times = 255
                else:
                    tmp_repeat_times = local_repeat_times
                ib.emit(tvm.call_extern(
                    dtype, "vconv_f162s32r",
                    dst.access_ptr("w", offset=repeat_dst_offset),
                    src.access_ptr("r", offset=repeat_src_offset),
                    tmp_repeat_times,
                    dst_stridem0, src_stridem0,
                    dst_stridem1, src_stridem1))
                repeat_dst_offset += 4*tmp_repeat_times
                repeat_src_offset += 2*tmp_repeat_times
                local_repeat_times -= 255
    if remain_len > 0:
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", base_pipe_line + total_pipe_line*5)
            mask1, mask2 = intrin.set_mask(remain_len)
            ib.emit(tvm.call_extern(
                dtype, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", base_pipe_line + total_pipe_line*6)
            ib.emit(tvm.call_extern(
                dtype, "vconv_f162s32r",
                dst.access_ptr("w", offset=repeat_times*4*128),
                src.access_ptr("r", offset=repeat_times*2*128),
                1,
                dst_stridem0, src_stridem0,
                dst_stridem1, src_stridem1))
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", base_pipe_line + total_pipe_line*7)
            mask1, mask2 = intrin.set_mask(128)
            ib.emit(tvm.call_extern(
                dtype, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
                tvm.const(mask2, dtype="uint64")))


def embed_ir(input_tensor, weight, bias, bias_term, output):
    ib = tvm.ir_builder.create()

    _, n = weight.shape
    m = input_tensor.shape[0]

    # load input_tensor
    input_ub = allocate_UB(ib, input_tensor.dtype, m, "input_ub")
    emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", input_tensor.dtype, m, input_ub, 0, input_tensor, 0, 5)
    if input_tensor.dtype == "float16":
        input_ub_int32 = allocate_UB(ib, 'int32', m, "input_ub_int32")
        emit_vconv_f162s32r(ib, input_tensor.dtype, m, input_ub_int32, input_ub)
    else:
        input_ub_int32 = input_ub

    # load bias
    if bias_term:
        bias_ub = allocate_UB(ib, bias.dtype, n, "bias_ub")
        emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", bias.dtype, n, bias_ub, 0, bias, 0, 5)

    # load weight, then compute output
    weight_ub = allocate_UB(ib, weight.dtype, n, "weight_ub")
    reg = ib.allocate("int32", (1,), name="reg_buf", scope=param.scope_reg)
    with ib.for_range(0, m, name="i") as i:
        with ib.new_scope():
            ib.scope_attr(param.CCE_AXIS, "coproc_scope", 1)
            ib.emit(tvm.call_extern("int32", "reg_mov",
                                    tvm.call_extern(reg.dtype, "reg", reg[0]),
                                    input_ub_int32.access_ptr("rw", offset=i)))
        emit_copy_gm_ubuf(ib, "copy_gm_to_ubuf", weight.dtype, n, weight_ub, 0, weight, reg[0]*n,
                          5)
        if bias_term:
            emit_vadd(ib, weight.dtype, n, weight_ub, weight_ub, bias_ub)
        emit_copy_gm_ubuf(ib, 'copy_ubuf_to_gm', output.dtype, n, output, i*n, weight_ub, 0, 6)
    body = ib.get()
    return body


@tvm.tag_scope(tag="embed")
def compute_embed_cce(input_tensor, weight, bias, bias_term):
    _, n = weight.shape
    m = input_tensor.shape[0]
    output = tvm.extern([(m, n)], [input_tensor, weight, bias],
                        lambda ins, outs: embed_ir(ins[0], ins[1], ins[2], bias_term, outs[0]),
                        dtype=[weight.dtype], name="output")
    return output
