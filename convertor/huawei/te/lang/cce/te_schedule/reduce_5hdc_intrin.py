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

reduce 5hd c axis intrinsic functions
"""
import math
import te

from te import tvm

from te.platform.cce_emitinsn_params import cceEmitParamsIns
from te.platform.cce_util import get_buffer, get_align_factor
from te.platform.cce_intrin_md import reset_mask_insn


INTRIN_MAPPING_GROUP = {
    "max": "vcgmax",
    "min": "vcgmin",
    "sum": "vcgadd"
}

INTRIN_MAPPING_NORMAL = {
    "max": "vmax",
    "min": "vmin",
    "sum": "vadd"
}

DEBUG_SWITCH = True


@tvm.register_func("tvm.intrin.cce.5hdc_reduce_min")
def reduce_5hdc_reduce_min(stmt):
    """5HDC reduce for reduce_min"""
    return reduce_5hdc(stmt, "min")


@tvm.register_func("tvm.intrin.cce.5hdc_reduce_max")
def reduce_5hdc_reduce_max(stmt):
    """5HDC reduce for reduce_min"""
    return reduce_5hdc(stmt, "max")


@tvm.register_func("tvm.intrin.cce.5hdc_reduce_sum")
def reduce_5hdc_reduce_sum(stmt):
    """5HDC reduce for reduce_min"""
    return reduce_5hdc(stmt, "sum")


def reduce_5hdc(stmt, intrin_cmd):  # pylint: disable=R0914, R0915
    """5HDC reduce"""
    ir_builder = tvm.ir_builder.create()
    c_var = cceEmitParamsIns.get_param("CAxisVar")
    c_inv_size = cceEmitParamsIns.get_param("CAxisInvalidSize")
    if DEBUG_SWITCH:
        print("[Reduce5HDCIntrin]", "Var:", c_var, "InvSize:", c_inv_size)
    # Get original shape and target shape
    original_shape = []
    original_layout = []
    target_shape = []
    target_layout = []
    loop_var_dict = {}

    def store_ori(_var):
        if not isinstance(_var, (tvm.expr.Var, tvm.expr.IntImm)):
            store_ori(_var.a)
            store_ori(_var.b)
            return
        if isinstance(_var, tvm.expr.Var):
            if _var.name in original_layout:
                last_index = original_layout.index(_var.name)
                del original_shape[last_index]
                del original_layout[last_index]
            original_shape.append(loop_var_dict[str(_var.name)])
            original_layout.append(_var.name)
            return
        if isinstance(_var, tvm.expr.IntImm):
            return
        raise RuntimeError("Backend Error: Received unexpected statement: " + str(type(_var)))

    def store_tgt(_var):
        if not isinstance(_var, (tvm.expr.Var, tvm.expr.IntImm)):
            store_tgt(_var.a)
            store_tgt(_var.b)
            return
        if isinstance(_var, tvm.expr.Var):
            if str(_var.name) in loop_var_dict:
                if _var.name in original_layout:
                    last_index = target_layout.index(_var.name)
                    del target_shape[last_index]
                    del target_layout[last_index]
                target_shape.append(loop_var_dict[str(_var.name)])
                target_layout.append(_var.name)
            return
        if isinstance(_var, tvm.expr.IntImm):
            return
        raise RuntimeError("Backend Error: Received unexpected statement: " + str(type(_var)))

    def interpret_statement(_stmt):
        if isinstance(_stmt, tvm.stmt.Store):
            store_tgt(_stmt.index)
            return
        if isinstance(_stmt, tvm.expr.Load):
            store_ori(_stmt.index)
            return
        if isinstance(_stmt, tvm.stmt.For):
            loop_var_dict[str(_stmt.loop_var)] = int(_stmt.extent)
            return
        raise RuntimeError("Backend Error: Received unexpected statement: " + str(type(_stmt)))

    def list_product(list_slice):
        result = 1
        for i in list_slice:
            result *= i
        return int(result)

    tvm.ir_pass.IRTransform(stmt, None, interpret_statement, ["For"])
    tvm.ir_pass.IRTransform(stmt, None, interpret_statement, ["Load", "Store"])
    if DEBUG_SWITCH:
        print("[Reduce5HDCIntrin]", "Input:", original_shape, "InputVar:", original_layout)
        print("[Reduce5HDCIntrin]", "Input:", target_shape, "InputVar:", target_layout)
    reduce_schedule = []
    reduce_src = 1
    offset = 0
    # reduce_src reduce_unit reduce_factor
    mid_clean_enabled = False
    for idx, var in enumerate(original_layout):
        if idx - offset < len(target_layout) and var == target_layout[idx - offset]:
            reduce_src *= original_shape[idx]
        else:
            # Mid clean
            is_c1 = False
            if str(var) == str(c_var):
                is_c1 = True
                mid_clean_enabled = True
            reduce_schedule.append((reduce_src,
                                    list_product(original_shape[idx + 1:]),
                                    original_shape[idx],
                                    is_c1,
                                    c_inv_size))
            reduce_src = math.ceil(reduce_src / original_shape[idx])
            offset += 1
    # Prepare buffer
    ins, outs = get_buffer(stmt, need_origin_adress=True)
    input_buffer = ins[1]
    output_buffer = outs[0]
    if DEBUG_SWITCH:
        print("[Reduce5HDCIntrin]", "InputBuffers:", ins, "OutputBuffers:", outs)
        print("[Reduce5HDCIntrin]", "ReduceSchedule", reduce_schedule)
    for reduce_sch in reduce_schedule[:-1]:
        do_reduce(ir_builder, intrin_cmd, reduce_sch, input_buffer)
    do_reduce(ir_builder, intrin_cmd, reduce_schedule[-1],
              input_buffer, output_buffer, mid_clean_enabled)
    return ir_builder.get()


def do_reduce(ir_builder, intrin_cmd, sch,  # pylint: disable=R0913
              input_buffer, output_buffer=None, mid_clean_enabled=False):
    """Distribute reduce algorithm by reduce_sch"""
    if output_buffer is None:
        output_buffer = input_buffer
    reduce_src, reduce_unit, reduce_factor, is_c1, c1_inv_size = sch
    if reduce_unit == 1:
        last_axis_reduce(ir_builder, intrin_cmd, reduce_src,
                         reduce_factor, input_buffer,
                         output_buffer, not mid_clean_enabled, c1_inv_size)
    else:
        mid_axis_reduce(ir_builder, intrin_cmd, reduce_src,
                        reduce_unit, reduce_factor, input_buffer,
                        output_buffer, is_c1, c1_inv_size)


def last_axis_reduce(ir_builder, intrin_cmd, reduce_src, reduce_factor,  # pylint: disable=R0913
                     input_buffer, output_buffer, need_clean, clean_factor):
    """last axis reduce"""
    if reduce_factor != 16:
        raise RuntimeError("Last axis reduce currently doesn't support unaligned reduce")
    if DEBUG_SWITCH:
        print("[LastAxisReduce] Need clean:", need_clean)
    element_size = get_align_factor(input_buffer.dtype)[1]
    element_per_block = 32 // element_size
    vector_insn_rep_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // element_size
    block_per_vector_insn = vector_insn_rep_size // element_per_block
    repeat_times = reduce_src * reduce_factor // vector_insn_rep_size
    remains = reduce_src * reduce_factor - vector_insn_rep_size * repeat_times
    if need_clean and clean_factor < 16:
        vector_insn_factor_clean(ir_builder, intrin_cmd, input_buffer,
                                 repeat_times, remains, 0, clean_factor)
    if input_buffer.dtype == "float16":
        # Use vcg
        vector_insn_factory_vcg(ir_builder, INTRIN_MAPPING_GROUP[intrin_cmd], output_buffer,
                                input_buffer, repeat_times, block_per_vector_insn,
                                vector_insn_rep_size, remains, 0, 0)
    elif element_size == 4:
        # Use normal
        raise RuntimeError("Last axis reduce does not support fp32 or int32")


def mid_axis_reduce(ir_builder, intrin_cmd, reduce_src,  # pylint: disable=R0913, R0914
                    reduce_unit, reduce_factor,
                    input_buffer, output_buffer, need_clean, clean_factor):
    """non-last axis reduce"""
    if reduce_unit % 16 != 0:
        raise RuntimeError("Mid axis reduce currently doesn't support unaligned reduce")
    element_size = get_align_factor(input_buffer.dtype)[1]
    vector_insn_rep_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // element_size
    repeat_times = reduce_unit // vector_insn_rep_size
    remains = reduce_unit - vector_insn_rep_size * repeat_times
    with ir_builder.for_range(0, reduce_src, name="loop_reduce_src") as loop_src:
        src_offset = loop_src * reduce_factor * reduce_unit
        dst_offset = loop_src * reduce_unit
        with ir_builder.for_range(0, reduce_factor - 1, name="loop_reduce_factor") as loop_factor:
            src_offset = src_offset + (loop_factor + 1) * reduce_unit
            if need_clean and clean_factor < 16:
                with ir_builder.if_scope(loop_factor == reduce_factor - 2):
                    vector_insn_factor_clean(ir_builder, intrin_cmd, input_buffer,
                                             repeat_times, remains, src_offset, clean_factor)
            vector_insn_factory_normal(ir_builder, INTRIN_MAPPING_NORMAL[intrin_cmd],
                                       output_buffer, input_buffer, repeat_times,
                                       vector_insn_rep_size, vector_insn_rep_size,
                                       remains, dst_offset, src_offset)


def vector_insn_factory_vcg(ir_b, cmd, dst_buffer, src_buffer,  # pylint: disable=R0913
                            repeat, dst_stride, src_stride, rem, dst_offset, src_offset):
    """Generate vcgxxx intrin, factory function"""
    reset_mask_insn(ir_b, dst_buffer.dtype)
    if repeat > 255:
        outer_repeat_times = repeat // 255
        remain_repeat_times = repeat % 255
        for outer_repeat in range(outer_repeat_times):
            ir_b.emit(tvm.call_extern(
                dst_buffer.dtype,
                cmd,
                dst_buffer.access_ptr("rw", offset=outer_repeat * 255 * dst_stride + dst_offset),
                src_buffer.access_ptr("r", offset=outer_repeat * 255 * src_stride + src_offset),
                255, 1, 1, 8))
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=outer_repeat_times * 255 * dst_stride + dst_offset),
            src_buffer.access_ptr("r", offset=outer_repeat_times * 255 * src_stride + src_offset),
            remain_repeat_times, 1, 1, 8))
    else:
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=0 + dst_offset),
            src_buffer.access_ptr("r", offset=0 + src_offset),
            repeat, 1, 1, 8))
    # Remain part
    if rem > 0:
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=repeat * dst_stride + dst_offset),
            src_buffer.access_ptr("r", offset=repeat * src_stride + src_offset),
            1, 1, 1, 8))
    reset_mask_insn(ir_b, dst_buffer.dtype)


def vector_insn_factory_normal(ir_b, cmd, dst_buffer, src_buffer,  # pylint: disable=R0913
                               repeat, dst_stride, src_stride, rem, dst_offset, src_offset):
    """Generate normal vector intrin, factory function"""
    reset_mask_insn(ir_b, dst_buffer.dtype)
    if repeat > 255:
        outer_repeat_times = repeat // 255
        remain_repeat_times = repeat % 255
        for outer_repeat in range(outer_repeat_times):
            ir_b.emit(tvm.call_extern(
                dst_buffer.dtype,
                cmd,
                dst_buffer.access_ptr("rw", offset=outer_repeat * 255 * dst_stride + dst_offset),
                dst_buffer.access_ptr("rw", offset=outer_repeat * 255 * dst_stride + dst_offset),
                src_buffer.access_ptr("r", offset=outer_repeat * 255 * src_stride + src_offset),
                255, 1, 1, 1, 8, 8, 8))
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=outer_repeat_times * 255 * dst_stride + dst_offset),
            dst_buffer.access_ptr("rw", offset=outer_repeat_times * 255 * dst_stride + dst_offset),
            src_buffer.access_ptr("r", offset=outer_repeat_times * 255 * src_stride + src_offset),
            remain_repeat_times, 1, 1, 1, 8, 8, 8))
    else:
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=0 + dst_offset),
            dst_buffer.access_ptr("rw", offset=0 + dst_offset),
            src_buffer.access_ptr("r", offset=0 + src_offset),
            repeat, 1, 1, 1, 8, 8, 8))
    # Remain part
    if rem > 0:
        ir_b.emit(tvm.call_extern(
            dst_buffer.dtype,
            cmd,
            dst_buffer.access_ptr("rw", offset=repeat * dst_stride + dst_offset),
            dst_buffer.access_ptr("rw", offset=repeat * dst_stride + dst_offset),
            src_buffer.access_ptr("r", offset=repeat * src_stride + src_offset),
            1, 1, 1, 1, 8, 8, 8))
    reset_mask_insn(ir_b, dst_buffer.dtype)


def vector_insn_factor_clean(ir_builder, cmd, src_buffer,  # pylint: disable=R0913, R0914, R0912
                             repeat_times, remains, src_offset, clean_factor):
    """Repair 5HD format, for vcgmax, replace 0 with fp16_min, for vcgadd, replace with 0"""
    num_fill_dict = {
        "sum": {
            "float16": 0,
            "float32": 0,
            "int32": 0
        },
        "min": {
            "float16": 65504,
            "float32": (2 - 2 ** (-23)) * (2 ** 127),
            "int32": 2 ** 31 - 1
        },
        "max": {
            "float16": -65504,
            "float32": -(2 - 2 ** (-23)) * (2 ** 127),
            "int32": -2 ** 31 - 1
        },
    }
    element_size = get_align_factor(src_buffer.dtype)[1]
    element_per_block = 32 // element_size
    vector_insn_rep_size = te.platform.cce_params.VECTOR_INST_BLOCK_WIDTH // element_size
    block_per_vector_insn = vector_insn_rep_size // element_per_block
    mask_num = 16 - clean_factor
    if cmd in num_fill_dict:
        num_fill = num_fill_dict[cmd][str(src_buffer.dtype)]
    else:
        raise RuntimeError("Unsupported cmd: " + str(cmd))
    mask_unit = (16 - mask_num) * "1" + "0" * mask_num
    mask = int(mask_unit * (block_per_vector_insn // 2), 2)
    ir_builder.emit(tvm.call_extern(
        src_buffer.dtype,
        "set_vector_mask",
        tvm.const(mask, dtype="uint64"),
        tvm.const(mask, dtype="uint64")))
    if repeat_times > 255:
        outer_repeat_times = repeat_times // 255
        remain_repeat_times = repeat_times % 255
        for outer_repeat in range(outer_repeat_times):
            ir_builder.emit(tvm.call_extern(
                src_buffer.dtype,
                "vector_dup",
                src_buffer.access_ptr("rw", offset=outer_repeat * 255
                                      * vector_insn_rep_size + src_offset),
                num_fill,
                255, 1, 1, 8, 8))
        ir_builder.emit(tvm.call_extern(
            src_buffer.dtype,
            "vector_dup",
            src_buffer.access_ptr("rw", offset=outer_repeat_times * 255
                                  * vector_insn_rep_size + src_offset),
            num_fill,
            remain_repeat_times, 1, 1, 8, 8))
    else:
        ir_builder.emit(tvm.call_extern(
            src_buffer.dtype,
            "vector_dup",
            src_buffer.access_ptr("rw", offset=src_offset),
            num_fill,
            repeat_times, 1, 1, 8, 8))
    # Remain part
    if remains > 0:
        remain_block_num = remains // element_per_block
        if remain_block_num > block_per_vector_insn // 2:
            higher_mask = int((remain_block_num - block_per_vector_insn // 2) * mask_unit, 2)
            lower_mask = mask
        else:
            higher_mask = 0
            lower_mask = int(remain_block_num * mask_unit, 2)
        ir_builder.emit(tvm.call_extern(
            src_buffer.dtype,
            "set_vector_mask",
            tvm.const(higher_mask, dtype="uint64"),
            tvm.const(lower_mask, dtype="uint64")))
        ir_builder.emit(tvm.call_extern(
            src_buffer.dtype,
            "vector_dup",
            src_buffer.access_ptr("rw", offset=repeat_times * vector_insn_rep_size + src_offset),
            num_fill,
            1, 1, 1, 8, 8))
