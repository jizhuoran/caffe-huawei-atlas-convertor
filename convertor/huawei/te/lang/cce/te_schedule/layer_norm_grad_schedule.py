#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=too-many-lines
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

batch_normalization_forward_training_reduce
"""
# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
import math
from functools import reduce

import te.lang.cce
from te import tvm
import te.platform.cce_params as cce
from te import platform as cceconf
from te.platform import cce_util

MAX_SHAPE_NUM = 10000000

DTYPE_WIDTH_MAP = {"float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}


def _reset_mask_insn(ib_expr, type_, bits=128, mask_func=None):
    """
    :describe: caculate the mask, and set vector mask
    :param ib: ir builder
    :param type_: the type of mask dst
    :param bits: the bit of mask, default : 128
    """
    # argmin/argmax has his own set_mask func
    if mask_func is not None:
        mask1, mask2 = mask_func(bits)
    else:
        mask1, mask2 = cce_util.set_mask(bits)

    ib_expr.emit(tvm.call_extern(
        type_, "set_vector_mask", tvm.const(mask1, dtype="uint64"),
        tvm.const(mask2, dtype="uint64")))


def _need_dichotomy_add(loop_size, loop_tail_size, dtype):
    if dtype == "float16":
        vector_inst_one_repeat_size = 128
    else:
        vector_inst_one_repeat_size = 64

    return loop_size % vector_inst_one_repeat_size == 0 and \
        loop_tail_size % vector_inst_one_repeat_size == 0


def _get_factors_of_positive_integer(n_value):
    factors = []
    if n_value <= 0:
        return factors
    sqrt_n = int(math.sqrt(n_value))
    for i in range(1, sqrt_n + 1, 1):
        if n_value % i == 0:
            r_val = n_value // i
            factors.append(i)
            if r_val != i:
                factors.append(r_val)
    factors.sort()
    return factors


def _find_closest_factor(factors, m_val):
    if not factors:
        return None
    factors.sort()
    index = 0
    is_find = False
    for i in range(0, len(factors), 1):
        if factors[i] > m_val:
            index = i
            is_find = True
            break
    if is_find:
        if index > 0:
            index = index - 1
    else:
        index = len(factors) - 1

    closest_factor = factors[index]
    return closest_factor


def get_max_ub_count(dtype):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    # div 2 for align to fp16
    total_size = cceconf.get_soc_spec("UB_SIZE") // 2
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)
    total_size = total_size // dtype_size
    if dtype == "float16":
        total_width = 15.08
    else:
        total_width = 7.01

    align_to = 128

    max_bound = total_width * align_to
    max_ub_count = int(total_size / max_bound * align_to)

    return max_ub_count


# pylint: disable=too-many-branches, too-many-statements, unused-argument
def get_tiling(shape, dtype, reduce_axis_idx, max_ub_count):
    """
    get tiling
    """
    core_num = cceconf.get_soc_spec("CORE_NUM")

    dim = len(shape)

    # find block tiling axis
    block_split_axis = 0
    block_size = 1
    temp_size = 1

    i = 0
    for i in range(reduce_axis_idx + 1):
        temp_size *= shape[i]

        if temp_size >= core_num:
            block_split_axis = i
            block_size = temp_size // shape[i]
            break

    if i == dim - 2:
        block_split_axis = dim - 2

    block_factor = 1

    for i in range(shape[block_split_axis], 0, -1):
        if shape[block_split_axis] % i != 0:
            continue

        if block_size*i > core_num:
            continue
        else:
            block_factor = shape[block_split_axis] // i
            break

    # find ub tiling
    ub_split_inner = 1
    ub_split_axis = dim - 1

    if shape[-1] > max_ub_count:
        # need split shape[-1]
        ub_split_axis = dim - 1

        for i in range(shape[-1], 0, -1):
            if shape[-1] % i != 0:
                continue
            if i > max_ub_count:
                continue
            else:
                ub_split_inner = i
                break
    else:
        temp_size = shape[-1]
        i = dim - 2
        can_break = False
        for i in range(dim - 2, block_split_axis - 1, -1):
            temp_size = temp_size*shape[i]
            if temp_size < max_ub_count:
                continue
            else:
                temp_size = temp_size // shape[i]
                for j in range(shape[i], 0, -1):
                    if shape[i] % j != 0:
                        continue
                    if temp_size*j > max_ub_count:
                        continue
                    else:
                        ub_split_axis = i
                        ub_split_inner = j
                        can_break = True
                        break
            if can_break:
                break

        if i == block_split_axis:
            ub_split_axis = block_split_axis
            for j in range(block_factor, 0, -1):
                if block_factor % j != 0:
                    continue
                if temp_size * j > max_ub_count:
                    continue
                else:
                    ub_split_inner = j
                    break

    return block_split_axis, block_factor, ub_split_axis, ub_split_inner


# pylint: disable=too-many-locals, too-many-nested-blocks, unused-argument
def get_tiling_fractal_z(shape, dtype, max_ub_count):
    """
    get tiling fractal_z
    """
    core_num = cceconf.get_soc_spec("CORE_NUM")

    dim = len(shape)

    if dim == 3:
        m1_val = shape[0]
        n_val = shape[1]
        m0_val = 16

        if m1_val >= core_num:
            block_split_axis = 0
            if m1_val % core_num == 0:
                block_factor = m1_val // core_num
            else:
                for i in range(core_num, 0, -1):
                    if m1_val % i == 0:
                        block_factor = i
                if block_factor < core_num // 2:
                    block_factor = 1

            ub_split_axis = 1

            if n_val*m0_val > max_ub_count:
                max_factor = (max_ub_count + m0_val - 1) // m0_val
                ub_split_inner = 1

                for i in range(max_factor, 0, -1):
                    if n_val % i == 0:
                        ub_split_inner = i
                        break
            else:
                ub_split_axis = 1
                ub_split_inner = n_val
        else:
            ub_split_axis = 1
            ub_split_inner = 1
            if n_val*m0_val > max_ub_count:
                for i in range(n_val, 0, -1):
                    if i*m0_val > max_ub_count:
                        continue
                    else:
                        for j in range(i, 0, -1):
                            if n_val % j == 0:
                                ub_split_inner = j
                                break
                        break

                ub_split_outer = n_val // ub_split_inner
                if ub_split_outer > m0_val:
                    block_split_axis = 1
                    block_factor = 1
                else:
                    block_split_axis = 0
                    block_factor = 1
            else:
                block_split_axis = 1
                block_factor = n_val
                ub_split_axis = 1
                ub_split_inner = n_val
    else:
        batch = shape[0]
        m1_val = shape[1]
        n_val = shape[2]
        m0_val = 16

        block_split_axis = 0
        block_factor = batch

        if batch >= core_num:
            block_split_axis = 0
            if batch % core_num == 0:
                block_factor = batch // core_num
            else:
                for i in range(core_num, 0, -1):
                    if batch % i == 0:
                        block_factor = i
                if block_factor < core_num // 2:
                    block_factor = 1
        elif m1_val >= core_num:
            block_split_axis = 1
            if m1_val % core_num == 0:
                block_factor = m1_val // core_num
            else:
                for i in range(core_num, 0, -1):
                    if m1_val % i == 0:
                        block_factor = i
                if block_factor < core_num // 2:
                    block_factor = 1

        ub_split_axis = 2

        if n_val*m0_val > max_ub_count:
            max_factor = (max_ub_count + m0_val - 1) // m0_val
            ub_split_inner = 1
            for i in range(max_factor, 0, -1):
                if n_val % i == 0:
                    ub_split_inner = i
                    break
        else:
            ub_split_inner = n_val


    return block_split_axis, block_factor, ub_split_axis, ub_split_inner


def _map_apend(input_map, key, value):
    """
    map apend
    """
    if input_map.get(key):
        if isinstance(value, list):
            for val in value:
                if not val in input_map[key]:
                    input_map[key].append(val)
        else:
            if not value in input_map[key]:
                input_map[key].append(value)
    else:
        if isinstance(value, list):
            input_map[key] = value
        else:
            input_map[key] = [value]


def _get_emit_insn_map(tensor):
    insn_map = {"elewise_single_cast": "vector_conv",
                "elewise_single_VS_max": "vector_maxs",
                "elewise_single_VS_min": "vector_mins",
                "elewise_single_log": "vector_ln",
                "elewise_single_exp": "vector_exp",
                "elewise_single_relu": "vector_relu",
                "elewise_single_abs": "vector_abs",
                "elewise_single_not": "vector_not",
                "elewise_single_sqrt": "vector_sqrt",
                "elewise_single_rsqrt": "vector_rsqrt",
                "elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                "elewise_binary_div": "vector_div",
                "elewise_binary_add": "vector_add",
                "elewise_single_VS_add": "vector_adds",
                "elewise_binary_min": "vector_min",
                "elewise_binary_max": "vector_max",
                "elewise_binary_vcmpv_gt": "vector_gt",
                "elewise_binary_vcmpv_ge": "vector_ge",
                "elewise_binary_vcmpv_lt": "vector_lt",
                "elewise_binary_vcmpv_le": "vector_le",
                "elewise_binary_vcmpv_eq": "vector_eq",
                "elewise_binary_vcmpv_ne": "vector_ne",
                "elewise_binary_or": "vector_or",
                "elewise_binary_and": "vector_and",
                "elewise_multiple_mla": "vector_multiple",
                "elewise_multiple_madd": "vector_multiple",
                "elewise_multiple_maddrelu": "vector_multiple",
                "broadcast": "vector_dup",
                "elewise_binary_sub": "vector_sub",
                "broadcast_for_tensor": "broadcast_for_tensor"}
    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn


def _get_emit_insn_map_for_broadcast(tensor):
    insn_map = {"elewise_binary_mul": "vector_mul_with_broadcast",
                "elewise_binary_div": "vector_div_with_broadcast",
                "elewise_binary_add": "vector_add_with_broadcast",
                "elewise_binary_sub": "vector_sub_with_broadcast",
                }
    if tensor.op.tag not in insn_map:
        raise RuntimeError("Invalid tag of with broadcast vector instric!")

    return insn_map.get(tensor.op.tag)


def _gen_reversed_subgraph_list(out_tensor, tensor_list,
                                tensor_list_dst_tensor_map, visited_list,
                                input_broadcast_tensors):
    """traverse tensors by Depth-First-Search

    Parameters
    ----------
    out_tensor : tensor
        traverse tensors from this tensor,
        traversing its input tensors recursively.

    tensor_list : list
        record tensors in the order of Depth-First-Search.

    """
    if out_tensor is None:
        return
    stack = [out_tensor]
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_list.append(in_tensor)

                if in_tensor.op.tag.find("broadcast_for_tensor") != -1:
                    input_broadcast_tensors.append(cur_tensor)

            _map_apend(tensor_list_dst_tensor_map, in_tensor, cur_tensor)


def _do_cache_read(sch_list, input_tensor_dst_tensor_map, exclude_tensor):
    """
    do cache read
    """
    sch = sch_list[0]
    input_tensor_buffer_map = {}
    for tensor in input_tensor_dst_tensor_map:
        if tensor not in exclude_tensor:
            buffer_tensor = sch.cache_read(tensor, cce.scope_ubuf,
                                           input_tensor_dst_tensor_map[tensor])
            input_tensor_buffer_map[tensor] = buffer_tensor
    sch_list[0] = sch
    return input_tensor_buffer_map


def _do_cache_write(sch_list, mid_tensor_dst_tensor_map,
                    exclude_tensor, input_broadcast_tensors):
    """
    do cache write
    """
    sch = sch_list[0]
    mid_tensor_buffer_map = {}
    input_broadcast_tensor_buffers = []
    for tensor in mid_tensor_dst_tensor_map:
        if tensor not in exclude_tensor:
            buffer_tensor = sch.cache_write(tensor, cce.scope_ubuf)
            mid_tensor_buffer_map[tensor] = buffer_tensor

            if tensor in input_broadcast_tensors:
                input_broadcast_tensor_buffers.append(buffer_tensor)

    sch_list[0] = sch
    return mid_tensor_buffer_map, input_broadcast_tensor_buffers


def _do_compute_inline(sch_list, mid_tensor_dst_tensor_map, exclude_tensor):
    """
    do cache inline
    """
    sch = sch_list[0]
    for tensor in mid_tensor_dst_tensor_map:
        if tensor not in exclude_tensor:
            sch[tensor].compute_inline()
    sch_list[0] = sch


# pylint: disable=too-many-arguments, unused-argument
def _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map, mid_tensor_buffer_map,
                  input_broadcast_tensor_buffers, shape_input,
                  ub_split_axis, broadcast_axis_idx):
    """
    do emit insn
    """
    sch = sch_list[0]
    for tensor in input_tensor_buffer_map:
        buffer_tensor = input_tensor_buffer_map[tensor]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    for tensor in mid_out_read_buffer_map:
        buffer_tensor = mid_out_read_buffer_map[tensor]
        sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "phony_insn")

    for tensor in mid_tensor_buffer_map:
        buffer_tensor = mid_tensor_buffer_map[tensor]

        insn = _get_emit_insn_map(tensor)

        insn_axis = 0
        if insn.find("broadcast") != -1:
            if ub_split_axis < broadcast_axis_idx:
                insn_axis = broadcast_axis_idx
            else:
                insn_axis = -1

        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[insn_axis], insn)

    sch_list[0] = sch


# pylint: disable=too-many-arguments, unused-argument
def _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   final_out_buffer_list, compute_at_axis_list):
    """
    do compute at
    """
    if not isinstance(sch_list, list) or not isinstance(final_out_buffer_list, list):
        raise RuntimeError("sch or final_out_buffer is not list .")

    sch = sch_list[0]
    final_out_buffer_1 = final_out_buffer_list[0]
    final_out_buffer_2 = final_out_buffer_1
    compute_at_axis_1 = compute_at_axis_list[0]
    compute_at_axis_2 = compute_at_axis_1

    if len(final_out_buffer_list) > 1:
        final_out_buffer_2 = final_out_buffer_list[1]
        compute_at_axis_2 = compute_at_axis_list[1]

    for tensor in input_tensor_buffer_map:
        buffer_tensor = input_tensor_buffer_map[tensor]
        shape = te.lang.cce.util.shape_to_list(tensor.shape)
        if shape == shape_input:
            sch[buffer_tensor].compute_at(sch[final_out_buffer_1], compute_at_axis_1)
            # small shape input to be broadcast
        else:
            sch[buffer_tensor].compute_at(sch[final_out_buffer_2], compute_at_axis_2)

    for tensor in mid_out_read_buffer_map:
        buffer_tensor = mid_out_read_buffer_map[tensor]
        sch[buffer_tensor].compute_at(sch[final_out_buffer_1], compute_at_axis_1)
        sch[tensor].compute_at(sch[final_out_buffer_1], compute_at_axis_1)

    for tensor in mid_tensor_buffer_map:
        buffer_tensor = mid_tensor_buffer_map[tensor]
        sch[buffer_tensor].compute_at(sch[final_out_buffer_1], compute_at_axis_1)
    sch_list[0] = sch


def _do_double_buffer(sch_list, shape_input, outer_loop, input_tensor_buffer_map):
    """
    do double buffer
    """
    sch = sch_list[0]
    if outer_loop > 2:
        for tensor in input_tensor_buffer_map:
            shape = te.lang.cce.util.shape_to_list(tensor.shape)
            if shape == shape_input:
                buffer_tensor = input_tensor_buffer_map[tensor]
                sch[buffer_tensor].double_buffer()
                break
    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-arguments
def schedule_cut_nlstaxis_twice(sch_list, res, shape_x,
                                block_split_axis, block_factor,
                                ub_split_axis, ub_split_inner,
                                reduce_axis_idx, broadcast_axis_idx,
                                input_tensor_buffer_map,
                                mid_out_read_buffer_map,
                                mid_tensor_buffer_map,
                                final_out_tensor_list,
                                input_broadcast_tensor_buffers,
                                is_keep_dim):
    """
    schedule cut nlstaxis twice
    """
    sch = sch_list[0]

    final_out_tensor = final_out_tensor_list[0]

    reorder_list = final_out_tensor.op.reduce_axis[:] +\
                        final_out_tensor.op.axis[:]

    sch[final_out_tensor].reorder(*reorder_list)

    final_out_tensor_block_outer, final_out_tensor_block_inner = \
        sch[final_out_tensor].split(
            final_out_tensor.op.reduce_axis[block_split_axis],
            factor=block_factor)

    if block_split_axis == 0:
        fused_axis = final_out_tensor_block_outer
    else:
        fuse_axis_list = final_out_tensor.op.reduce_axis[0:block_split_axis] + \
                         [final_out_tensor_block_outer]
        fused_axis = sch[final_out_tensor].fuse(*fuse_axis_list)

    sch[final_out_tensor].split(final_out_tensor_block_inner,
                                factor=ub_split_inner)

    final_out_tensor_ub_rf, _ = sch.rfactor(final_out_tensor, fused_axis)
    final_out_tensor_global_list = sch.cache_write(final_out_tensor_list, "global")
    # final out tensor list index in res
    list_index_in_res = []
    for tensor in final_out_tensor_list:
        # pylint: disable=consider-using-enumerate
        for i in range(0, len(res)):
            if tensor == res[i]:
                list_index_in_res.append(i)
                break

    # pylint: disable=consider-using-enumerate
    for i in range(0, len(final_out_tensor_global_list)):
        res[list_index_in_res[i]] =\
            final_out_tensor_global_list[i]

    sch[final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

    final_out_tensor_global = final_out_tensor_global_list[0]

    reorder_list = final_out_tensor_global.op.reduce_axis[:] + \
                    final_out_tensor_global.op.axis[:]
    sch[final_out_tensor_global].reorder(*reorder_list)

    reorder_list = []
    reorder_list.append(final_out_tensor_ub_rf.op.axis[0])

    reduce_axis_len = len(final_out_tensor_ub_rf.op.reduce_axis)

    reduce_axis = final_out_tensor_ub_rf.op.reduce_axis

    for axis in reduce_axis[reduce_axis_len - 2:]:
        reorder_list.append(axis)

    for axis in reduce_axis[0:reduce_axis_len - 2]:
        reorder_list.append(axis)

    for axis in final_out_tensor_ub_rf.op.axis[1:]:
        reorder_list.append(axis)

    sch[final_out_tensor_ub_rf].reorder(*reorder_list)

    if ub_split_axis >= reduce_axis_idx:
        compute_at_axis = reduce_axis[0]
    else:
        compute_at_axis = reduce_axis[reduce_axis_len - 1]

    sch[final_out_tensor_ub_rf].compute_at(sch[final_out_tensor_global],
                                           final_out_tensor_global.op.axis[0])

    _do_compute_at(sch_list, shape_x, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_tensor_ub_rf], [compute_at_axis])

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor_global].bind(final_out_tensor_global.op.reduce_axis[0], block)

    if ub_split_axis >= reduce_axis_idx:
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_tensor_ub_rf.op.reduce_axis[1],
            "vector_reduce_sum")
    else:
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_tensor_ub_rf.op.reduce_axis[0],
            "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map, mid_tensor_buffer_map,
                  input_broadcast_tensor_buffers, shape_x,
                  ub_split_axis, broadcast_axis_idx)

    sch[final_out_tensor_global].emit_insn(
        final_out_tensor_global.op.axis[1],
        "dma_copy")
    sch[final_out_tensor].emit_insn(
        sch[final_out_tensor].op.axis[0],
        "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, unused-argument
def schedule_cut_diff_axis(sch_list, res, shape_x,
                           block_split_axis, block_factor,
                           ub_split_axis, ub_split_inner,
                           reduce_axis_idx, broadcast_axis_idx,
                           input_tensor_buffer_map,
                           mid_out_read_buffer_map,
                           mid_tensor_buffer_map,
                           final_out_tensor_list,
                           input_broadcast_tensor_buffers,
                           is_keep_dim):
    """
    schedule cut diff nlstaxis
    """
    sch = sch_list[0]

    final_out_tensor = final_out_tensor_list[0]

    reorder_list = final_out_tensor.op.reduce_axis[:] +\
                        final_out_tensor.op.axis[:]

    sch[final_out_tensor].reorder(*reorder_list)

    final_out_tensor_block_outer, _ = \
        sch[final_out_tensor].split(
            final_out_tensor.op.reduce_axis[block_split_axis],
            factor=block_factor)

    if block_split_axis == 0:
        fused_axis = final_out_tensor_block_outer
    else:
        fuse_axis_list = final_out_tensor.op.reduce_axis[0:block_split_axis] +\
                         [final_out_tensor_block_outer]
        fused_axis = sch[final_out_tensor].fuse(*fuse_axis_list)

    final_out_tensor_ub_rf, _ = sch.rfactor(final_out_tensor, fused_axis)
    final_out_tensor_global_list = sch.cache_write(final_out_tensor_list, "global")

    # final out tensor list index in res
    list_index_in_res = []
    for tensor in final_out_tensor_list:
        # pylint: disable=consider-using-enumerate
        for i in range(0, len(res)):
            if tensor == res[i]:
                list_index_in_res.append(i)
                break
    # pylint: disable=consider-using-enumerate
    for i in range(0, len(final_out_tensor_global_list)):
        res[list_index_in_res[i]] =\
            final_out_tensor_global_list[i]

    sch[final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

    final_out_tensor_global = final_out_tensor_global_list[0]


    if ub_split_axis <= reduce_axis_idx:
        ub_outer, ub_inner = \
            sch[final_out_tensor_ub_rf].split(
                final_out_tensor_ub_rf.op.reduce_axis[
                    ub_split_axis - block_split_axis - 1],
                factor=ub_split_inner)

        reorder_list = final_out_tensor_global.op.reduce_axis[:] + \
                       final_out_tensor_global.op.axis[:]
        sch[final_out_tensor_global].reorder(*reorder_list)

        reorder_list = []
        reorder_list.append(final_out_tensor_ub_rf.op.axis[0])
        reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[-1])
        reorder_list += final_out_tensor_ub_rf.op. \
                            reduce_axis[0:ub_split_axis - block_split_axis - 1]
        reorder_list.append(ub_outer)
        reorder_list.append(ub_inner)
        reorder_list += final_out_tensor_ub_rf.op. \
                            reduce_axis[ub_split_axis - block_split_axis: -1]
        for axis in final_out_tensor_ub_rf.op.axis[1:]:
            reorder_list.append(axis)
        sch[final_out_tensor_ub_rf].reorder(*reorder_list)

    else:
        ub_outer, ub_inner = \
            sch[final_out_tensor_ub_rf].split(
                final_out_tensor_ub_rf.op.axis[ub_split_axis + 1],
                #final_out_tensor_ub_rf.op.reduce_axis[
                #    ub_split_axis - block_split_axis - 1],
                factor=ub_split_inner)

        reorder_list = final_out_tensor_global.op.reduce_axis[:] + \
                       final_out_tensor_global.op.axis[:]
        sch[final_out_tensor_global].reorder(*reorder_list)

        reorder_list = []
        reorder_list.append(final_out_tensor_ub_rf.op.axis[0])
        reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[-1])
        reorder_list += final_out_tensor_ub_rf.op. \
                            reduce_axis[0: -1]
        reorder_list.append(ub_outer)
        reorder_list.append(ub_inner)
        for axis in final_out_tensor_ub_rf.op.axis[1:ub_split_axis + 1]:
            reorder_list.append(axis)
        for axis in final_out_tensor_ub_rf.op.axis[ub_split_axis + 2:]:
            reorder_list.append(axis)
        sch[final_out_tensor_ub_rf].reorder(*reorder_list)

    compute_at_axis = ub_outer

    rf_compute_at_axis = final_out_tensor_global.op.reduce_axis[0]
    global_dma_insn_axis = final_out_tensor_global.op.axis[1]
    rf_reduce_axis = ub_inner

    if ub_split_axis > reduce_axis_idx:
        global_outer, global_inner = sch[final_out_tensor_global].split(
            final_out_tensor_global.op.axis[ub_split_axis],
            factor=ub_split_inner)
        rf_compute_at_axis = global_outer
        global_dma_insn_axis = global_inner

    sch[final_out_tensor_ub_rf].compute_at(
        sch[final_out_tensor_global],
        rf_compute_at_axis)

    _do_compute_at(sch_list, shape_x, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_tensor_ub_rf], [compute_at_axis])

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor_global].bind(
        final_out_tensor_global.op.reduce_axis[0], block)

    sch[final_out_tensor_ub_rf].emit_insn(
        rf_reduce_axis, "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map, mid_tensor_buffer_map,
                  input_broadcast_tensor_buffers, shape_x,
                  ub_split_axis, broadcast_axis_idx)

    sch[final_out_tensor_global].emit_insn(
        global_dma_insn_axis,
        "dma_copy")
    sch[final_out_tensor].emit_insn(
        sch[final_out_tensor].op.axis[0],
        "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
def schedule_cut_general(sch_list, shape_input, ub_split_reduce_axis, split_factor,
                         reduce_axis_idx, broadcast_axis_idx,
                         input_tensor_buffer_map, mid_out_read_buffer_map,
                         mid_tensor_buffer_map, final_out_tensor_list,
                         input_broadcast_tensor_buffers, is_keep_dim):
    """
    schedule cut general
    """
    sch = sch_list[0]

    final_out_buffer_list = sch.cache_write(final_out_tensor_list, cce.scope_ubuf)
    final_out_tensor = final_out_tensor_list[0]
    final_out_buffer = final_out_buffer_list[0]

    sum_x_ub_outer, sum_x_ub_inner = sch[final_out_buffer].split(
        final_out_buffer.op.reduce_axis[ub_split_reduce_axis], factor=split_factor)
    if ub_split_reduce_axis == 1:
        ub_split_axis = 2
    elif ub_split_reduce_axis == 2:
        ub_split_axis = 3
    else:
        raise RuntimeError("Batch normalization only support 5D format.")
    outer_loop = shape_input[ub_split_axis] // split_factor
    if ub_split_axis == 3:
        outer_loop = outer_loop * shape_input[2]

    if is_keep_dim:
        sum_x_c1_axis = final_out_tensor.op.axis[1]
        sum_x_c0_axis = final_out_tensor.op.axis[4]
        sum_x_ub_n_axis = final_out_buffer.op.axis[0]
        sum_x_ub_c1_axis = final_out_buffer.op.axis[1]
        sum_x_ub_h_axis = final_out_buffer.op.axis[2]
        sum_x_ub_w_axis = final_out_buffer.op.axis[3]
        sum_x_ub_c0_axis = final_out_buffer.op.axis[4]
    else:
        sum_x_c1_axis = final_out_tensor.op.axis[0]
        sum_x_c0_axis = final_out_tensor.op.axis[1]
        sum_x_ub_c1_axis = final_out_buffer.op.axis[0]
        sum_x_ub_c0_axis = final_out_buffer.op.axis[1]

    sum_x_ub_n_reduce_axis = final_out_buffer.op.reduce_axis[0]
    sum_x_ub_h_reduce_axis = final_out_buffer.op.reduce_axis[1]
    sum_x_ub_w_reduce_axis = final_out_buffer.op.reduce_axis[2]

    if ub_split_reduce_axis == 1:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                                          sum_x_ub_outer, sum_x_ub_inner,
                                          sum_x_ub_w_axis, sum_x_ub_w_reduce_axis,
                                          sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c1_axis, sum_x_ub_n_reduce_axis,
                                          sum_x_ub_outer, sum_x_ub_inner,
                                          sum_x_ub_w_reduce_axis, sum_x_ub_c0_axis)
    else:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_c1_axis, sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_outer, sum_x_ub_inner, sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_w_axis, sum_x_ub_outer,
                                          sum_x_ub_inner, sum_x_ub_c0_axis)

    _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_buffer], [sum_x_ub_outer])

    sch[final_out_buffer].compute_at(sch[final_out_tensor], sum_x_c1_axis)

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor].bind(sum_x_c1_axis, block)

    _do_double_buffer(sch_list, shape_input, outer_loop, input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    c0_size = 16
    loop_size = split_factor * c0_size
    outer_factor = shape_input[ub_split_axis] // split_factor
    size = shape_input[ub_split_axis] * c0_size
    if ub_split_axis == 2:
        loop_size = loop_size * shape_input[3]
        size = size * shape_input[3]
    loop_tail_size = size - outer_factor * loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[final_out_buffer].emit_insn(sum_x_ub_inner, "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_out_buffer].emit_insn(sum_x_ub_inner, "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map, mid_out_read_buffer_map,
                  mid_tensor_buffer_map, input_broadcast_tensor_buffers,
                  shape_input,
                  ub_split_axis, broadcast_axis_idx)

    sch[final_out_tensor].emit_insn(sum_x_c0_axis, "dma_copy")

    sch_list[0] = sch


# pylint: disable=too-many-arguments, unused-argument
def schedule_cut_m1_nz(sch_list, res, shape_x,
                       block_split_axis, block_factor,
                       ub_split_axis, ub_split_inner,
                       input_tensor_buffer_map,
                       mid_out_read_buffer_map,
                       mid_tensor_buffer_map,
                       final_out_tensor_list,
                       input_broadcast_tensor_buffers,
                       is_keep_dim):
    """
    schedule cut m1_nz
    """
    sch = sch_list[0]

    final_ub_tensor_list = sch.cache_write(final_out_tensor_list,
                                           cce.scope_ubuf)

    final_ub_tensor = final_ub_tensor_list[0]

    res = final_out_tensor_list[0]

    final_out_tensor_block_outer, final_out_tensor_block_inner = \
        sch[final_ub_tensor].split(
            final_ub_tensor.op.axis[block_split_axis],
            factor=block_factor)

    ub_outer, ub_inner = sch[final_ub_tensor].\
        split(final_ub_tensor.op.reduce_axis[ub_split_axis - 1],
              factor=ub_split_inner)

    res_outer, res_inner = \
        sch[res].split(res.op.axis[block_split_axis],
                       factor=block_factor)

    dim = len(shape_x)
    if dim == 3:
        reorder_list = [final_out_tensor_block_outer,
                        final_out_tensor_block_inner] + \
                       [ub_outer, ub_inner] + \
                       final_ub_tensor.op.axis[(dim - 1):]
    else:
        reorder_list = [final_out_tensor_block_outer] + \
                       final_ub_tensor.op.reduce_axis[0:ub_split_axis-1] + \
                       final_ub_tensor.op.axis[0:block_split_axis] +\
                       [final_out_tensor_block_inner] + \
                       final_ub_tensor.op.axis[block_split_axis+1:dim-1] +\
                       [ub_outer, ub_inner] + \
                       final_ub_tensor.op.axis[(dim - 1):]

    sch[final_ub_tensor].reorder(*reorder_list)

    compute_at_axis = ub_outer

    _do_compute_at(sch_list, shape_x, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_ub_tensor], [compute_at_axis])

    sch[final_ub_tensor].compute_at(sch[res], res_outer)

    block = tvm.thread_axis("blockIdx.x")
    sch[res].bind(res_outer, block)

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map, mid_tensor_buffer_map,
                  input_broadcast_tensor_buffers, shape_x,
                  ub_split_axis, -1)

    vector_inst_one_repeat_size = 64

    if ub_split_inner*16 % vector_inst_one_repeat_size == 0:
        sch[final_ub_tensor].emit_insn(ub_inner,
                                       "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_ub_tensor].emit_insn(ub_inner, "vector_reduce_sum")

    sch[res].emit_insn(res_inner, "dma_copy")

    sch_list[0] = sch


def _is_general_schedule(is_fractal_z, shape_x,
                         block_split_axis, ub_split_axis):
    """
    check is general schedule, if general, not use this schedule
    """
    general_schedule = False
    if is_fractal_z:
        # Nz input
        if len(shape_x) == 3:
            if block_split_axis < ub_split_axis:
                general_schedule = False
            else:
                general_schedule = True
        else:
            if block_split_axis < ub_split_axis:
                general_schedule = False
            else:
                general_schedule = True
    else:
        #ND input
        dim = len(shape_x)
        if block_split_axis == ub_split_axis < dim - 1:
            general_schedule = False
        elif block_split_axis < ub_split_axis < dim - 1:
            general_schedule = False
        elif ub_split_axis == dim - 1:
            general_schedule = False
        else:
            general_schedule = True

    return general_schedule


def _get_reduce_axis(shape_x, shape_res):
    """
    get reduce axis according to shape_x and shape_res
    no reduce case will not come into this schedule
    """
    reduce_axis_idx = 0
    if len(shape_x) != len(shape_res):
        reduce_axis_idx = len(shape_x) - len(shape_res) - 1
    else:
        for i in range(len(shape_x) - 1, -1, -1):
            if shape_x[i] != shape_res[i]:
                reduce_axis_idx = i
                break

    return reduce_axis_idx


def _get_broadcast_axis(shape_x, shape_mean):
    """
    get broadcast axis according to shape_x and shape_mean
    """
    broadcast_axis_idx = len(shape_x) - 1
    for i, val in enumerate(shape_x):
        if val != shape_mean[i]:
            broadcast_axis_idx = i
            break

    return broadcast_axis_idx


def _is_reduce_align(shape_x, dtype, reduce_axis_idx,
                     block_split_axis, block_factor,
                     ub_split_axis, ub_split_inner):
    """
    check whether reduce compute is 32B aligned.
    if not 32B aligned, this sch can't process
    """
    # reduce compute must be 32B aligned
    shape_before_reduce = []

    shape_before_reduce.append(ub_split_inner)
    shape_before_reduce += shape_x[ub_split_axis + 1:]

    shape_reduce_res = []
    if ub_split_axis <= reduce_axis_idx:
        shape_reduce_res = shape_x[reduce_axis_idx + 1:]
    else:
        shape_reduce_res.append(ub_split_inner)
        shape_reduce_res += shape_x[ub_split_axis + 1:]

    aligned_data_size = 16
    if dtype == "float32":
        aligned_data_size = 8

    reduce_res_size = reduce(lambda i, j: i * j, shape_reduce_res)

    if shape_reduce_res != shape_before_reduce:
        if reduce_res_size % aligned_data_size != 0:
            return False

    # one core res data size must be greater equal 32B
    block_outer_size = shape_x[block_split_axis] // block_factor
    for val in shape_x[0:block_split_axis]:
        block_outer_size *= val

    if block_outer_size > 1 and reduce_res_size < aligned_data_size:
        return False

    return True


# pylint: disable=too-many-locals, too-many-branches
# pylint: disable=too-many-statements, too-many-function-args
def layer_norm_grad_schedule(res, input_tensors):
    """
    layer norm grad schedule
    """
    if len(res) != 2:
        raise RuntimeError("LayerNorm_grad_beta_gamma output nums should be 2.")

    if len(input_tensors) != 4:
        raise RuntimeError("LayerNorm_grad_beta_gamma input nums should be 4.")

    data_x_tensor = input_tensors[-1]
    shape_x = te.lang.cce.util.shape_to_list(data_x_tensor.shape)

    dtype = data_x_tensor.dtype.lower()

    tensor_list = []
    tensor_list_dst_tensor_map = {}
    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}
    mid_out_tensor_list = []
    broadcast_tensor_list = []
    visited_list = []
    input_broadcast_tensors = []
    final_out_tensor_list = []
    for tensor in res[1:]:
        _gen_reversed_subgraph_list(tensor, tensor_list,
                                    tensor_list_dst_tensor_map, visited_list,
                                    input_broadcast_tensors)

    input_broadcast_tensors = list(set(input_broadcast_tensors))

    for tensor in res:
        if tensor in tensor_list:
            mid_out_tensor_list.append(tensor)
        else:
            final_out_tensor_list.append(tensor)

    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[tensor]
        else:
            mid_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[tensor]
        if tensor.op.tag.find("broadcast") != -1:
            broadcast_tensor_list.append(tensor)

    out_tensor = final_out_tensor_list[0]
    shape_res = te.lang.cce.util.shape_to_list(out_tensor.shape)

    is_keep_dim = True
    if len(shape_x) != len(shape_res):
        is_keep_dim = False

    max_ub_count = get_max_ub_count(dtype)

    is_fractal_z = shape_res[-1] == 16 and \
                   ((len(shape_x) == 4 and shape_res[2] == 1) or\
                    (len(shape_x) == 3 and shape_res[1] == 1))

    reduce_axis_idx = _get_reduce_axis(shape_x, shape_res)

    data_mean = input_tensors[0]
    shape_mean = te.lang.cce.util.shape_to_list(data_mean.shape)

    broadcast_axis_idx = _get_broadcast_axis(shape_x, shape_mean)

    general_schedule = False
    if is_fractal_z:
        block_split_axis, block_factor, ub_split_axis, ub_split_inner = \
            get_tiling_fractal_z(shape_x, dtype, max_ub_count)
    else:
        block_split_axis, block_factor, ub_split_axis, ub_split_inner = \
            get_tiling(shape_x, dtype, reduce_axis_idx, max_ub_count)

        is_reduce_align = \
            _is_reduce_align(shape_x, dtype, reduce_axis_idx,
                             block_split_axis, block_factor,
                             ub_split_axis, ub_split_inner)

        if not is_reduce_align:
            general_schedule = True

    if not general_schedule:
        general_schedule = \
            _is_general_schedule(is_fractal_z, shape_x,
                                 block_split_axis, ub_split_axis)

    if general_schedule:
        # pylint: disable=consider-using-enumerate
        for i in range(len(res)):
            input_tensor = res[i].op.input_tensors[i]
            input_shape = te.lang.cce.util.shape_to_list(input_tensor.shape)
            output_shape = te.lang.cce.util.shape_to_list(res[i].shape)
            reduce_axes = []
            for cur_dim in range(len(output_shape)):
                if output_shape[cur_dim] != input_shape[cur_dim]:
                    reduce_axes.append(cur_dim)
            res[i] = te.lang.cce.sum(input_tensor, reduce_axes, keepdims=True)
        return None

    sch = tvm.create_schedule([out_tensor.op])
    sch_list = [sch]
    exclude_tensor = []
    input_tensor_buffer_map = _do_cache_read(sch_list,
                                             input_tensor_dst_tensor_map,
                                             exclude_tensor)
    mid_out_tensor_dst_tensor_map = {}
    mid_out_read_buffer_map = {}
    for tensor in mid_out_tensor_list:
        dst_tensor = mid_tensor_dst_tensor_map[tensor]
        mid_out_tensor_dst_tensor_map[tensor] = dst_tensor
        mid_out_read_buffer_map =\
            _do_cache_read(sch_list,
                           mid_out_tensor_dst_tensor_map,
                           exclude_tensor)

    cache_write_exclude_tensor = []

    mid_tensor_buffer_map, input_broadcast_tensor_buffers =\
        _do_cache_write(sch_list,
                        mid_tensor_dst_tensor_map,
                        cache_write_exclude_tensor,
                        input_broadcast_tensors)

    _do_compute_inline(sch_list, mid_tensor_dst_tensor_map, mid_out_tensor_list)

    if is_fractal_z:
        # Nz input
        if len(shape_x) == 3:
            # no batch
            if block_split_axis < ub_split_axis:
                # cut m0
                schedule_cut_m1_nz(
                    sch_list, res, shape_x,
                    block_split_axis, block_factor,
                    ub_split_axis, ub_split_inner,
                    input_tensor_buffer_map, mid_out_read_buffer_map,
                    mid_tensor_buffer_map, final_out_tensor_list,
                    input_broadcast_tensor_buffers, is_keep_dim)
        else:
            if block_split_axis < ub_split_axis:
                # cut m0
                schedule_cut_m1_nz(
                    sch_list, res, shape_x,
                    block_split_axis, block_factor,
                    ub_split_axis, ub_split_inner,
                    input_tensor_buffer_map, mid_out_read_buffer_map,
                    mid_tensor_buffer_map, final_out_tensor_list,
                    input_broadcast_tensor_buffers, is_keep_dim)
    else:
        #ND input
        dim = len(shape_x)
        if block_split_axis == ub_split_axis < dim - 1:
            schedule_cut_nlstaxis_twice(
                sch_list, res, shape_x,
                block_split_axis, block_factor,
                ub_split_axis, ub_split_inner,
                reduce_axis_idx, broadcast_axis_idx,
                input_tensor_buffer_map, mid_out_read_buffer_map,
                mid_tensor_buffer_map, final_out_tensor_list,
                input_broadcast_tensor_buffers, is_keep_dim)
        elif block_split_axis < ub_split_axis:
            schedule_cut_diff_axis(
                sch_list, res, shape_x,
                block_split_axis, block_factor,
                ub_split_axis, ub_split_inner,
                reduce_axis_idx, broadcast_axis_idx,
                input_tensor_buffer_map, mid_out_read_buffer_map,
                mid_tensor_buffer_map, final_out_tensor_list,
                input_broadcast_tensor_buffers, is_keep_dim)
        else:
            schedule_cut_general(
                sch_list, res, shape_x,
                block_split_axis, block_factor,
                ub_split_axis, ub_split_inner,
                reduce_axis_idx, broadcast_axis_idx,
                input_tensor_buffer_map, mid_out_read_buffer_map,
                mid_tensor_buffer_map, final_out_tensor_list,
                input_broadcast_tensor_buffers, is_keep_dim)

    sch = sch_list[0]

    return sch
