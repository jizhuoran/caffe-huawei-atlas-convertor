#!/usr/bin/env python # pylint: disable=too-many-lines, unused-import
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

batch_normalization_forward_training_reduce
"""
from __future__ import absolute_import
from __future__ import division
import math
from functools import reduce

import te.lang.cce
from te import tvm
import te.platform.cce_params as cce
from te import platform as cceconf
from te.platform import cce_util
from .reduce_atomic_schedule import ReduceAtomicSchedule
from .util import get_nearest_factor
from .util import DTYPE_WIDTH_MAP

MAX_SHAPE_NUM = 10000000


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


def _need_dichotomy_add(loop_size, dtype):
    if dtype == "float16":
        vector_inst_one_repeat_size = 128
    else:
        vector_inst_one_repeat_size = 64

    return loop_size > vector_inst_one_repeat_size


def _get_factors_of_positive_integer(value):
    factors = []
    if value <= 0:
        return factors
    sqrt_n = int(math.sqrt(value))
    for i in range(1, sqrt_n + 1, 1):
        if value % i == 0:
            tmp = value // i
            factors.append(i)
            if tmp != i:
                factors.append(tmp)
    factors.sort()
    return factors


def _find_closest_factor(factors, value):
    """
    find closest factor
    """
    if not factors:
        return None
    factors.sort()
    index = 0
    is_find = False
    for i in range(0, len(factors), 1):
        if factors[i] > value:
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


RESNET_50_SPECIAL_MAX_UB_COUNT_FP16_MAP = {
    "32_4_112_112_16": 10,
    "32_4_57_57_16": 8.5,
    "32_8_57_57_16": 8.5,
    "32_16_57_57_16": 8.5,
}


RESNET_50_SPECIAL_MAX_UB_COUNT_FP32_MAP = {
    "32_4_112_112_16": 4,
}

RESNET_50_SHAPE_LIST = [
    [32, 1024 // 16, 7, 7, 16],
    [32, 1024 // 16, 14, 14, 16],
    [32, 128 // 16, 14, 14, 16],
    [32, 128 // 16, 28, 28, 16],
    [32, 128 // 16, 56, 56, 16],
    [32, 2048 // 16, 7, 7, 16],
    [32, 256 // 16, 14, 14, 16],
    [32, 256 // 16, 28, 28, 16],
    [32, 256 // 16, 56, 56, 16],
    [32, 256 // 16, 7, 7, 16],
    [32, 512 // 16, 14, 14, 16],
    [32, 512 // 16, 28, 28, 16],
    [32, 512 // 16, 7, 7, 16],
    [32, 64 // 16, 112, 112, 16],
    [32, 64 // 16, 28, 28, 16],
    [32, 64 // 16, 56, 56, 16],

    [32, 1, 224, 224, 16],
    [32, 4, 57, 57, 16],
    [32, 4, 112, 112, 16],
    [32, 8, 29, 29, 16],
    [32, 8, 57, 57, 16],
    [32, 16, 15, 15, 16],
    [32, 16, 29, 29, 16],
    [32, 16, 57, 57, 16],
    [32, 32, 15, 15, 16],
    [32, 32, 29, 29, 16],
    [32, 32, 8, 8, 16],
    [32, 64, 15, 15, 16],
]


def get_max_ub_count(dtype, shape):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    # div 2 for align to fp16
    total_size = cceconf.get_soc_spec("UB_SIZE") // 2
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)
    total_size = total_size // dtype_size
    if shape not in RESNET_50_SHAPE_LIST:
        # div 2 for double buffer
        total_size = total_size // 2
        if dtype == "float16":
            total_width = 4.5
        else:
            total_width = 2.5
    else:
        key_shape = "_".join(str(i) for i in shape)
        if dtype == "float16":
            if RESNET_50_SPECIAL_MAX_UB_COUNT_FP16_MAP.get(key_shape):
                total_width = \
                    RESNET_50_SPECIAL_MAX_UB_COUNT_FP16_MAP[key_shape]
            else:
                total_width = 5.5
        else:
            if RESNET_50_SPECIAL_MAX_UB_COUNT_FP32_MAP.get(key_shape):
                total_width = \
                    RESNET_50_SPECIAL_MAX_UB_COUNT_FP32_MAP[key_shape]
            else:
                total_width = 3.5

    align_to = 128

    max_bound = total_width * align_to
    max_ub_count = int(total_size / max_bound * align_to)

    return max_ub_count

# pylint: disable=too-many-locals, too-many-branches
def get_ub_tiling(shape, block_tiling_axis,
                  block_tiling_inner_loop, max_ub_count):
    '''
    find ub tiling
    '''
    last_axis = len(shape) - 1
    ub_split_inner = 1
    ub_split_axis = 0
    if block_tiling_axis < 0 or block_tiling_axis > last_axis:
        return ub_split_axis, ub_split_inner

    core_num = cceconf.get_soc_spec("CORE_NUM")

    n_size = shape[0]
    c1_size = shape[1]
    h_size = shape[2]
    w_size = shape[3]
    c0_size = shape[4]

    if max_ub_count // (h_size*w_size*c0_size) >= 2 \
            and ((c1_size >= core_num and c1_size % core_num == 0) \
                or (n_size >= core_num and n_size % core_num == 0)):
        # ub utilization ratio is small, so use "model parallel"
        # c1_size axis as block_axis and n_size axis as ub split axis
        # can raise dma copy data size and dichotomy efficiency
        ub_split_axis = 0
        ub_split_inner = 1
        if c1_size >= core_num and c1_size % core_num == 0:
            n_inner = n_size
        else:
            n_inner = n_size // core_num

        for i in range(n_inner, 0, -1):
            if n_inner % i != 0:
                continue
            if h_size*w_size*c0_size*i > max_ub_count:
                continue

            ub_split_inner = i
            break
        return ub_split_axis, ub_split_inner

    bound_size = max_ub_count
    split_axis = block_tiling_axis
    step = -1
    temp_size = 1
    need_split = False
    for i in range(last_axis, block_tiling_axis + step, step):
        temp_size = temp_size * shape[i]
        if temp_size >= bound_size:
            split_axis = i
            temp_size = temp_size / shape[i]
            need_split = True
            break

    split_size = 1
    # split the split axis
    if need_split:
        for i in range(1, shape[split_axis] + 1, 1):
            if (temp_size * i) == bound_size:
                split_size = i
                break
            if (temp_size * i) > bound_size:
                split_size = i - 1
                split_size = get_nearest_factor(shape[split_axis],
                                                split_size)
                break
    else:
        split_size = block_tiling_inner_loop

    if split_axis == block_tiling_axis \
                      and split_size > block_tiling_inner_loop:
        split_size = block_tiling_inner_loop

    ub_split_inner = split_size
    ub_split_axis = split_axis

    return ub_split_axis, ub_split_inner


def _map_apend(input_map, key, value):
    if input_map.get(key):
        if isinstance(value, list):
            for sub_v in value:
                if sub_v not in input_map[key]:
                    input_map[key].append(sub_v)
        else:
            if value not in input_map[key]:
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
                "elewise_binary_sub": "vector_sub"}
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


def _do_cache_read(sch_list, input2dst_tensor_map, exclude_tensor):
    """
    cache read
    """
    sch = sch_list[0]
    input_tensor_buffer_map = {}
    for tensor in input2dst_tensor_map:
        if tensor not in exclude_tensor:
            input_tensor = sch.cache_read(tensor, cce.scope_ubuf,
                                          input2dst_tensor_map[tensor])
            input_tensor_buffer_map[tensor] = input_tensor
    sch_list[0] = sch
    return input_tensor_buffer_map


def _do_cache_write(sch_list, mid_tensor_dst_tensor_map,
                    exclude_tensor, broadcast_tensors):
    """
    cache write
    """
    sch = sch_list[0]
    mid_tensor_buffer_map = {}
    broadcast_tensor_buffers = []
    for tensor in mid_tensor_dst_tensor_map:
        if tensor not in exclude_tensor:
            buffer_tensor = sch.cache_write(tensor, cce.scope_ubuf)
            mid_tensor_buffer_map[tensor] = buffer_tensor

            if tensor in broadcast_tensors:
                broadcast_tensor_buffers.append(buffer_tensor)

    sch_list[0] = sch
    return mid_tensor_buffer_map, broadcast_tensor_buffers


def _do_compute_inline(sch_list, mid_tensor_dst_tensor_map, exclude_tensor):
    """
    compute inline
    """
    sch = sch_list[0]
    for tensor in mid_tensor_dst_tensor_map:
        if tensor not in exclude_tensor:
            sch[tensor].compute_inline()
    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-arguments
def _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map, mid_tensor_buffer_map,
                  broadcast_tensor_buffers, shape_input,
                  ub_split_axis, split_factor):
    """
    emit insn
    """
    sch = sch_list[0]
    for tensor in input_tensor_buffer_map:
        buffer_tensor = input_tensor_buffer_map[tensor]
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "dma_copy")

    for tensor in mid_out_read_buffer_map:
        buffer_tensor = mid_out_read_buffer_map[tensor]
        sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")
        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], "phony_insn")

    batch = shape_input[0]
    c1_size = shape_input[1]
    c0_size = shape_input[4]

    for tensor in mid_tensor_buffer_map:
        buffer_tensor = mid_tensor_buffer_map[tensor]
        if buffer_tensor in broadcast_tensor_buffers:
            shape = tensor.shape
            shape_size = reduce(lambda i, j: i * j, shape)
            if shape_size.value // (batch * c1_size * c0_size) == 1 or \
                    (ub_split_axis == 3 and split_factor == 1):
                insn = _get_emit_insn_map(tensor)
            else:
                insn = _get_emit_insn_map_for_broadcast(tensor)
        else:
            insn = _get_emit_insn_map(tensor)

        sch[buffer_tensor].emit_insn(buffer_tensor.op.axis[0], insn)
    sch_list[0] = sch


# pylint: disable=too-many-arguments
def _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   final_out_buffer_list, compute_at_axis_list):
    """
    compute at
    """
    if not isinstance(sch_list, list) \
            or not isinstance(final_out_buffer_list, list):
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
        shape = te.lang.cce.util.shape_to_list(tensor.shape)
        if shape == shape_input:
            sch[buffer_tensor].compute_at(sch[final_out_buffer_1], compute_at_axis_1)
        else:
            sch[buffer_tensor].compute_at(sch[final_out_buffer_2], compute_at_axis_2)

    sch_list[0] = sch


def _do_double_buffer(sch_list, shape_input, outer_loop,
                      input_tensor_buffer_map):
    """
    double buffer
    """
    sch = sch_list[0]
    if outer_loop > 2:
        # pylint: disable=consider-using-enumerate
        for tensor in input_tensor_buffer_map:
            shape = te.lang.cce.util.shape_to_list(tensor.shape)
            if shape == shape_input:
                buffer_tensor = input_tensor_buffer_map[tensor]
                sch[buffer_tensor].double_buffer()
                break
    sch_list[0] = sch


def schedule_cut_h_or_w_twice(
        sch_list, res, shape_input, ub_split_axis, split_factor,
        input_tensor_buffer_map, mid_out_read_buffer_map,
        mid_tensor_buffer_map, final_out_tensor_list,
        broadcast_tensor_buffers, is_keep_dim):
    """
    cut h
    """
    # pylint: disable=too-many-locals, too-many-statements
    sch = sch_list[0]

    final_out_tensor = final_out_tensor_list[0]

    core_num = cceconf.get_soc_spec("CORE_NUM")
    final_out_tensor_block_outer, final_out_tensor_block_inner =\
        sch[final_out_tensor].split(
            final_out_tensor.op.reduce_axis[ub_split_axis - 1],
            nparts=core_num)

    inner_loop = shape_input[ub_split_axis] // core_num
    factors = _get_factors_of_positive_integer(inner_loop)
    split_factor = _find_closest_factor(factors, split_factor)

    sch[final_out_tensor].split(final_out_tensor_block_inner,
                                factor=split_factor)

    final_out_tensor_ub_rf, _ = sch.rfactor(final_out_tensor,
                                            final_out_tensor_block_outer)
    final_out_tensor_global_list = sch.cache_write(final_out_tensor_list,
                                                   "")

    final_tensors_index_res = []
    for tensor in final_out_tensor_list:
        # pylint: disable=consider-using-enumerate
        for i in range(0, len(res)):
            if tensor == res[i]:
                final_tensors_index_res.append(i)
                break
    # pylint: disable=consider-using-enumerate
    for i in range(0, len(final_out_tensor_global_list)):
        res[final_tensors_index_res[i]] = final_out_tensor_global_list[i]


    sch[final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

    final_out_tensor_global = final_out_tensor_global_list[0]

    if is_keep_dim:
        sch[final_out_tensor_global].reorder(
            final_out_tensor_global.op.reduce_axis[0],
            final_out_tensor_global.op.axis[0],
            final_out_tensor_global.op.axis[1],
            final_out_tensor_global.op.axis[2],
            final_out_tensor_global.op.axis[3],
            final_out_tensor_global.op.axis[4])

        reorder_list = []
        for i in range(5):
            reorder_list.append(final_out_tensor_ub_rf.op.axis[i])

        if ub_split_axis == 2:
            reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[0])
            reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[2])
            reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[3])
            reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[1])
        else:
            for i in range(4):
                reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[i])
        reorder_list.append(final_out_tensor_ub_rf.op.axis[5])

        sch[final_out_tensor_ub_rf].reorder(*reorder_list)
    else:
        sch[final_out_tensor_global].reorder(
            final_out_tensor_global.op.reduce_axis[0],
            final_out_tensor_global.op.axis[0],
            final_out_tensor_global.op.axis[1])

        reorder_list = []
        for i in range(2):
            reorder_list.append(final_out_tensor_ub_rf.op.axis[i])

        if ub_split_axis == 2:
            reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[0])
            reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[2])
            reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[3])
            reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[1])
        else:
            for i in range(4):
                reorder_list.append(final_out_tensor_ub_rf.op.reduce_axis[i])
        reorder_list.append(final_out_tensor_ub_rf.op.axis[5])

        sch[final_out_tensor_ub_rf].reorder(*reorder_list)

    sch[final_out_tensor_ub_rf].compute_at(
        sch[final_out_tensor_global],
        final_out_tensor_global.op.axis[0])

    _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_tensor_ub_rf],
                   [final_out_tensor_ub_rf.op.reduce_axis[2]])

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor_global].bind(
        final_out_tensor_global.op.reduce_axis[0], block)

    outer_loop = shape_input[2] // split_factor
    outer_loop = outer_loop * shape_input[0] * shape_input[1]
    _do_double_buffer(sch_list, shape_input,
                      outer_loop, input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    c0_size = 16
    if ub_split_axis == 2:
        loop_size = split_factor * shape_input[3] * c0_size
    else:
        loop_size = split_factor * c0_size

    if _need_dichotomy_add(loop_size, dtype):
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_tensor_ub_rf.op.reduce_axis[3],
            "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_tensor_ub_rf.op.reduce_axis[3],
            "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map,
                  mid_tensor_buffer_map,
                  broadcast_tensor_buffers,
                  shape_input, ub_split_axis, split_factor)

    sch[final_out_tensor_global].emit_insn(
        final_out_tensor_global.op.axis[1], "dma_copy")
    sch[final_out_tensor].emit_insn(
        sch[final_out_tensor].op.axis[0], "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_fuse_h_n(sch_list, res, shape_input, split_factor,
                      input_tensor_buffer_map, mid_out_read_buffer_map,
                      mid_tensor_buffer_map, final_out_tensor_list,
                      broadcast_tensor_buffers, is_keep_dim):
    """
    fuse h
    """
    sch = sch_list[0]

    final_out_tensor = final_out_tensor_list[0]

    core_num = cceconf.get_soc_spec("CORE_NUM")
    half_core_num = core_num // 2
    final_out_tensor_block_outer, final_out_tensor_block_inner = \
        sch[final_out_tensor].split(final_out_tensor.op.reduce_axis[1],
                                    nparts=half_core_num)

    inner_loop = shape_input[2] // half_core_num
    factors = _get_factors_of_positive_integer(inner_loop)
    split_factor = _find_closest_factor(factors, split_factor)

    sch[final_out_tensor].split(final_out_tensor_block_inner,
                                factor=split_factor)
    fused = sch[final_out_tensor].fuse(final_out_tensor.op.reduce_axis[0],
                                       final_out_tensor_block_outer)
    final_out_tensor_ub_rf, _ = sch.rfactor(final_out_tensor, fused)

    final_out_tensor_global_list = sch.cache_write(final_out_tensor_list, "")

    final_tensors_index_res = []
    for tensor in final_out_tensor_list:
        # pylint: disable=consider-using-enumerate
        for i in range(0, len(res)):
            if tensor == res[i]:
                final_tensors_index_res.append(i)
                break
    # pylint: disable=consider-using-enumerate
    for i in range(0, len(final_out_tensor_global_list)):
        res[final_tensors_index_res[i]] = final_out_tensor_global_list[i]

    final_out_tensor_global = final_out_tensor_global_list[0]

    if is_keep_dim:
        sum_x_global_c1_axis = final_out_tensor_global.op.axis[1]
        sum_x_global_c0_axis = final_out_tensor_global.op.axis[4]

    else:
        sum_x_global_c1_axis = final_out_tensor_global.op.axis[0]
        sum_x_global_c0_axis = final_out_tensor_global.op.axis[1]

    sch[final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)
    if is_keep_dim:
        sch[final_out_tensor_global].reorder(
            final_out_tensor_global.op.reduce_axis[0],
            sum_x_global_c1_axis,
            final_out_tensor_global.op.axis[0],
            final_out_tensor_global.op.axis[2],
            final_out_tensor_global.op.axis[3],
            sum_x_global_c0_axis)
        sch[final_out_tensor_ub_rf].reorder(
            final_out_tensor_ub_rf.op.axis[0],
            final_out_tensor_ub_rf.op.axis[1],
            final_out_tensor_ub_rf.op.reduce_axis[1],
            final_out_tensor_ub_rf.op.reduce_axis[2],
            final_out_tensor_ub_rf.op.reduce_axis[0],
            final_out_tensor_ub_rf.op.axis[5])
    else:
        sch[final_out_tensor_global].reorder(
            final_out_tensor_global.op.reduce_axis[0],
            sum_x_global_c1_axis,
            sum_x_global_c0_axis)
        sch[final_out_tensor_ub_rf].reorder(
            final_out_tensor_ub_rf.op.axis[0],
            final_out_tensor_ub_rf.op.axis[1],
            final_out_tensor_ub_rf.op.reduce_axis[1],
            final_out_tensor_ub_rf.op.reduce_axis[2],
            final_out_tensor_ub_rf.op.reduce_axis[0],
            final_out_tensor_ub_rf.op.axis[2])

    sch[final_out_tensor_ub_rf].compute_at(sch[final_out_tensor_global],
                                           sum_x_global_c1_axis)

    _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_tensor_ub_rf],
                   [final_out_tensor_ub_rf.op.reduce_axis[1]])

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor_global].bind(final_out_tensor_global.op.reduce_axis[0], block)

    outer_loop = shape_input[2] // split_factor
    outer_loop = outer_loop * shape_input[0] * shape_input[1]
    _do_double_buffer(sch_list, shape_input, outer_loop, input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    c0_size = 16
    loop_size = split_factor * shape_input[3] * c0_size

    if _need_dichotomy_add(loop_size, dtype):
        sch[final_out_tensor_ub_rf].emit_insn(final_out_tensor_ub_rf.op.reduce_axis[2],
                                              "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_out_tensor_ub_rf].emit_insn(final_out_tensor_ub_rf.op.reduce_axis[2],
                                              "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map, mid_out_read_buffer_map,
                  mid_tensor_buffer_map, broadcast_tensor_buffers,
                  shape_input, 2, split_factor)

    if is_keep_dim:
        sch[final_out_tensor_global].emit_insn(final_out_tensor_global.op.axis[4], "dma_copy")
        sch[final_out_tensor].emit_insn(sch[final_out_tensor].op.axis[1], "phony_insn")
    else:
        sch[final_out_tensor_global].emit_insn(final_out_tensor_global.op.axis[1], "dma_copy")
        sch[final_out_tensor].emit_insn(sch[final_out_tensor].op.axis[0], "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-arguments
def schedule_cut_c1(sch_list, shape_input, ub_split_axis, split_factor,
                    input_tensor_buffer_map, mid_out_read_buffer_map,
                    mid_tensor_buffer_map, final_out_tensor_list,
                    broadcast_tensor_buffers, is_keep_dim, is_do_double_buffer):
    '''
    bn_update_grad schedule for cut c1
    '''
    sch = sch_list[0]

    final_out_buffer_list = sch.cache_write(final_out_tensor_list,
                                            cce.scope_ubuf)

    final_out_tensor = final_out_tensor_list[0]
    final_out_buffer = final_out_buffer_list[0]

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

    core_num = cceconf.get_soc_spec("CORE_NUM")

    inner_loop = shape_input[ub_split_axis]

    factors = _get_factors_of_positive_integer(inner_loop)
    split_factor = _find_closest_factor(factors, split_factor)

    sum_x_block_outer, sum_x_block_inner =\
        sch[final_out_tensor].split(sum_x_c1_axis, nparts=core_num)

    c1_size = shape_input[1]
    is_need_mte3_opt = False
    if c1_size % core_num == 0 and c1_size // core_num > 1:
        is_need_mte3_opt = True
        sum_x_block_inner_outer, sum_x_block_inner_inner = \
            sch[final_out_tensor].split(sum_x_block_inner, nparts=1)

    if ub_split_axis == 0:
        ub_split_reduce_axis = 0
    else:
        ub_split_reduce_axis = ub_split_axis - 1
    sum_x_ub_outer, sum_x_ub_inner = \
        sch[final_out_buffer].split(
            final_out_buffer.op.reduce_axis[ub_split_reduce_axis],
            factor=split_factor)

    if ub_split_axis not in [0, 2, 3]:
        raise RuntimeError("Batch normalization only support 5D format.")

    if ub_split_axis == 0:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c1_axis,
                                          sum_x_ub_h_axis,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_w_reduce_axis,
                                          sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c1_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_w_reduce_axis,
                                          sum_x_ub_c0_axis)
    elif ub_split_axis == 2:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_w_reduce_axis,
                                          sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_w_reduce_axis,
                                          sum_x_ub_c0_axis)
    else:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_axis,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_c0_axis)

    _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_buffer, final_out_tensor],
                   [sum_x_ub_outer, sum_x_block_outer])

    if is_need_mte3_opt:
        sch[final_out_buffer].compute_at(sch[final_out_tensor], sum_x_block_inner_outer)
    else:
        sch[final_out_buffer].compute_at(sch[final_out_tensor], sum_x_block_inner)

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor].bind(sum_x_block_outer, block)

    if is_do_double_buffer:
        outer_loop = shape_input[ub_split_axis] // split_factor
        if ub_split_axis == 3:
            outer_loop = outer_loop * shape_input[2]

        _do_double_buffer(sch_list, shape_input,
                          outer_loop, input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    c0_size = 16
    loop_size = split_factor * c0_size

    if ub_split_axis == 2:
        loop_size = loop_size * shape_input[3]

    if ub_split_axis == 0:
        loop_size = loop_size * shape_input[2]

    if _need_dichotomy_add(loop_size, dtype):
        sch[final_out_buffer].emit_insn(sum_x_ub_inner,
                                        "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_out_buffer].emit_insn(sum_x_ub_inner,
                                        "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map, mid_out_read_buffer_map,
                  mid_tensor_buffer_map, broadcast_tensor_buffers,
                  shape_input, ub_split_axis, split_factor)

    if is_need_mte3_opt:
        sch[final_out_tensor].emit_insn(sum_x_block_inner_inner, "dma_copy")
    else:
        sch[final_out_tensor].emit_insn(sum_x_c0_axis, "dma_copy")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_general(sch_list, shape_input, ub_split_reduce_axis, split_factor,
                         input_tensor_buffer_map, mid_out_read_buffer_map,
                         mid_tensor_buffer_map, final_out_tensor_list,
                         broadcast_tensor_buffers, is_keep_dim):
    """
    cut general
    """
    sch = sch_list[0]

    final_out_buffer_list = sch.cache_write(final_out_tensor_list, cce.scope_ubuf)
    final_out_tensor = final_out_tensor_list[0]
    final_out_buffer = final_out_buffer_list[0]

    if ub_split_reduce_axis == 1:
        ub_split_axis = 2
    elif ub_split_reduce_axis == 2:
        ub_split_axis = 3
    else:
        raise RuntimeError("Batch normalization only support 5D format.")

    inner_loop = shape_input[ub_split_axis]
    factors = _get_factors_of_positive_integer(inner_loop)
    split_factor = _find_closest_factor(factors, split_factor)

    sum_x_ub_outer, sum_x_ub_inner =\
        sch[final_out_buffer].split(
            final_out_buffer.op.reduce_axis[ub_split_reduce_axis],
            factor=split_factor)

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
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_w_reduce_axis,
                                          sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_w_reduce_axis,
                                          sum_x_ub_c0_axis)
    else:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_c0_axis)
        else:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c1_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_axis,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_c0_axis)

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
    if ub_split_axis == 2:
        loop_size = loop_size*shape_input[3]

    if ub_split_axis == 3:
        sch[final_out_buffer].emit_insn(sum_x_ub_inner, "vector_reduce_sum")
    else:
        if _need_dichotomy_add(loop_size, dtype):
            sch[final_out_buffer].emit_insn(
                sum_x_ub_inner,
                "vector_dichotomy_add_for_bn_reduce")
        else:
            sch[final_out_buffer].emit_insn(sum_x_ub_inner, "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map, mid_tensor_buffer_map,
                  broadcast_tensor_buffers, shape_input,
                  ub_split_axis, split_factor)

    sch[final_out_tensor].emit_insn(sum_x_c0_axis, "dma_copy")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_batch_model_para(
        sch_list, res, shape_input, ub_split_reduce_axis,
        split_factor, input_tensor_buffer_map,
        mid_out_read_buffer_map, mid_tensor_buffer_map,
        final_out_tensor_list,
        broadcast_tensor_buffers, is_keep_dim):
    """
    cut batch
    """
    sch = sch_list[0]

    final_out_tensor = final_out_tensor_list[0]

    batch_split_factor = 2
    c1_split_factor = 8

    final_out_batch_outer, _ = \
        sch[final_out_tensor].split(
            final_out_tensor.op.reduce_axis[0],
            factor=batch_split_factor)

    final_out_tensor_ub_rf, _ = sch.rfactor(final_out_tensor,
                                            final_out_batch_outer)

    final_out_tensor_global_list = sch.cache_write(final_out_tensor_list,
                                                   "")

    final_tensors_index_res = []
    for tensor in final_out_tensor_list:
        for i, tensor_res in enumerate(res):
            if tensor == tensor_res:
                final_tensors_index_res.append(i)
                break

    for i, tensor in enumerate(final_out_tensor_global_list):
        res[final_tensors_index_res[i]] = tensor

    sch[final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

    final_out_tensor_global = final_out_tensor_global_list[0]


    if is_keep_dim:
        rf_c1_outer, rf_c1_inner = \
            sch[final_out_tensor_ub_rf].split(
                final_out_tensor_ub_rf.op.axis[2],
                factor=c1_split_factor)

        out_c1_outer, out_c1_inner = \
            sch[final_out_tensor_global].split(
                final_out_tensor_global.op.axis[1],
                factor=c1_split_factor)

        sch[final_out_tensor_global].reorder(
            final_out_tensor_global.op.reduce_axis[0],
            out_c1_outer,
            final_out_tensor_global.op.axis[0],
            out_c1_inner,
            final_out_tensor_global.op.axis[2],
            final_out_tensor_global.op.axis[3],
            final_out_tensor_global.op.axis[4]  # C0 axis
        )
        sch[final_out_tensor_ub_rf].reorder(
            final_out_tensor_ub_rf.op.axis[0],
            rf_c1_outer,
            rf_c1_inner,
            final_out_tensor_ub_rf.op.axis[1],
            final_out_tensor_ub_rf.op.axis[3],
            final_out_tensor_ub_rf.op.axis[4],
            final_out_tensor_ub_rf.op.reduce_axis[2],
            final_out_tensor_ub_rf.op.reduce_axis[0],
            final_out_tensor_ub_rf.op.reduce_axis[1],
            final_out_tensor_ub_rf.op.axis[5]  # C0 axis
        )
    else:
        rf_c1_outer, rf_c1_inner = \
            sch[final_out_tensor_ub_rf].split(
                final_out_tensor_ub_rf.op.axis[1],
                factor=c1_split_factor)

        out_c1_outer, out_c1_inner = \
            sch[final_out_tensor_global].split(
                final_out_tensor_global.op.axis[0],
                factor=c1_split_factor)

        sch[final_out_tensor_global].reorder(
            final_out_tensor_global.op.reduce_axis[0],
            out_c1_outer,
            out_c1_inner,
            final_out_tensor_global.op.axis[1]  # C0 axis
        )
        sch[final_out_tensor_ub_rf].reorder(
            final_out_tensor_ub_rf.op.axis[0],
            rf_c1_outer,
            rf_c1_inner,
            final_out_tensor_ub_rf.op.reduce_axis[2],
            final_out_tensor_ub_rf.op.reduce_axis[0],
            final_out_tensor_ub_rf.op.reduce_axis[1],
            final_out_tensor_ub_rf.op.axis[2]  # C0 axis
        )

    _ = sch[final_out_tensor_ub_rf].fuse(
        final_out_tensor_ub_rf.op.axis[0],
        rf_c1_outer,)

    out_fused_axis = sch[final_out_tensor_global].fuse(
        final_out_tensor_global.op.reduce_axis[0],
        out_c1_outer,)

    sch[final_out_tensor_ub_rf].compute_at(
        sch[final_out_tensor_global],
        out_fused_axis)

    _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_tensor_ub_rf, final_out_tensor_global],
                   [rf_c1_inner, out_fused_axis])

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor_global].bind(out_fused_axis, block)

    sch[final_out_tensor_ub_rf].emit_insn(
        final_out_tensor_ub_rf.op.reduce_axis[2],
        "vector_dichotomy_add_for_bn_reduce")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map,
                  mid_tensor_buffer_map,
                  broadcast_tensor_buffers,
                  shape_input, ub_split_reduce_axis, split_factor)

    sch[final_out_tensor_global].emit_insn(out_c1_inner, "dma_copy")

    sch[final_out_tensor].emit_insn(sch[final_out_tensor].op.axis[0],
                                    "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_batch(sch_list, res, shape_input, ub_split_reduce_axis,
                       split_factor, input_tensor_buffer_map,
                       mid_out_read_buffer_map, mid_tensor_buffer_map,
                       final_out_tensor_list,
                       broadcast_tensor_buffers, is_keep_dim,
                       is_do_double_buffer):
    """
    cut batch
    """
    if shape_input in ([32, 16, 14, 14, 16], [32, 16, 15, 15, 16]):
        schedule_cut_batch_model_para(
            sch_list, res, shape_input, ub_split_reduce_axis,
            split_factor, input_tensor_buffer_map,
            mid_out_read_buffer_map, mid_tensor_buffer_map,
            final_out_tensor_list,
            broadcast_tensor_buffers, is_keep_dim)
        return

    sch = sch_list[0]

    final_out_tensor = final_out_tensor_list[0]

    core_num = cceconf.get_soc_spec("CORE_NUM")
    final_out_tensor_block_outer, _ = \
        sch[final_out_tensor].split(
            final_out_tensor.op.reduce_axis[0], nparts=core_num)

    final_out_tensor_ub_rf, _ = sch.rfactor(final_out_tensor,
                                            final_out_tensor_block_outer)

    final_out_tensor_global_list = sch.cache_write(final_out_tensor_list,
                                                   "")

    final_tensors_index_res = []
    for tensor in final_out_tensor_list:
        for i, tensor_res in enumerate(res):
            if tensor == tensor_res:
                final_tensors_index_res.append(i)
                break

    for i, tensor in enumerate(final_out_tensor_global_list):
        res[final_tensors_index_res[i]] = tensor

    sch[final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)

    final_out_tensor_global = final_out_tensor_global_list[0]

    if ub_split_reduce_axis == 0:
        final_out_rf_outer, final_out_rf_inner = \
            sch[final_out_tensor_ub_rf].split(
                final_out_tensor_ub_rf.op.reduce_axis[-1],
                factor=split_factor)
    else:
        final_out_rf_outer, final_out_rf_inner = \
            sch[final_out_tensor_ub_rf].split(
                final_out_tensor_ub_rf.op.reduce_axis[
                    ub_split_reduce_axis - 1],
                factor=split_factor)

    if is_keep_dim:
        sch[final_out_tensor_global].reorder(
            final_out_tensor_global.op.reduce_axis[0],
            final_out_tensor_global.op.axis[0],
            final_out_tensor_global.op.axis[1],  # C1 axis
            final_out_tensor_global.op.axis[2],
            final_out_tensor_global.op.axis[3],
            final_out_tensor_global.op.axis[4]  # C0 axis
        )
        if ub_split_reduce_axis == 0:
            sch[final_out_tensor_ub_rf].reorder(
                final_out_tensor_ub_rf.op.axis[0],  # N axis
                final_out_tensor_ub_rf.op.axis[1],
                final_out_tensor_ub_rf.op.axis[2],  # C1 axis
                final_out_tensor_ub_rf.op.axis[3],
                final_out_tensor_ub_rf.op.axis[4],
                final_out_rf_outer,
                final_out_rf_inner,
                final_out_tensor_ub_rf.op.reduce_axis[0],
                final_out_tensor_ub_rf.op.reduce_axis[1],
                final_out_tensor_ub_rf.op.axis[5]  # C0 axis
            )
        elif ub_split_reduce_axis == 1:
            sch[final_out_tensor_ub_rf].reorder(
                final_out_tensor_ub_rf.op.axis[0],  # N axis
                final_out_tensor_ub_rf.op.reduce_axis[2],
                final_out_tensor_ub_rf.op.axis[1],
                final_out_tensor_ub_rf.op.axis[2],  # C1 axis
                final_out_tensor_ub_rf.op.axis[3],
                final_out_tensor_ub_rf.op.axis[4],
                final_out_rf_outer,
                final_out_rf_inner,
                final_out_tensor_ub_rf.op.reduce_axis[1],
                final_out_tensor_ub_rf.op.axis[5]  # C0 axis
            )
        elif ub_split_reduce_axis == 2:
            sch[final_out_tensor_ub_rf].reorder(
                final_out_tensor_ub_rf.op.axis[0],  # N axis
                final_out_tensor_ub_rf.op.reduce_axis[2],
                final_out_tensor_ub_rf.op.axis[1],
                final_out_tensor_ub_rf.op.axis[2],  # C1 axis
                final_out_tensor_ub_rf.op.axis[3],
                final_out_tensor_ub_rf.op.axis[4],
                final_out_tensor_ub_rf.op.reduce_axis[0],
                final_out_rf_outer,
                final_out_rf_inner,
                final_out_tensor_ub_rf.op.axis[5]  # C0 axis
            )
    else:
        sch[final_out_tensor_global].reorder(
            final_out_tensor_global.op.reduce_axis[0],
            final_out_tensor_global.op.axis[0],  # C1 axis
            final_out_tensor_global.op.axis[1]  # C0 axis
        )

        if ub_split_reduce_axis == 0:
            sch[final_out_tensor_ub_rf].reorder(
                final_out_tensor_ub_rf.op.axis[0],
                final_out_tensor_ub_rf.op.axis[1],  # C1 axis
                final_out_rf_outer,
                final_out_rf_inner,
                final_out_tensor_ub_rf.op.reduce_axis[0],
                final_out_tensor_ub_rf.op.reduce_axis[1],
                final_out_tensor_ub_rf.op.axis[2]  # C0 axis
            )
        elif ub_split_reduce_axis == 1:
            sch[final_out_tensor_ub_rf].reorder(
                final_out_tensor_ub_rf.op.axis[0],
                final_out_tensor_ub_rf.op.reduce_axis[2],
                final_out_tensor_ub_rf.op.axis[1],  # C1 axis
                final_out_rf_outer,
                final_out_rf_inner,
                final_out_tensor_ub_rf.op.reduce_axis[1],
                final_out_tensor_ub_rf.op.axis[2]  # C0 axis
            )
        elif ub_split_reduce_axis == 2:
            sch[final_out_tensor_ub_rf].reorder(
                final_out_tensor_ub_rf.op.axis[0],
                final_out_tensor_ub_rf.op.reduce_axis[2],
                final_out_tensor_ub_rf.op.axis[1],  # C1 axis
                final_out_tensor_ub_rf.op.reduce_axis[0],
                final_out_rf_outer,
                final_out_rf_inner,
                final_out_tensor_ub_rf.op.axis[2]  # C0 axis
            )

    sch[final_out_tensor_ub_rf].compute_at(
        sch[final_out_tensor_global],
        final_out_tensor_global.op.reduce_axis[0])

    _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_tensor_ub_rf, final_out_tensor_global],
                   [final_out_rf_outer, final_out_tensor_global.op.reduce_axis[0]])

    sch[final_out_tensor_ub_rf].pragma(final_out_rf_outer,
                                       "json_info_batchBindOnly", 1)

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor_global].bind(
        final_out_tensor_global.op.reduce_axis[0], block)

    c0_size = 16
    n_size = shape_input[0]
    c1_size = shape_input[1]
    h_size = shape_input[2]
    w_size = shape_input[3]

    if is_do_double_buffer:
        if ub_split_reduce_axis == 0:
            outer_loop = c1_size
        elif ub_split_reduce_axis == 1:
            outer_loop = h_size // split_factor
            outer_loop = outer_loop*n_size*c1_size
        else:
            outer_loop = w_size // split_factor
            outer_loop = outer_loop*n_size*c1_size*h_size

        _do_double_buffer(sch_list, shape_input, outer_loop,
                          input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    if ub_split_reduce_axis == 0:
        loop_size = n_size // core_num * h_size * w_size * c0_size
    elif ub_split_reduce_axis == 1:
        loop_size = split_factor*shape_input[3]*c0_size
    else:
        loop_size = split_factor*c0_size

    if _need_dichotomy_add(loop_size, dtype):
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_rf_inner,
            "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_rf_inner, "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map,
                  mid_tensor_buffer_map,
                  broadcast_tensor_buffers,
                  shape_input, ub_split_reduce_axis + 1, split_factor)

    if is_keep_dim:
        sch[final_out_tensor_global].emit_insn(
            final_out_tensor_global.op.axis[1], "dma_copy")
    else:
        sch[final_out_tensor_global].emit_insn(
            final_out_tensor_global.op.axis[0], "dma_copy")

    sch[final_out_tensor].emit_insn(sch[final_out_tensor].op.axis[0],
                                    "phony_insn")

    sch_list[0] = sch


def bn_update_grad_schedule(res, input_tensors):
    """
    bn update grad schedule
    """
    if len(res) != 2:
        raise RuntimeError("Batch normalization update grad output nums should be 2.")

    if len(input_tensors) != 4:
        raise RuntimeError("Batch normalization update grad input nums should be 4.")

    input_x_tensor = input_tensors[-1]
    shape_x = te.lang.cce.util.shape_to_list(input_x_tensor.shape)
    if len(shape_x) != 5:
        raise RuntimeError("Batch normalization only support 5D format.")

    old_way_process_shape_list = [
        [2, 1, 2, 1000000, 16],
        [1, 60370, 1, 13, 16],
    ]

    if shape_x in old_way_process_shape_list:
        # pylint: disable=consider-using-enumerate
        for i in range(len(res)):
            if res[i].op.tag != "tuple_reduce_sum":
                continue

            input_tensor = res[i].op.input_tensors[i]
            input_shape = te.lang.cce.util.shape_to_list(input_tensor.shape)
            output_shape = te.lang.cce.util.shape_to_list(res[i].shape)
            reduce_axes = []
            for cur_dim in range(len(output_shape)):
                if output_shape[cur_dim] != input_shape[cur_dim]:
                    reduce_axes.append(cur_dim)
            res[i] = te.lang.cce.sum(input_tensor, reduce_axes, keepdims=True)

        return None

    tensor_list = []
    tensor_list_dst_tensor_map = {}
    input2dst_tensor_map = {}
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
            input2dst_tensor_map[tensor] = list(set(tensor_list_dst_tensor_map[tensor]))
        else:
            mid_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[tensor]
        if tensor.op.tag.find("broadcast") != -1:
            broadcast_tensor_list.append(tensor)

    out_tensor = final_out_tensor_list[0]
    shape_res = te.lang.cce.util.shape_to_list(out_tensor.shape)

    is_keep_dim = True
    if len(shape_x) != len(shape_res):
        is_keep_dim = False

    dtype = input_x_tensor.dtype.lower()
    max_ub_count = get_max_ub_count(dtype, shape_x)
    ub_split_axis, ub_split_inner = get_ub_tiling(shape_x, 2,
                                                  shape_x[2], max_ub_count)
    if ub_split_axis == 3:
        ub_split_reduce_axis = 2
    elif ub_split_axis == 2:
        ub_split_reduce_axis = 1
    elif ub_split_axis == 0:
        ub_split_reduce_axis = 0
    else:
        raise RuntimeError("Batch normalization only support 5D format.")
    split_factor = ub_split_inner
    outer_loop = shape_x[ub_split_axis] // ub_split_inner
    # get device_core_num
    core_num = cceconf.get_soc_spec("CORE_NUM")
    half_core_num = core_num // 2
    batch = shape_x[0]
    c1_size = shape_x[1]

    if c1_size >= core_num:
        pass
    elif ub_split_axis in (2, 3)\
        and outer_loop >= core_num \
        and shape_x[ub_split_axis] % core_num == 0:
        pass
    elif ub_split_axis == 2 and shape_x[ub_split_axis] >= half_core_num \
        and shape_x[ub_split_axis] % half_core_num == 0 \
        and shape_x[0] < core_num:
        pass
    elif batch >= core_num:
        pass
    else:
        sch = tvm.create_schedule(
            [out.op for out in final_out_tensor_list])
        sch_list = [sch]
        atomic_sch = ReduceAtomicSchedule()
        schedule_valid = atomic_sch.do_schedule(res, sch_list, [])
        if schedule_valid:
            return sch_list[0]
    sch = tvm.create_schedule([out_tensor.op])
    sch_list = [sch]
    exclude_tensor = []
    input_tensor_buffer_map = _do_cache_read(sch_list,
                                             input2dst_tensor_map,
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
    for tensor in broadcast_tensor_list:
        if tensor.op.tag.find("broadcast_for_tensor") != -1:
            cache_write_exclude_tensor.append(tensor)

    mid_tensor_buffer_map, broadcast_tensor_buffers =\
        _do_cache_write(sch_list,
                        mid_tensor_dst_tensor_map,
                        cache_write_exclude_tensor,
                        input_broadcast_tensors)

    for tensor in mid_out_tensor_list:
        tensor_ub = mid_tensor_buffer_map[tensor]
        reuse_tensor_ub = mid_out_read_buffer_map[tensor]
        sch[tensor_ub].reused_by(reuse_tensor_ub)

    _do_compute_inline(sch_list, mid_tensor_dst_tensor_map, mid_out_tensor_list)

    is_do_double_buffer = True
    if shape_x in RESNET_50_SHAPE_LIST:
        is_do_double_buffer = False

    batch = shape_x[0]
    c1_size = shape_x[1]
    if c1_size >= core_num:
        schedule_cut_c1(sch_list, shape_x, ub_split_axis, split_factor,
                        input_tensor_buffer_map, mid_out_read_buffer_map,
                        mid_tensor_buffer_map, final_out_tensor_list,
                        broadcast_tensor_buffers, is_keep_dim,
                        is_do_double_buffer)
    elif ub_split_axis in (2, 3)\
        and outer_loop >= core_num \
        and shape_x[ub_split_axis] % core_num == 0:
        schedule_cut_h_or_w_twice(
            sch_list, res, shape_x, ub_split_axis, split_factor,
            input_tensor_buffer_map, mid_out_read_buffer_map,
            mid_tensor_buffer_map, final_out_tensor_list,
            broadcast_tensor_buffers,
            is_keep_dim)
    elif ub_split_axis == 2 and shape_x[ub_split_axis] >= half_core_num \
        and shape_x[ub_split_axis] % half_core_num == 0 \
        and shape_x[0] < core_num:
        schedule_fuse_h_n(sch_list, res, shape_x, split_factor,
                          input_tensor_buffer_map, mid_out_read_buffer_map,
                          mid_tensor_buffer_map, final_out_tensor_list,
                          broadcast_tensor_buffers, is_keep_dim)
    elif batch >= core_num:
        schedule_cut_batch(sch_list, res, shape_x, ub_split_reduce_axis,
                           split_factor, input_tensor_buffer_map,
                           mid_out_read_buffer_map, mid_tensor_buffer_map,
                           final_out_tensor_list,
                           broadcast_tensor_buffers,
                           is_keep_dim, is_do_double_buffer)
    else:
        schedule_cut_general(sch_list, shape_x, ub_split_reduce_axis,
                             split_factor, input_tensor_buffer_map,
                             mid_out_read_buffer_map, mid_tensor_buffer_map,
                             final_out_tensor_list,
                             broadcast_tensor_buffers,
                             is_keep_dim)

    sch = sch_list[0]

    return sch
