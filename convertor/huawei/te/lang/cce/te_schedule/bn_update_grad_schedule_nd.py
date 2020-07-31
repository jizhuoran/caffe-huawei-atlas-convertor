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

import te.lang.cce
from te import tvm
import te.platform.cce_params as cce
from te import platform as cceconf
from te.platform import cce_util
from .reduce_atomic_schedule import ReduceAtomicSchedule

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

    return loop_size % vector_inst_one_repeat_size == 0 \
           and loop_tail_size % vector_inst_one_repeat_size == 0


def _need_tuple_reduce_sum_convert_to_dichotomy_add(shape_input, dtype):
    """
    tuple_reduce_sum convert to dichotomy_add for enhancing performance
    there are some restrictions to use dichotomy_add:
    1.4D shape
    2.shape[0]=shape[2]=1 and shape_input[3] % one_repeat_size != 0
    """
    if dtype == "float16":
        one_repeat_size = 128
    else:
        one_repeat_size = 64

    if len(shape_input) != 4:
        raise RuntimeError("input shape is not 4D:", shape_input)

    if shape_input[0] != 1 or\
            shape_input[2] != 1 or \
            shape_input[3] % one_repeat_size != 0:
        return False
    return True


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


def get_max_ub_count(dtype):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    # div 2 for align to fp16
    total_size = cceconf.get_soc_spec("UB_SIZE") // 2
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)
    total_size = total_size // dtype_size
    total_size = total_size // 2  # div 2 for double buffer
    if dtype == "float16":
        total_width = 4.5
    else:
        total_width = 2.5

    align_to = 128

    max_bound = total_width * align_to
    max_ub_count = int(total_size / max_bound * align_to)

    return max_ub_count


def need_ub_tiling_for_cut_c(max_ub_count, shape, core_num):
    """
    need ub tiling for cut c
    """
    n_size = shape[0]
    c_size = shape[1]
    h_size = shape[2]
    w_size = shape[3]

    ub_split_axis = 0
    if max_ub_count // (h_size*w_size*n_size) >= 2:
        # NHW_axis total size less than max_ub_count/2,
        # so cut C as ub and block tiling axis.
        # And this case will use schedule_cut_c_for_ub_and_block_tiling
        if c_size < core_num or c_size % core_num != 0 or n_size != 1:
            n_inner = n_size
        else:
            n_inner = c_size // core_num
            ub_split_axis = 1
    else:
        if c_size >= core_num and c_size % core_num == 0:
            n_inner = n_size
        else:
            n_inner = n_size // core_num
    return n_inner, ub_split_axis


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
    c_size = shape[1]
    h_size = shape[2]
    w_size = shape[3]

    if max_ub_count // (h_size*w_size) >= 2 \
            and ((c_size >= core_num and c_size % core_num == 0) \
                 or (n_size >= core_num and n_size % core_num == 0)):
        # ub utilization ratio is small, so use "model parallel"
        # c1_size axis as block_axis and n_size axis as ub split axis
        # can raise dma copy data size and dichotomy efficiency
        ub_split_inner = 1
        n_inner, ub_split_axis = \
            need_ub_tiling_for_cut_c(max_ub_count, shape, core_num)

        for i in range(n_inner, 0, -1):
            if n_inner % i != 0:
                continue
            if h_size*w_size*i > max_ub_count:
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
            input_tensor_buffer = sch.cache_read(tensor, cce.scope_ubuf,
                                                 input2dst_tensor_map[tensor])
            input_tensor_buffer_map[tensor] = input_tensor_buffer
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
            mid_tensor_buffer = sch.cache_write(tensor, cce.scope_ubuf)
            mid_tensor_buffer_map[tensor] = mid_tensor_buffer

            if tensor in broadcast_tensors:
                broadcast_tensor_buffers.append(mid_tensor_buffer)

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


def _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map, mid_tensor_buffer_map):
    """
    emit insn
    """
    sch = sch_list[0]
    for tensor in input_tensor_buffer_map:
        input_tensor = input_tensor_buffer_map[tensor]
        sch[input_tensor].emit_insn(input_tensor.op.axis[0], "dma_copy")

    for tensor in mid_out_read_buffer_map:
        mid_out = mid_out_read_buffer_map[tensor]
        sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")
        sch[mid_out].emit_insn(mid_out.op.axis[0], "phony_insn")

    for tensor in mid_tensor_buffer_map:
        mid_tensor = mid_tensor_buffer_map[tensor]
        insn = _get_emit_insn_map(tensor)
        sch[mid_tensor].emit_insn(mid_tensor.op.axis[0], insn)

    sch_list[0] = sch


def _do_emit_insn_for_cut_c(sch_list, input_tensor_buffer_map,
                            mid_out_read_buffer_map, mid_tensor_buffer_map):
    """
    emit insn for schedule_cut_c_for_ub_and_block_tiling
    some changes comparing to _do_emit_insn:
    replace vadds and vmuls insn
    to realize pragma into inner axis for increase repeat times
    """
    sch = sch_list[0]
    for tensor in input_tensor_buffer_map:
        buffer1 = input_tensor_buffer_map[tensor]
        sch[buffer1].emit_insn(buffer1.op.axis[0], "dma_copy")

    for tensor in mid_out_read_buffer_map:
        buffer2 = mid_out_read_buffer_map[tensor]
        sch[tensor].emit_insn(tensor.op.axis[0], "dma_copy")
        sch[buffer2].emit_insn(buffer2.op.axis[0], "phony_insn")

    for tensor in mid_tensor_buffer_map:
        buffer3 = mid_tensor_buffer_map[tensor]

        insn = _get_emit_insn_map(tensor)

        if insn == "vector_add":
            sch[buffer3].emit_insn(buffer3.op.axis[-1], insn)
        elif insn == "vector_mul":
            if buffer3.op.input_tensors[1].name.find("broadcast") != -1:
                sch[buffer3].emit_insn(buffer3.op.axis[-1], insn)
            else:
                sch[buffer3].emit_insn(buffer3.op.axis[0], insn)
        else:
            sch[buffer3].emit_insn(buffer3.op.axis[0], insn)

    sch_list[0] = sch


# pylint: disable=too-many-arguments
def _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   final_out_buffer_list, compute_at_axis_list):
    """
    compute at
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
        input_tensor = input_tensor_buffer_map[tensor]
        shape = te.lang.cce.util.shape_to_list(tensor.shape)
        if shape == shape_input:
            sch[input_tensor].compute_at(sch[final_out_buffer_1],
                                         compute_at_axis_1)
            # small shape input to be broadcast
        else:
            sch[input_tensor].compute_at(sch[final_out_buffer_2],
                                         compute_at_axis_2)

    for tensor in mid_out_read_buffer_map:
        mid_out_buffer = mid_out_read_buffer_map[tensor]
        sch[mid_out_buffer].compute_at(sch[final_out_buffer_1],
                                       compute_at_axis_1)
        sch[tensor].compute_at(sch[final_out_buffer_1], compute_at_axis_1)

    for tensor in mid_tensor_buffer_map:
        mid_tensor_buffer = mid_tensor_buffer_map[tensor]
        shape = te.lang.cce.util.shape_to_list(tensor.shape)
        if shape == shape_input:
            sch[mid_tensor_buffer].compute_at(sch[final_out_buffer_1],
                                              compute_at_axis_1)
        else:
            sch[mid_tensor_buffer].compute_at(sch[final_out_buffer_2],
                                              compute_at_axis_2)

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
                double_buffer = input_tensor_buffer_map[tensor]
                sch[double_buffer].double_buffer()
                break
    sch_list[0] = sch


def schedule_cut_h_or_w_twice(
        sch_list, res, shape_input, ub_split_axis, split_factor,
        input_tensor_buffer_map, mid_out_read_buffer_map,
        mid_tensor_buffer_map, final_out_tensor_list,
        is_keep_dim):
    """
    cut h
    """
    # pylint: disable=too-many-locals, too-many-statements
    sch = sch_list[0]

    final_out_tensor = final_out_tensor_list[0]

    core_num = cceconf.get_soc_spec("CORE_NUM")
    final_out_tensor_block_outer, final_out_tensor_block_inner = \
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
            final_out_tensor_global.op.axis[3])

        reorder_list = []
        for i in range(4):
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
    sch[final_out_tensor_global].bind(final_out_tensor_global.op.reduce_axis[0]
                                      , block)

    outer_loop = shape_input[2] // split_factor
    outer_loop = outer_loop * shape_input[0] * shape_input[1]
    _do_double_buffer(sch_list, shape_input,
                      outer_loop, input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    c0_size = 16
    if ub_split_axis == 2:
        loop_size = split_factor * shape_input[3] * c0_size
        size = shape_input[2] * shape_input[3] * c0_size
        outer_factor = shape_input[2] // split_factor
    else:
        loop_size = split_factor * c0_size
        size = shape_input[3] * c0_size
        outer_factor = shape_input[3] // split_factor

    loop_tail_size = size - outer_factor * loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_tensor_ub_rf.op.reduce_axis[3],
            "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_tensor_ub_rf.op.reduce_axis[3],
            "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map,
                  mid_tensor_buffer_map)

    sch[final_out_tensor_global].emit_insn(
        final_out_tensor_global.op.axis[1], "dma_copy")
    sch[final_out_tensor].emit_insn(
        sch[final_out_tensor].op.axis[0], "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_fuse_h_n(sch_list, res, shape_input, split_factor,
                      input_tensor_buffer_map, mid_out_read_buffer_map,
                      mid_tensor_buffer_map, final_out_tensor_list,
                      is_keep_dim):
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
    sch[final_out_tensor_global].bind(final_out_tensor_global.op.reduce_axis[0]
                                      , block)

    outer_loop = shape_input[2] // split_factor
    outer_loop = outer_loop * shape_input[0] * shape_input[1]
    _do_double_buffer(sch_list, shape_input, outer_loop,
                      input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    c0_size = 16
    loop_size = split_factor * shape_input[3] * c0_size
    outer_factor = shape_input[2] // split_factor
    size = shape_input[2] * shape_input[3] * c0_size
    loop_tail_size = size - outer_factor * loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_tensor_ub_rf.op.reduce_axis[2],
            "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_tensor_ub_rf.op.reduce_axis[2], "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map, mid_out_read_buffer_map,
                  mid_tensor_buffer_map)

    if is_keep_dim:
        sch[final_out_tensor_global].emit_insn(
            final_out_tensor_global.op.axis[4], "dma_copy")
        sch[final_out_tensor].emit_insn(sch[final_out_tensor].op.axis[1],
                                        "phony_insn")
    else:
        sch[final_out_tensor_global].emit_insn(
            final_out_tensor_global.op.axis[1], "dma_copy")
        sch[final_out_tensor].emit_insn(sch[final_out_tensor].op.axis[0],
                                        "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-arguments
def schedule_cut_c1(sch_list, shape_input, ub_split_reduce_axis, split_factor,
                    input_tensor_buffer_map, mid_out_read_buffer_map,
                    mid_tensor_buffer_map, final_out_tensor_list,
                    is_keep_dim):
    '''
    bn_update_grad schedule for cut c1
    '''
    sch = sch_list[0]

    final_out_buffer_list = sch.cache_write(final_out_tensor_list,
                                            cce.scope_ubuf)

    final_out_tensor = final_out_tensor_list[0]
    final_out_buffer = final_out_buffer_list[0]

    if is_keep_dim:
        sum_x_c_axis = final_out_tensor.op.axis[1]
        sum_x_ub_n_axis = final_out_buffer.op.axis[0]
        sum_x_ub_c_axis = final_out_buffer.op.axis[1]
        sum_x_ub_h_axis = final_out_buffer.op.axis[2]
        sum_x_ub_w_axis = final_out_buffer.op.axis[3]
    else:
        sum_x_c_axis = final_out_tensor.op.axis[0]
        sum_x_ub_c_axis = final_out_buffer.op.axis[0]

    sum_x_ub_n_reduce_axis = final_out_buffer.op.reduce_axis[0]
    sum_x_ub_h_reduce_axis = final_out_buffer.op.reduce_axis[1]
    sum_x_ub_w_reduce_axis = final_out_buffer.op.reduce_axis[2]

    core_num = cceconf.get_soc_spec("CORE_NUM")

    if ub_split_reduce_axis == 0:
        inner_loop = shape_input[ub_split_reduce_axis]
    else:
        inner_loop = shape_input[ub_split_reduce_axis + 1]

    factors = _get_factors_of_positive_integer(inner_loop)
    split_factor = _find_closest_factor(factors, split_factor)

    sum_x_block_outer, sum_x_block_inner = \
        sch[final_out_tensor].split(sum_x_c_axis, nparts=core_num)

    sum_x_ub_outer, sum_x_ub_inner = \
        sch[final_out_buffer].split(
            final_out_buffer.op.reduce_axis[ub_split_reduce_axis],
            factor=split_factor)

    if ub_split_reduce_axis == 1:
        ub_split_axis = 2
    elif ub_split_reduce_axis == 2:
        ub_split_axis = 3
    elif ub_split_reduce_axis == 0:
        ub_split_axis = 0
    else:
        raise RuntimeError("Batch normalization only support ND format.")

    if ub_split_reduce_axis == 0:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c_axis,
                                          sum_x_ub_h_axis,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_w_reduce_axis,
                                          )
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_w_reduce_axis,
                                          )
    elif ub_split_reduce_axis == 1:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_w_reduce_axis,
                                          )
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          sum_x_ub_w_reduce_axis,
                                          )
    else:
        if is_keep_dim:
            sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                          sum_x_ub_c_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_axis,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_w_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          )
        else:
            sch[final_out_buffer].reorder(sum_x_ub_c_axis,
                                          sum_x_ub_n_reduce_axis,
                                          sum_x_ub_h_reduce_axis,
                                          sum_x_ub_outer,
                                          sum_x_ub_inner,
                                          )

    _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_buffer, final_out_tensor],
                   [sum_x_ub_outer, sum_x_block_outer])

    sch[final_out_buffer].compute_at(sch[final_out_tensor], sum_x_block_outer)

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor].bind(sum_x_block_outer, block)

    outer_loop = shape_input[ub_split_axis] // split_factor
    if ub_split_axis == 3:
        outer_loop = outer_loop * shape_input[2]

    _do_double_buffer(sch_list, shape_input,
                      outer_loop, input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    loop_size = split_factor
    outer_factor = shape_input[ub_split_axis] // split_factor
    size = shape_input[ub_split_axis]
    if ub_split_axis == 2:
        loop_size = loop_size * shape_input[3]
        size = size * shape_input[3]
    loop_tail_size = size - outer_factor * loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        # sch[final_out_buffer].emit_insn(sum_x_ub_inner,
        #                                 "vector_dichotomy_add_for_bn_reduce")
        sch[final_out_buffer].emit_insn(sum_x_ub_inner,
                                        "vector_reduce_sum")
    else:
        sch[final_out_buffer].emit_insn(sum_x_ub_inner,
                                        "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map, mid_out_read_buffer_map,
                  mid_tensor_buffer_map)

    sch[final_out_tensor].emit_insn(sum_x_block_inner, "dma_copy")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_general(sch_list, shape_input, ub_split_reduce_axis,
                         split_factor, input_tensor_buffer_map,
                         mid_out_read_buffer_map, mid_tensor_buffer_map,
                         final_out_tensor_list, is_keep_dim):
    """
    cut general
    """
    sch = sch_list[0]

    final_out_buffer_list = sch.cache_write(final_out_tensor_list,
                                            cce.scope_ubuf)
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

    sum_x_ub_outer, sum_x_ub_inner = \
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

    _do_double_buffer(sch_list, shape_input, outer_loop,
                      input_tensor_buffer_map)

    dtype = final_out_tensor.dtype.lower()
    c0_size = 16
    loop_size = split_factor * c0_size
    outer_factor = shape_input[ub_split_axis] // split_factor
    size = shape_input[ub_split_axis] * c0_size
    if ub_split_axis == 2:
        loop_size = loop_size*shape_input[3]
        size = size*shape_input[3]
    loop_tail_size = size - outer_factor*loop_size

    if ub_split_axis == 3:
        sch[final_out_buffer].emit_insn(sum_x_ub_inner, "vector_reduce_sum")
    else:
        if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
            sch[final_out_buffer].emit_insn(
                sum_x_ub_inner,
                "vector_dichotomy_add_for_bn_reduce")
        else:
            sch[final_out_buffer].emit_insn(sum_x_ub_inner,
                                            "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map, mid_tensor_buffer_map)

    sch[final_out_tensor].emit_insn(sum_x_c0_axis, "dma_copy")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_batch(sch_list, res, shape_input, ub_split_reduce_axis,
                       split_factor, input_tensor_buffer_map,
                       mid_out_read_buffer_map, mid_tensor_buffer_map,
                       final_out_tensor_list, is_keep_dim):
    """
    cut batch
    """
    sch = sch_list[0]

    final_out_tensor = final_out_tensor_list[0]

    core_num = cceconf.get_soc_spec("CORE_NUM")
    final_out_tensor_block_outer, _ = \
        sch[final_out_tensor].split(
            final_out_tensor.op.reduce_axis[0], nparts=core_num)

    if ub_split_reduce_axis == 0:
        inner_loop = shape_input[ub_split_reduce_axis] // core_num
    else:
        inner_loop = shape_input[ub_split_reduce_axis + 1]

    factors = _get_factors_of_positive_integer(inner_loop)
    split_factor = _find_closest_factor(factors, split_factor)

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
                   [final_out_tensor_ub_rf],
                   [final_out_rf_outer])

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor_global].bind(
        final_out_tensor_global.op.reduce_axis[0], block)

    c0_size = 16
    n_size = shape_input[0]
    c1_size = shape_input[1]
    h_size = shape_input[2]
    w_size = shape_input[3]

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
    if ub_split_reduce_axis == 1:
        loop_size = n_size // core_num * h_size * w_size * c0_size
        size = split_factor * h_size * w_size * c0_size
        outer_factor = n_size // core_num // split_factor
    elif ub_split_reduce_axis == 1:
        loop_size = split_factor*shape_input[3]*c0_size
        size = shape_input[2]*shape_input[3]*c0_size
        outer_factor = shape_input[2] // split_factor
    else:
        loop_size = split_factor*c0_size
        size = shape_input[3]*c0_size
        outer_factor = shape_input[3] // split_factor

    loop_tail_size = size - outer_factor * loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_rf_inner,
            "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[final_out_tensor_ub_rf].emit_insn(
            final_out_rf_inner, "vector_reduce_sum")

    _do_emit_insn(sch_list, input_tensor_buffer_map,
                  mid_out_read_buffer_map,
                  mid_tensor_buffer_map)

    if is_keep_dim:
        sch[final_out_tensor_global].emit_insn(
            final_out_tensor_global.op.axis[1], "dma_copy")
    else:
        sch[final_out_tensor_global].emit_insn(
            final_out_tensor_global.op.axis[0], "dma_copy")

    sch[final_out_tensor].emit_insn(sch[final_out_tensor].op.axis[0],
                                    "phony_insn")

    sch_list[0] = sch


def schedule_cut_c_for_tiling(sch_list, shape_input,
                              ub_split_axis, split_factor,
                              input_tensor_buffer_map,
                              mid_out_read_buffer_map,
                              mid_tensor_buffer_map,
                              final_out_tensor_list,
                              is_keep_dim):
    '''
    bn_update_grad schedule cut c for ub and block tiling
    case that can use this schedule need satisfy some conditons as flows:
        1.n_size == 1 (not split N axis)
        2.n_size*h_size*w_size < max_ub_count//2
        3.c_size >= core_num and c_size % core_num == 0
    so, this schedule will cut C to increase repeat times as ub and
    block tiling strategy.
    '''
    sch = sch_list[0]

    final_out_buffer_list = sch.cache_write(final_out_tensor_list,
                                            cce.scope_ubuf)
    final_out_tensor = final_out_tensor_list[0]
    final_out_buffer = final_out_buffer_list[0]

    # get axis and reduce_axis for reordering
    if is_keep_dim:
        sum_x_c_axis = final_out_tensor.op.axis[1]
        sum_x_ub_n_axis = final_out_buffer.op.axis[0]
        sum_x_ub_c_axis = final_out_buffer.op.axis[1]
        sum_x_ub_h_axis = final_out_buffer.op.axis[2]
        sum_x_ub_w_axis = final_out_buffer.op.axis[3]
    else:
        sum_x_c_axis = final_out_tensor.op.axis[0]
        sum_x_ub_c_axis = final_out_buffer.op.axis[0]
    sum_x_ub_n_reduce_axis = final_out_buffer.op.reduce_axis[0]
    sum_x_ub_h_reduce_axis = final_out_buffer.op.reduce_axis[1]
    sum_x_ub_w_reduce_axis = final_out_buffer.op.reduce_axis[2]

    core_num = cceconf.get_soc_spec("CORE_NUM")

    input_c_shape = shape_input[ub_split_axis]

    factors = _get_factors_of_positive_integer(input_c_shape)
    split_factor = _find_closest_factor(factors, split_factor)

    # split C axis for ub and block tiling
    sum_x_block_outer, sum_x_block_inner = \
        sch[final_out_tensor].split(sum_x_c_axis, nparts=core_num)
    sum_x_ub_outer, sum_x_ub_inner = \
        sch[final_out_buffer].split(
            sum_x_ub_c_axis,
            factor=split_factor)

    if is_keep_dim:
        sch[final_out_buffer].reorder(sum_x_ub_n_axis,
                                      sum_x_ub_outer,
                                      sum_x_ub_inner,
                                      sum_x_ub_h_axis,
                                      sum_x_ub_w_axis,
                                      sum_x_ub_n_reduce_axis,
                                      sum_x_ub_h_reduce_axis,
                                      sum_x_ub_w_reduce_axis,
                                      )
    else:
        sch[final_out_buffer].reorder(sum_x_ub_outer,
                                      sum_x_ub_inner,
                                      sum_x_ub_n_reduce_axis,
                                      sum_x_ub_h_reduce_axis,
                                      sum_x_ub_w_reduce_axis,
                                      )
    # do compute and bind multi core
    _do_compute_at(sch_list, shape_input, input_tensor_buffer_map,
                   mid_out_read_buffer_map, mid_tensor_buffer_map,
                   [final_out_buffer, final_out_tensor],
                   [sum_x_ub_outer, sum_x_block_outer])

    sch[final_out_buffer].compute_at(sch[final_out_tensor], sum_x_block_outer)

    block = tvm.thread_axis("blockIdx.x")
    sch[final_out_tensor].bind(sum_x_block_outer, block)

    # do double buffer
    outer_loop = shape_input[ub_split_axis] // split_factor // core_num
    _do_double_buffer(sch_list, shape_input,
                      outer_loop, input_tensor_buffer_map)

    # do emit insn
    dtype = final_out_tensor.dtype.lower()
    if _need_tuple_reduce_sum_convert_to_dichotomy_add(shape_input, dtype):
        # use this branch to optimize dichotomy add for tuple_reduce_sum
        sch[final_out_buffer].emit_insn(
            sum_x_ub_inner,
            "vector_tuple_reduce_sum_for_bn_update_grad")
    else:
        sch[final_out_buffer].emit_insn(sum_x_ub_w_reduce_axis,
                                        "vector_reduce_sum")

    _do_emit_insn_for_cut_c(sch_list, input_tensor_buffer_map,
                            mid_out_read_buffer_map, mid_tensor_buffer_map)

    sch[final_out_tensor].emit_insn(sum_x_block_inner, "dma_copy")

    sch_list[0] = sch


def bn_update_grad_schedule_nd(res, input_tensors):
    """
    bn update grad schedule
    """
    if len(res) != 2:
        raise RuntimeError("Batch normalization update grad output nums \
                            should be 3.")

    if len(input_tensors) != 4:
        raise RuntimeError("Batch normalization update grad input nums \
                            should be 4.")

    # input_x shape: Nz(NCW1H1H0W0)
    input_x_tensor = input_tensors[-1]
    shape_x = te.lang.cce.util.shape_to_list(input_x_tensor.shape)

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
            input2dst_tensor_map[tensor] = list(
                set(tensor_list_dst_tensor_map[tensor]))
        else:
            mid_tensor_dst_tensor_map[tensor] = \
                tensor_list_dst_tensor_map[tensor]
        if tensor.op.tag.find("broadcast") != -1:
            broadcast_tensor_list.append(tensor)

    out_tensor = final_out_tensor_list[0]
    shape_res = te.lang.cce.util.shape_to_list(out_tensor.shape)

    is_keep_dim = True
    if len(shape_x) != len(shape_res):
        is_keep_dim = False

    dtype = input_x_tensor.dtype.lower()
    max_ub_count = get_max_ub_count(dtype)
    ub_split_axis, ub_split_inner = get_ub_tiling(shape_x, 2, shape_x[2],
                                                  max_ub_count)

    def get_split_reduce_axis(ub_split_axis):
        """
        get split reduce axis
        """
        ub_split_reduce_axis = 1
        if ub_split_axis == 3:
            ub_split_reduce_axis = 2
        elif ub_split_axis == 2:
            ub_split_reduce_axis = 1
        elif ub_split_axis == 0:
            ub_split_reduce_axis = 0
        elif ub_split_axis == 1:
            pass
        else:
            raise RuntimeError("Cannot find UB tiling.")
        return ub_split_reduce_axis

    ub_split_reduce_axis = get_split_reduce_axis(ub_split_axis)

    split_factor = ub_split_inner
    outer_loop = shape_x[ub_split_axis] // ub_split_inner
    # get device_core_num
    core_num = cceconf.get_soc_spec("CORE_NUM")
    half_core_num = core_num // 2
    batch = shape_x[0]
    c_size = shape_x[1]

    if c_size >= core_num:
        pass
    elif ub_split_axis in (2, 3) \
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
    input_tensor_buffer_map = _do_cache_read(
        sch_list, input2dst_tensor_map, exclude_tensor)
    mid_out_tensor_dst_tensor_map = {}
    mid_out_read_buffer_map = {}
    for tensor in mid_out_tensor_list:
        dst_tensor = mid_tensor_dst_tensor_map[tensor]
        mid_out_tensor_dst_tensor_map[tensor] = dst_tensor
        mid_out_read_buffer_map = _do_cache_read(
            sch_list, mid_out_tensor_dst_tensor_map, exclude_tensor)

    cache_write_exclude_tensor = []
    for tensor in broadcast_tensor_list:
        if tensor.op.tag.find("broadcast_for_tensor") != -1:
            cache_write_exclude_tensor.append(tensor)

    mid_tensor_buffer_map, _ = _do_cache_write(
        sch_list, mid_tensor_dst_tensor_map,
        cache_write_exclude_tensor, input_broadcast_tensors)

    for tensor in mid_out_tensor_list:
        tensor_ub = mid_tensor_buffer_map[tensor]
        reuse_tensor_ub = mid_out_read_buffer_map[tensor]
        sch[tensor_ub].reused_by(reuse_tensor_ub)

    _do_compute_inline(sch_list, mid_tensor_dst_tensor_map,
                       mid_out_tensor_list)

    batch = shape_x[0]
    c_size = shape_x[1]
    if c_size >= core_num:
        if ub_split_axis == 1:
            schedule_cut_c_for_tiling(sch_list, shape_x,
                                      ub_split_axis, split_factor,
                                      input_tensor_buffer_map,
                                      mid_out_read_buffer_map,
                                      mid_tensor_buffer_map,
                                      final_out_tensor_list,
                                      is_keep_dim)
        else:
            schedule_cut_c1(sch_list, shape_x, ub_split_reduce_axis,
                            split_factor, input_tensor_buffer_map,
                            mid_out_read_buffer_map,
                            mid_tensor_buffer_map, final_out_tensor_list,
                            is_keep_dim)

    elif ub_split_axis in (2, 3) \
            and outer_loop >= core_num \
            and shape_x[ub_split_axis] % core_num == 0:
        schedule_cut_h_or_w_twice(
            sch_list, res, shape_x, ub_split_axis, split_factor,
            input_tensor_buffer_map, mid_out_read_buffer_map,
            mid_tensor_buffer_map, final_out_tensor_list,
            is_keep_dim)
    elif ub_split_axis == 2 and shape_x[ub_split_axis] >= half_core_num \
            and shape_x[ub_split_axis] % half_core_num == 0 \
            and shape_x[0] < core_num:
        schedule_fuse_h_n(sch_list, res, shape_x, split_factor,
                          input_tensor_buffer_map, mid_out_read_buffer_map,
                          mid_tensor_buffer_map, final_out_tensor_list,
                          is_keep_dim)
    elif batch >= core_num:
        schedule_cut_batch(sch_list, res, shape_x, ub_split_reduce_axis,
                           split_factor, input_tensor_buffer_map,
                           mid_out_read_buffer_map, mid_tensor_buffer_map,
                           final_out_tensor_list,
                           is_keep_dim)
    else:
        schedule_cut_general(sch_list, shape_x, ub_split_reduce_axis,
                             split_factor, input_tensor_buffer_map,
                             mid_out_read_buffer_map, mid_tensor_buffer_map,
                             final_out_tensor_list,
                             is_keep_dim)

    sch = sch_list[0]

    return sch
