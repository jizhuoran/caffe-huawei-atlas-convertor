#!/usr/bin/env python # pylint: disable=too-many-lines
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
from .util import get_nearest_factor
from .util import DTYPE_WIDTH_MAP

MAX_SHAPE_NUM = 10000000

def reset_mask_insn(ib_expr, type_, bits=128, mask_func=None):
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


# pylint: disable=unused-argument
def _need_dichotomy_add(loop_size, loop_tail_size, dtype):
    """
    need dichotomy add
    """
    return True


def _get_factors_of_positive_integer(value):
    """
    get factors of positive integer
    """
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
    total_width = 4
    if not total_width:
        raise RuntimeError("Can not calculate with no compute")
    align_to = 128

    max_bound = total_width * align_to
    max_ub_count = int(total_size // max_bound * align_to)

    return max_ub_count


# pylint: disable=too-many-branches,too-many-locals
def get_ub_tiling(shape, block_tiling_axis,
                  block_tiling_inner_loop, max_ub_count):
    '''
    do ub tiling, find ub tiling axis
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
                split_size = get_nearest_factor(shape[split_axis], split_size)
                break
    else:
        split_size = block_tiling_inner_loop

    if split_axis == block_tiling_axis and split_size > block_tiling_inner_loop:
        split_size = block_tiling_inner_loop

    ub_split_inner = split_size
    ub_split_axis = split_axis

    return ub_split_axis, ub_split_inner


# pylint: disable=too-many-locals, too-many-statements, too-many-arguments
def schedule_cut_h_twice(sch_list, res_list, sum_x, square_sum_x,
                         data_ub, cast_0_ub, data_mul_ub,
                         split_factor, is_keep_dim):
    """
    cut h twice
    """
    sch = sch_list[0]
    shape_input = te.lang.cce.util.shape_to_list(data_ub.shape)

    core_num = cceconf.get_soc_spec("CORE_NUM")

    sum_x_block_outer, sum_x_block_inner =\
        sch[sum_x].split(sum_x.op.reduce_axis[1], nparts=core_num)
    inner_loop = shape_input[2] // core_num

    factors = _get_factors_of_positive_integer(inner_loop)
    split_factor = _find_closest_factor(factors, split_factor)

    sch[sum_x].split(sum_x_block_inner, factor=split_factor)

    outer_loop = shape_input[2] // split_factor
    outer_loop = outer_loop * shape_input[0] * shape_input[1]

    sum_x_ub_rf, _ = sch.rfactor(sum_x, sum_x_block_outer)

    sum_x_global, square_sum_x_global = sch.cache_write([sum_x, square_sum_x],
                                                        "")
    res_list[0] = sum_x_global
    res_list[1] = square_sum_x_global

    sch[sum_x_ub_rf].set_scope(cce.scope_ubuf)

    if is_keep_dim:
        sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0],
                                  sum_x_global.op.axis[0],
                                  sum_x_global.op.axis[1], # C1 axis
                                  sum_x_global.op.axis[2],
                                  sum_x_global.op.axis[3],
                                  sum_x_global.op.axis[4]) # C0 axis

        sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],
                                 sum_x_ub_rf.op.axis[1], # N axis
                                 sum_x_ub_rf.op.axis[2], # C1 axis
                                 sum_x_ub_rf.op.axis[3],
                                 sum_x_ub_rf.op.axis[4],
                                 sum_x_ub_rf.op.reduce_axis[0],
                                 sum_x_ub_rf.op.reduce_axis[2],
                                 sum_x_ub_rf.op.reduce_axis[3],
                                 sum_x_ub_rf.op.reduce_axis[1],
                                 sum_x_ub_rf.op.axis[5]) # C0 axis
    else:
        sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0],
                                  sum_x_global.op.axis[0], # C1 axis
                                  sum_x_global.op.axis[1]) # C0 axis

        sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],
                                 sum_x_ub_rf.op.axis[1], # C1 axis
                                 sum_x_ub_rf.op.reduce_axis[0],
                                 sum_x_ub_rf.op.reduce_axis[2],
                                 sum_x_ub_rf.op.reduce_axis[3],
                                 sum_x_ub_rf.op.reduce_axis[1],
                                 sum_x_ub_rf.op.axis[2]) # C0 axis

    if is_keep_dim:
        sch[sum_x_ub_rf].compute_at(sch[sum_x_global], sum_x_global.op.axis[1])
    else:
        sch[sum_x_ub_rf].compute_at(sch[sum_x_global], sum_x_global.op.axis[0])

    sch[data_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[2])
    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[2])
    sch[data_mul_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[2])
    #
    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x_global].bind(sum_x_global.op.reduce_axis[0], block)

    if outer_loop > 2:
        sch[data_ub].double_buffer()

    dtype = sum_x.dtype.lower()
    c0_size = 16
    loop_size = split_factor * shape_input[3] * c0_size
    size = shape_input[2] * shape_input[3] * c0_size
    outer_factor = shape_input[2] // split_factor
    loop_tail_size = size - outer_factor * loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[sum_x_ub_rf].emit_insn(sum_x_ub_rf.op.reduce_axis[3],
                                   "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[sum_x_ub_rf].emit_insn(sum_x_ub_rf.op.reduce_axis[3],
                                   "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")
    if is_keep_dim:
        sch[sum_x_global].emit_insn(sum_x_global.op.axis[2], "dma_copy")
    else:
        sch[sum_x_global].emit_insn(sum_x_global.op.axis[1], "dma_copy")

    sch[sum_x].emit_insn(sch[sum_x].op.axis[0], "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_fuse_h_n(sch_list, res_list, sum_x, square_sum_x,
                      data_ub, cast_0_ub, data_mul_ub,
                      split_factor, is_keep_dim):
    """
    fuse h
    """
    sch = sch_list[0]
    shape_input = te.lang.cce.util.shape_to_list(data_ub.shape)

    core_num = cceconf.get_soc_spec("CORE_NUM")
    half_core_num = core_num // 2

    sum_x_block_outer, sum_x_block_inner = sch[sum_x].split(
        sum_x.op.reduce_axis[1], nparts=half_core_num)
    inner_loop = shape_input[2] // half_core_num

    factors = _get_factors_of_positive_integer(inner_loop)
    split_factor = _find_closest_factor(factors, split_factor)

    _, _ = sch[sum_x].split(
        sum_x_block_inner, factor=split_factor)

    outer_loop = shape_input[2] // split_factor
    outer_loop = outer_loop * shape_input[0] * shape_input[1]

    fused = sch[sum_x].fuse(sum_x.op.reduce_axis[0], sum_x_block_outer)

    sum_x_ub_rf, _ = sch.rfactor(sum_x, fused)

    sum_x_global, square_sum_x_global = sch.cache_write([sum_x, square_sum_x], "")
    res_list[0] = sum_x_global
    res_list[1] = square_sum_x_global

    if is_keep_dim:
        sum_x_global_c1_axis = sum_x_global.op.axis[1]
        sum_x_global_c0_axis = sum_x_global.op.axis[4]

    else:
        sum_x_global_c1_axis = sum_x_global.op.axis[0]
        sum_x_global_c0_axis = sum_x_global.op.axis[1]

    sch[sum_x_ub_rf].set_scope(cce.scope_ubuf)
    if is_keep_dim:
        sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0],
                                  sum_x_global_c1_axis,
                                  sum_x_global.op.axis[0],
                                  sum_x_global.op.axis[2],
                                  sum_x_global.op.axis[3],
                                  sum_x_global_c0_axis)
        sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],
                                 sum_x_ub_rf.op.axis[1],
                                 sum_x_ub_rf.op.reduce_axis[1],
                                 sum_x_ub_rf.op.reduce_axis[2],
                                 sum_x_ub_rf.op.reduce_axis[0],
                                 sum_x_ub_rf.op.axis[5])
    else:
        sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0],
                                  sum_x_global_c1_axis,
                                  sum_x_global_c0_axis)
        sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],
                                 sum_x_ub_rf.op.axis[1],
                                 sum_x_ub_rf.op.reduce_axis[1],
                                 sum_x_ub_rf.op.reduce_axis[2],
                                 sum_x_ub_rf.op.reduce_axis[0],
                                 sum_x_ub_rf.op.axis[2])

    sch[sum_x_ub_rf].compute_at(sch[sum_x_global], sum_x_global_c1_axis)

    sch[data_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[1])
    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[1])
    sch[data_mul_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[1])

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x_global].bind(sum_x_global.op.reduce_axis[0], block)

    if outer_loop > 2:
        sch[data_ub].double_buffer()

    dtype = sum_x.dtype.lower()
    c0_size = 16
    loop_size = split_factor * shape_input[3] * c0_size
    outer_factor = shape_input[2] // split_factor
    size = shape_input[2] * shape_input[3] * c0_size
    loop_tail_size = size - outer_factor * loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[sum_x_ub_rf].emit_insn(sum_x_ub_rf.op.reduce_axis[2],
                                   "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[sum_x_ub_rf].emit_insn(sum_x_ub_rf.op.reduce_axis[2], "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

    if is_keep_dim:
        sch[sum_x_global].emit_insn(sum_x_global.op.axis[4], "dma_copy")
        sch[sum_x].emit_insn(sch[sum_x].op.axis[1], "phony_insn")
    else:
        sch[sum_x_global].emit_insn(sum_x_global.op.axis[1], "dma_copy")
        sch[sum_x].emit_insn(sch[sum_x].op.axis[0], "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_c1(sch_list, res_list, sum_x, square_sum_x,
                    data_ub, cast_0_ub, data_mul_ub, ub_split_reduce_axis,
                    split_factor, is_keep_dim):
    '''
    bn_reduce schedule for cut c1
    '''
    sch = sch_list[0]
    shape_input = te.lang.cce.util.shape_to_list(data_ub.shape)
    res_list[0] = sum_x
    res_list[1] = square_sum_x

    _, sum_x_ub = sch.cache_write([square_sum_x, sum_x],
                                  cce.scope_ubuf)

    if is_keep_dim:
        sum_x_c1_axis = sum_x.op.axis[1]
        sum_x_c0_axis = sum_x.op.axis[4]
        sum_x_ub_n_axis = sum_x_ub.op.axis[0]
        sum_x_ub_c1_axis = sum_x_ub.op.axis[1]
        sum_x_ub_h_axis = sum_x_ub.op.axis[2]
        sum_x_ub_w_axis = sum_x_ub.op.axis[3]
        sum_x_ub_c0_axis = sum_x_ub.op.axis[4]
    else:
        sum_x_c1_axis = sum_x.op.axis[0]
        sum_x_c0_axis = sum_x.op.axis[1]
        sum_x_ub_c1_axis = sum_x_ub.op.axis[0]
        sum_x_ub_c0_axis = sum_x_ub.op.axis[1]

    sum_x_ub_n_reduce_axis = sum_x_ub.op.reduce_axis[0]
    sum_x_ub_h_reduce_axis = sum_x_ub.op.reduce_axis[1]
    sum_x_ub_w_reduce_axis = sum_x_ub.op.reduce_axis[2]

    core_num = cceconf.get_soc_spec("CORE_NUM")

    sum_x_block_outer, sum_x_block_inner = sch[sum_x].split(sum_x_c1_axis,
                                                            nparts=core_num)

    if ub_split_reduce_axis == 0:
        sum_x_block_inner_outer, sum_x_block_inner_inner =\
            sch[sum_x].split(sum_x_block_inner, nparts=1)

    sum_x_ub_outer, sum_x_ub_inner = \
        sch[sum_x_ub].split(sum_x_ub.op.reduce_axis[ub_split_reduce_axis],
                            factor=split_factor)
    if ub_split_reduce_axis == 1:
        ub_split_axis = 2
    elif ub_split_reduce_axis == 2:
        ub_split_axis = 3
    elif ub_split_reduce_axis == 0:
        ub_split_axis = 0
    else:
        raise RuntimeError("Batch normalization only support 5D format.")

    outer_loop = shape_input[ub_split_axis] // split_factor

    if ub_split_axis == 2:
        outer_loop = outer_loop * shape_input[0]
    elif ub_split_axis == 3:
        outer_loop = outer_loop * shape_input[2]

    if ub_split_reduce_axis == 0:
        if is_keep_dim:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis,
                                  sum_x_ub_h_axis, sum_x_ub_w_axis,
                                  sum_x_ub_outer, sum_x_ub_inner,
                                  sum_x_ub_h_reduce_axis,
                                  sum_x_ub_w_reduce_axis,
                                  sum_x_ub_c0_axis)
        else:
            sch[sum_x_ub].reorder(sum_x_ub_c1_axis, sum_x_ub_outer,
                                  sum_x_ub_inner, sum_x_ub_h_reduce_axis,
                                  sum_x_ub_w_reduce_axis, sum_x_ub_c0_axis)
    elif ub_split_reduce_axis == 1:
        if is_keep_dim:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis,
                                  sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                                  sum_x_ub_outer, sum_x_ub_inner,
                                  sum_x_ub_w_axis, sum_x_ub_w_reduce_axis,
                                  sum_x_ub_c0_axis)
        else:
            sch[sum_x_ub].reorder(sum_x_ub_c1_axis, sum_x_ub_n_reduce_axis,
                                  sum_x_ub_outer, sum_x_ub_inner,
                                  sum_x_ub_w_reduce_axis, sum_x_ub_c0_axis)
    else:
        if is_keep_dim:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis,
                                  sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                                  sum_x_ub_h_reduce_axis, sum_x_ub_w_axis,
                                  sum_x_ub_outer, sum_x_ub_inner,
                                  sum_x_ub_c0_axis)
        else:
            sch[sum_x_ub].reorder(sum_x_ub_c1_axis, sum_x_ub_n_reduce_axis,
                                  sum_x_ub_h_reduce_axis, sum_x_ub_outer,
                                  sum_x_ub_inner, sum_x_ub_c0_axis)

    sch[data_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
    sch[data_mul_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)

    if ub_split_reduce_axis == 0:
        sch[sum_x_ub].compute_at(sch[sum_x], sum_x_block_inner_outer)
    else:
        sch[sum_x_ub].compute_at(sch[sum_x], sum_x_block_inner)

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x].bind(sum_x_block_outer, block)

    if outer_loop >= 2:
        sch[data_ub].double_buffer()

    dtype = sum_x.dtype.lower()
    c0_size = 16
    loop_size = split_factor * c0_size
    outer_factor = shape_input[ub_split_axis] // split_factor
    size = shape_input[ub_split_axis] * c0_size
    if ub_split_axis == 2:
        loop_size = loop_size * shape_input[3]
        size = size * shape_input[3]
    loop_tail_size = size - outer_factor * loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[sum_x_ub].emit_insn(sum_x_ub_inner, "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[sum_x_ub].emit_insn(sum_x_ub_inner, "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

    if ub_split_reduce_axis == 0:
        sch[sum_x].emit_insn(sum_x_block_inner_inner, "dma_copy")
    else:
        sch[sum_x].emit_insn(sum_x_c0_axis, "dma_copy")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_general(sch_list, res_list, sum_x, square_sum_x,
                         data_ub, cast_0_ub, data_mul_ub, ub_split_reduce_axis,
                         split_factor, is_keep_dim):
    """
    cut general
    """
    if not is_keep_dim:
        raise RuntimeError("Bn_reduce only support keep_dim is True.")

    sch = sch_list[0]
    shape_input = te.lang.cce.util.shape_to_list(data_ub.shape)
    res_list[0] = sum_x
    res_list[1] = square_sum_x

    _, sum_x_ub = sch.cache_write([square_sum_x, sum_x], cce.scope_ubuf)
    sum_x_ub_outer, sum_x_ub_inner = sch[sum_x_ub].split(
        sum_x_ub.op.reduce_axis[ub_split_reduce_axis], factor=split_factor)
    if ub_split_reduce_axis == 0:
        ub_split_axis = 0
    elif ub_split_reduce_axis == 1:
        ub_split_axis = 2
    elif ub_split_reduce_axis == 2:
        ub_split_axis = 3
    else:
        raise RuntimeError("Batch normalization only support 5D format.")
    outer_loop = shape_input[ub_split_axis] // split_factor
    if ub_split_axis == 3:
        outer_loop = outer_loop * shape_input[2]

    sum_x_c1_axis = sum_x.op.axis[1]
    sum_x_c0_axis = sum_x.op.axis[4]
    sum_x_ub_n_axis = sum_x_ub.op.axis[0]
    sum_x_ub_c1_axis = sum_x_ub.op.axis[1]
    sum_x_ub_h_axis = sum_x_ub.op.axis[2]
    sum_x_ub_w_axis = sum_x_ub.op.axis[3]
    sum_x_ub_c0_axis = sum_x_ub.op.axis[4]

    sum_x_ub_n_reduce_axis = sum_x_ub.op.reduce_axis[0]
    sum_x_ub_h_reduce_axis = sum_x_ub.op.reduce_axis[1]
    sum_x_ub_w_reduce_axis = sum_x_ub.op.reduce_axis[2]

    if ub_split_reduce_axis == 0:
        sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis,
                              sum_x_ub_outer, sum_x_ub_inner,
                              sum_x_ub_h_axis, sum_x_ub_w_axis,
                              sum_x_ub_h_reduce_axis, sum_x_ub_w_reduce_axis,
                              sum_x_ub_c0_axis)
    elif ub_split_reduce_axis == 1:
        sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis,
                              sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                              sum_x_ub_outer, sum_x_ub_inner,
                              sum_x_ub_w_axis, sum_x_ub_w_reduce_axis,
                              sum_x_ub_c0_axis)
    else:
        sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis,
                              sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                              sum_x_ub_h_reduce_axis, sum_x_ub_w_axis,
                              sum_x_ub_outer, sum_x_ub_inner,
                              sum_x_ub_c0_axis)

    sch[data_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
    sch[data_mul_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)

    sch[sum_x_ub].compute_at(sch[sum_x], sum_x_c1_axis)

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x].bind(sum_x_c1_axis, block)

    if outer_loop > 2:
        sch[data_ub].double_buffer()

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")
    sch[sum_x_ub].emit_insn(sum_x_ub_inner, "vector_reduce_sum")
    sch[sum_x].emit_insn(sum_x_c0_axis, "dma_copy")

    sch_list[0] = sch


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_batch_model_parallerl(
        sch_list, shape_input, res_list, sum_x,
        square_sum_x, data_ub, cast_0_ub,
        data_mul_ub, ub_split_axis,
        split_factor, is_keep_dim):
    """
    cut batch for special shape [32, 16, 14, 14, 16], [32, 16, 15, 15, 16]
    """

    sch = sch_list[0]

    core_num = cceconf.get_soc_spec("CORE_NUM")

    batch_split_factor = 4
    c1_split_factor = 4

    sum_x_block_outer, _ =\
        sch[sum_x].split(sum_x.op.reduce_axis[0], factor=batch_split_factor)

    sum_x_ub_rf, _ = sch.rfactor(sum_x, sum_x_block_outer)

    sum_x_global, square_sum_x_global =\
        sch.cache_write([sum_x, square_sum_x], "")

    res_list[0] = sum_x_global
    res_list[1] = square_sum_x_global

    sch[sum_x_ub_rf].set_scope(cce.scope_ubuf)

    if is_keep_dim:
        rf_c1_outer, rf_c1_inner = \
            sch[sum_x_ub_rf].split(
                sum_x_ub_rf.op.axis[2],
                factor=c1_split_factor)

        out_c1_outer, out_c1_inner = \
            sch[sum_x_global].split(
                sum_x_global.op.axis[1],
                factor=c1_split_factor)

        sch[sum_x_global].reorder(
            sum_x_global.op.reduce_axis[0],
            out_c1_outer,
            sum_x_global.op.axis[0],
            out_c1_inner,
            sum_x_global.op.axis[2],
            sum_x_global.op.axis[3],
            sum_x_global.op.axis[4]  # C0 axis
        )
        sch[sum_x_ub_rf].reorder(
            sum_x_ub_rf.op.axis[0],
            rf_c1_outer,
            rf_c1_inner,
            sum_x_ub_rf.op.axis[1],
            sum_x_ub_rf.op.axis[3],
            sum_x_ub_rf.op.axis[4],
            sum_x_ub_rf.op.reduce_axis[2],
            sum_x_ub_rf.op.reduce_axis[0],
            sum_x_ub_rf.op.reduce_axis[1],
            sum_x_ub_rf.op.axis[5]  # C0 axis
        )

    else:
        rf_c1_outer, rf_c1_inner = \
            sch[sum_x_ub_rf].split(
                sum_x_ub_rf.op.axis[1],
                factor=c1_split_factor)

        out_c1_outer, out_c1_inner = \
            sch[sum_x_global].split(
                sum_x_global.op.axis[0],
                factor=c1_split_factor)

        sch[sum_x_global].reorder(
            sum_x_global.op.reduce_axis[0],
            out_c1_outer,
            out_c1_inner,
            sum_x_global.op.axis[1]  # C0 axis
        )
        sch[sum_x_ub_rf].reorder(
            sum_x_ub_rf.op.axis[0],
            rf_c1_outer,
            rf_c1_inner,
            sum_x_ub_rf.op.reduce_axis[2],
            sum_x_ub_rf.op.reduce_axis[0],
            sum_x_ub_rf.op.reduce_axis[1],
            sum_x_ub_rf.op.axis[2]  # C0 axis
        )

    _ = sch[sum_x_ub_rf].fuse(
        sum_x_ub_rf.op.axis[0], rf_c1_outer)

    out_fused_axis = sch[sum_x_global].fuse(
        sum_x_global.op.reduce_axis[0], out_c1_outer)

    sch[sum_x_ub_rf].compute_at(sch[sum_x_global], out_fused_axis)

    sch[data_ub].compute_at(sch[sum_x_ub_rf], rf_c1_inner)

    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub_rf], rf_c1_inner)
    sch[data_mul_ub].compute_at(sch[sum_x_ub_rf], rf_c1_inner)

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x_global].bind(out_fused_axis, block)

    if ub_split_axis == 0:
        if shape_input[1] > 2:
            sch[data_ub].double_buffer()
    else:
        outer_loop = shape_input[ub_split_axis] // split_factor
        if outer_loop > 2:
            sch[data_ub].double_buffer()

    dtype = sum_x.dtype.lower()
    c0_size = 16
    n_size = shape_input[0]
    h_size = shape_input[2]
    w_size = shape_input[3]
    if ub_split_axis == 0:
        loop_size = n_size // core_num*h_size*w_size*c0_size
        size = split_factor*h_size*w_size*c0_size
        outer_factor = n_size // core_num // split_factor
    elif ub_split_axis == 2:
        loop_size = split_factor*shape_input[3]*c0_size
        size = shape_input[2]*shape_input[3]*c0_size
        outer_factor = shape_input[2] // split_factor
    else:
        loop_size = split_factor*c0_size
        size = shape_input[3]*c0_size
        outer_factor = shape_input[3] // split_factor

    loop_tail_size = size - outer_factor*loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[sum_x_ub_rf].emit_insn(sum_x_ub_rf.op.reduce_axis[2],
                                   "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[sum_x_ub_rf].emit_insn(sum_x_ub_rf.op.reduce_axis[2],
                                   "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

    sch[sum_x_global].emit_insn(out_c1_inner, "dma_copy")

    sch[sum_x].emit_insn(sch[sum_x].op.axis[0], "phony_insn")

    sch = sch_list[0]


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def schedule_cut_batch(sch_list, res_list, sum_x,
                       square_sum_x, data_ub, cast_0_ub,
                       data_mul_ub, ub_split_axis,
                       split_factor, is_keep_dim):
    '''
    bn_reduce schedule for cut batch
    '''
    shape_input = te.lang.cce.util.shape_to_list(data_ub.shape)

    if shape_input in ([32, 16, 14, 14, 16], [32, 16, 15, 15, 16]):
        schedule_cut_batch_model_parallerl(
            sch_list, shape_input, res_list, sum_x,
            square_sum_x, data_ub, cast_0_ub,
            data_mul_ub, ub_split_axis,
            split_factor, is_keep_dim)
        return

    sch = sch_list[0]

    core_num = cceconf.get_soc_spec("CORE_NUM")

    sum_x_block_outer, _ =\
        sch[sum_x].split(sum_x.op.reduce_axis[0], nparts=core_num)

    sum_x_ub_rf, _ = sch.rfactor(sum_x, sum_x_block_outer)

    sum_x_global, square_sum_x_global =\
        sch.cache_write([sum_x, square_sum_x], "")
    res_list[0] = sum_x_global
    res_list[1] = square_sum_x_global

    sch[sum_x_ub_rf].set_scope(cce.scope_ubuf)

    if ub_split_axis == 0:
        sum_x_rf_outer, sum_x_rf_inner = sch[sum_x_ub_rf].split(
            sum_x_ub_rf.op.reduce_axis[-1], factor=split_factor)
    else:
        sum_x_rf_outer, sum_x_rf_inner = sch[sum_x_ub_rf].split(
            sum_x_ub_rf.op.reduce_axis[ub_split_axis - 2], factor=split_factor)

    if is_keep_dim:
        sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0],
                                  sum_x_global.op.axis[0],
                                  sum_x_global.op.axis[1],  # C1 axis
                                  sum_x_global.op.axis[2],
                                  sum_x_global.op.axis[3],
                                  sum_x_global.op.axis[4])  # C0 axis

        if ub_split_axis == 0:
            sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],  # N axis
                                     sum_x_ub_rf.op.axis[1],
                                     sum_x_ub_rf.op.axis[2],  # C1 axis
                                     sum_x_ub_rf.op.axis[3],
                                     sum_x_ub_rf.op.axis[4],
                                     sum_x_rf_outer, sum_x_rf_inner,
                                     sum_x_ub_rf.op.reduce_axis[0],
                                     sum_x_ub_rf.op.reduce_axis[1],
                                     sum_x_ub_rf.op.axis[5])  # C0 axis
        elif ub_split_axis == 2:
            sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],  # N axis
                                     sum_x_ub_rf.op.reduce_axis[2],
                                     sum_x_ub_rf.op.axis[1],
                                     sum_x_ub_rf.op.axis[2],  # C1 axis
                                     sum_x_ub_rf.op.axis[3],
                                     sum_x_ub_rf.op.axis[4],
                                     sum_x_rf_outer, sum_x_rf_inner,
                                     sum_x_ub_rf.op.reduce_axis[1],
                                     sum_x_ub_rf.op.axis[5])  # C0 axis
        elif ub_split_axis == 3:
            sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],  # N axis
                                     sum_x_ub_rf.op.reduce_axis[2],
                                     sum_x_ub_rf.op.axis[1],
                                     sum_x_ub_rf.op.axis[2],  # C1 axis
                                     sum_x_ub_rf.op.axis[3],
                                     sum_x_ub_rf.op.axis[4],
                                     sum_x_ub_rf.op.reduce_axis[0],
                                     sum_x_rf_outer, sum_x_rf_inner,
                                     sum_x_ub_rf.op.axis[5])  # C0 axis
    else:
        sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0],
                                  sum_x_global.op.axis[0],  # C1 axis
                                  sum_x_global.op.axis[1])  # C0 axis

        if ub_split_axis == 2:
            sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],
                                     sum_x_ub_rf.op.reduce_axis[2],
                                     sum_x_ub_rf.op.axis[1],  # C1 axis
                                     sum_x_rf_outer, sum_x_rf_inner,
                                     sum_x_ub_rf.op.reduce_axis[1],
                                     sum_x_ub_rf.op.axis[2])  # C0 axis
        elif ub_split_axis == 3:
            sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0],
                                     sum_x_ub_rf.op.reduce_axis[2],
                                     sum_x_ub_rf.op.axis[1],  # C1 axis
                                     sum_x_ub_rf.op.reduce_axis[0],
                                     sum_x_rf_outer, sum_x_rf_inner,
                                     sum_x_ub_rf.op.axis[2])  # C0 axis

    sch[sum_x_ub_rf].compute_at(sch[sum_x_global],
                                sum_x_global.op.reduce_axis[0])

    sch[data_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)

    if cast_0_ub is not None:
        sch[cast_0_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)
    sch[data_mul_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)

    block = tvm.thread_axis("blockIdx.x")
    sch[sum_x_global].bind(sum_x_global.op.reduce_axis[0], block)

    if ub_split_axis == 0:
        if shape_input[1] > 2:
            sch[data_ub].double_buffer()
    else:
        outer_loop = shape_input[ub_split_axis] // split_factor
        if outer_loop > 2:
            sch[data_ub].double_buffer()

    dtype = sum_x.dtype.lower()
    c0_size = 16
    n_size = shape_input[0]
    h_size = shape_input[2]
    w_size = shape_input[3]
    if ub_split_axis == 0:
        loop_size = n_size // core_num*h_size*w_size*c0_size
        size = split_factor*h_size*w_size*c0_size
        outer_factor = n_size // core_num // split_factor
    elif ub_split_axis == 2:
        loop_size = split_factor*shape_input[3]*c0_size
        size = shape_input[2]*shape_input[3]*c0_size
        outer_factor = shape_input[2] // split_factor
    else:
        loop_size = split_factor*c0_size
        size = shape_input[3]*c0_size
        outer_factor = shape_input[3] // split_factor

    loop_tail_size = size - outer_factor*loop_size

    if _need_dichotomy_add(loop_size, loop_tail_size, dtype):
        sch[sum_x_ub_rf].emit_insn(sum_x_rf_inner,
                                   "vector_dichotomy_add_for_bn_reduce")
    else:
        sch[sum_x_ub_rf].emit_insn(sum_x_rf_inner, "vector_reduce_sum")

    sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
    if cast_0_ub is not None:
        sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
    sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

    if is_keep_dim:
        sch[sum_x_global].emit_insn(sum_x_global.op.axis[1], "dma_copy")
    else:
        sch[sum_x_global].emit_insn(sum_x_global.op.axis[0], "dma_copy")

    sch[sum_x].emit_insn(sch[sum_x].op.axis[0], "phony_insn")

    sch_list[0] = sch


# pylint: disable=too-many-statements
def _is_in_shape_white_list(shape):
    shape_white_list = (
        [1, 2, 56, 19, 16],
        [1, 1, 56, 5, 16],
        [1, 1, 28, 3, 16])
    for i in shape_white_list:
        if shape == i:
            return True
    return False


def bn_reduce_schedule(res, input_tensors):
    """
    bn reduce schedule
    """
    if len(res) != 2:
        raise RuntimeError("Batch normalization reduce output nums should be 2.")

    if len(input_tensors) != 1:
        raise RuntimeError("Batch normalization reduce input nums should be 1.")

    input_tensor = input_tensors[0]
    sum_x = res[0]
    square_sum_x = res[1]
    shape_input = te.lang.cce.util.shape_to_list(input_tensor.shape)
    if len(shape_input) != 5:
        raise RuntimeError("Batch normalization only support 5D format.")

    shape_res = te.lang.cce.util.shape_to_list(sum_x.shape)

    is_keep_dim = True
    if len(shape_input) != len(shape_res):
        is_keep_dim = False

    dtype = input_tensor.dtype.lower()
    max_ub_count = get_max_ub_count(dtype)
    ub_split_axis, ub_split_inner = get_ub_tiling(shape_input,
                                                  2, shape_input[2],
                                                  max_ub_count)
    if ub_split_axis == 3:
        ub_split_reduce_axis = 2
    elif ub_split_axis == 2:
        ub_split_reduce_axis = 1
    elif ub_split_axis == 0:
        ub_split_reduce_axis = 0
    else:
        raise RuntimeError("Batch normalization only support 5D format.")

    split_factor = ub_split_inner
    outer_loop = shape_input[ub_split_axis] // ub_split_inner
    # get device_core_num
    core_num = cceconf.get_soc_spec("CORE_NUM")
    half_core_num = core_num // 2
    batch = shape_input[0]
    c1_size = shape_input[1]

    threshold = 16
    if c1_size >= core_num and shape_input[2] * shape_input[3] > threshold:
        pass
    elif ub_split_axis == 2 and \
            outer_loop >= core_num and \
            shape_input[ub_split_axis] % core_num == 0:
        pass
    elif ub_split_axis == 2 and \
            shape_input[ub_split_axis] >= half_core_num and \
            shape_input[ub_split_axis] % half_core_num == 0 and \
            shape_input[0] < core_num and shape_input[0] == 2:
        pass
    elif batch >= core_num and shape_input[2] * shape_input[3] > threshold:
        pass
    else:
        if not _is_in_shape_white_list(shape_input):
            sch = tvm.create_schedule(
                [out.op for out in res])
            sch_list = [sch]
            atomic_sch = ReduceAtomicSchedule()
            schedule_valid = atomic_sch.do_schedule(res, sch_list, [])
            if schedule_valid:
                return sch_list[0]

    sch = tvm.create_schedule([sum_x.op])
    sch_list = [sch]
    if input_tensor.dtype == "float16":
        data_mul = square_sum_x.op.input_tensors[1]
        cast_0 = data_mul.op.input_tensors[0]
    else:
        data_mul = square_sum_x.op.input_tensors[1]
        cast_0 = None

    data = input_tensor
    if cast_0 is not None:
        data_ub = sch.cache_read(data, cce.scope_ubuf, [cast_0])
        cast_0_ub = sch.cache_read(cast_0, cce.scope_ubuf, [data_mul, sum_x])
    else:
        data_ub = sch.cache_read(data, cce.scope_ubuf, [data_mul, sum_x])
        cast_0_ub = None

    data_mul_ub = sch.cache_read(data_mul, cce.scope_ubuf, [sum_x])

    sch[data_mul].compute_inline()
    if cast_0 is not None:
        sch[cast_0].compute_inline()

    if c1_size >= core_num and shape_input[2] * shape_input[3] > threshold:
        schedule_cut_c1(sch_list, res, sum_x, square_sum_x,
                        data_ub, cast_0_ub, data_mul_ub,
                        ub_split_reduce_axis, split_factor, is_keep_dim)
    elif ub_split_axis == 2 and \
            outer_loop >= core_num and \
            shape_input[ub_split_axis] % core_num == 0:
        schedule_cut_h_twice(sch_list, res, sum_x,
                             square_sum_x, data_ub, cast_0_ub,
                             data_mul_ub, split_factor, is_keep_dim)
    elif ub_split_axis == 2 and \
            shape_input[ub_split_axis] >= half_core_num and \
            shape_input[ub_split_axis] % half_core_num == 0 and \
            shape_input[0] < core_num and shape_input[0] == 2:
        schedule_fuse_h_n(sch_list, res, sum_x, square_sum_x,
                          data_ub, cast_0_ub, data_mul_ub,
                          split_factor, is_keep_dim)
    elif batch >= core_num:
        schedule_cut_batch(sch_list, res, sum_x,
                           square_sum_x, data_ub, cast_0_ub,
                           data_mul_ub, ub_split_axis,
                           split_factor, is_keep_dim)
    else:
        schedule_cut_general(sch_list, res, sum_x, square_sum_x,
                             data_ub, cast_0_ub, data_mul_ub,
                             ub_split_reduce_axis, split_factor, is_keep_dim)

    sch = sch_list[0]

    return sch
