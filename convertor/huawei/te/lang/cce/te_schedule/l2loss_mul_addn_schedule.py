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

l2loss_mull_addn_schedule
"""

from __future__ import absolute_import
from functools import reduce
import te
from te import tvm
from te import platform as cce
from .util import get_emit_insn_map
from .util import gen_reversed_subgraph_list
from .util import DTYPE_WIDTH_MAP


def get_max_ub_count(dtype):
    """
    caculate the max element num loaded in UB buffer
    :return: max element num loaded in UB buffer
    """
    # div 2 for align to fp16
    total_size = cce.get_soc_spec("UB_SIZE") // 2
    dtype_size = DTYPE_WIDTH_MAP.get(dtype)
    total_size = total_size // dtype_size
    # two input, not need to do double buffer
    total_width = 4.0001

    align_to = 32
    max_bound = total_width * align_to
    max_ub_count = int(total_size // max_bound * align_to)

    return max_ub_count

def _get_block_tiling(shape, one_core_data_threadhold):
    """
    find block tiling
    """
    core_num = cce.get_soc_spec("CORE_NUM")
    block_split_axis = 0
    block_factor = 1
    tmp_size = 1
    find_block_tiling = False

    for i, dim in enumerate(shape):
        tmp_size = tmp_size*dim
        if tmp_size < core_num:
            continue
        if tmp_size == core_num:
            block_split_axis = i
            block_factor = 1
            break

        tmp_size = tmp_size // dim
        for j in range(dim, 0, -1):
            if dim % j != 0:
                continue
            if tmp_size*j > core_num:
                continue
            block_split_axis = i
            block_factor = dim // j

            remain_size = 1
            remain_size = reduce(lambda i, j: i * j,
                                 shape[block_split_axis + 1:])

            remain_size = remain_size * block_factor

            if remain_size < one_core_data_threadhold:
                remain_size = remain_size // block_factor
                k = 0
                for k in range(j, 0, -1):
                    if dim % k != 0:
                        continue
                    if remain_size*(dim // k) < one_core_data_threadhold:
                        continue
                    block_factor = dim // k
                    break
                if k == 1:
                    block_split_axis = 0 if block_split_axis == 0 \
                        else block_split_axis - 1
                    block_factor = 1

            find_block_tiling = True
            break

        if find_block_tiling:
            break

    return block_split_axis, block_factor

def _get_ub_tiling(shape, block_split_axis, block_factor, max_ub_count):
    """
    find ub tiling
    """
    tmp_size = 1
    find_ub_tiling = False
    ub_split_axis = block_split_axis
    ub_factor = block_factor

    for i in range(len(shape) - 1, block_split_axis, -1):
        tmp_size = tmp_size*shape[i]
        if tmp_size < max_ub_count:
            continue
        if tmp_size == max_ub_count:
            ub_split_axis = i
            ub_factor = shape[i]
            break
        dim = shape[i]
        tmp_size = tmp_size // dim
        for j in range(dim, 0, -1):
            if dim % j != 0:
                continue
            if tmp_size*j > max_ub_count:
                continue
            ub_split_axis = i
            ub_factor = j
            find_ub_tiling = True
            break
        if find_ub_tiling:
            break

    if not find_ub_tiling:
        ub_split_axis = block_split_axis
        block_inner = block_factor

        for j in range(block_inner, 0, -1):
            if block_inner % j != 0:
                continue
            if tmp_size*j > max_ub_count:
                continue
            ub_factor = j
            break

    return ub_split_axis, ub_factor


def get_tiling(shape, dtype):
    """
    ubtiling for l2loss + mul + addn fusion
    :param one_core_data_cnt: one_core_data_cnt
    :param dtype: data type
    :return: ubtiling factor
    """
    max_ub_count = get_max_ub_count(dtype)
    one_core_data_threadhold = 1024

    total_size = 1
    for i in shape:
        total_size *= i

    if total_size < one_core_data_threadhold:
        return 0, shape[0], 0, shape[0]

    block_split_axis, block_factor = \
        _get_block_tiling(shape, one_core_data_threadhold)

    remain_size = 1
    for i in range(len(shape) - 1, block_split_axis, -1):
        remain_size = remain_size*shape[i]

    remain_size = remain_size*block_factor

    if remain_size < one_core_data_threadhold:
        ub_split_axis = block_split_axis
        ub_factor = block_factor
        return block_split_axis, block_factor, ub_split_axis, ub_factor

    ub_split_axis, ub_factor = \
        _get_ub_tiling(shape, block_split_axis, block_factor, max_ub_count)

    return block_split_axis, block_factor, ub_split_axis, ub_factor


def _check_params(res, input_tensors):
    """
    check params
    """
    if len(res) != 2:
        raise RuntimeError("L2loss mul addn output nums should be 2!")

    if len(input_tensors) != 3:
        raise RuntimeError("L2loss mul addn input nums should be 3!")

def _do_emit_insn(sch_list, cache_read_buffer_list,
                  mid_out_tensor_list, mid_out_tensor_read_buffer_map,
                  cache_write_buffer_map, phony_tensor):
    # pylint: too-many-arguments
    sch = sch_list[0]
    for tensor_u in cache_read_buffer_list:
        sch[tensor_u].emit_insn(tensor_u.op.axis[0], 'dma_copy')

    for tensor_u in mid_out_tensor_list:
        sch[tensor_u].emit_insn(tensor_u.op.axis[0], 'dma_copy')

    for tensor_u in mid_out_tensor_read_buffer_map:
        buffer = mid_out_tensor_read_buffer_map[tensor_u]
        sch[buffer].emit_insn(buffer.op.axis[0], 'phony_insn')

    for tensor in cache_write_buffer_map:
        buffer = cache_write_buffer_map[tensor]
        if tensor in phony_tensor:
            sch[buffer].emit_insn(buffer.op.axis[0], 'phony_insn')
        else:
            emit_insn_pragma = get_emit_insn_map(buffer)
            sch[buffer].emit_insn(buffer.op.axis[0], emit_insn_pragma)
    sch_list[0] = sch

def _do_compute_at(sch_list, cache_read_buffer_list,
                   mid_out_tensor_list, mid_out_tensor_read_buffer_map,
                   cache_write_buffer_map, compute_at_tensor,
                   compute_at_axis):
    # pylint: too-many-arguments
    sch = sch_list[0]
    for i in cache_read_buffer_list:
        sch[i].compute_at(sch[compute_at_tensor], compute_at_axis)

    for i in mid_out_tensor_list:
        sch[i].compute_at(sch[compute_at_tensor], compute_at_axis)

    for i in mid_out_tensor_read_buffer_map:
        buffer = mid_out_tensor_read_buffer_map[i]
        sch[buffer].compute_at(sch[compute_at_tensor], compute_at_axis)

    for i in cache_write_buffer_map:
        buffer = cache_write_buffer_map[i]
        sch[buffer].compute_at(sch[compute_at_tensor], compute_at_axis)
    sch_list[0] = sch

def l2loss_mul_addn_schedule(res, input_tensors):
    '''
    l2loss + mul + addn fusion schedule for float32 and dim cnt equal to 1
    :param res: res tensor
    :param input_tensors: input tensors
    :return: sch
    '''
    # pylint: too-many-locals, too-many-branches, too-many-statements
    _check_params(res, input_tensors)

    res_add = res[0]
    res_l2l0ss = res[1]

    dtype = res_add.dtype
    if dtype != "float32":
        raise RuntimeError("L2loss mul addn only support float32 input!")

    mul_3 = res_l2l0ss.op.input_tensors[0]

    phony_mul = te.lang.cce.vmuls(res_add, 0.0)
    phony_add = te.lang.cce.vadd(phony_mul, mul_3)
    axis = [i for i in range(len(res_add.shape))]
    new_res = te.lang.cce.sum(phony_add, axis=axis, keepdims=True)

    shape_add = te.lang.cce.util.shape_to_list(res_add.shape)

    tensor_list_map = {}
    tensor_list_dst_tensor_map = {}
    mid_out_tensor_list = [res_add,]
    phony_tensor = [phony_mul, phony_add]

    gen_reversed_subgraph_list(new_res, tensor_list_map,
                               tensor_list_dst_tensor_map)

    input_tensor_dst_tensor_map = {}
    mid_tensor_dst_tensor_map = {}
    cache_read_tensor_list = []
    cache_write_tensor_list = []
    for tensor in tensor_list_dst_tensor_map:
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            input_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[
                tensor]
            cache_read_tensor_list.append(tensor)
        else:
            mid_tensor_dst_tensor_map[tensor] = tensor_list_dst_tensor_map[
                tensor]
            cache_write_tensor_list.append(tensor)

    sch = tvm.create_schedule([new_res.op])

    block_split_axis, block_factor, ub_split_axis, ub_factor = \
        get_tiling(shape_add, dtype)

    if ub_split_axis < block_split_axis:
        raise RuntimeError("Invalid tiling!")

    res_block_outer, _ =\
        sch[new_res].split(new_res.op.reduce_axis[block_split_axis],
                           block_factor)

    fused_axis = res_block_outer
    if block_split_axis > 0:
        fuse_axis_list = []
        for i in range(block_split_axis):
            fuse_axis_list.append(new_res.op.reduce_axis[i])
        fuse_axis_list.append(res_block_outer)
        fused_axis = sch[new_res].fuse(*fuse_axis_list)

    res_ub_rf = sch.rfactor(new_res, fused_axis)

    # ---------cache read/write--------------
    cache_read_buffer_list = []
    for tensor in cache_read_tensor_list:
        cache_read_buffer_list.append(
            sch.cache_read(tensor, cce.scope_ubuf,
                           input_tensor_dst_tensor_map[tensor]))

    mid_out_tensor_read_buffer_map = {}
    for i in mid_out_tensor_list:
        read_buffer = sch.cache_read(i, cce.scope_ubuf, mid_tensor_dst_tensor_map[i])
        mid_out_tensor_read_buffer_map[i] = read_buffer

    cache_write_buffer_list = []
    cache_write_buffer_map = {}
    for tensor in cache_write_tensor_list:
        buffer = sch.cache_write(tensor, cce.scope_ubuf)
        cache_write_buffer_list.append(buffer)
        cache_write_buffer_map[tensor] = buffer

    new_res_global = sch.cache_write(new_res, cce.scope_gm)

    sch[res_ub_rf].set_scope(cce.scope_ubuf)

    # ---------compute inline----------------
    for tensor in cache_write_tensor_list:
        if tensor not in mid_out_tensor_list:
            sch[tensor].compute_inline()

    # reuse buffer of mul_3 and phony_add
    tensor_ub = cache_write_buffer_map[mul_3]
    reuse_tensor_ub = cache_write_buffer_map[phony_add]
    sch[tensor_ub].reused_by(reuse_tensor_ub)

    reorder_axis_list = []
    if ub_split_axis == block_split_axis:
        ub_outer, ub_inner = sch[res_ub_rf].split(res_ub_rf.op.reduce_axis[-1],
                                                  factor=ub_factor)
        reorder_axis_list += res_ub_rf.op.axis
        reorder_axis_list.append(ub_outer)
        reorder_axis_list.append(ub_inner)
        reorder_axis_list += res_ub_rf.op.reduce_axis[0:-1]
        sch[res_ub_rf].reorder(*reorder_axis_list)
    else:
        ub_outer, ub_inner = \
            sch[res_ub_rf].split(res_ub_rf.op.reduce_axis[ub_split_axis - 1],
                                 factor=ub_factor)
        reorder_axis_list += res_ub_rf.op.axis
        reorder_axis_list.append(res_ub_rf.op.reduce_axis[-1])
        reorder_axis_list += res_ub_rf.op.reduce_axis[0:ub_split_axis - 1]
        reorder_axis_list.append(ub_outer)
        reorder_axis_list.append(ub_inner)
        reorder_axis_list += res_ub_rf.op.reduce_axis[ub_split_axis:-1]
        sch[res_ub_rf].reorder(*reorder_axis_list)

    reorder_axis_list = []
    reorder_axis_list.append(new_res_global.op.reduce_axis[0])
    reorder_axis_list += new_res_global.op.axis
    sch[new_res_global].reorder(*reorder_axis_list)

    compute_at_axis = ub_outer

    sch_list = [sch]
    _do_compute_at(sch_list, cache_read_buffer_list,
                   mid_out_tensor_list, mid_out_tensor_read_buffer_map,
                   cache_write_buffer_map, res_ub_rf,
                   compute_at_axis)
    sch = sch_list[0]

    sch[res_ub_rf].compute_at(sch[new_res_global],
                              new_res_global.op.reduce_axis[0])

    res[0] = res_add
    res[1] = new_res_global

    sch_list = [sch]
    _do_emit_insn(sch_list, cache_read_buffer_list,
                  mid_out_tensor_list, mid_out_tensor_read_buffer_map,
                  cache_write_buffer_map, phony_tensor)
    sch = sch_list[0]

    sch[res_ub_rf].emit_insn(ub_inner, "reduce_last_axis_reduce_sum")
    sch[new_res_global].emit_insn(new_res_global.op.axis[0], "dma_copy")
    sch[new_res].emit_insn(sch[new_res].op.axis[0], "phony_insn")

    # ------------------bind----------------------
    block = tvm.thread_axis("blockIdx.x")
    sch[new_res_global].bind(new_res_global.op.reduce_axis[0], block)

    return sch
