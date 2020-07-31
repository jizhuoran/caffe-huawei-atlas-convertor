#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

split schedule
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import reduce as functools_reduce
import math

from te import tvm
from te import platform as cce
from te.platform import get_soc_spec
from te.platform.fusion_manager import fusion_manager
from te.lang.cce.rl_bank import rl_bank


# pylint: disable=locally-disabled,too-many-locals
def _tile_axis(shape, dtype, split_dim):
    """Calculate the tile parameters.

    Parameters
    ----------
    shape: list or tuple
        shape of tensor.
    dtype: str
        dtype of tensor.
    split_dim: int
        the dimension along which to split.

    Returns
    -------
    tile_axis: int
        the target axis that is used for tile the tensor.
    tile_factor: int
        the factor used when tile the target axis.
    """
    ub_size = get_soc_spec("UB_SIZE") - 1024
    dtype_size = cce.cce_intrin.get_bit_len(dtype) // 8
    total_cnt = ub_size // dtype_size
    block_cnt = 32 // dtype_size
    split_cnt = functools_reduce(lambda x, y: x * y, shape[split_dim:])

    tile_shape = []
    for dim in shape:
        tile_shape.append(dim)

    if split_cnt % block_cnt != 0 and split_dim != 0:
        last_ele = math.ceil(shape[-1] / block_cnt) * block_cnt
        tile_shape[-1] = int(last_ele)

    tile_axis = 0
    tile_factor = 1
    for i, _ in enumerate(tile_shape):
        ele_cnt = functools_reduce(lambda x, y: x * y, tile_shape[i:])
        if ele_cnt <= total_cnt:
            tile_axis = i - 1
            tile_factor = total_cnt // ele_cnt
            break

    if tile_shape[-1] > total_cnt:
        tile_axis = len(tile_shape) - 1
        tile_factor = total_cnt

    if tile_axis < 0:
        tile_axis = 0
        tile_factor = tile_shape[0]

    return tile_axis, tile_factor


def _check_align(shape_list, block_cnt, split_dim):
    """Check if the output is aligned.

    Parameters
    ----------
    shape_list: list
        the list of shapes.
    block_cnt: int
        the element count of one block.
    split_dim: int
        the dimension along which to split.

    Returns
    -------
    divide_flag: bool
        whether the outputs are equally divided.
    align_flag: bool
        whether the outputs are aligned.
    """
    divide_flag = True
    for i, _ in enumerate(shape_list):
        if shape_list[i][split_dim] != shape_list[0][split_dim]:
            divide_flag = False
            break

    align_flag = True
    for i, _ in enumerate(shape_list):
        split_ele = functools_reduce(lambda x, y: x * y, shape_list[i][split_dim:])
        if split_ele % block_cnt != 0:
            align_flag = False
            break

    return divide_flag, align_flag


def do_split_schedule(divide_flag, split_dim, align_flag, shape_list, i, dtype, sch, res, res_op,  # pylint: disable=too-many-arguments
                      block_idx, tensor_list, block_cnt):  # pylint: disable=too-many-arguments

    """
    do_split_schedule
    :param divide_flag:
    :param split_dim:
    :param align_flag:
    :param shape_list:
    :param i:
    :param dtype:
    :param sch:
    :param res:
    :param res_op:
    :param block_idx:
    :param tensor_list:
    :param block_cnt:
    :return:
    """
    if divide_flag and (split_dim == 0 or align_flag):
        tile_axis, tile_factor = _tile_axis(shape_list[i], dtype, split_dim)
        axis_outer, axis_inner = sch[res[i]].split(res_op[i].axis[tile_axis], factor=tile_factor)
        if tile_axis == 0:
            sch[res[i]].bind(axis_outer, block_idx)
        else:
            sch[res[i]].bind(res[i].op.axis[0], block_idx)
    elif not divide_flag and split_dim == 0:
        tile_axis, tile_factor = _tile_axis(shape_list[i], dtype, split_dim)
        axis_outer, axis_inner = sch[res[i]].split(res_op[i].axis[tile_axis], factor=tile_factor)
    elif not divide_flag and align_flag:
        tile_axis, tile_factor = _tile_axis(shape_list[i], dtype, split_dim)
        axis_outer, axis_inner = sch[res[i]].split(
            res_op[i].axis[split_dim],
            factor=shape_list[i][split_dim]) if tile_axis < split_dim else sch[res[i]].split(
                res_op[i].axis[tile_axis], factor=tile_factor)
        sch[res[i]].bind(res[i].op.axis[0], block_idx)
    else:
        tile_axis, tile_factor = _tile_axis(shape_list[i], dtype, split_dim)
        axis_outer, axis_inner = sch[res[i]].split(res_op[i].axis[tile_axis], factor=tile_factor)
        sch[tensor_list[i]].storage_align(tensor_list[i].op.axis[split_dim - 1], block_cnt, 0)

    sch[tensor_list[i]].compute_at(sch[res[i]], axis_outer)
    sch[tensor_list[i]].emit_insn(tensor_list[i].op.axis[tile_axis], "dma_copy")
    sch[res[i]].emit_insn(axis_inner, "dma_copy")


def split_schedule_com(data, split_dim, shape_list, tensor_list):
    """Create split schedule.

    Parameters
    ----------
    data: TVM tensor
        input tensor.
    split_dim: int
        the dimension along which to split.
    shape_list: list
        the list of output shapes.
    tensor_list: list
        the list of output tensors, tensor type is TVM tensor.

    Returns
    -------
    sch: schedule.Schedule
        The created schedule.
    build_list: list
        the list of input and output tensors, tensor type is TVM tensor.
    """
    res = []
    data_ub = None
    shape_ub = None
    for i, _ in enumerate(shape_list):
        data_ub = tensor_list[i]
        shape_ub = shape_list[i]
        # pylint: disable=locally-disabled,unnecessary-lambda
        data_gm = tvm.compute(shape_ub,
                              lambda *index: data_ub(*index),
                              name='res' + str(i),
                              tag='split_com|schedule_' + str(i))
        res.append(data_gm)
    # for RL tune getting res
    fusion_manager.set_op_res(res)

    res_op = []
    build_list = [data]
    for data_gm in res:
        res_op.append(data_gm.op)
        build_list.append(data_gm)

    _, sch = rl_bank.query_rl_bank(res)
    if sch:
        return sch, build_list

    sch = tvm.create_schedule(res_op)

    for tensor in tensor_list:
        sch[tensor].set_scope(cce.scope_ubuf)

    dtype = data.dtype
    dtype_size = cce.cce_intrin.get_bit_len(dtype) // 8
    block_cnt = 32 // dtype_size
    block_idx = tvm.thread_axis('blockIdx.x')
    divide_flag, align_flag = _check_align(shape_list, block_cnt, split_dim)

    for i, _ in enumerate(shape_list):
        do_split_schedule(divide_flag, split_dim, align_flag, shape_list, i, dtype, sch, res,
                          res_op, block_idx, tensor_list, block_cnt)

    return sch, build_list
