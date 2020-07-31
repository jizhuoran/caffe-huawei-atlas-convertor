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

read_select_schedule
"""
from functools import reduce as functools_reduce
from te import tvm
from te import platform as cce


def _tilling_axis(valid_shape, input_dtype):
    ub_size_bytes = cce.CceProductParams().getParams("Unified_Buffer") - 32
    dtype_bytes_size = cce.cce_intrin.get_bit_len(input_dtype) // 8

    total_ele = int(ub_size_bytes // dtype_bytes_size)
    split_axis = 0
    split_factor = 1

    for i, _ in enumerate(valid_shape):
        ele_cnt = int(functools_reduce(lambda x, y: x*y, valid_shape[i:]))
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            break
        elif i == len(valid_shape) - 1:
            split_axis = i
            split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = valid_shape[0]

    return split_axis, split_factor


def _get_tensor_map(res, tensor_map):
    """
    get the compute tensors

    Parameters
    ----------
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    None
    """
    if res is None:
        return
    stack = [res]
    visited_list = []
    while len(stack) > 0:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_map[in_tensor.name] = in_tensor


def read_select_schedule(res, input_tensors):# pylint: disable=locally-disabled,unused-argument
    """
    the schedule processes of read_select

    Parameters
    ----------
    res: the placeholder of result
    input_tensors: the placeholder of input

    Returns
    -------
    the result of schedule
    """
    tensor_map = {}
    _get_tensor_map(res, tensor_map)

    tensor_input = tensor_map.get("input_tensor")

    src_in_flag = "DDR"
    if "src_in_flag" in tensor_input.op.attrs:
        src_in_flag = tensor_input.op.attrs['src_in_flag']

    valid_shape = tensor_map.get("output_ub_5d").shape
    input_dtype = tensor_map.get("output_ub_5d").dtype

    sch = tvm.create_schedule(res.op)

    split_axis, split_factor = _tilling_axis(valid_shape, input_dtype)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis], factor=split_factor)
    sch[tensor_map.get("output_ub_5d")].compute_at(sch[res], axis_outer)

    if src_in_flag == "L1":
        sch[tensor_input].set_scope(cce.scope_cbuf_fusion)
    sch[tensor_map.get("output_ub_5d")].set_scope(cce.scope_ubuf)
    sch[tensor_map.get("output_ub_5d")].emit_insn(
        tensor_map.get("output_ub_5d").op.axis[split_axis], 'dma_copy')
    sch[res].emit_insn(axis_inner, 'dma_copy')

    return sch
