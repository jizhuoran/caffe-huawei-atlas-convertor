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

write_select_schedule
"""
from te import tvm
from te import platform as cce


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


def write_select_schedule(res, input_tensors):# pylint: disable=locally-disabled,unused-argument
    """
    the schedule processes of write_select

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

    tensor_input_ph = tensor_map.get("input_tensor_ph")
    tensor_input = tensor_map.get("input_tensor")

    dst_out_flag = "DDR"
    if "dst_out_flag" in tensor_input_ph.op.attrs:
        dst_out_flag = tensor_input_ph.op.attrs['dst_out_flag']

    valid_shape = tensor_input_ph.op.attrs['valid_shape']
    _, _, h_valid, w_valid, c0_valid = valid_shape
    h_valid = int(h_valid)
    w_valid = int(w_valid)
    c0_valid = int(c0_valid)

    sch = tvm.create_schedule(res.op)

    sch[tensor_input].set_scope(cce.scope_ubuf)
    if dst_out_flag == "L1":
        sch[res].set_scope(cce.scope_cbuf_fusion)

    sch[res].buffer_stride(res.op.axis[1], h_valid*w_valid*c0_valid, 0)
    sch[tensor_input].emit_insn(tensor_input.op.axis[0], 'dma_copy')
    sch[res].emit_insn(res.op.axis[0], 'dma_copy')

    return sch
