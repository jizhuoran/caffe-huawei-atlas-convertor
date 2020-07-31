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

ascend_dequant
"""
from functools import reduce as function_reduce
import te.lang.cce
from te import tvm
from te import platform as cceconf
from topi.cce.util import is_v200_version


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
    stack = [res]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_map[in_tensor.name] = in_tensor
    if "x" in tensor_map:
        tensor_map.pop("x")
    if "deq_scale" in tensor_map:
        tensor_map.pop("deq_scale")


def _tilling_axis(shape, dtype_size, tensor_num):
    """
    get the split axis and factor by ub size

    Parameters
    ----------
    shape: the shape of input
    dtype_size: the dtype size
    tensor_num: the number of tensor size

    Returns
    -------
    split_axis and split_factor
    """
    shape_new = list(shape).copy()
    total_ele = (cceconf.CceProductParams().getParams(
        "Unified_Buffer") - 1024 * 2) // dtype_size // tensor_num // 2
    block_num = cceconf.CceProductParams().getParams("Device_core_num")
    val_cnt = 1
    index_cnt = 0
    for i in range(0, len(shape_new) - 1):
        val_cnt = val_cnt * shape_new[i]
        index_cnt = i
        if val_cnt >= block_num:
            break

    block_size = val_cnt // block_num * \
                 function_reduce(lambda x, y: x * y, shape_new[index_cnt + 1:])
    if 256 <= block_size <= total_ele:
        total_ele = block_size
    split_axis = 0
    split_factor = 1
    size = function_reduce(lambda x, y: x * y, shape_new[1:])
    for index, _ in enumerate(shape_new):
        ele_cnt = function_reduce(lambda x, y: x * y, shape_new[index:])
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break
    if split_axis < 0 or (split_axis == 0 and size <= total_ele):
        split_axis = 0
        split_factor = 1
    return split_axis, split_factor


def _get_fuse_info(sch, res, res_split_shape, split_info):
    """
    get the fuse info

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    res_split_shape: the output shape
    split_info: split_axis and split_factor

    Returns
    -------
    fused_value, fused_list, axis_outer_num
    """
    split_axis = split_info[0]
    split_factor = split_info[1]
    if res_split_shape[split_axis] % split_factor > 0:
        axis_outer_num = res_split_shape[split_axis] // split_factor + 1
    else:
        axis_outer_num = res_split_shape[split_axis] // split_factor
    origin_list = [res_split_shape[i] for i in range(split_axis)]
    fused_value = 1
    for _, item in enumerate(origin_list):
        fused_value *= item
    fused_list = [sch[res].op.axis[i] for i in range(split_axis)]
    return fused_value, fused_list, axis_outer_num


def _set_buffer_scope(sch, tensor_map):
    """
    set the scope for tensors

    Parameters
    ----------
    sch: the schedule
    tensor_map: the compute tensors

    Returns
    -------
    None
    """
    for key, value in tensor_map.items():
        if key == "x_l0c":
            sch[value].set_scope(cceconf.scope_cc)
        else:
            sch[value].set_scope(cceconf.scope_ubuf)


def _bind_fuse(fused_value, fused_list, axis_outer_num, sch, res,
               axis_outer, out_shape):
    """
    bind the fused axis.
    """
    core_num = cceconf.CceProductParams().getParams("Device_core_num")
    bind_axis = axis_outer
    if fused_list:
        if fused_value * axis_outer_num <= core_num:
            fused_list.append(axis_outer)
            bind_axis = sch[res].fuse(*fused_list)
            axis_outer = bind_axis
        elif fused_value < core_num:
            num = core_num // fused_value
            thread_outer, axis_outer = sch[res].split(axis_outer,
                                                      nparts=num)
            fused_list.append(thread_outer)
            bind_axis = sch[res].fuse(*fused_list)
        else:
            val_cnt = 1
            index = 0
            for i in range(len(fused_list)):
                val_cnt = val_cnt * out_shape[i]
                if val_cnt >= core_num:
                    index = i
                    break
            num = core_num // (val_cnt // out_shape[index])
            thread_outer, _ = sch[res].split(res.op.axis[index], nparts=num)
            new_fused_list = fused_list[:index]
            new_fused_list.append(thread_outer)
            bind_axis = sch[res].fuse(*new_fused_list)
    sch[res].bind(bind_axis, tvm.thread_axis("blockIdx.x"))
    return axis_outer


def _bind_core(out_shape, sch, res, tensor_map):
    """
    bind multi-core

    Parameters
    ----------
    out_shape: the output shape
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    axis_outer, axis_inner
    """
    core_num = cceconf.CceProductParams().getParams("Device_core_num")
    split_axis, split_factor = _tilling_axis(out_shape, 4, 2)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                            factor=split_factor)
    fused_value, fused_list, axis_outer_num = _get_fuse_info(
        sch, res, out_shape, (split_axis, split_factor))
    bind_axis = 0
    can_bind = False
    for i in range(split_axis):
        if out_shape[i] >= core_num:
            bind_axis = i
            can_bind = True
            break
    if can_bind:
        thread_outer, _ = sch[res].split(res.op.axis[bind_axis],
                                         nparts=core_num)
        sch[res].bind(thread_outer, tvm.thread_axis("blockIdx.x"))
    elif axis_outer_num >= core_num:
        thread_outer, axis_outer = sch[res].split(axis_outer,
                                                  nparts=core_num)
        sch[res].bind(thread_outer, tvm.thread_axis("blockIdx.x"))
    else:
        axis_outer = _bind_fuse(fused_value, fused_list, axis_outer_num, sch,
                                res, axis_outer, out_shape)
    sch[tensor_map.get("x_ub")].double_buffer()
    sch[tensor_map.get("deq_ub")].double_buffer()
    return axis_outer, axis_inner


def _set_buffer_emit_insn(sch, res, tensor_map, axis_inner):
    """
    instruction mapping

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors
    axis_inner: the inner axis

    Returns
    -------
    None
    """
    sch[tensor_map.get("x_ub")].emit_insn(
        sch[tensor_map.get("x_ub")].op.axis[0], 'dma_copy')
    sch[tensor_map.get("deq_ub")].emit_insn(
        sch[tensor_map.get("deq_ub")].op.axis[0], 'dma_copy')
    sch[tensor_map.get("x_l0c")].emit_insn(
        sch[tensor_map.get("x_l0c")].op.axis[0], 'dma_copy')
    sch[res].emit_insn(axis_inner, 'dma_copy')
    for key, value in tensor_map.items():
        if key in ["x_ub", "deq_ub"]:
            pass
        elif key == "x_l0c":
            sch[value].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
        elif key == "dequant_to_fp16":
            sch[value].buffer_align((1, 1), (1, 1), (1, 16), (1, 16))
            if is_v200_version():
                sch[value].emit_insn(sch[value].op.axis[0], 'dma_copy')
            else:
                if res.op.attrs['is_scalar'].value == 1:
                    sch[value].pragma(value.op.axis[0], 'deq_scale', 'scalar')
                else:
                    sch[value].pragma(value.op.axis[2], 'deq_scale', 'vector')
        else:
            sch[value].emit_insn(
                sch[value].op.axis[0], 'vector_auto')


def ascend_dequant_schedule(res, input_tensors):
    """
    the schedule processes of dequant

    Parameters
    ----------
    res: the placeholder of result
    input_tensors: the placeholder of input

    Returns
    -------
    the result of schedule
    """
    sch = tvm.create_schedule(res.op)
    tensor_map = {}
    _get_tensor_map(res, tensor_map)
    out_shape = te.lang.cce.util.shape_to_list(res.shape)
    _set_buffer_scope(sch, tensor_map)
    axis_outer, axis_inner = _bind_core(out_shape, sch, res, tensor_map)

    for _, value in tensor_map.items():
        sch[value].compute_at(sch[res], axis_outer)

    _set_buffer_emit_insn(sch, res, tensor_map, axis_inner)
    return sch
