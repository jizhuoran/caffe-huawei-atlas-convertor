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

ascend_quant
"""
from functools import reduce as function_reduce
import te.lang.cce
from te import tvm
from te import platform as cceconf
from topi.cce.util import is_mini_version

# define the tensor name
CAST_F16_NAME = "cast_f16_ub"
INPUT_NAME = "input_ub"
VMULS_REFORM_NAME = "reform_by_vmuls"
SQRT_NAME = "scale_sqrt_ub"
OFFSET_NAME = "offset_ub"
CAST_I8_NAME = "cast_i8_ub"
VADDS_REFORM_NAME = "reform_by_vadds"

# define the Maximum number of cores
MAXIMUM_CORE_NUM = 65535

# define the map of dtype size
DTYPE_SIZE_MAP = {"float16": 2,
                  "float32": 4}


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
    total_size = (cceconf.CceProductParams().getParams(
        "Unified_Buffer") - 1024) // dtype_size
    max_ub_count = total_size // tensor_num
    total_ele = max_ub_count // 2
    split_axis = 0
    split_factor = 1
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

    for index, _ in enumerate(shape_new):
        ele_cnt = function_reduce(lambda x, y: x * y, shape_new[index:])
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break
    size = function_reduce(lambda x, y: x * y, shape_new[1:])
    if split_axis == 0 and size <= total_ele:
        split_axis = 0
        split_factor = 1
    if split_axis < 0:
        split_axis = 0
        split_factor = 1
    return split_axis, split_factor


def _round_emit_insn(round_mode):
    """
    Obtains the conv instruction by the round mode attr

    Parameters
    ----------
    round_mode: the attr of round mode

    Returns
    -------
    instruction
    """
    if is_mini_version():
        # mini
        emit_insn_str = "vector_conv"
    else:
        if round_mode == "Round":
            emit_insn_str = "vector_conv_round"
        elif round_mode == "Ceil":
            emit_insn_str = "vector_conv_ceil"
        elif round_mode == "Floor":
            emit_insn_str = "vector_conv_floor"
        elif round_mode == "Trunc":
            emit_insn_str = "vector_conv_trunc"
        else:
            emit_insn_str = "vector_conv"
    return emit_insn_str


def _reorder_by_split_c0(tensor):
    """
    reorder tensor by c1 axis

    Parameters
    ----------
    tensor: the tensor to be split

    Returns
    -------
    None
    """
    factor = 16
    c0o, c0i = tensor.split(tensor.op.axis[3], factor)
    tensor.reorder(tensor.op.axis[0],
                   tensor.op.axis[1],
                   tensor.op.axis[2],
                   c0o,
                   c0i)


def _reorder_by_split_c1(tensor):
    """
    reorder tensor by c0 axis

    Parameters
    ----------
    tensor: the tensor to be split

    Returns
    -------
    None
    """
    factor = 2
    c1o, c1i = tensor.split(tensor.op.axis[1], factor)
    tensor.reorder(tensor.op.axis[0],
                   c1o,
                   tensor.op.axis[2],
                   c1i,
                   tensor.op.axis[3])


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
    for _, value in tensor_map.items():
        sch[value].set_scope(cceconf.scope_ubuf)


def _set_buffer_compute_at(sch, res, tensor_map, axis_outer):
    """
    set the compute axis for tensors

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors
    axis_outer: the axis to be set

    Returns
    -------
    None
    """
    for _, value in tensor_map.items():
        sch[value].compute_at(sch[res], axis_outer)


def _reorder_buffer(sch, res, tensor_map):
    """
    reorder all tensors to the same shape

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    None
    """
    for key, value in tensor_map.items():
        if key in [INPUT_NAME, CAST_F16_NAME]:
            _reorder_by_split_c1(sch[value])
        else:
            _reorder_by_split_c0(sch[value])
    _reorder_by_split_c0(sch[res])


def _set_buffer_emit_insn(sch, tensor_list, axis_inner, attr_dic):
    """
    instruction mapping

    Parameters
    ----------
    sch: the schedule
    tensor_list: the list of tensors
    axis_inner: the inner axis
    attr_dic: the dict of attr

    Returns
    -------
    None
    """
    res = tensor_list[0]
    tensor_map = tensor_list[1]
    round_emit_insn = _round_emit_insn(attr_dic.get("round_mode"))
    input_c1 = attr_dic.get("input_c1")
    if input_c1 % 2 == 0:
        in_dma = "dma_copy"
    else:
        in_dma = "dma_padding"
    if CAST_F16_NAME in tensor_map:
        sch[tensor_map.get(CAST_F16_NAME)].emit_insn(
            sch[tensor_map.get(CAST_F16_NAME)].op.axis[0], 'vector_conv')
    if OFFSET_NAME in tensor_map:
        sch[tensor_map.get(OFFSET_NAME)].emit_insn(
            sch[tensor_map.get(OFFSET_NAME)].op.axis[0], 'vector_adds')
    if SQRT_NAME in tensor_map:
        sch[tensor_map.get(SQRT_NAME)].emit_insn(
            sch[tensor_map.get(SQRT_NAME)].op.axis[0], 'vector_muls')
    if VMULS_REFORM_NAME in tensor_map:
        sch[tensor_map.get(VMULS_REFORM_NAME)].emit_insn(
            sch[tensor_map.get(VMULS_REFORM_NAME)].op.axis[0], 'vector_muls')
    if VADDS_REFORM_NAME in tensor_map:
        sch[tensor_map.get(VADDS_REFORM_NAME)].emit_insn(
            sch[tensor_map.get(VADDS_REFORM_NAME)].op.axis[0], 'vector_adds')
    sch[tensor_map.get(CAST_I8_NAME)].emit_insn(
        sch[tensor_map.get(CAST_I8_NAME)].op.axis[0], round_emit_insn)
    sch[tensor_map.get(INPUT_NAME)].emit_insn(
        sch[tensor_map.get(INPUT_NAME)].op.axis[0], in_dma)
    sch[res].emit_insn(axis_inner, 'dma_copy')


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


def _bind_fuse(fused_value, fused_list, axis_outer_num, sch, res,
               axis_outer, res_split_shape):
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
                val_cnt = val_cnt * res_split_shape[i]
                if val_cnt >= core_num:
                    index = i
                    break
            num = core_num // (val_cnt // res_split_shape[index])
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
    res_split_shape = (out_shape[0],
                       out_shape[1],
                       out_shape[2],
                       2,
                       out_shape[3] // 2)
    core_num = cceconf.CceProductParams().getParams("Device_core_num")
    split_axis, split_factor = _tilling_axis(
        res_split_shape,
        DTYPE_SIZE_MAP.get(tensor_map.get(INPUT_NAME).dtype.lower()),
        4)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                            factor=split_factor)
    bind_axis = 0
    can_bind = False
    for i in range(split_axis):
        if res_split_shape[i] >= core_num:
            bind_axis = i
            can_bind = True
            break
    fused_value, fused_list, axis_outer_num = _get_fuse_info(
        sch, res, res_split_shape, (split_axis, split_factor))

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
                                res, axis_outer, res_split_shape)
    sch[tensor_map.get(INPUT_NAME)].double_buffer()
    return axis_outer, axis_inner


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
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_map[in_tensor.name] = in_tensor
    if "input_x" in tensor_map:
        tensor_map.pop("input_x")


def ascend_quant_schedule(res, input_tensors):
    """
    the schedule processes of quant

    Parameters
    ----------
    res: the placeholder of result
    input_tensors: the placeholder of input

    Returns
    -------
    the result of schedule
    """
    input_shape = te.lang.cce.util.shape_to_list(input_tensors[0].shape)
    out_shape = te.lang.cce.util.shape_to_list(res.shape)
    sch = tvm.create_schedule(res.op)
    tensor_map = {}
    _get_tensor_map(res, tensor_map)
    attr_dic = {
        "scale": res.op.attrs['scale'],
        "sqrt_mode": res.op.attrs['sqrt_mode'],
        "offset": res.op.attrs['offset'],
        "round_mode": res.op.attrs['round_mode'],
        "input_c1": input_shape[1]
    }
    _set_buffer_scope(sch, tensor_map)
    _reorder_buffer(sch, res, tensor_map)
    axis_outer, axis_inner = _bind_core(out_shape, sch, res, tensor_map)
    _set_buffer_compute_at(sch, res, tensor_map, axis_outer)
    _set_buffer_emit_insn(sch, (res, tensor_map), axis_inner, attr_dic)
    return sch
