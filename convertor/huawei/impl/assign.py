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

assign
"""

import operator
from functools import reduce as functools_reduce

import te.platform as cce
from te.platform import insn_cmd
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
# available ub size
UB_SIZE_B = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)


def _get_factor(ele_zero, ele_cnt, total_ele, no_remainder):
    """
    get split factor for _tilling_one_axis function

    Parameters
    ----------
    ele_zero: int
        the number of shape's first dimension elements
    ele_cnt: int
        the number of all elements
    total_ele: int
        the number of total elements in UB
    no_remainder: bool
        when split_axis == 0,
        the value of shape[0] whether divided by split_factor without remainder.

    Returns
    -------
    split_factor: int
        the factor used when tiling the target axis
    """
    split_factor = 1
    if no_remainder:
        for i in reversed(list(range(1, ele_zero))):
            if ele_zero % i == 0 and i*ele_cnt <= total_ele:
                split_factor = i
                break
    else:
        for i in reversed(list(range(1, ele_zero))):
            if i*ele_cnt <= total_ele:
                split_factor = i
                break

    return split_factor


def _tilling_axis(shape, dtype, no_remainder):
    """
    calculate the split parameters according to different shapes

    Parameters
    ----------
    shape: tuple
        shape of tensor
    dtype: str
        the data type
    no_remainder: bool
        when split_axis == 0,
        the value of shape[0] whether divided by split_factor without remainder.

    Returns
    -------
    split_axis: int
        the target axis that is used for tiling the tensor to find
    split_factor: int
        the factor used when tiling the target axis
    """
    ub_size_bytes = UB_SIZE_B - 32
    # 8 bit = 1byte, '8' below for this reason
    dtype_bytes_size = cce.cce_intrin.get_bit_len(dtype) // 8
    total_ele = ub_size_bytes // dtype_bytes_size
    split_axis = 0
    split_factor = 1

    for i, _ in enumerate(shape):
        ele_cnt = functools_reduce(lambda x, y: x*y, shape[i:])
        if ele_cnt <= total_ele:
            split_axis = i - 1
            split_factor = total_ele // ele_cnt
            if no_remainder and i == 1 and shape[0] % split_factor != 0:
                split_factor = _get_factor(shape[0], ele_cnt, total_ele,
                                           no_remainder)
            break
        elif i == len(shape) - 1:
            if len(shape) == 1:
                split_axis = 0
                split_factor = _get_factor(shape[0], 1, total_ele, no_remainder)
            else:
                split_axis = i
                split_factor = total_ele
            break

    if split_axis < 0:
        split_axis = 0
        split_factor = shape[0]

    return split_axis, split_factor


def _check_shape(ref_shape, value_shape):
    """
    check the ref_shape and value_shape

    Parameters
    ----------
    ref_shape: list or tuple
        shape of ref_tensor
    value_shape: list or tuple
        shape of value_tensor

    Returns
    -------
    None
    """
    if operator.ne(list(ref_shape), list(value_shape)):
        raise RuntimeError("ref and value must have the same shape !")

    util.check_shape_rule(ref_shape)
    util.check_shape_size(ref_shape, SHAPE_SIZE_LIMIT)


def _check_params(ref_shape, value_shape, dtype, kernel_name):
    """
    check the parameters including ref_shape, value_shape, dtype and kernel_name

    Parameters
    ----------
    ref_shape: list or tuple
        shape of ref_tensor
    value_shape: list or tuple
        shape of value_tensor
    dtype: str
        the data type
    kernel_name: str
        cce kernel name, default value is "cce_assign"

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)

    check_list = (
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
        "uint64",
        "float16", "float32")
    util.check_dtype_rule(dtype, check_list)

    _check_shape(ref_shape, value_shape)


def _get_target_core_num(first_axis_size):
    max_core_num = 65535
    cloud_core_num = 32

    if first_axis_size % cloud_core_num == 0:
        return cloud_core_num

    if first_axis_size <= max_core_num:
        return first_axis_size

    target_core_num = 1
    for i in reversed(list(range(1, max_core_num + 1))):
        if first_axis_size % i == 0:
            target_core_num = i
            break

    return target_core_num


def _core_bind_axis(input_shape):
    cloud_core_num = 32

    def __suite_factor(pre_size, axis_size):
        for index in reversed(range(2, axis_size + 1)):
            if pre_size * index <= cloud_core_num and axis_size % index == 0:
                return axis_size // index
        return 1

    shape_len = len(input_shape)
    cloud_core_num = 32
    shape_size = 1
    for i in range(shape_len):
        if shape_size * input_shape[i] <= cloud_core_num:
            shape_size *= input_shape[i]
            continue

        if i == 0:
            return 0, _get_target_core_num(input_shape[0])

        if shape_size * 2 > cloud_core_num:
            return i-1, 1
        else:
            factor = __suite_factor(shape_size, input_shape[i])
            if factor == 1:
                return i-1, 1
            else:
                return i, factor
    return shape_len-1, 1


# pylint: disable=locally-disabled,too-many-arguments,unnecessary-lambda
# pylint: disable=locally-disabled,too-many-branches,too-many-locals
# pylint: disable=locally-disabled,unused-argument,too-many-statements
@util.check_input_type(dict, dict, dict, str)
def assign(ref, value, output, kernel_name="assign"):
    """
    algorithm: assign
    calculating: update 'ref' by assigning 'value' to it

    Parameters
    ----------
    ref: dict
        dict of input_ref, include shape and dtype,
    value: dict
        dict of input_value, include shape and dtype,
        Must have the same shape and dtype as input_ref
    output: dict
        dict of output
    kernel_name : str
        cce kernel name, default value is assign

    Returns
    -------
    None
    """
    ref_shape = util.scalar2tensor_one(ref.get("shape"))
    value_shape = util.scalar2tensor_one(value.get("shape"))
    dtype = ref.get("dtype").lower()
    _check_params(ref_shape, value_shape, dtype, kernel_name)

    data_b = tvm.placeholder(value_shape, dtype=dtype, name='data_b')
    data_b_ub = tvm.compute(value_shape, lambda *i: data_b(*i),
                            name='data_b_ub')
    data_a = tvm.compute(ref_shape, lambda *i: data_b_ub(*i), name='data_a')
    sch = tvm.create_schedule(data_a.op)
    sch[data_b_ub].set_scope(cce.scope_ubuf)

    split_axis, split_factor = _tilling_axis(ref_shape, dtype, True)

    core_bind_axis, core_bind_split_factor = _core_bind_axis(ref_shape)
    if core_bind_axis < split_axis:
        core_bind_axis_outer, core_bind_axis_inner = sch[data_a].split(
            data_a.op.axis[core_bind_axis],
            factor=core_bind_split_factor)
        if core_bind_axis == 0:
            axis_outer = core_bind_axis_outer
        else:
            axis_outer = data_a.op.axis[0]
            for axis_index in range(1, core_bind_axis):
                axis_outer = sch[data_a].fuse(axis_outer,
                                              data_a.op.axis[axis_index])
            axis_outer = sch[data_a].fuse(axis_outer, core_bind_axis_outer)
        axis_inner = core_bind_axis_inner
        for axis_index in range(core_bind_axis + 1, split_axis):
            axis_inner = sch[data_a].fuse(axis_inner,
                                          data_a.op.axis[axis_index])
        tilling_axis_outer, tilling_axis_inner = sch[data_a].split(
            data_a.op.axis[split_axis], factor=split_factor)
        axis_inner = sch[data_a].fuse(axis_inner, tilling_axis_outer)
    else:
        if split_axis == 0:
            axis_outer, tilling_axis_inner = sch[data_a].split(
                data_a.op.axis[split_axis], factor=split_factor)
            core_num = _get_target_core_num(
                ref_shape[split_axis] // split_factor)
            axis_outer, axis_inner = sch[data_a].split(
                axis_outer,
                nparts=core_num)
        else:
            temp_shape = list(ref_shape[:split_axis])
            temp_shape.append(ref_shape[split_axis] // split_factor)
            if split_axis == core_bind_axis and \
                    core_bind_split_factor > split_factor:
                core_bind_axis, core_bind_split_factor \
                    = _core_bind_axis(temp_shape)
                axis_outer, tilling_axis_inner = sch[data_a].split(
                    data_a.op.axis[split_axis], factor=split_factor)

                if core_bind_axis == split_axis:
                    core_bind_axis_outer, axis_inner \
                        = sch[data_a].split(axis_outer,
                                            factor=core_bind_split_factor)
                else:
                    factor = ref_shape[split_axis] // split_factor
                    core_bind_axis_outer, axis_inner \
                        = sch[data_a].split(axis_outer,
                                            factor=factor)

                axis_outer = data_a.op.axis[0]
                for axis_index in range(1, split_axis):
                    axis_outer = sch[data_a].fuse(axis_outer,
                                                  data_a.op.axis[axis_index])
                axis_outer = sch[data_a].fuse(axis_outer, core_bind_axis_outer)
            else:
                core_bind_axis_outer, core_bind_axis_inner = sch[data_a].split(
                    data_a.op.axis[core_bind_axis],
                    factor=core_bind_split_factor)
                axis_outer = data_a.op.axis[0]
                for axis_index in range(1, core_bind_axis):
                    axis_outer = sch[data_a].fuse(axis_outer,
                                                  data_a.op.axis[axis_index])
                axis_outer = sch[data_a].fuse(axis_outer, core_bind_axis_outer)
                axis_inner = core_bind_axis_inner
                tilling_axis_inner = axis_inner
                split_axis = core_bind_axis

    sch[data_a].bind(axis_outer, tvm.thread_axis('blockIdx.x'))
    sch[data_b_ub].compute_at(sch[data_a], axis_inner)
    sch[data_b_ub].emit_insn(data_b_ub.op.axis[split_axis], insn_cmd.DMA_COPY)
    sch[data_a].emit_insn(tilling_axis_inner, insn_cmd.DMA_COPY)

    with build_config:
        tvm.build(sch, [data_a, data_b], "cce", name=kernel_name)
