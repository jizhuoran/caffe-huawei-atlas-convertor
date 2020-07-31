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

assign_sub

  Op_description :
    Update tensor 'var' by subtracting tensor 'value' from it

    # assign_sub(
    #   var,
    #   value,
    #   out,
    #   kernel_name='assign_sub')

  Supportive_dtype_format :
    ['float16', 'float32', 'int8', 'uint8', 'int32']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : 'var' and 'value' must have the same type and shape.
    [2] All : shape size limit is 2147483648.

"""

import operator
from functools import reduce as function_reduce

from impl.util.util_build import set_bool_storage_config
from te import tvm
from te import platform as cce
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
from topi.cce import util

# tiling size
TILING_SIZE = 32
# Bytes to Bits Conversion
BYTES_TO_BITS = 8

INDEX_ZERO = 0
INDEX_ONE = 1
INDEX_TWO = 2
INDEX_THREE = 3
INDEX_FOUR = 4
INDEX_FIVE = 5

SHAPE_THREHOLD = 293953


def _tilling_axis(shape, dtype):
    """
    calculate the split parameters according to different shapes

    Parameters
    ----------
    shape : list or tuple
        shape of tensor
    dtype : string
        buffer date type

    Returns
    -------
    split_axis : the target axis that is used for spliting the tensor to find
        the maximum amount of data can be stored and processed every time on UB.
    split_factor : the factor used when spliting the target axis.
        For example, for data of float16, [1024, 1024, 256] will be split to
        [1024, 7, 164, 256], UB processes 164*256 elements every time.
        In this case, the split_axis is 1 and the split_factor is 164.
    """

    # Number of Tensor in assign_sub
    tensor_num = 2

    # ub_size_bytes is the size of the UB expressed by bytes(mod 8 bits).
    ub_size_bytes = cce.CceProductParams().getParams("Unified_Buffer")

    # dtype_bytes_size for float16 is 2, for float32 is 4
    dtype_bytes_size = cce.cce_intrin.get_bit_len(dtype) // BYTES_TO_BITS
    # total_ele is the maximum amount of data that can be stored in UB.
    if dtype in ("int8", "uint8"):
        dtype_bytes_size_fp16 = cce.cce_intrin.get_bit_len(
            "float16") // BYTES_TO_BITS
        total_ele = ub_size_bytes // (dtype_bytes_size +
                                      dtype_bytes_size_fp16) // tensor_num
    else:
        total_ele = ub_size_bytes // dtype_bytes_size // tensor_num

    shape_value = shape[-1]
    if dtype in ("int8", "uint8"):
        bytes_size = dtype_bytes_size + dtype_bytes_size_fp16
    else:
        bytes_size = dtype_bytes_size

    ele_num = total_ele // 16 * (shape_value * bytes_size // SHAPE_THREHOLD +
                                 1)

    if ele_num > total_ele // 2:
        total_ele = total_ele // 2
    else:
        total_ele = total_ele // 16 * \
                    (shape_value * bytes_size // SHAPE_THREHOLD // 2 + 1)

    # To initialize the split_axis and the split_factor.
    split_axis = 0
    split_factor = 1

    # To find the appropriate axis from the first one to the last
    # by comparing the amount of the elements of the split tensor with
    # the maximum amount of data that can be stored in UB.
    for index, _ in enumerate(shape):
        ele_cnt = function_reduce(lambda x, y: x * y, shape[index:])
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break

    # when the last axis is still over the size of UB, we choose to split the
    # last axis, and the split_factor is set as the maximum amount of data
    # that can be stored in UB.
    if shape[-1] > total_ele:
        split_axis = len(shape) - 1
        split_factor = (total_ele // TILING_SIZE) * TILING_SIZE

    # when the amount of the elements of the tensor is less than the size of UB,
    # it means UB can process the whole tensor in one time. But the split_axis
    # has already been set to "-1", split_axis and split_factor
    # should be initialized into "0" and shape[0]
    if split_axis < 0:
        split_axis = 0
        split_factor = shape[0]

    return split_axis, split_factor


def _assign_sub_schedule(schedule_list, res, shape, dtype, data_a):
    """
    assign_sub schedule function

    Parameters
    ----------
    schedule_list : list
        list of tensors for schedule.
    res : tvm.tensor
        tensor of result
    shape : list or tuple
        shape of ref and value.
    dtype : str
        the type of ref and value.

    Returns
    -------
    sch: tvm.schedule
        the compute schedule
    """


    # list of tensors for 'elewise_single_cast'
    cast_list = (schedule_list[INDEX_TWO], schedule_list[INDEX_THREE],
                 schedule_list[INDEX_FIVE])

    sch = tvm.create_schedule(res.op)

    for cal_res in schedule_list:
        sch[cal_res].set_scope(cce.scope_ubuf)

    for cal_res in schedule_list:
        sch[cal_res].double_buffer()

    # choose a appropriate method of tiling the tensor
    split_axis, split_factor = _tilling_axis(shape, dtype=dtype)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                            factor=split_factor)
    out_extent = (int(res.shape[0]) + split_factor - 1) // split_factor
    # if out extent > 1, bind to multi core thread axis
    if out_extent > 1:
        block_index = tvm.thread_axis('blockIdx.x')
        if out_extent > cce.CceProductParams().getParams("Device_core_num"):
            thread_axis, axis_outer = sch[res].split(
                axis_outer,
                nparts=cce.CceProductParams().getParams("Device_core_num"))
            sch[res].bind(thread_axis, block_index)
        else:
            sch[res].bind(axis_outer, block_index)

    # compute_at
    for cal_res in schedule_list:
        sch[cal_res].compute_at(sch[res], axis_outer)

    # rewrite the variable
    sch[data_a].reused_by(res)
    sch[schedule_list[INDEX_ZERO]].reused_by(schedule_list[INDEX_FIVE])

    sch[schedule_list[INDEX_ZERO]].emit_insn(
        schedule_list[INDEX_ZERO].op.axis[split_axis], 'dma_copy')
    sch[schedule_list[INDEX_ONE]].emit_insn(
        schedule_list[INDEX_ONE].op.axis[split_axis], 'dma_copy')

    if dtype in ("int8", "uint8"):
        for cal_res in cast_list:
            sch[cal_res].emit_insn(cal_res.op.axis[split_axis],
                                   'elewise_single_cast')
        sch[schedule_list[INDEX_TWO]].reused_by(schedule_list[INDEX_FOUR])

    sch[schedule_list[INDEX_FOUR]].emit_insn(
        schedule_list[INDEX_FOUR].op.axis[split_axis], 'elewise_binary_sub')
    sch[res].emit_insn(axis_inner, 'dma_copy')

    return sch


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,unnecessary-lambda
@fusion_manager.register("assign_sub")
def _assign_sub_compute(data_var, data_value, out, kernel_name='assign_sub'):
    """
    assign_sub compute function

    Parameters
    ----------
    data_var : tvm.tensor
        tensor of var
    data_value : tvm.tensor
        tensor of value
    out : dict
        dict of out.
    kernel_name : str
        cce kernel name, default value is "assign_sub"

    Returns
    -------
    sch : tvm.schedule
        the compute schedule
    res : tvm.tensor
        tensor of result
    """

    shape = data_var.shape
    shape = [i.value for i in shape]
    data_var_ub = tvm.compute(shape,
                              lambda *i: data_var(*i),
                              name='data_var_ub')
    data_value_ub = tvm.compute(shape,
                                lambda *i: data_value(*i),
                                name='data_value_ub')
    if data_var.dtype == "int8" or data_var.dtype == "uint8":
        data_var_cast = tvm.compute(
            shape,
            lambda *i: data_var_ub(*i).astype("float16"),
            name="data_var_cast")
        data_value_cast = tvm.compute(
            shape,
            lambda *i: data_value_ub(*i).astype("float16"),
            name="data_value_cast")
    else:
        data_var_cast = data_var_ub
        data_value_cast = data_value_ub
    res_ub = tvm.compute(shape,
                         lambda *i: data_var_cast(*i) - data_value_cast(*i),
                         name='res_ub.local.UB')
    if data_var.dtype == "int8" or data_var.dtype == "uint8":
        res_ub_cast = tvm.compute(shape,
                                  lambda *i: res_ub(*i).astype(data_var.dtype),
                                  name="res_ub_cast")
    else:
        res_ub_cast = res_ub
    res = tvm.compute(shape, lambda *i: res_ub_cast(*i), name='res')
    schedule_list = (data_var_ub, data_value_ub, data_var_cast,
                     data_value_cast, res_ub, res_ub_cast)
    sch = _assign_sub_schedule(schedule_list, res, shape, data_var.dtype,
                               data_var)

    return sch, res


@util.check_input_type(dict, dict, dict, str)
def assign_sub(var, value, out, kernel_name='assign_sub'):
    """
    Update var by subtracting value from it.

    Parameters:
    ----------
    var : dict
        dict of input_var, include shape and dtype,
        dtype support int8, uint8, int32, float16, float32

    value : dict
        dict of input_value, include shape and dtype,
        dtype support int8, uint8, int32, float16, float32.
        Must have the same shape and dtype as input_var

    out : dict
        dict of out

    kernel_name : str
        cce kernel name, default value is "assign_sub"

    Returns
    -------
    None
    """

    # get the shape and dtype
    shape_var = var.get("shape")
    shape_value = value.get("shape")
    dtype_var = var.get("dtype")
    dtype_value = value.get("dtype")

    # kernel name check: should be unique
    util.check_kernel_name(kernel_name)

    # check whether the shape is right
    check_shape(shape_var)
    check_shape(shape_value)
    if not operator.eq(shape_var, shape_value):
        raise RuntimeError("all input shape must be the equal")

    # check whether dtypes are fp16, fp32, int8, uint8, int32
    # and whether they are the same
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    check_dtype(dtype_var, check_list)
    check_dtype(dtype_value, check_list)
    dtype_var = dtype_var.lower()
    dtype_value = dtype_value.lower()
    if dtype_var != dtype_value:
        raise RuntimeError("all input dtype must be same")

    shape, _ = refine_shape_axes(shape_var, [])
    data_var = tvm.placeholder(shape, dtype=dtype_var, name='data_var')
    data_value = tvm.placeholder(shape, dtype=dtype_value, name='data_value')
    sch, res = _assign_sub_compute(data_var, data_value, out, kernel_name)

    with set_bool_storage_config():
        tvm.build(sch, [data_var, data_value, res], "cce", name=kernel_name)
