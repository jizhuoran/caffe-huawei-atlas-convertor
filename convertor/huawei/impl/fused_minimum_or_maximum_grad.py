#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fused_minimum_or_maximum_grad
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# shape size limit for aicore is 2**31
SHAPE_SIZE_LIMIT = 2147483647


def _compare_value_int32(x_data, y_data, shape_dz):
    """
    The input data type of this function only support int32;
    The return value of the function: if x_data >= y_data return 1; else return 0.
    """
    min_value_int = tvm.const(1, dtype="int32")
    data_zero_int = tvm.const(0, dtype="int32")
    min_value_tensor = te.lang.cce.broadcast(min_value_int, shape_dz)
    data_zero_int_tensor = te.lang.cce.broadcast(data_zero_int, shape_dz)
    sub_xy = te.lang.cce.vsub(x_data, y_data)
    add_min = te.lang.cce.vadd(sub_xy, min_value_tensor)
    vmax_zero = te.lang.cce.vmax(add_min, data_zero_int_tensor)
    result = te.lang.cce.vmin(vmax_zero, min_value_tensor)

    return result


# pylint: disable = locally-disabled,too-many-locals
def _compare_value_float(x_data, y_data):
    """
    The input data type of the function only support float;
    The return value of the function: if x_data >= y_data return 1; else return 0.
    """
    # The smallest positive subnormal number of float32 is 2**(-126)
    min_value = tvm.const(2**(-126), dtype="float32")
    # (2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1
    # so min_value*max_value*max_value*max_value_1 = 1
    max_value = tvm.const(2**(62), dtype="float32")
    max_value_1 = tvm.const(2**(2), dtype="float32")

    data_zero = te.lang.cce.vmuls(x_data, 0)
    min_value_tensor = te.lang.cce.vadds(data_zero, min_value)
    max_value_tensor = te.lang.cce.vadds(data_zero, max_value)
    max_value_1_tensor = te.lang.cce.vadds(data_zero, max_value_1)
    sub_xy = te.lang.cce.vsub(x_data, y_data)
    add_min_value = te.lang.cce.vadds(sub_xy, min_value)
    vmax_zero = te.lang.cce.vmax(add_min_value, data_zero)
    vmin_min_value = te.lang.cce.vmin(vmax_zero, min_value_tensor)
    vmul_max_value = te.lang.cce.vmul(vmin_min_value, max_value_tensor)
    vmul_max_value_1 = te.lang.cce.vmul(vmul_max_value, max_value_tensor)
    result = te.lang.cce.vmul(vmul_max_value_1, max_value_1_tensor)

    return result


def _compare_value(x_data, y_data, dtype, shape_dz):
    """
    The input data type of the function only support float and int32;
    The return value of the function: if x_data >= y_data return 1; else return 0.
    """
    dtype = dtype.lower()
    if dtype == "int32":
        compare_value_data = _compare_value_int32(x_data, y_data, shape_dz)
    else:
        compare_value_data = _compare_value_float(x_data, y_data)

    return compare_value_data


def _calculate_result_le(x_data, y_data, dz_data, dtype, shape_dz):
    """
    The input data type of the function only support float int32 dtype;
    The return value of the function: if y_data >= x_data : result_dx = dz_data, result_dy = 0;
    else result_dx = 0,result_dx = dz_data.
    """
    minus_one = tvm.const(-1, dtype="int32")

    minus_one_tensor = te.lang.cce.broadcast(minus_one, shape_dz)
    # if y_data >= x_data ; datax_select_le = 1; else datax_select_le =0;
    datax_select_le = _compare_value(y_data, x_data, dtype, shape_dz)
    result_dx = te.lang.cce.vmul(dz_data, datax_select_le)
    select_reverse = te.lang.cce.vadd(datax_select_le, minus_one_tensor)
    select_dy = te.lang.cce.vmul(select_reverse, minus_one_tensor)
    result_dy = te.lang.cce.vmul(dz_data, select_dy)

    return result_dx, result_dy


def _calculate_result_ge(x_data, y_data, dz_data, dtype, shape_dz):
    """
    The input data type of the function only support float int32 dtype;
    The return value of the function: if x_data >= y_data : result_dx = dz_data, result_dy = 0;
    else result_dx = 0,result_dx = dz_data.
    """
    minus_one = tvm.const(-1, dtype="int32")

    minus_one_tensor = te.lang.cce.broadcast(minus_one, shape_dz)
    # if x_data >= y_data ; datax_select_ge = 1; else datax_select_ge =0;
    datax_select_ge = _compare_value(x_data, y_data, dtype, shape_dz)
    result_dx = te.lang.cce.vmul(dz_data, datax_select_ge)
    select_reverse = te.lang.cce.vadd(datax_select_ge, minus_one_tensor)
    select_dy = te.lang.cce.vmul(select_reverse, minus_one_tensor)
    result_dy = te.lang.cce.vmul(dz_data, select_dy)

    return result_dx, result_dy


def _reduce_result(shape_x, shape_y, shape_dz, result_dx, result_dy):
    """
    If the shapes of the two input data are not equal,
    we need to call this function to do reduce operation.
    """
    if list(shape_x) != list(shape_dz):
        reduce_axis = []
        for i, shape_x_i in enumerate(shape_x):
            if shape_x_i == 1:
                reduce_axis.append(i)
        result_dx = te.lang.cce.sum(result_dx, axis=reduce_axis, keepdims=None)

    if list(shape_y) != list(shape_dz):
        reduce_axis = []
        for i, shape_y_i in enumerate(shape_y):
            if shape_y_i == 1:
                reduce_axis.append(i)
        result_dy = te.lang.cce.sum(result_dy, axis=reduce_axis, keepdims=None)

    return result_dx, result_dy


# pylint: disable = locally-disabled,invalid-name,too-many-arguments,unused-argument,no-member
@fusion_manager.register("fused_minimum_or_maximum_grad_cce")
def fused_minimum_or_maximum_grad_compute(placeholders, shape_x, shape_y, shape_dz, cmp_type,
                                          dtype,
                                          kernel_name="cce_fused_minimum_or_maximum_grad",
                                          need_build=False, need_print=False):
    """
    algorithm:
    calculating minimum or maximum_grad of the two input data

    Parameters
    ----------
    placeholders:TVM tensor.
        The tensor of inputs data
    shape_x: list or tuple.
        shape of data_inputx
    shape_y: list or tuple.
        shape of data_inputy
    shape_dz: list or tuple.
        shape of data_inputdz
    cmp_type: str
        LessEqual or GreatEqual
    dtype: str
        the data type, assume src_dtype equals dst_dtype,
        only support float16, float32, int32
    kernel_name: str
        cce kernel name, default value is "cce_fused_minimum_or_maximum_grad"
    need_build: bool
        if need to build CCEC kernel, default value is False
    need_print: bool
        if need to print the ir, default value is False

    Returns:
    -------
    results of minimum or maximum_grad of the two input data.
    """
    dz_data, inputx_data, inputy_data = placeholders
    if dtype == "float16":
        inputx_data = te.lang.cce.cast_to(inputx_data, "float32")
        inputy_data = te.lang.cce.cast_to(inputy_data, "float32")
        dz_data = te.lang.cce.cast_to(dz_data, "float32")
    inputx_data = te.lang.cce.broadcast(inputx_data, shape_dz)
    inputy_data = te.lang.cce.broadcast(inputy_data, shape_dz)

    if cmp_type == "LE":
        result_dx, result_dy = _calculate_result_le(inputx_data, inputy_data,
                                                    dz_data, dtype, shape_dz)
    if cmp_type == "GE":
        result_dx, result_dy = _calculate_result_ge(inputx_data, inputy_data,
                                                    dz_data, dtype, shape_dz)
    if list(shape_x) != list(shape_dz) or list(shape_y) != list(shape_dz):
        result_dx, result_dy = _reduce_result(shape_x, shape_y, shape_dz, result_dx, result_dy)

    if dtype == "float16":
        result_dx = te.lang.cce.cast_to(result_dx, "float16")
        result_dy = te.lang.cce.cast_to(result_dy, "float16")
    outs = [result_dx, result_dy]

    return outs


@util.check_input_type((list, tuple), (list, tuple), (list, tuple), bool, bool, str,
                       str, str, bool, bool)
def fused_minimum_or_maximum_grad_cce(shape_dz, shape_x, shape_y, grad_x=True, grad_y=True,
                                      cmp_type="LE", dtype="float32",
                                      kernel_name="cce_fused_minimum_or_maximum_grad",
                                      need_build=False, need_print=False):
    """
    algorithm:
    calculating minimum or maximum_grad of the two input data

    Parameters
    ----------
    shape_dz: list or tuple.
        shape of data_inputdz
    shape_x: list or tuple.
        shape of data_inputx
    shape_y: list or tuple.
        shape of data_inputy
    grad_x: bool
        if grad_x is true,output need return dx
    grad_y: bool
        if grad_y is true,output need return dy
    cmp_type: str
        LessEqual or GreatEqual
    dtype: str
        the data type, assume src_dtype equals dst_dtype,
        only support float16, float32, int32
    kernel_name: str
        cce kernel name, default value is "cce_fused_minimum_or_maximum_grad"
    need_build: bool
        if need to build CCEC kernel, default value is False
    need_print: bool
        if need to print the ir, default value is False

    Returns:
    -------
    none.
    """
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_rule(shape_max)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    if list(shape_dz) != list(shape_max):
        raise RuntimeError("fused_minimum_or_maximum_grad_cce shape_dz != shape_max")

    dtype = dtype.lower()
    if dtype not in ["float16", "float32", "int32"]:
        raise RuntimeError("fused_minimum_or_maximum_grad_cce only support"
                           " float16, float32, int32")

    if (grad_x, grad_y) == (False, False):
        raise RuntimeError("grad_x and grad_x at least one is true")

    placeholders = []
    placeholders.append(tvm.placeholder(shape_dz, name="input_dz", dtype=dtype))
    placeholders.append(tvm.placeholder(shape_x, name="input_x", dtype=dtype))
    placeholders.append(tvm.placeholder(shape_y, name="input_y", dtype=dtype))

    outs = fused_minimum_or_maximum_grad_compute(placeholders, shape_x, shape_y,
                                                 shape_dz, cmp_type, dtype)

    with tvm.target.cce():
        if (grad_x, grad_y) == (True, False):
            sch = generic.auto_schedule(outs[0])
            outs = [outs[0]]
        if (grad_x, grad_y) == (False, True):
            sch = generic.auto_schedule(outs[1])
            outs = [outs[1]]
        if (grad_x, grad_y) == (True, True):
            sch = generic.auto_schedule(outs)

    config = {"print_ir": need_print,
              "need_build": need_build,
              "name": kernel_name,
              "tensor_list": placeholders + outs}

    te.lang.cce.cce_build_code(sch, config)
