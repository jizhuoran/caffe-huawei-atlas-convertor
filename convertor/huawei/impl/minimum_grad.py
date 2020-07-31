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

minimum_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


def _compare_value_int32(data_x, data_y, shape_dz):
    """
    The input data type of this function only support int32;
    The return value of the function: if data_x >= data_y return 1;
    else return 0.
    """
    min_value_int = tvm.const(1, dtype="int32")
    data_zero_int = tvm.const(0, dtype="int32")
    min_value_tensor = te.lang.cce.broadcast(min_value_int, shape_dz)
    data_zero_int_tensor = te.lang.cce.broadcast(data_zero_int, shape_dz)
    sub_xy = te.lang.cce.vsub(data_x, data_y)
    add_min = te.lang.cce.vadd(sub_xy, min_value_tensor)
    vmax_zero = te.lang.cce.vmax(add_min, data_zero_int_tensor)
    result = te.lang.cce.vmin(vmax_zero, min_value_tensor)

    return result


# pylint: disable = locally-disabled,invalid-name,too-many-arguments
# pylint: disable = unused-argument
def _compare_value_float(data_x, data_y):
    """
    The input data type of the function only support float;
    The return value of the function: if data_x >= data_y return 1;
    else return 0.
    """
    # The smallest positive subnormal number of float32 is 2**(-126)
    min_value = tvm.const(2 ** (-126), dtype="float32")
    # (2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1
    # so min_value*max_value*max_value*max_value_1 = 1
    max_value = tvm.const(2 ** (62), dtype="float32")
    max_value_1 = tvm.const(2 ** (2), dtype="float32")

    data_zero = te.lang.cce.vmuls(data_x, 0)
    min_value_tensor = te.lang.cce.vadds(data_zero, min_value)
    max_value_tensor = te.lang.cce.vadds(data_zero, max_value)
    max_value_1_tensor = te.lang.cce.vadds(data_zero, max_value_1)
    sub_xy = te.lang.cce.vsub(data_x, data_y)
    add_min_value = te.lang.cce.vadds(sub_xy, min_value)
    vmax_zero = te.lang.cce.vmax(add_min_value, data_zero)
    vmin_min_value = te.lang.cce.vmin(vmax_zero, min_value_tensor)
    vmul_max_value = te.lang.cce.vmul(vmin_min_value, max_value_tensor)
    vmul_max_value_1 = te.lang.cce.vmul(vmul_max_value, max_value_tensor)
    result = te.lang.cce.vmul(vmul_max_value_1, max_value_1_tensor)

    return result


def _compare_value(data_x, data_y, dtype, shape_dz):
    """
    The input data type of the function only support float and int32;
    The return value of the function: if data_x >= data_y return 1;
    else return 0.
    """
    if dtype == "int32":
        compare_value_data = _compare_value_int32(data_x, data_y, shape_dz)
    else:
        compare_value_data = _compare_value_float(data_x, data_y)

    return compare_value_data


def _calculate_result_le(data_x, data_y, data_dz, dtype, shape_dz):
    """
    The input data type of the function only support float int32 dtype;
    The return value of the function: if data_y >= data_x :
    result_dx = data_dz, result_dy = 0;
    else result_dx = 0,result_dx = data_dz.
    """
    minus_one = tvm.const(-1, dtype="int32")
    minus_one_tensor = te.lang.cce.broadcast(minus_one, shape_dz)

    # if data_y >= data_x ; datax_select_le = 1; else datax_select_le =0;
    datax_select_le = _compare_value(data_y, data_x, dtype, shape_dz)
    result_dx = te.lang.cce.vmul(data_dz, datax_select_le)

    select_reverse = te.lang.cce.vadd(datax_select_le, minus_one_tensor)
    select_dy = te.lang.cce.vmul(select_reverse, minus_one_tensor)
    result_dy = te.lang.cce.vmul(data_dz, select_dy)

    return result_dx, result_dy


def _reduce_result(shape_x, shape_y, shape_dz, result_dx, result_dy):
    """
    If the shapes of the two input data are not equal,
    we need to call this function to do reduce operation.
    """
    if shape_x != shape_dz:
        reduce_axis = []
        for i, shape_x_i in enumerate(shape_x):
            if shape_x_i == 1:
                reduce_axis.append(i)
        result_dx = te.lang.cce.sum(result_dx, axis=reduce_axis, keepdims=None)

    if shape_y != shape_dz:
        reduce_axis = []
        for i, shape_y_i in enumerate(shape_y):
            if shape_y_i == 1:
                reduce_axis.append(i)
        result_dy = te.lang.cce.sum(result_dy, axis=reduce_axis, keepdims=None)

    return result_dx, result_dy


@fusion_manager.register("minimum_grad")
def minimum_grad_compute(data_x, data_y, data_dz, y1, y2,
                         kernel_name="minimum_grad"):
    """
    algorithm:
    calculating minimum_grad of the two input data

    Parameters
    ----------
    data_x:TVM tensor.
        the placeholder of data_x
    data_y:TVM tensor.
        the placeholder of data_y
    data_dz:TVM tensor.
        the placeholder of data_dz
    y1: dict:
        dict with keys(shape and dtype) of y1
    y2: dict:
        dict with keys(shape and dtype) of y2
    kernel_name: str
        cce kernel name, default value is "minimum_grad"

    Returns:
    -------
    results of minimum or maximum_grad of the two input data.
    """
    dtype = data_x.dtype
    if data_x.dtype == "float16":
        data_x = te.lang.cce.cast_to(data_x, "float32")
        data_y = te.lang.cce.cast_to(data_y, "float32")
        data_dz = te.lang.cce.cast_to(data_dz, "float32")

    shape_dz = te.lang.cce.util.shape_to_list(data_dz.shape)
    shape_x = te.lang.cce.util.shape_to_list(data_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(data_y.shape)
    data_x = te.lang.cce.broadcast(data_x, shape_dz)
    data_y = te.lang.cce.broadcast(data_y, shape_dz)

    result_dx, result_dy = _calculate_result_le(data_x, data_y, data_dz,
                                                dtype, shape_dz)

    if shape_x != shape_dz or shape_y != shape_dz:
        if dtype == "int32":
            raise RuntimeError("sum not support int32")
        result_dx, result_dy = _reduce_result(shape_x, shape_y, shape_dz,
                                              result_dx, result_dy)

    if dtype == "float16":
        result_dx = te.lang.cce.cast_to(result_dx, "float16")
        result_dy = te.lang.cce.cast_to(result_dy, "float16")

    res = [result_dx, result_dy]

    return res


@util.check_input_type(dict, dict, dict, dict, dict, bool, bool, str)
def minimum_grad(grads, x1, x2, y1, y2, grad_x=True, grad_y=True,
                 kernel_name="minimum_grad"):
    """
    algorithm: minimum_grad
    calculating the reversed outputs of the function "minimum"
    "minimum" : z = vmin(x,y),  dx, dy = minimum_grad(...)

    Parameters
    ----------
    x1: dict
        dict with keys(shape and dtype) of x1
    x2: dict
        dict with keys(shape and dtype) of x2
    grads: dict
        dict with keys(shape and dtype) of grads
    y1: dict:
        dict with keys(shape and dtype) of y1
    y2: dict:
        dict with keys(shape and dtype) of y2
    kernel_name: str
        kernel name, default value is "minimum_grad"

    Returns:
    -------
    none.
    """


    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    shape_dz = grads.get("shape")
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    dtype_dz = grads.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_rule(shape_max)
    util.check_tensor_shape_size(shape_max)

    if list(shape_dz) != list(shape_max):
        raise RuntimeError("minimum_grad shape_dz != shape_max")

    if dtype_x != dtype_y != dtype_dz:
        raise RuntimeError("the dtypes of intputs should be same")

    check_list = ("float16", "float32", "int32")
    util.check_dtype_rule(dtype_dz, check_list)
    util.check_dtype_rule(dtype_x, check_list)
    util.check_dtype_rule(dtype_y, check_list)

    data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_y")
    data_dz = tvm.placeholder(shape_dz, dtype=dtype_dz, name="data_dz")
    res = minimum_grad_compute(data_x, data_y, data_dz, y1, y2, kernel_name)

    with tvm.target.cce():
        if (grad_x, grad_y) == (True, False):
            sch = generic.auto_schedule(res[0])
            res = [res[0]]
        if (grad_x, grad_y) == (False, True):
            sch = generic.auto_schedule(res[1])
            res = [res[1]]
        if (grad_x, grad_y) == (True, True):
            sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_dz, data_x, data_y] + res}
    te.lang.cce.cce_build_code(sch, config)
