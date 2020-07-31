#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except
in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

acosh_grad

  Op_description :
    Computes gradients for Acosh operation

    # acosh_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_acosh_grad")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : 'y' and 'dy' must have the same type and shape.
    [2] All : shape size limit is 2147483648.
"""

import operator

from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
from topi import generic
from topi.cce import util

NUM_ONE = 1
NUM_TWO = 2
NUM_REPEAT = 0.125

TAYLOR_SECOND_ORDER_PARAM = 0.1666666666666666666666666666666666
TAYLOR_THIRD_ORDER_PARAM = 0.0083333333333333333333333333333333
TAYLOR_FOURTH_ORDER_PARAM = 0.0001984126984126984126984126984126


def _taylor_sinh_compute(input_data):
    """
    do taylor sinh compute
    Parameters:
    ----------------
    input_data: input tensor
    return: sinh result
    ----------------
    """

    # x^2 / 7!
    data_power_2 = te.lang.cce.vmul(input_data, input_data)
    data_power_res = te.lang.cce.vmuls(
        data_power_2,
        tvm.const(TAYLOR_FOURTH_ORDER_PARAM, input_data.dtype))

    # 1/5! + x^2 / 7!
    data_power_res = te.lang.cce.vadds(
        data_power_res,
        tvm.const(TAYLOR_THIRD_ORDER_PARAM, input_data.dtype))

    # 1/3! + x^2( 1/5! + x^2/7!)
    data_power_res = te.lang.cce.vmul(data_power_res, data_power_2)
    data_power_res = te.lang.cce.vadds(
        data_power_res,
        tvm.const(TAYLOR_SECOND_ORDER_PARAM, input_data.dtype))

    # 1 + x^2( 1/3! + x^2(1/5! + x^2/7!))
    data_power_res = te.lang.cce.vmul(data_power_res, data_power_2)

    data_power_res = te.lang.cce.vadds(data_power_res, \
                     tvm.const(NUM_ONE, input_data.dtype))

    # x * (1 + x^2( 1/3! + x^2(1/5! + x^2/7!)))
    data_power_res = te.lang.cce.vmul(data_power_res, input_data)
    return data_power_res


def _sinh_repeat_with_sqrt(data):
    """
    do sinh convert compute with sqrt
    Calculate f(2x) = 2f(x) * (f(x)^2 + 1)^0.5
    Parameters:
    ----------------
    data: input tensor
    return: sinh repeat result
    ----------------
    """

    data_square = te.lang.cce.vmul(data, data)
    data_square = te.lang.cce.vadds(data_square, tvm.const(NUM_ONE,
                                                           data.dtype))

    data_square = te.lang.cce.vsqrt(data_square, 1)

    data_square = te.lang.cce.vmul(data_square, data)
    data_square = te.lang.cce.vmuls(data_square, tvm.const(NUM_TWO,
                                                           data.dtype))

    return data_square


# pylint: disable=unused-argument
@fusion_manager.register("acosh_grad")
def acosh_grad_compute(y, dy, z, kernel_name="acos_grad"):
    """
    do acosh_grad compute
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "acosh_grad"
    return: dy * (1 / sinh(y))
    ----------------
    """

    dtype = y.dtype
    dtype_1 = dtype
    if dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        y = te.lang.cce.cast_to(y, "float32")
        dy = te.lang.cce.cast_to(dy, "float32")
        dtype = "float32"

    data_y = te.lang.cce.vmuls(y, tvm.const(NUM_REPEAT, dtype))
    sinh_value_0 = _taylor_sinh_compute(data_y)
    sinh_value_1 = _sinh_repeat_with_sqrt(sinh_value_0)
    sinh_value_2 = _sinh_repeat_with_sqrt(sinh_value_1)
    data_sinh = _sinh_repeat_with_sqrt(sinh_value_2)

    res = te.lang.cce.vdiv(dy, data_sinh)

    if dtype_1 == "float16":
        res = te.lang.cce.cast_to(res, "float16")
    return res


@util.check_input_type(dict, dict, dict, str)
def acosh_grad(y, dy, z, kernel_name="acosh_grad"):
    """
    do element-wise acosh_grad operation between two input tensors
    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32
    dy : dict of dy, include shape and dtype, dtype support float16, float32
    z : dict of z
    kernel_name : cce kernel name, default value is "acosh_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype = y.get("dtype")
    dtype_dy = dy.get("dtype")

    util.check_kernel_name(kernel_name)
    check_shape(shape_y)
    check_shape(shape_dy)
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    if not operator.eq(shape_y, shape_dy):
        raise RuntimeError(
            "acosh_grad only support input shape while input_shape1 equals to input_shape2"
        )
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    # raise runtimeerror if the input paras are invalid
    check_list = ("float16", "float32")
    check_dtype(dtype, check_list)
    check_dtype(dtype_dy, check_list)
    dtype = dtype.lower()
    dtype_dy = dtype_dy.lower()

    if dtype != dtype_dy:
        raise RuntimeError(
            "acosh_grad only support dtype while input_dtype_dy equals to input_dtype2"
        )

    data_y = tvm.placeholder(shape_y, dtype=dtype, name="data1")
    data_dy = tvm.placeholder(shape_dy, dtype=dtype_dy, name="data2")

    res = acosh_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (data_y, data_dy, res)}
    te.lang.cce.cce_build_code(sch, config)
