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

asinh_grad

  Op_description :
    Computes gradients for Asinh operation

    # asinh_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_asinh_grad")

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

# scalar in asinh_grad
NUM_MINUS_ONE = -1
NUM_TWO = 2
NUM_ONE = 1
NUM_REPEAT = 0.125

# scalar 1/2! , 1/4! and 1/6! used in taylor
TAYLOR_SECOND = 0.5
TAYLOR_FOURTH = 1 / 24.0
TAYLOR_SIXTH = 1 / 720.0


def _cosh_taylor_compute(data):
    """
    Calculate cosh  = 1 + x^2( 1/2! + x^2( 1/4! + x^2/6!))

    Parameters:
    ----------
    data : the placeholder of data input

    Returns
    -------
    A Tensor represents cosh(data). Has the same type as data.
    """

    # x^2 / 6!
    pow_2 = te.lang.cce.vmul(data, data)
    pow_2_div = te.lang.cce.vmuls(pow_2, tvm.const(TAYLOR_SIXTH, data.dtype))

    # 1/4! + x^2 / 6!
    pow_2_plus = te.lang.cce.vadds(pow_2_div, tvm.const(TAYLOR_FOURTH,
                                                        data.dtype))

    # 1/2! + x^2( 1/4! + x^2/6!)
    pow_4 = te.lang.cce.vmul(pow_2_plus, pow_2)
    pow_4_plus = te.lang.cce.vadds(pow_4, tvm.const(TAYLOR_SECOND,
                                                    data.dtype))

    # 1 + x^2( 1/2! + x^2( 1/4! + x^2/6!))
    pow_6 = te.lang.cce.vmul(pow_4_plus, pow_2)
    res = te.lang.cce.vadds(pow_6, tvm.const(NUM_ONE, data.dtype))

    return res


def _cosh_repeat(data):
    """
    Calculate f(2x) = 2f(x)^2 -1

    Parameters:
    ----------
    data : the placeholder of data input

    Returns
    -------
    A Tensor represents f(2x). Has the same type as data.
    """

    data_square = te.lang.cce.vmul(data, data)
    data_mul = te.lang.cce.vmuls(data_square, tvm.const(NUM_TWO, data.dtype))
    res = te.lang.cce.vadds(data_mul, tvm.const(NUM_MINUS_ONE, data.dtype))

    return res


# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("asinh_grad")
def asinh_grad_compute(y, dy, output_res, kernel_name="cce_asinh_grad"):
    """
    do element-wise asinh_grad compute

    Parameters:
    ----------
    y : the placeholders of input y

    dy : the placeholders of input dy

    output_res : output dict

    kernel_name : cce kernel name, default value is "cce_asinh_grad"

    Return :
    -------
    dy * (1/cosh(y))
    """

    dtype = y.dtype
    if dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        y = te.lang.cce.cast_to(y, "float32")
        dy = te.lang.cce.cast_to(dy, "float32")

    if api_check_support('te.lang.cce.vexp', 'float32'):
        # use vexp,vdiv api for high efficiency computation
        # cosh(y) = (e^y + e^-y) / 2
        #           (e^2y + 1) / 2e^y
        exp_pos = te.lang.cce.vexp(y)
        res = te.lang.cce.vmul(exp_pos, exp_pos)
        res = te.lang.cce.vadds(res, tvm.const(NUM_ONE, y.dtype))
        data_dy1 = te.lang.cce.vmuls(dy, tvm.const(NUM_TWO, y.dtype))
        data_dy1 = te.lang.cce.vmul(data_dy1, exp_pos)
        res = te.lang.cce.vdiv(data_dy1, res)
    else:
        # use taylor's method for high accuracy result
        y = te.lang.cce.vmuls(y, tvm.const(NUM_REPEAT, y.dtype))
        cosh_value_0 = _cosh_taylor_compute(y)
        # repeat 3 times
        cosh_value_1 = _cosh_repeat(cosh_value_0)
        cosh_value_2 = _cosh_repeat(cosh_value_1)
        cosh_value = _cosh_repeat(cosh_value_2)
        res = te.lang.cce.vrec(cosh_value)
        res = te.lang.cce.vmul(res, dy)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, dict, str)
def asinh_grad(y, dy, z, kernel_name="cce_asinh_grad"):
    """
    do element-wise asinh_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of output

    kernel_name : cce kernel name, default value is "cce_asinh_grad"

    Returns
    -------
    None
    """

    # get the shape and dtype
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype_y = y.get("dtype")
    dtype_dy = dy.get("dtype")

    # kernel name check: should be unique
    util.check_kernel_name(kernel_name)

    # check whether the shape is right
    check_shape(shape_y)
    check_shape(shape_dy)
    if not operator.eq(shape_y, shape_dy):
        raise RuntimeError("all input shape must be the same")
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32")
    check_dtype(dtype_y, check_list)
    check_dtype(dtype_dy, check_list)
    dtype_y = dtype_y.lower()
    if dtype_y != dtype_dy.lower():
        raise RuntimeError("all input dtype must be same")

    # get 2 input tensors: data_y, data_dy
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
    data_dy = tvm.placeholder(shape_y, name="data_dy", dtype=dtype_y)
    res = asinh_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
