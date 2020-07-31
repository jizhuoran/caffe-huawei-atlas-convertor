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

gelu grad
"""
from __future__ import absolute_import

import operator

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# CSVALUE equals 0.044715
CSVALUE = tvm.const(0.044715, "float32")
# SQURT equals np.sqrt(2 / np.pi)
SQURT = tvm.const(0.7978846, "float32")
# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648
# CSVALUE_A equals 2*np.sqrt(2 / np.pi)
CSVALUE_A = tvm.const(1.59576912, "float32")
# CSVALUE_3B equals 3*2*np.sqrt(2 / np.pi)*CSVALUE
CSVALUE_3B = tvm.const(0.2140644488, "float32")


def _math_four_compute(placeholders):
    """
    placeholders: data_x
    return: math_four
    math_four equals 2*(np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))
    """
    data_x = placeholders
    datax_pow = te.lang.cce.vmul(data_x, data_x)
    datax_pow1 = te.lang.cce.vmul(datax_pow, data_x)
    datax_muls_c = te.lang.cce.vmuls(datax_pow1, CSVALUE)
    datax_addx = te.lang.cce.vadd(datax_muls_c, data_x)
    datax_muls_s = te.lang.cce.vmuls(datax_addx, CSVALUE_A)

    return datax_muls_s


def _result_grad_compute(placeholders):
    """
    placeholders: data_x, data_gelu
    return: res_grad
    res_grad equals res/x +
    x*0.5*(1 - tanh(math_four)*tanh(math_four))
    *np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)
    """
    data_x = placeholders[0]
    data_gelu = placeholders[1]

    # common part
    math_four = _math_four_compute(data_x)
    math_four_abs = te.lang.cce.vabs(math_four)  # abs(y)
    math_four_abs_neg = te.lang.cce.vmuls(math_four_abs,
                                          tvm.const(-1.0, "float32"))
    math_four_abs_neg_exp = te.lang.cce.vexp(math_four_abs_neg)  # exp(-abs(y))
    math_four_min = te.lang.cce.vmins(math_four,
                                      tvm.const(0.0, "float32"))  # min(y,0)

    # dividend part
    datax_pow = te.lang.cce.vmul(data_x, data_x)  # x^2
    datax_pow_mul = te.lang.cce.vmuls(datax_pow, CSVALUE_3B)
    datax_pow_mul_add = te.lang.cce.vadds(datax_pow_mul,
                                          CSVALUE_A)  # (a+3bx^2)
    data_gelu_mul = te.lang.cce.vmul(data_gelu, datax_pow_mul_add)
    math_four_min_2 = te.lang.cce.vmuls(math_four_min,
                                        tvm.const(2.0, "float32"))
    div_right = te.lang.cce.vmul(data_gelu_mul, math_four_abs_neg_exp)
    div_left = te.lang.cce.vexp(math_four_min_2)
    dividend = te.lang.cce.vadd(div_left, div_right)

    # divisor part
    div_0 = te.lang.cce.vadds(math_four_abs_neg_exp,
                              tvm.const(1.0, "float32"))  # (1+exp(-abs(y)))
    div_1 = te.lang.cce.vexp(math_four_min)  # exp(min(y,0))
    divisor = te.lang.cce.vmul(div_1, div_0)

    res_grad = te.lang.cce.vdiv(dividend, divisor)
    return res_grad


# pylint:disable = locally-disabled,too-many-arguments
# pylint:disable = unused-argument,no-member
@fusion_manager.register("gelu_grad")
def gelu_grad_compute(data_dy,
                      data_x,
                      data_gelu,
                      output_z,
                      kernel_name="gelu_grad"):
    """
    algorithm: gelu_grad
    calculating: dy*res'
    res' = res/x +
           x*0.5*(1 - tanh(math_four)*tanh(math_four))*
           np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)
    math_four = (np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))

    Parameters
    ----------
    placeholders: TVM tensor.
        input placeholder tensors data
    shape_dy: list or tuple.
        shape of dy
    shape_x: list or tuple.
        shape of x
    shape_y: list or tuple.
        shape of gelu
    dtype: str
        the data type, assume src_dtype equals dst_dtype,
        only support float16, float32
    kernel_name: str
        cce kernel name, default value is "cce_gelu_grad"
    need_build: str
        if need to build CCEC kernel, default value is False
    need_print: str
        if need to print the ir, default value is False

    Returns:
    -------
    A TVM tensor same as input placeholders.
    """
    dtype = data_dy.dtype.lower()
    if dtype == "float16":
        data_dy = te.lang.cce.cast_to(data_dy, "float32")
        data_x = te.lang.cce.cast_to(data_x, "float32")
        data_gelu = te.lang.cce.cast_to(data_gelu, "float32")

    # compute res'
    result5 = _result_grad_compute([data_x, data_gelu])
    # compute dy*res'
    result = te.lang.cce.vmul(result5, data_dy)

    if dtype == "float16":
        result = te.lang.cce.cast_to(result, "float16")

    return result


@util.check_input_type(dict, dict, dict, dict, str)
def gelu_grad(input_dy, input_x, input_y, output_z, kernel_name="gelu_grad"):
    """
    algorithm: gelu_grad
    calculating: dy*res'
    res' = res/x +
           x*0.5*(1 - tanh(math_four)*
           tanh(math_four))*np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)
    math_four = (np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    input_y : dict
        shape and dtype of y input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is gelu_grad

    Returns:
    -------
    none.
    """
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_dy)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)
    input_dtype = input_dy.get("dtype").lower()
    check_list = ("float16", "float32")
    util.check_dtype_rule(input_dtype, check_list)
    shape_dy = list(shape_dy)
    shape_x = list(shape_x)
    shape_y = list(shape_y)
    if not (operator.eq(shape_dy, shape_x) and operator.eq(shape_dy, shape_y)):
        raise RuntimeError("all input shape must be equal")

    data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=input_dtype)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_dtype)
    data_gelu = tvm.placeholder(shape_y, name="data_gelu", dtype=input_dtype)
    res = gelu_grad_compute(data_dy, data_x, data_gelu, output_z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": [data_dy, data_x, data_gelu, res]
    }

    te.lang.cce.cce_build_code(sch, config)
