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

gelu
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te import platform as tbe_platform

# const CSVALUE equals 0.044715
CSVALUE = tvm.const(0.044715, "float32")
# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648


def _tanh_parameter_compute(placeholders):
    """
    compute the parameter of tanh:
    return: result equals (x+0.044715*tf.pow(x,3))
    """
    data = placeholders
    mul_0 = te.lang.cce.vmul(data, data)
    pow_0 = te.lang.cce.vmul(mul_0, data)
    mul_1 = te.lang.cce.vmuls(pow_0, CSVALUE)
    result = te.lang.cce.vadd(data, mul_1)

    return result

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@fusion_manager.register("gelu")
def gelu_compute(input_x, output_y, kernel_name="gelu"):
    """
    mathematical formula of gelu(x):
    gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    tanh(y) = 2/(1+exp(-2y)) - 1
    convert gelu to result(x) =
      x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is gelu

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    dtype = input_x.dtype.lower()
    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                    "float32"):
        has_improve_precision = True
        input_x = te.lang.cce.cast_to(input_x, "float32")

    # gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    # tanh(y) = 2/(1+exp(-2y)) - 1

    # simplize
    # gelu(x) = x/(1+e^(-y))
    # the formula is y = 2*np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))

    # to avoid overflow, keep exp negative
    # gelu(x) = x/(1+e^(-|y|)) * v_const
    # the formula is y = 2*np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))
    # v_const = 1 if x > 0  , e^y  if x < 0
    const_0 = tvm.const(1.5957691, "float32") # 2*np.sqrt(2/np.pi)
    const_1 = tvm.const(1.0, "float32")
    const_2 = tvm.const(-1.0, "float32")
    const_3 = tvm.const(0.0, "float32")
    tanh_parameter = _tanh_parameter_compute(input_x)
    mul_0 = te.lang.cce.vmuls(tanh_parameter, const_0)  # y

    mul_0_min = te.lang.cce.vmins(mul_0, const_3)
    right_mul = te.lang.cce.vexp(mul_0_min)

    mul_0_abs = te.lang.cce.vabs(mul_0)   # abs(y)
    mul_0_abs_neg = te.lang.cce.vmuls(mul_0_abs, const_2)  # -abs(y)

    # the formula is e^(-abs(y))
    mul_0_abs_neg_exp = te.lang.cce.vexp(mul_0_abs_neg)

    # the formula is e^(-abs(y)) + 1
    mul_0_abs_neg_exp_add = te.lang.cce.vadds(mul_0_abs_neg_exp, const_1)
    left_mul = te.lang.cce.vdiv(input_x, mul_0_abs_neg_exp_add)

    result = te.lang.cce.vmul(left_mul, right_mul)

    if has_improve_precision:
        result = te.lang.cce.cast_to(result, "float16")

    return result


@util.check_input_type(dict, dict, str)
def gelu(input_x, output_y, kernel_name="gelu"):
    """
    mathematical formula of gelu(x):
    gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    tanh(y) = 2/(1+exp(-2y)) - 1
    convert gelu to result(x) =
     x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is gelu

    Returns
    -------
    none.
    """
    shape = input_x.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(shape)

    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    util.check_dtype_rule(input_dtype, check_list)

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=input_dtype)
    result = gelu_compute(data, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, result]}

    te.lang.cce.cce_build_code(sch, config)
