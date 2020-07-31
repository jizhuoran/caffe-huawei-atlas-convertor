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

sin
"""
import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from functools import reduce as reduceIns
from topi import generic
from topi.cce import util

# define a string name of "float16"
FLOAT_16 = "float16"
# define a string name of "float32"
FLOAT_32 = "float32"

PI = 3.14159265358979

# the first factor to use Taylor series in circle
FIRST_ORDER = 5
# the last factor to use Taylor series in circle
LAST_ORDER = 13
# the first factor of Taylor series
FIRST_FACTOR = -1.0 / 6.0


# pylint: disable=invalid-name
def _sin(x):
    """
    algorithm: sin
    calculating data's sin x = x-x^3/3!+x ^5/5!-x^7/7!+x^9/9!-x^11/11! (-pai/2 < x < pai/2)

    Parameters
    ----------
    x : TVM tensor
        the placeholders of input data

    Returns
    -------
    res : the res of sin
    """
    input_x_power = te.lang.cce.vmul(x, x)
    iter_value = te.lang.cce.vmul(te.lang.cce.vmuls(input_x_power, FIRST_FACTOR), x)
    res = te.lang.cce.vadd(x, iter_value)

    i = FIRST_ORDER
    while i < LAST_ORDER:
        iter_value = te.lang.cce.vmuls(te.lang.cce.vmul(input_x_power, iter_value),
                                       -1.0 / (i*(i - 1)))
        res = te.lang.cce.vadd(res, iter_value)
        # add 2 to get the next order
        i = i + 2

    return res


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("sin")
def sin_compute(x, y, kernel_name="sin"):
    """
    algorithm: sin
    calculating data's sin x = x - x^3/3! + x ^5/5! + ... + (-1)^k*x^2(k+1)/(2(k+1))!

    Parameters
    ----------
    x : TVM tensor
        the placeholders of input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "sin"

    Returns
    -------
    res : the res of sin
    """
    dtype = x.dtype
    shape = te.lang.cce.util.shape_to_list(x.shape)

    has_improve_precision = False
    cast_dtype = FLOAT_16
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul", "float32"):
        has_improve_precision = True
        cast_dtype = FLOAT_32

    # cast to type float32 when type is float16
    if dtype == FLOAT_16 and has_improve_precision:
        x = te.lang.cce.cast_to(x, FLOAT_32)

    pai_multiple = te.lang.cce.vmuls(x, 1 / PI)
    round_float = te.lang.cce.cast_to(te.lang.cce.round(pai_multiple), cast_dtype)
    # to adjust x to [-pai/2,pai/2]
    x = te.lang.cce.vsub(x, te.lang.cce.vmuls(round_float, PI))

    res = _sin(x)

    # if round is odd, the final result need to mutiply -1.Need to multipy 1/2 to get the ceil value
    ceil_value = te.lang.cce.ceil(te.lang.cce.vmuls(round_float, 1 / 2))
    # if odd, ceil*2-round is 1,if even, the value is 0
    sub_value = te.lang.cce.vsub(te.lang.cce.vmuls(ceil_value, tvm.const(2, dtype)), round_float)
    tensor_one = te.lang.cce.broadcast(tvm.const(1, cast_dtype), shape)
    odd_tensor = te.lang.cce.vsub(tensor_one, sub_value)
    even_tensor = te.lang.cce.vsub(odd_tensor, tensor_one)
    odd_even_tensor = te.lang.cce.vadd(odd_tensor, even_tensor)
    res = te.lang.cce.vmul(res, odd_even_tensor)

    # cast the dtype to float16
    if dtype == FLOAT_16 and has_improve_precision:
        res = te.lang.cce.cast_to(res, FLOAT_16)

    return res


@util.check_input_type(dict, dict, str)
def sin(x, y, kernel_name="sin"):
    """
    algorithm: sin
    calculating data's sin x = x - x^3/3! + x^5/5! + ... + (-1)^k*x^2(k+1)/(2(k+1))!

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "sin"

    Returns
    -------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype").lower()

    util.check_shape_rule(shape_input)
    util.check_kernel_name(kernel_name)
    check_list = (FLOAT_16, FLOAT_32)
    util.check_dtype_rule(dtype_input, check_list)
    util.check_tensor_shape_size(shape_input)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_input)
    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=dtype_input)
    res = sin_compute(data_input, y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_input, res)}
    te.lang.cce.cce_build_code(sch, config)
