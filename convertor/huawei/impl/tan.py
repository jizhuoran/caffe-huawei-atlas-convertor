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

tan
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te import platform as tbe_platform

# define a string name of "float16"
FLOAT_16 = "float16"
# define a string name of "float32"
FLOAT_32 = "float32"
# define a string name of "int32"
INT_32 = "int32"
# define the PI
PI = 3.14159265
# define the expansion order of Tan series
TAN_EXPANSION_ORDER = 5
# define the number of times using the tan2x formula
TAN_2X_TIMES = 6


def _tan_expand(input_x):
    """
    calculating tan x = x + x^3/3 + 2*x^5/15 + 17*x^7/315 +
                        62*x^9/2835 + 1382*x^11/155925...(|x|<pi/2)
    """
    # Taylor expansion coefficient
    factors = [1 / 3, 2 / 15, 17 / 315, 62 / 2835, 1382 / 155925]

    input_x_power = te.lang.cce.vmul(input_x, input_x)
    iter_value = input_x
    res = input_x

    for i, _ in enumerate(range(TAN_EXPANSION_ORDER)):
        iter_value = te.lang.cce.vmuls(
            te.lang.cce.vmul(input_x_power, iter_value), factors[i])
        res = te.lang.cce.vadd(res, iter_value)

    return res


def _tan_2x_multi(input_x, times):
    """
    calculating tan x by calculating tan (x/2^times) and
    using formula tan 2x = 2*tan x/(1-tan x*tan x) multiple times
    """
    # calculate tan (x/2^times)
    input_x_divide = te.lang.cce.vmuls(input_x, 1.0 / (2.0**times))
    res = _tan_expand(input_x_divide)

    while times != 0:
        # using double angle formula: tan 2x = 2*tan x/(1-tan x*tan x)
        res_denominator = te.lang.cce.vmuls(res, 2.0)
        tanx_square = te.lang.cce.vmul(res, res)
        res_numerator = te.lang.cce.vadds(te.lang.cce.vmuls(tanx_square, -1.0),
                                          1.0)
        res = te.lang.cce.vdiv(res_denominator, res_numerator)
        times = times - 1

    return res


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("tan")
def tan_compute(x, y, kernel_name="tan"):
    """
    algorithm: tan
    calculating tan x using _tan_2x_multi

    Parameters
    ----------
    x : TVM tensor
        the placeholders of x
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "tan"

    Returns
    -------
    res: TVM tensor
        the result of tan(x)
    """
    dtype = x.dtype

    has_improve_precision = False
    cast_dtype = FLOAT_16
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        has_improve_precision = True
        cast_dtype = FLOAT_32

    # cast to type float32 when type is float16 or int32
    if dtype in (FLOAT_16, INT_32):
        if has_improve_precision:
            x = te.lang.cce.cast_to(x, FLOAT_32)

    # adjust x to [-pi/2,pi/2] using x = x-round(x/pi)*pi
    round_pi_div = te.lang.cce.round(
        te.lang.cce.vmuls(x, tvm.const(1.0 / PI, cast_dtype)))
    if has_improve_precision:
        round_pi_div = te.lang.cce.cast_to(round_pi_div, FLOAT_32)
    input_x = te.lang.cce.vsub(
        x, te.lang.cce.vmuls(round_pi_div, tvm.const(PI, cast_dtype)))

    res = _tan_2x_multi(input_x, TAN_2X_TIMES)

    # cast the dtype to original dtype
    if dtype in (FLOAT_16, INT_32):
        if has_improve_precision:
            res = te.lang.cce.cast_to(res, dtype)

    return res


@util.check_input_type(dict, dict, str)
def tan(x, y, kernel_name="tan"):
    """
    algorithm: tan
    calculating tan x = x + x^3/3 + 2*x^5/5 + 17*x^7/315 +
                        62*x^9/2835 + 1382*x^11/155925...(|x|<pi/2)

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of input
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "tan"

    Returns
    -------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype").lower()

    util.check_shape_rule(shape_input)
    util.check_kernel_name(kernel_name)
    check_list = (FLOAT_16, FLOAT_32, INT_32)
    util.check_dtype_rule(dtype_input, check_list)
    util.check_tensor_shape_size(shape_input)

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_input)
    data_input = tvm.placeholder(fuseshape,
                                 name="data_input",
                                 dtype=dtype_input)
    res = tan_compute(data_input, y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (data_input, res)}
    te.lang.cce.cce_build_code(sch, config)
