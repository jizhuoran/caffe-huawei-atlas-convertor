#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

bessel_i0e

  Op_description :
    Computes the Bessel i0e function of `x` element-wise

    # bessel_i0e(
    #   x,
    #   y,
    #   kernel_name="bessel_i0e")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.
"""

from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
import topi
from topi.cce import util

# const value
ITR_BEFORE = (1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813)
ITR_AFTER = (0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281,
             -0.02057706, 0.02635537, -0.01647633, 0.00392377)
LEN_BEFORE = 7
LEN_AFTER = 9
CONST_LIMIT = 15.0 / 4


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals,
@fusion_manager.register("bessel_i0e")
def bessel_i0e_compute(x, y, kernel_name="bessel_i0e"):
    """
    Algrithm:
    I0 = 1 + ( (z/2) / (1!) )^2 + ((z/2)^2 / (2!))^2 + ... + ((z/2)^n / (n!)) ^2
    I0e = I0 / exp(x)

    t = x / 3.75
    I0(x) = e^-|x|(1 + 3.5156229t^2 + 3.0899424t^4 + 1.2067492t^6 + 0.2659732t^8
            + 0.0360768t^10 + 0.0045813t^12)), |x| <= 3.75
    I0(x) = (1 / sqrt(|x|))*(0.39894228 + 0.01328592t^-1 + 0.00225319t^-2 + -0.00157565t^-3
            + 0.00916281t^-4 + -0.02057706t^-5 + 0.02635537t^-6 + -0.01647633t^-7
            + 0.00392377t^-8), |x| >= 3.75

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "bessel_i0e"

    Returns
    -------
    A tensor. Has the same type as x.

    """

    shape_input = x.shape
    dtype_input = x.dtype

    # chose the type of data in begin
    if dtype_input == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        x = te.lang.cce.cast_to(x, "float32")
    abs_data = te.lang.cce.vabs(x)

    # compute bessel_i0e for data in (-3.75, 3.75)
    broad_const_limit = te.lang.cce.broadcast(tvm.const(CONST_LIMIT, x.dtype),
                                              shape_input)
    before_abs_data = te.lang.cce.vmin(abs_data, broad_const_limit)
    data = te.lang.cce.vdiv(before_abs_data, broad_const_limit)
    square_data = te.lang.cce.vmul(data, data)

    before_res = te.lang.cce.vmuls(square_data, tvm.const(ITR_BEFORE[LEN_BEFORE - 1]))
    before_res = te.lang.cce.vadds(before_res, ITR_BEFORE[LEN_BEFORE - 2])
    for index in reversed(range(LEN_BEFORE - 2)):
        before_res = te.lang.cce.vmul(before_res, square_data)
        before_res = te.lang.cce.vadds(before_res, ITR_BEFORE[index])

    exp_data = te.lang.cce.vexp(before_abs_data)
    before_res = te.lang.cce.vdiv(before_res, exp_data)

    # compute bessel_i0e for data in other domain
    data = te.lang.cce.vdiv(broad_const_limit, abs_data)

    after_res = te.lang.cce.vmuls(data, tvm.const(ITR_AFTER[LEN_AFTER - 1]))
    after_res = te.lang.cce.vadds(after_res, ITR_AFTER[LEN_AFTER - 2])
    for index in reversed(range(LEN_AFTER - 2)):
        after_res = te.lang.cce.vmul(after_res, data)
        after_res = te.lang.cce.vadds(after_res, ITR_AFTER[index])

    sqrt_data = te.lang.cce.vsqrt(abs_data, 1)

    after_res = te.lang.cce.vdiv(after_res, sqrt_data)
    after_res = te.lang.cce.vmin(before_res, after_res)

    # chose the type of data in end
    if dtype_input == "float16":
        after_res = te.lang.cce.cast_to(after_res, "float16")

    return after_res


@util.check_input_type(dict, dict, str)
def bessel_i0e(x, y, kernel_name="bessel_i0e"):
    """
    Computes the Bessel i0e function of x element-wise.

    Parameters
    ----------
    x: the dict of input, only support float16, float32

    y : the dict of output

    kernel_name : cce kernel name, default value is "bessel_i0e"

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    util.check_kernel_name(kernel_name)
    check_shape(shape_input)
    shape_input, _ = refine_shape_axes(shape_input, [])

    check_list = ("float16", "float32")
    check_dtype(dtype_input, check_list)

    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_input, dtype=input_dtype, name="data_input")

    res = bessel_i0e_compute(data, y, kernel_name)

    with tvm.target.cce():
        sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data, res)}
    te.lang.cce.cce_build_code(sch, config)
