#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

acos

  Op_description :
    Computes acos of x element-wise

    # acos(
    #   x,
    #   y,
    #   kernel_name='cce_acos')

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : shape size limit is 2147483648.

"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_conf import api_check_support
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
import topi
from topi.cce import util
from impl.util import util_compute


NUM_ONE = 1.0
NEG_NUM_ONE = -1.0
HALF_PI = 1.5707963267948966192313216916398
BOUNDARY_1 = 0.70710678118654752440084436210485
# Taylor coefficient
COEF = (1.0,
        0.16666666666666666666666666666667,
        0.075,
        0.04464285714285714285714285714286,
        0.03038194444444444444444444444444,
        0.02237215909090909090909090909091,
        0.01735276442307692307692307692308,
        0.01396484375)
# TAYLOR COUNT
TAYLOR_COUNT = 7
# negative min float16 value
NEG_MIN_FP16 = -2**(-24)
# min float16 * 2
TWO_MIN_FP16 = 2**(-23)


def _taylor_compute(data_x, x_square=None):
    """
    do arcsinx compute use the 15th order taylor expansion when
     0 <= x <= BOUNDARY_1
    asin(x) = x + 1/6*x^3 + 3/40*x^5 + 5/112*x^7 + ... + 13!!/(14!!*15)*x^15

    Parameters:
    ----------
    data_x : the placeholder of data input

    x_square : the placeholder of the square of data_x

    Returns : A Tensor. Has the same type as data.
    -------
    """

    if x_square is None:
        x_square = te.lang.cce.vmul(data_x, data_x)

    res = te.lang.cce.vmuls(x_square, tvm.const(COEF[TAYLOR_COUNT], "float32"))
    for temp in reversed(range(TAYLOR_COUNT)):
        res = te.lang.cce.vadds(res, tvm.const(COEF[temp], "float32"))
        if temp == 0:
            res = te.lang.cce.vmul(res, data_x)
        else:
            res = te.lang.cce.vmul(x_square, res)

    return res


# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("acos")
def acos_compute(x, y, kernel_name="acos"):
    """
    do element-wise acos compute using asin op
    acos(x) = HALF_PI - asin(x)

    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to (-1, -2^(-0.5))
              | the 15th order taylor expansion, x belongs to (-2^(-0.5),
              | 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)

    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "acos"

    Returns : A Tensor. Has the same type as x.
    -------
    """

    shape = x.shape
    dtype = x.dtype

    has_improve_precision = False
    # Change dtype to float32
    if dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        x = te.lang.cce.cast_to(x, "float32")
        boundary_mask1 = te.lang.cce.broadcast(
            tvm.const(BOUNDARY_1, "float32"), shape)
        has_improve_precision = True
    else:
        boundary_mask1 = te.lang.cce.broadcast(
            tvm.const(BOUNDARY_1, "float16"), shape)

    # to fix bug for input data is 1.0
    x = te.lang.cce.vadds(x, NEG_MIN_FP16)
    # Sign mask
    sign = util_compute.sign(x)

    # All positive
    x = te.lang.cce.vmul(x, sign)

    # x belongs to (0, 2^(-0.5))
    choice_1 = te.lang.cce.vmin(x, boundary_mask1)
    choice_1 = te.lang.cce.vsub(choice_1, boundary_mask1)
    choice_1 = te.lang.cce.vmuls(te.lang.cce.floor(choice_1), NEG_NUM_ONE)

    res_1 = _taylor_compute(x)
    res_1 = te.lang.cce.vmul(res_1, choice_1)

    # x belongs to (2^(-0.5), 1)
    choice_2 = te.lang.cce.vmuls(choice_1, tvm.const(NEG_NUM_ONE, x.dtype))
    choice_2 = te.lang.cce.vadds(choice_2, tvm.const(NUM_ONE, x.dtype))

    # to fix bug for input data is 1.0
    x = te.lang.cce.vadds(x, TWO_MIN_FP16)
    res_2 = te.lang.cce.vmul(x, x)
    res_2 = te.lang.cce.vmuls(res_2, tvm.const(NEG_NUM_ONE, x.dtype))
    res_2 = te.lang.cce.vadds(res_2, tvm.const(NUM_ONE, x.dtype))
    res_2_sqrt = te.lang.cce.vsqrt(res_2, 1)

    res_2 = _taylor_compute(res_2_sqrt, res_2)
    res_2 = te.lang.cce.vmuls(res_2, tvm.const(NEG_NUM_ONE, x.dtype))
    res_2 = te.lang.cce.vadds(res_2, tvm.const(HALF_PI, x.dtype))
    res_2 = te.lang.cce.vmul(res_2, choice_2)

    # Restore sign of asin
    res_1 = te.lang.cce.vadd(res_1, res_2)
    res_1 = te.lang.cce.vmul(res_1, sign)
    res_1 = te.lang.cce.vmuls(res_1, tvm.const(NEG_NUM_ONE, x.dtype))
    res_1 = te.lang.cce.vadds(res_1, tvm.const(HALF_PI, x.dtype))

    # Restore dtype
    if has_improve_precision:
        res_1 = te.lang.cce.cast_to(res_1, "float16")

    return res_1


@util.check_input_type(dict, dict, str)
def acos(x, y, kernel_name="acos"):
    """
    ----------
    acos(x) = HALF_PI - asin(x)

    Parameters:
    ----------
    x : the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "acos"

    Returns : None
    -------
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    util.check_kernel_name(kernel_name)
    check_shape(shape_input)
    shape_input, _ = refine_shape_axes(shape_input, [])

    check_list = ("float16", "float32")
    check_dtype(dtype_input, check_list)

    inp_dtype = dtype_input.lower()
    data_input = tvm.placeholder(shape_input, dtype=inp_dtype,
                                 name="data_input")

    res = acos_compute(data_input, y, kernel_name)

    with tvm.target.cce():
        auto_sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data_input, res),
              "bool_storage_as_1bit": False}

    te.lang.cce.cce_build_code(auto_sch, config)
