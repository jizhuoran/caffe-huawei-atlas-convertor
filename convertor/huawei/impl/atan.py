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

atan

  Op_description :
    Computes the trignometric inverse tangent of x element-wise

    # atan(
    #   x,
    #   y,
    #   kernel_name="cce_atan")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.

"""

from impl.util import util_compute
from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
import topi
from topi.cce import util

CONST_POS_ONE = 1.0
CONST_PI_BY_FOUR = 0.78539816339744830961566084581988
CONST_PI_BY_EIGHT = 0.39269908169872415480783042290994
CONST_ITERTOR = 6
CONST_ITERTOR2 = 4
TAN_PI_BY_EIGHT = 0.4142135623730950

TAYLOR = (1.0, -1.0 / 3, 1.0 / 5, -1.0 / 7, 1.0 / 9, -1.0 / 11, 1.0 / 13)


def _do_taylor(input_data):
    """
    Algorithm:
        if x > 0 and x < tan(pi/8):
            atan(x) = x - x^3/3 + x^5/5 - x^7/7 ...
        elif x > tan(pi/8) and x < tan(pi/4):
            atan(x) = atan(y) + atan((x-y)/(1+xy))

    ----------------------------------
    Parameters:

    input_data: Input data

    ----------------------------------
    Returns:

        A Tensor of atan(x).

    """

    shape_input = input_data.shape
    dtype_input = input_data.dtype

    tensor_offset = te.lang.cce.broadcast(tvm.const(TAN_PI_BY_EIGHT, dtype_input), shape_input)
    denominator_data = te.lang.cce.vmuls(input_data, TAN_PI_BY_EIGHT)
    denominator_data = te.lang.cce.vadds(denominator_data, CONST_POS_ONE)
    molecule = te.lang.cce.vsub(input_data, tensor_offset)
    data = te.lang.cce.vdiv(molecule, denominator_data)
    data = te.lang.cce.vabs(data)

    square_data = te.lang.cce.vmul(data, data)
    res = te.lang.cce.broadcast(tvm.const(TAYLOR[CONST_ITERTOR], dtype_input), shape_input)
    for i in reversed(range(CONST_ITERTOR)):
        res = te.lang.cce.vmul(res, square_data)
        res = te.lang.cce.vadds(res, TAYLOR[i])
    res = te.lang.cce.vmul(res, data)
    res = te.lang.cce.vadds(res, CONST_PI_BY_EIGHT)

    square_data = te.lang.cce.vmul(input_data, input_data)
    res2 = te.lang.cce.broadcast(tvm.const(TAYLOR[CONST_ITERTOR2], dtype_input), shape_input)
    for i in reversed(range(CONST_ITERTOR2)):
        res2 = te.lang.cce.vmul(res2, square_data)
        res2 = te.lang.cce.vadds(res2, TAYLOR[i])
    res2 = te.lang.cce.vmul(res2, input_data)

    res = te.lang.cce.vmin(res, res2)

    return res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("atan")
def atan_compute(x, y, kernel_name="atan"):
    """
    Algorithm: atan

    ----------------------------------
    Parameters:

    x: Input data

    y : the dict of output

    kernel_name: cce kernel name, default value is "atan"

    ----------------------------------
    Returns:

        A Tensor of atan(x).

    """

    dtype = x.dtype
    shape = x.shape

    if dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        x = te.lang.cce.cast_to(x, "float32")
    abs_data = te.lang.cce.vabs(x)

    tensor_one = te.lang.cce.broadcast(tvm.const(CONST_POS_ONE, x.dtype),
                                       shape)

    abs_data_sub_one = te.lang.cce.vsub(abs_data, tensor_one)
    abs_data_add_one = te.lang.cce.vadd(abs_data, tensor_one)
    abs_data2 = te.lang.cce.vdiv(abs_data_sub_one, abs_data_add_one)
    abs_data2 = te.lang.cce.vabs(abs_data2)

    # calucate data less than one
    res = _do_taylor(abs_data)
    # calucate data more than one
    res_mt_one = _do_taylor(abs_data2)
    res_mt_one = te.lang.cce.vadds(res_mt_one, CONST_PI_BY_FOUR)

    res = te.lang.cce.vmin(res, res_mt_one)

    sign_mask = util_compute.sign(x)
    res = te.lang.cce.vmul(res, sign_mask)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")
    return res


@util.check_input_type(dict, dict, str)
def atan(x, y, kernel_name="atan"):
    """
    Algorithm: atan

    ----------------------------------
    Parameters:

    x: the dict of input data, only support float16, float32.

    y: the dict of output

    kernel_name: cce kernel name, default value is "atan".

    ----------------------------------
    Returns:

        None

    """
    shape = x.get("shape")
    dtype = x.get("dtype")

    util.check_kernel_name(kernel_name)
    check_shape(shape)
    shape, _ = refine_shape_axes(shape, [])

    check_list = ("float16", "float32")
    check_dtype(dtype, check_list)

    with tvm.target.cce():
        dtype = dtype.lower()
        input_data = tvm.placeholder(shape, dtype=dtype, name="input_data")
        res = atan_compute(input_data, y, kernel_name)
        res = te.lang.cce.cast_to(res, dtype)
        auto_sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (input_data, res)}

    te.lang.cce.cce_build_code(auto_sch, config)
