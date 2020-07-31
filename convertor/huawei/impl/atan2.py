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

atan2

  Op_description :
    Computes arctangent of y/x element-wise, respecting signs of the arguments

    # atan2(
    #   x1,
    #   x2,
    #   y,
    #   kernel_name="atan2")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : 'y' and 'x' must have the same type and shape.
    [2] All : shape size limit is 2147483648.

"""

import operator

from impl.util import util_compute
from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import refine_shapes_for_broadcast
import topi
from topi.cce import util

CONST_POS_ONE = 1.0
CONST_NA_ONE = -1.0
CONST_PI = 3.1415926535897932384626433832795
CONST_PI_BY_TWO = 1.5707963267948966192313216916398
CONST_PI_BY_FOUR = 0.78539816339744830961566084581988
CONST_PI_BY_EIGHT = 0.39269908169872415480783042290994
CONST_ITERTOR = 6
CONST_ITERTOR2 = 4
TAN_PI_BY_EIGHT = 0.4142135623730950
TAN_PI_BY_EIGHT_NA = -0.4142135623730950

CONST_ZERO = 0
CONST_ONE = 1

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

        input_data: Input data.

    ----------------------------------
    Returns:

        A Tensor of atan(x).

    """

    denominator_data = te.lang.cce.vmuls(input_data, TAN_PI_BY_EIGHT)
    denominator_data = te.lang.cce.vadds(denominator_data, CONST_POS_ONE)
    molecule = te.lang.cce.vadds(input_data, TAN_PI_BY_EIGHT_NA)
    data = te.lang.cce.vdiv(molecule, denominator_data)
    data = te.lang.cce.vabs(data)

    square_data = te.lang.cce.vmul(data, data)
    res = te.lang.cce.vmuls(square_data, TAYLOR[CONST_ITERTOR])
    res = te.lang.cce.vadds(res, TAYLOR[CONST_ITERTOR - 1])
    for i in reversed(range(CONST_ITERTOR - 1)):
        res = te.lang.cce.vmul(res, square_data)
        res = te.lang.cce.vadds(res, TAYLOR[i])
    res = te.lang.cce.vmul(res, data)
    res = te.lang.cce.vadds(res, CONST_PI_BY_EIGHT)

    square_data = te.lang.cce.vmul(input_data, input_data)
    res2 = te.lang.cce.vmuls(square_data, TAYLOR[CONST_ITERTOR2])
    res2 = te.lang.cce.vadds(res2, TAYLOR[CONST_ITERTOR2 - 1])
    for i in reversed(range(CONST_ITERTOR2 - 1)):
        res2 = te.lang.cce.vmul(res2, square_data)
        res2 = te.lang.cce.vadds(res2, TAYLOR[i])
    res2 = te.lang.cce.vmul(res2, input_data)

    res = te.lang.cce.vmin(res, res2)

    return res


def _atan_compute(input_x):
    """
    Algorithm: atan

    ----------------------------------
    Parameters:

        input_x: Input data.

    ----------------------------------
    Returns:

        A Tensor of atan(x).

    """

    shape = input_x.shape
    dtype = input_x.dtype
    if dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
    abs_data = te.lang.cce.vabs(input_x)

    tensor_one = te.lang.cce.broadcast(tvm.const(CONST_POS_ONE, input_x.dtype),
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

    sign_mask = util_compute.sign(input_x)
    res = te.lang.cce.vmul(res, sign_mask)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")
    return res


def _init_atan2_mask(data_y, data_x):
    """
    Algorithm: atan2

    ----------------------------------
    Parameters:

        data_y: the y of atan2(y, x)

        data_x: the x of atan2(y, x)
    ----------------------------------
    Returns:

        mask: the mask of x's and y's value

    """

    shape_input = data_y.shape
    dtype_input = data_y.dtype

    tensor_one = te.lang.cce.broadcast(tvm.const(CONST_POS_ONE, dtype_input),
                                       shape_input)
    tensor_zero = te.lang.cce.broadcast(tvm.const(CONST_ZERO, dtype_input),
                                        shape_input)
    tensor_na_one = te.lang.cce.vmuls(tensor_one,
                                      tvm.const(CONST_NA_ONE, dtype_input))

    y_me_zero = te.lang.cce.vsel(te.lang.cce.vcmp(data_y, tensor_zero, 'ge'),
                                 tensor_one, tensor_na_one)
    x_lt_zero_y_mask = te.lang.cce.vsel(
        te.lang.cce.vcmp(data_x, tensor_zero, 'lt'), y_me_zero, tensor_zero)

    y_cmp_zero = te.lang.cce.vsel(te.lang.cce.vcmp(data_y, tensor_zero, 'ge'),
                                  tensor_one, tensor_na_one)

    mask = (x_lt_zero_y_mask, y_cmp_zero)
    return mask


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("atan2")
def atan2_compute(y, x, output_dict, kernel_name="atan2"):
    """
    Algorithm: atan2
    ----------------------------------
    Parameters:

        y: Input data y.

        x: Input data x.

        kernel_name: cce kernel name, default value is "atan2"
    ----------------------------------
    Returns:

        A Tensor of atan2(x).

    """

    shape_y = y.shape
    dtype_y = y.dtype
    shape_x = x.shape

    shape_y = te.lang.cce.util.shape_to_list(shape_y)
    shape_x = te.lang.cce.util.shape_to_list(shape_x)
    shape_y, shape_x, shape_broadcast = broadcast_shapes(shape_y, shape_x)
    y = te.lang.cce.broadcast(y, shape_broadcast)
    x = te.lang.cce.broadcast(x, shape_broadcast)

    if dtype_y == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        y = te.lang.cce.cast_to(y, "float32")
        x = te.lang.cce.cast_to(x, "float32")

    tensor_zero = te.lang.cce.broadcast(tvm.const(CONST_ZERO, "float32"),
                                        shape_broadcast)
    mask = _init_atan2_mask(y, x)

    # caculate the atan(y/x) when x > 0
    res = te.lang.cce.vdiv(y, x)
    res = _atan_compute(res)

    x_equal_zero = te.lang.cce.vcmp(x, tensor_zero, 'eq')
    y_cmp_zero = te.lang.cce.vmuls(mask[CONST_ONE],
                                   tvm.const(CONST_PI_BY_TWO, y.dtype))
    res_x_lt_zero = te.lang.cce.vmuls(mask[CONST_ZERO],
                                      tvm.const(CONST_PI, y.dtype))

    res = te.lang.cce.vsel(x_equal_zero, y_cmp_zero, res)

    res = te.lang.cce.vadd(res, res_x_lt_zero)

    if dtype_y == "float16":
        res = te.lang.cce.cast_to(res, "float16")
    return res

@util.check_input_type(dict, dict, dict, str)
def atan2(x1, x2, y, kernel_name="atan2"):
    """
    Algorithm: arctan2
        arctan2(y, x) = arctan(y/x)
    ----------------------------------
    Parameters:

        x1: the dict of input data x1, only support float16, float32.

        x2: the dict of input data x2, only support float16, float32.

        y: the dict of output

        kernel_name: default value is "atan2".
    ----------------------------------
    Returns:
        None
    """

    y_shape = x1.get("shape")
    x_shape = x2.get("shape")

    y_dtype = x1.get("dtype")
    x_dtype = x2.get("dtype")
    util.check_kernel_name(kernel_name)

    check_shape(y_shape)
    check_shape(x_shape)

    shape_y, shape_x, shape_max = broadcast_shapes(
        y_shape, x_shape)

    check_shape(shape_max)
    check_list = ("float16", "float32")
    check_dtype(y_dtype, check_list)
    check_dtype(x_dtype, check_list)
    if y_dtype.lower() != x_dtype.lower():
        raise RuntimeError("The input tensor must have identical dtype!")
    shape_y, shape_x = refine_shapes_for_broadcast(shape_y, shape_x)
    input_y = tvm.placeholder(shape_y, dtype=y_dtype.lower(), name="input_y")
    input_x = tvm.placeholder(shape_x, dtype=x_dtype.lower(), name="input_x")

    res = atan2_compute(input_y, input_x, y, kernel_name)
    res = te.lang.cce.cast_to(res, x_dtype.lower())
    with tvm.target.cce():
        auto_sch = topi.generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": (input_y, input_x, res),
        "print_ir": False,
        "bool_storage_as_1bit": False
    }

    te.lang.cce.cce_build_code(auto_sch, config)
