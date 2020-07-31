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

expm1
"""
from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

# define a scalar , value = -1
SCALAR_NEGATIVE_ONE = -1.0
# define taylor negative threshold , value = -0.7
TAYLOR_NEGATIVE_THRESHOLD = -0.7
# define taylor positive threshold , value = 1.7
TAYLOR_POSITIVE_THRESHOLD = 1.7
# define second order parameter , value = 1 / 2.0
TAYLOR_SECOND_ORDER_PARAM = 1 / 2.0
# define third order parameter , value = 1 / 6.0
TAYLOR_THIRD_ORDER_PARAM = 1 / 6.0
# define fourth order parameter , value = 1 / 24.0
TAYLOR_FOURTH_ORDER_PARAM = 1 / 24.0
# define fifth order parameter , value = 1 / 120.0
TAYLOR_FIFTH_ORDER_PARAM = 1 / 120.0
# define sixth order parameter , value = 1 / 720.0
TAYLOR_SIXTH_ORDER_PARAM = 1 / 720.0
# define seventh order parameter , value = 1 / 5040.0
TAYLOR_SEVENTH_ORDER_PARAM = 1 / 5040.0


# pylint: disable=locally-disabled,too-many-locals
def _expm1_taylor_compute(input_x):
    """
    Calculate e^x - 1, Use seventh order taylor expansion
    e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!) + \
          (x^6 / 6!) + (x^7 / 7!)
    e^x - 1 = x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!) + \
            (x^6 / 6!) + (x^7 / 7!)

    Parameters:
    ----------
    input_x : TVM tensor, the placeholder of input_x

    Returns : A Tensor. Has the same type as input_x.
    -------
    """
    # calculate second order tayloy section : x^2 / 2!
    taylor_second_order_param = tvm.const(TAYLOR_SECOND_ORDER_PARAM, "float32")
    data_power_2 = te.lang.cce.vmul(input_x, input_x)
    data_power_2_div_2 = te.lang.cce.vmuls(data_power_2,
                                           taylor_second_order_param)

    # calculate third order tayloy section : x^3 / 3!
    taylor_third_order_param = tvm.const(TAYLOR_THIRD_ORDER_PARAM, "float32")
    data_power_3 = te.lang.cce.vmul(data_power_2, input_x)
    data_power_3_div_6 = te.lang.cce.vmuls(data_power_3,
                                           taylor_third_order_param)

    # calculate fourth order tayloy section : x^4 / 4!
    taylor_fourth_order_param = tvm.const(TAYLOR_FOURTH_ORDER_PARAM, "float32")
    data_power_4 = te.lang.cce.vmul(data_power_3, input_x)
    data_power_4_div_24 = te.lang.cce.vmuls(data_power_4,
                                            taylor_fourth_order_param)

    # calculate fifth order tayloy section : x^5 / 5!
    taylor_fifth_order_param = tvm.const(TAYLOR_FIFTH_ORDER_PARAM, "float32")
    data_power_5 = te.lang.cce.vmul(data_power_4, input_x)
    data_power_5_div_120 = te.lang.cce.vmuls(data_power_5,
                                             taylor_fifth_order_param)

    # xcalculate sixth order tayloy section : ^6 / 6!
    taylor_sixth_order_param = tvm.const(TAYLOR_SIXTH_ORDER_PARAM, "float32")
    data_power_6 = te.lang.cce.vmul(data_power_5, input_x)
    data_power_6_div_720 = te.lang.cce.vmuls(data_power_6,
                                             taylor_sixth_order_param)

    # calculate seventh order tayloy section : x^7 / 7!
    taylor_seventh_order_param = tvm.const(TAYLOR_SEVENTH_ORDER_PARAM,
                                           "float32")
    data_power_7 = te.lang.cce.vmul(data_power_6, input_x)
    data_power_7_div_5040 = te.lang.cce.vmuls(data_power_7,
                                              taylor_seventh_order_param)

    res_second_taylor = te.lang.cce.vadd(input_x, data_power_2_div_2)
    res_third_taylor = te.lang.cce.vadd(res_second_taylor, data_power_3_div_6)
    res_fourth_taylor = te.lang.cce.vadd(res_third_taylor, data_power_4_div_24)
    res_fifth_taylor = te.lang.cce.vadd(res_fourth_taylor, data_power_5_div_120)
    res_sixth_taylor = te.lang.cce.vadd(res_fifth_taylor, data_power_6_div_720)
    res = te.lang.cce.vadd(res_sixth_taylor, data_power_7_div_5040)

    return res


def _expm1_mini_compute(mini_res, input_x, shape):
    """
    do element-wise e^x - 1 compute in mini scene
    f(x) = e^x - 1,
    x <= TAYLOR_NEGATIVE_THRESHOLD or x >= TAYLOR_POSITIVE_THRESHOLD
    f(x) = seventh taylor computer,
    TAYLOR_NEGATIVE_THRESHOLD < x < TAYLOR_POSITIVE_THRESHOLD

    Parameters:
    ----------
    mini_res: TVM tensor, the tensor of e^x - 1

    input_x : TVM tensor, the placeholder of input data

    shape : tuple, the shape of input data

    Returns : A Tensor. Has the same type as mini_res.
    -------
    """
    taylor_res = _expm1_taylor_compute(input_x)

    input_right_border = tvm.const(TAYLOR_POSITIVE_THRESHOLD, "float32")
    tensor_input_right_border = te.lang.cce.broadcast(input_right_border, shape)

    input_left_border = tvm.const(TAYLOR_NEGATIVE_THRESHOLD, "float32")
    tensor_input_left_border = te.lang.cce.broadcast(input_left_border, shape)

    b_gt_left_border = te.lang.cce.vcmp(input_x, tensor_input_left_border, 'gt')
    exp_taylor_neg = te.lang.cce.vsel(b_gt_left_border, taylor_res, mini_res)

    b_lt_right_border = te.lang.cce.vcmp(input_x, tensor_input_right_border,
                                         'lt')
    mini_res = te.lang.cce.vsel(b_lt_right_border, exp_taylor_neg, mini_res)

    return mini_res

# pylint: disable=locally-disabled,too-many-locals,unused-argument
@fusion_manager.register("expm1")
def expm1_compute(input_x, output_y, kernel_name="expm1"):
    """
    algorithm: expm1
    calculating data's expm1, y = (e^x) - 1
    in cloud scene, for all inputs :
    f(x) = e^x - 1,
    in mini scene :
    f(x) = e^x - 1,
    x <= TAYLOR_NEGATIVE_THRESHOLD or x >= TAYLOR_POSITIVE_THRESHOLD
    f(x) = seventh taylor computer,
    TAYLOR_NEGATIVE_THRESHOLD < x < TAYLOR_POSITIVE_THRESHOLD

    Parameters
    ----------
    input_x : TVM tensor, the placeholders of input data
    output_y : dict, shape and dtype of output, should be same shape
               and type as input
    kernel_name : str, kernel name, default value is "expm1"

    Returns
    -------
    res : the result of compute
    """
    dtype = input_x.dtype
    shape = input_x.shape
    flag_cloud = tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                         "float32")
    flag_mini = tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd",
                                                        "float32")
    if dtype.lower() == "float16" and flag_cloud:
        input_x = te.lang.cce.cast_to(input_x, "float32")

    scalar_negative_one = tvm.const(SCALAR_NEGATIVE_ONE, "float32")
    exp_res = te.lang.cce.vexp(input_x)
    res = te.lang.cce.vadds(exp_res, scalar_negative_one)

    if (not flag_cloud) and flag_mini:
        input_x = te.lang.cce.cast_to(input_x, "float32")
        res = _expm1_mini_compute(res, input_x, shape)

    if dtype.lower() == "float16" and (flag_cloud or flag_mini):
        res = te.lang.cce.cast_to(res, dtype)

    return res


@util.check_input_type(dict, dict, str)
def expm1(input_x, output_y, kernel_name="expm1"):
    """
    algorithm: expm1
    calculating data's expm1, y = (e^x) - 1

    Parameters
    ----------
    input_x : dict,shape and dtype of input, only support float16, float32
    output_y: dict,shape and dtype of output, should be same shape
              and type as input
    kernel_name : str, kernel name, default value is "expm1"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")

    util.check_tensor_shape_size(shape)
    util.check_shape_rule(shape)
    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    util.check_dtype_rule(input_dtype, check_list)

    shape_x = (functools_reduce(lambda x, y: x * y, shape[:]),)
    data_input = tvm.placeholder(shape_x, name="data_input", dtype=input_dtype)

    res = expm1_compute(data_input, output_y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res],
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(sch, config)
