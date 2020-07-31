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

xlogy
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util


# define a scalar, value = -1
SCALAR_NEG_ONE = -1.0
# define a scalar, value = 1
SCALAR_ONE = 1.0
# define taylor negative threshold , value = -1.7
TAYLOR_NEGATIVE_THRESHOLD = -1.7
# define taylor positive threshold , value = 0.7
TAYLOR_POSITIVE_THRESHOLD = 0.7
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

# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("xlogy")
def xlogy_compute(input_x, input_y, output_z, kernel_name="xlogy"):
    """
    algorithm: xlogy
    calculating data's xlogy, res = 0 if x == 0 else x*log(y)
    in cloud scene, for all inputs :
    res = 0 if x == 0 else x*log(y)
    in mini scene :
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)
    f(z) = e^(z(n)*x(n)^-1)
    z(n)*x(n)^-1 <= TAYLOR_NEGATIVE_THRESHOLD or z(n)*x(n)^-1 >=
    TAYLOR_POSITIVE_THRESHOLD
    f(z) = seventh taylor computer
    TAYLOR_NEGATIVE_THRESHOLD < z(n)*x(n)^-1 < TAYLOR_POSITIVE_THRESHOLD

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict info of output_z
    kernel_name: str
        kernel name, default value is "xlogy"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_list = util.produce_shapes(
        te.lang.cce.util.shape_to_list(input_x.shape),
        te.lang.cce.util.shape_to_list(input_y.shape))
    util.check_tensor_shape_size(shape_list[2])

    shape = shape_list[2]
    dtype = input_x.dtype

    cloud_check = tbe_platform.cce_conf.api_check_support("te.lang.cce.vlog",
                                                          "float32")
    mini_check = tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul",
                                                         "float32")
    if dtype == "float16" and cloud_check:
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")

    data_x_broad = te.lang.cce.broadcast(input_x, shape_list[2])
    data_y_broad = te.lang.cce.broadcast(input_y, shape_list[2])
    data_log = te.lang.cce.vlog(data_y_broad)
    res = te.lang.cce.vmul(data_log, data_x_broad)

    if (not cloud_check) and mini_check:
        data_x_broad = te.lang.cce.cast_to(data_x_broad, "float32")
        data_x_broad = te.lang.cce.cast_to(data_x_broad, "float32")
        res = _xlogy_mini_compute(res, data_x_broad, data_y_broad, shape)

    if dtype == "float16" and (cloud_check or mini_check):
        res = te.lang.cce.cast_to(res, "float16")

    return res


def _xlogy_mini_compute(res_mini, input_x, input_y, shape):
    """
    do element-wise x*log(y) compute in mini scene
    f(z) = e^(z(n)*x(n)^-1)
    z(n)*x(n)^-1 <= TAYLOR_NEGATIVE_THRESHOLD or z(n)*x(n)^-1 >=
    TAYLOR_POSITIVE_THRESHOLD
    f(z) = seventh taylor computer
    TAYLOR_NEGATIVE_THRESHOLD < z(n)*x(n)^-1 < TAYLOR_POSITIVE_THRESHOLD

    Parameters:
    ----------
    mini_res: TVM tensor, the tensor of x*log(y)
    input_x : TVM tensor, the placeholder of input_x
    input_y : TVM tensor, the placeholder of input_y
    shape : tuple, the shape of mini_res

    Returns : A Tensor. Has the same type as mini_res.
    -------
    """
    input_z = te.lang.cce.cast_to(res_mini, "float32")
    input_x = te.lang.cce.cast_to(input_x, "float32")
    input_y = te.lang.cce.cast_to(input_y, "float32")
    input_x_rec = te.lang.cce.vrec(input_x)
    input_z_compare = te.lang.cce.vmul(input_z, input_x_rec)

    newton_taylor_res = _newton_taylor_xlogy(input_x, input_y, input_z)
    newton_exp_res = _newton_exp_xlogy(input_x, input_y, input_z)

    input_left_border = tvm.const(TAYLOR_NEGATIVE_THRESHOLD, "float32")
    tensor_input_left_border = te.lang.cce.broadcast(input_left_border, shape)

    input_right_border = tvm.const(TAYLOR_POSITIVE_THRESHOLD, "float32")
    tensor_input_right_border = te.lang.cce.broadcast(input_right_border, shape)

    b_gt_left_border = te.lang.cce.vcmp(input_z_compare,
                                        tensor_input_left_border, 'gt')
    exp_taylor_neg = te.lang.cce.vsel(b_gt_left_border,
                                      newton_taylor_res, newton_exp_res)

    b_lt_right_border = te.lang.cce.vcmp(input_z_compare,
                                         tensor_input_right_border, 'lt')
    data_xlogy = te.lang.cce.vsel(b_lt_right_border,
                                  exp_taylor_neg, newton_exp_res)

    return data_xlogy


def _exp_taylor_compute(input_x):
    """
    Calculate e^x, Use seventh order taylor expansion
    e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) +
    (x^5 / 5!) + (x^6 / 6!) + (x^7 / 7!)

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

    # calculate first order tayloy plus one section : 1 + x
    res_first_taylor = te.lang.cce.vadds(input_x, tvm.const(SCALAR_ONE,
                                                            "float32"))
    res_second_taylor = te.lang.cce.vadd(res_first_taylor, data_power_2_div_2)
    res_third_taylor = te.lang.cce.vadd(res_second_taylor, data_power_3_div_6)
    res_fourth_taylor = te.lang.cce.vadd(res_third_taylor, data_power_4_div_24)
    res_fifth_taylor = te.lang.cce.vadd(res_fourth_taylor, data_power_5_div_120)
    res_sixth_taylor = te.lang.cce.vadd(res_fifth_taylor, data_power_6_div_720)
    res = te.lang.cce.vadd(res_sixth_taylor, data_power_7_div_5040)

    return res


def _newton_exp_iter(input_x, input_y, input_z):
    """
    do element-wise Newton compute
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: TVM tensor, the placeholder of input_y
    input_z: start value of Newton iteration

    Returns : A Tensor. Has the same type as input_z.
    -------
    """
    #Newton begin:z(n+1) = z(n) - x(n) + x(n)*y(n)*e^(-z(n)*x(n)^-1)
    input_x_mul = te.lang.cce.vmuls(input_x, tvm.const(SCALAR_NEG_ONE,
                                                       "float32"))
    newton_exp = te.lang.cce.vadd(input_x_mul, input_z)
    input_xy = te.lang.cce.vmul(input_x, input_y)
    input_x_rec = te.lang.cce.vrec(input_x)
    input_x_res = te.lang.cce.vmuls(input_x_rec, tvm.const(SCALAR_NEG_ONE,
                                                           "float32"))
    input_z_mul = te.lang.cce.vmul(input_x_res, input_z)
    input_z_exp = te.lang.cce.vexp(input_z_mul)
    input_z_res = te.lang.cce.vmul(input_z_exp, input_xy)
    newton_exp = te.lang.cce.vadd(newton_exp, input_z_res)

    return newton_exp


def _newton_taylor_iter(input_x, input_y, input_z):
    """
    do element-wise Newton compute
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: TVM tensor, the placeholder of input_y
    input_z: start value of Newton iteration

    Returns : A Tensor. Has the same type as input_z.
    -------
    """
    #Newton begin:z(n+1) = z(n) - x(n) + x(n)*y(n)*e^(-z(n)*x(n)^-1)
    input_x_mul = te.lang.cce.vmuls(input_x, tvm.const(SCALAR_NEG_ONE,
                                                       "float32"))
    newton_taylor = te.lang.cce.vadd(input_x_mul, input_z)
    input_xy = te.lang.cce.vmul(input_x, input_y)
    input_x_rec = te.lang.cce.vrec(input_x)
    input_x_res = te.lang.cce.vmuls(input_x_rec, tvm.const(SCALAR_NEG_ONE,
                                                           "float32"))
    input_z_mul = te.lang.cce.vmul(input_x_res, input_z)
    input_z_taylor = _exp_taylor_compute(input_z_mul)
    input_z_res = te.lang.cce.vmul(input_z_taylor, input_xy)
    newton_taylor = te.lang.cce.vadd(newton_taylor, input_z_res)

    return newton_taylor


def _newton_exp_xlogy(input_x, input_y, output_z):
    """
    do element-wise Newton compute
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: TVM tensor, the placeholder of input_y
    output_z: TVM tensor, start value of xlogy's Newton iteration

    Returns : A Tensor. Has the same type as output_z.
    -------
    """
    for _ in range(2):
        output_z = _newton_exp_iter(input_x, input_y, output_z)
    return output_z


def _newton_taylor_xlogy(input_x, input_y, output_z):
    """
    do element-wise Newton compute
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: TVM tensor, the placeholder of input_y
    output_z: TVM tensor, start value of xlogy's Newton iteration

    Returns : A Tensor. Has the same type as output_z.
    -------
    """
    for _ in range(2):
        output_z = _newton_taylor_iter(input_x, input_y, output_z)
    return output_z


@util.check_input_type(dict, dict, dict, str)
def xlogy(input_x, input_y, output_z, kernel_name="xlogy"):
    """
    algorithm: xlogy
    calculating data's xlogy, res = 0 if x == 0 else x*log(y)

    Parameters
    ----------
    input_x: dict
        dict of input_x, include keys(shape and dtype)
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    output_z: dict
        dict info of output_z
    kernel_name: str
        kernel name, default value is "xlogy"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_x.get("dtype")
    dtype_y = input_y.get("dtype")

    util.compare_tensor_dict_key(input_x, input_y, "dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)

    input_dtype = dtype.lower()
    input_dtype_y = dtype_y.lower()
    check_list = ("float16", "float32")
    util.check_dtype_rule(input_dtype, check_list)
    util.check_dtype_rule(input_dtype_y, check_list)
    shape_list = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_list[2])

    shape_x, shape_y = refine_shapes_for_broadcast(shape_list[0],
                                                   shape_list[1])
    data1 = tvm.placeholder(shape_x, name="data1", dtype=input_dtype)
    data2 = tvm.placeholder(shape_y, name="data2", dtype=input_dtype)
    res = xlogy_compute(data1, data2, output_z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data1, data2, res],
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(sch, config)
