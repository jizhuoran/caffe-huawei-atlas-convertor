#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2018-2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

relu6_grad

"""
from functools import reduce as reduce_ins
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2**30  # shape limit


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("relu6_grad")
def relu6_grad_compute(input_x, input_grad, output_y, kernel_name="relu6_grad"):
    """
    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input_grad : TVM tensor
        the placeholder of input_grad
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6_grad"

    Returns
    ------
    compute result of relu6grad
    """
    # input_x<=6 and input_x>=0
    # min(input,6)
    min_positive_6 = te.lang.cce.vmins(input_x, 6)
    # max(input,0)
    max_zero_min_6 = te.lang.cce.vmaxs(min_positive_6, 0)

    # (X-6), X*(X-6)
    x_sub_6 = te.lang.cce.vadds(max_zero_min_6, -6)
    x_mul_x_6 = te.lang.cce.vmul(max_zero_min_6, x_sub_6)

    input_dtype = input_x.dtype
    if input_dtype == "float16":
        # algrithm : Y = X*(X-6)*1024/(X*(X-6)*1024+ESP_MIN)
        # for float16, add a small number which value is 1.18e-7, so that the
        # divisor is not equal to 0, and for accuracy, multiply by a number
        # which value is 1024.
        x_mul_x_6_big = te.lang.cce.vmuls(x_mul_x_6, 1024)
        y_add_espmin = te.lang.cce.vadds(x_mul_x_6_big, 1.18e-7)
        y_y_esp_min = te.lang.cce.vdiv(x_mul_x_6_big, y_add_espmin)
    if input_dtype == "float32":
        # algrithm : Y = X*(X-6)/(X*(X-6)+ESP_MIN)
        # for float32, add a small number which value is 1.18e-38, so that
        # the divisor is not equal to 0.
        y_add_espmin = te.lang.cce.vadds(x_mul_x_6, 1.18e-38)
        y_y_esp_min = te.lang.cce.vdiv(x_mul_x_6, y_add_espmin)

    final_res = te.lang.cce.vmul(y_y_esp_min, input_grad)

    return final_res


@util.check_input_type(dict, dict, dict, str)
# pylint: disable=locally-disabled,too-many-locals
def relu6_grad(input_grad, input_x, output_y, kernel_name="relu6_grad"):
    """
    Parameters
    ----------
    input_grad : dict
        shape and dtype of input_grad
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6_grad"

    Returns
    ------
    None
    """
    # check input shape
    shape_x = input_x.get("shape")
    shape_grad = input_grad.get("shape")
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_grad)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_grad, SHAPE_SIZE_LIMIT)
    if list(shape_x) != list(shape_grad):
        raise RuntimeError("input_grad and input_x must have the same shape.")

    # check input tensor data_type and kernel_name
    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    grad_dtype = input_grad.get("dtype").lower()
    util.check_dtype_rule(input_dtype, check_list)
    util.check_dtype_rule(grad_dtype, check_list)
    util.check_kernel_name(kernel_name)
    if input_dtype == "float32" and not tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmuls", "float32"):
        raise RuntimeError(
            "Input dtype only support float16 while input dtype is float32")

    shape_x = [reduce_ins(lambda x, y: x * y, shape_x[:])]
    input_data_orginal = tvm.placeholder(shape_x,
                                         name="input_data",
                                         dtype=input_dtype)
    input_grad = tvm.placeholder(shape_x,
                                 name="input_grad",
                                 dtype=grad_dtype)

    final_res = relu6_grad_compute(input_data_orginal,
                                   input_grad,
                                   output_y,
                                   kernel_name="relu6_grad")
    with tvm.target.cce():
        auto_sch = generic.auto_schedule(final_res)

    config = {
        "name": kernel_name,
        "tensor_list": (input_grad, input_data_orginal, final_res)
    }

    te.lang.cce.cce_build_code(auto_sch, config)
