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

xdivy_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
SHAPE_SIZE_LIMIT = 2**30


# pylint: disable=invalid-name,too-many-locals
# pylint: disable=too-many-arguments,unused-argument
def _broadcast_gradient_args(x, y):
    """
    Return the reduction indices for computing gradients of
    x op y with broadcast.

    Parameters
    ----------
    x : the shape of data input
    y : the shape of data input

    Returns
    -------
    rx : the reduction indices for computing gradients of x
    ry : the reduction indices for computing gradients of y
    """
    rx = []
    ry = []
    for i, item in enumerate(x):
        if item < y[i]:
            rx.append(i)
        elif item > y[i]:
            ry.append(i)

    return rx, ry


@fusion_manager.register("xdivy_grad")
def xdivy_grad_compute(placeholders, shape_max, dtype, rx, ry):
    """
    do element-wise xdivy_grad compute

    Parameters
    ----------
    placeholders : the placeholder of data input
    shape_max : the shape of broadcast
    dtype : the type of data input
    rx : the reduction indices of data input with broadcast
    ry : the reduction indices for data input with broadcast

    Returns
    -------
    output_y1 : result of xdivy_grad
    output_y2 : result of xdivy_grad
    None
    """
    x1_ori = placeholders[0]
    x2_ori = placeholders[1]
    grad_ori = placeholders[2]

    fp32_support = tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32")
    if dtype == "float32" and not fp32_support:
        raise RuntimeError("Don't support float32 in the platform.")

    if dtype == "float16" and fp32_support:
        x1 = te.lang.cce.cast_to(x1_ori, "float32")
        x2 = te.lang.cce.cast_to(x2_ori, "float32")
        grad = te.lang.cce.cast_to(grad_ori, "float32")
        x1 = te.lang.cce.broadcast(x1, shape_max)
        x2 = te.lang.cce.broadcast(x2, shape_max)
        grad = te.lang.cce.broadcast(grad, shape_max)
    else:
        x1 = te.lang.cce.broadcast(x1_ori, shape_max)
        x2 = te.lang.cce.broadcast(x2_ori, shape_max)
        grad = te.lang.cce.broadcast(grad_ori, shape_max)

    if dtype == "float16" and not fp32_support:
        esp_min = tvm.const(1.18e-7, dtype="float16")
        neg_one = tvm.const(-1, dtype="float16")
    else:
        esp_min = tvm.const(1.18e-38, dtype="float32")
        neg_one = tvm.const(-1, dtype="float32")
    x1_addespmin = te.lang.cce.vadds(x1, esp_min)
    not_zero_x1 = te.lang.cce.vdiv(x1, x1_addespmin)
    partial_x1 = te.lang.cce.vdiv(not_zero_x1, x2)
    partial_x1g = te.lang.cce.vmul(partial_x1, grad)

    neg_x1 = te.lang.cce.vmuls(x1, neg_one)
    partial_x1pow = te.lang.cce.vmul(partial_x1, partial_x1)
    partial_x2 = te.lang.cce.vmul(neg_x1, partial_x1pow)
    partial_x2g = te.lang.cce.vmul(partial_x2, grad)

    output_y1 = te.lang.cce.sum(partial_x1g, rx, keepdims=True)
    output_y2 = te.lang.cce.sum(partial_x2g, ry, keepdims=True)

    if dtype == "float16" and fp32_support:
        output_y1 = te.lang.cce.cast_to(output_y1, "float16")
        output_y2 = te.lang.cce.cast_to(output_y2, "float16")

    return output_y1, output_y2


@util.check_input_type(dict, dict, dict, dict, dict, str)
def xdivy_grad(x1, x2, grad, y1, y2, kernel_name="xdivy_grad"):
    """
    Returns gradient of xdivy(x, y) with respect to x and y.

    Parameters
    ----------
    x1 : dict
        shape and dtype of input, only support float16, float32
    x2 : dict
        shape and dtype of input, only support float16, float32
    grad : dict
        shape and dtype of input, only support float16, float32
    y1 : dict
        shape and dtype of output, should be same shape and type as input
    y2 : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "xdivygrad"

    Returns
    -------
    None
    """
    shape_x1 = x1.get("shape")
    dtype_x1 = x1.get("dtype").lower()
    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype").lower()
    shape_grad = grad.get("shape")
    dtype_grad = grad.get("dtype").lower()
    if dtype_x1 != dtype_x2 or dtype_x2 != dtype_grad or dtype_grad != dtype_x1:
        raise RuntimeError(
            "the type of x1, x2 and grad must be the same.")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x1)
    util.check_shape_rule(shape_x2)
    util.check_shape_rule(shape_grad)
    util.check_shape_size(shape_x1, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_x2, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_grad, SHAPE_SIZE_LIMIT)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype_x1, check_list)
    shape_x1, shape_x2, shape_max_x1x2 = util.produce_shapes(shape_x1, shape_x2)
    if len(shape_max_x1x2) < len(shape_grad):
        raise RuntimeError(
            "the length of shape_grad can not be longer than the maximum "
            "length of x1 and x2.")

    shape_grad, _, shape_max = util.produce_shapes(shape_grad, shape_max_x1x2)

    for (x, y) in zip(shape_max_x1x2, shape_grad):
        if x < y:
            raise RuntimeError("this shape is not supported.")

    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    rx, ry = _broadcast_gradient_args(shape_x1, shape_x2)

    x1 = tvm.placeholder(shape_x1, name="x", dtype=dtype_x1)
    x2 = tvm.placeholder(shape_x2, name="y", dtype=dtype_x1)
    grad = tvm.placeholder(shape_grad, name="grad", dtype=dtype_x1)

    output_y1, output_y2 = xdivy_grad_compute([x1, x2, grad], shape_max,
                                              dtype_x1, rx, ry)

    with tvm.target.cce():
        sch = generic.auto_schedule([output_y1, output_y2])

    config = {"name": kernel_name,
              "tensor_list": [x1, x2, grad, output_y1, output_y2]}
    te.lang.cce.cce_build_code(sch, config)
