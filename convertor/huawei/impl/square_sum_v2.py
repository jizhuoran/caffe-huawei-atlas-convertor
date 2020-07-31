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

square_sum_v2
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform

MIN_FP32 = 2**(-126)
# min float16 value
MIN_FP16 = 2**(-24)
VALUE_ONE = 1

SHAPE_SIZE_LIMIT = 200000000


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
def reduce_sum_d_compute(x,
                         y,
                         axis=None,
                         keepdims=None,
                         kernel_name="reduce_sum_d"):
    """redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    dtype = x.dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        x = te.lang.cce.cast_to(x, "float32")
    res_sum = te.lang.cce.sum(x, axis=axis, keepdims=keepdims)
    res = te.lang.cce.cast_to(res_sum, dtype)

    return res


def square_compute(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    res : tvm.tensor
        the result of square
    """
    res = te.lang.cce.vmul(input_x, input_x)
    return res


@fusion_manager.register("square_sum_v2")
def suqare_sum_v2_compute(input_x,
                          output1,
                          output2,
                          attr1,
                          attr2,
                          kernel_name="square_sum_v2"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """
    shape = te.lang.cce.util.shape_to_list(input_x.shape)
    axis_d = []
    if not attr1:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    else:
        axis_d = attr1
    square = square_compute(input_x, {}, kernel_name)

    sum0 = reduce_sum_d_compute(square, {},
                                axis_d,
                                keepdims=attr2,
                                kernel_name=kernel_name)

    return [sum0, square]


def square_sum_v2(input_x,
                  output1,
                  output2,
                  attr1,
                  attr2=True,
                  kernel_name="square_sum_v2"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")

    input_dtype = dtype.lower()

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)

    res = suqare_sum_v2_compute(data_input, output1, output2, attr1, attr2,
                                kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input] + list(res)}

    te.lang.cce.cce_build_code(sch, config)
