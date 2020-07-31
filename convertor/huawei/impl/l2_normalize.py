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

l2_normalize
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.lang.cce.te_compute.util import shape_to_list
from topi import generic
from topi.cce import util
from te import platform as tbe_platform


# pylint: disable=unused-argument
@fusion_manager.register("l2_normalize")
def l2_normalize_compute(input_x,
                         output_y,
                         axis,
                         epsilon,
                         kernel_name="l2_normalize"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    axis : list
        the axis which to be computed
    epsilon : float
        the minimum value, in case the denominator is zero
    kernel_name : str
        kernel name, default value is "l2_normalize"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
    x_square = te.lang.cce.vmul(input_x, input_x)
    x_square_sum = te.lang.cce.sum(x_square, axis, keepdims=True)
    const_epsilon = tvm.const(epsilon, "float32")
    x_l2norm = te.lang.cce.vmaxs(x_square_sum, const_epsilon)
    x_l2norm_sqrt = te.lang.cce.vsqrt(x_l2norm)
    x_l2norm_sqrt = te.lang.cce.broadcast(x_l2norm_sqrt,
                                          shape_to_list(input_x.shape))

    result = te.lang.cce.vdiv(input_x, x_l2norm_sqrt)

    if dtype == "float16":
        result = te.lang.cce.cast_to(result, "float16")
    return result


@util.check_input_type(dict, dict, (list, tuple), float, str)
def l2_normalize(input_x, output_y, axis, epsilon, kernel_name="l2_normalize"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    axis : list
        the axis which to be computed
    epsilon : float
        the minimum value, in case the denominator is zero
    kernel_name : str
        kernel name, default value is "l2_normalize"

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
    util.check_dtype_rule(input_dtype, ("float16", "float32"))

    if len(axis) != len(list(set(axis))):
        raise RuntimeError("the axis elements are duplicated")

    for i in axis:
        if not isinstance(i, int):
            raise RuntimeError("the axis element must be int")
        if i >= len(shape) or i < -len(shape):
            raise RuntimeError("the axis is invalid")

    if epsilon < 0.:
        raise RuntimeError("epsilon must be greater than 0")

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = l2_normalize_compute(data_input, output_y,
                               axis, epsilon, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(sch, config)

