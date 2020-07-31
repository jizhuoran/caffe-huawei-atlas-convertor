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

sigmoid_grad
"""
import operator
from functools import reduce as reduce_ins
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
import te.lang.cce
import topi
from topi.cce import util
# General limitation of the reduce size for input shape: 2**30
SHAPE_SIZE_LIMIT = 2**30

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("sigmoid_grad")
def sigmoid_grad_compute(x, y, z, kernel_name="sigmoid_grad"):
    """
    algorithm : sigmoid grad compute

    sigmoid_grad = (sigmoid - sigmoid*sigmoid)*grad

    Parameters:
    ----------
    x : a tensor of input data

    y : a tensor of grad

    z : output dict

    kernel_name : cce kernel name, default value is "sigmoid_grad"

    Returns
    -------
    a tenosr
    """
    dtype = x.dtype.lower()
    cast_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.cast_to", "f322f16")
    if dtype == "float32" and not cast_support:
        raise RuntimeError(
            "float32 transfer to float16 is only supported on mini and cloud platform")
    vmul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    vsub_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vsub", "float32")
    if dtype == "float16":
        x = te.lang.cce.cast_to(x, "float32")
        y = te.lang.cce.cast_to(y, "float32")
    sigmoid_square = te.lang.cce.vmul(x, x)
    if dtype == "float32" and not vmul_support:
        sigmoid_square = te.lang.cce.cast_to(sigmoid_square, "float16")
    tensor_sub = te.lang.cce.vsub(x, sigmoid_square)
    if dtype == "float32" and not vsub_support:
        tensor_sub = te.lang.cce.cast_to(tensor_sub, "float16")
    res = te.lang.cce.vmul(tensor_sub, y)
    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")
    return res


@util.check_input_type(dict, dict, dict, str)
def sigmoid_grad(x,
                 y,
                 z,
                 kernel_name="sigmoid_grad"):
    """
    do sigmoid grad

    sigmoid_grad = (sigmoid - sigmoid*sigmoid)*grad

    Parameters:
    ----------
    x : dictionary shape of sigmoid input

    y : dictionary shape of grad

    z: dictionary output

    kernel_name : cce kernel name, default value is "sigmoid_grad_cce"

    Returns
    -------
    None
    """
    shape_sig = x.get("shape")
    shape_d = y.get("shape")
    dtype = x.get("dtype")
    dtype_y = y.get("dtype")
    util.check_kernel_name(kernel_name)
    if dtype != dtype_y:
        raise RuntimeError("Input dtype must be equal")
    if not operator.eq(list(shape_sig), list(shape_d)):
        raise RuntimeError("Input shapes must be equal")
    util.check_shape_rule(shape_sig)
    util.check_shape_size(shape_sig, SHAPE_SIZE_LIMIT)
    input_dtype = dtype.lower()
    util.check_dtype_rule(input_dtype, ["float16", "float32"])

    shape_sig = [reduce_ins(lambda x, y: x * y, shape_sig[:])]
    input_sigmoid = tvm.placeholder(shape_sig, name="input_sigmoid",
                                    dtype=input_dtype)
    input_grad = tvm.placeholder(shape_sig, name="input_grad",
                                 dtype=input_dtype)

    with tvm.target.cce():
        res = sigmoid_grad_compute(input_sigmoid, input_grad, z, kernel_name)
        auto_sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_sigmoid, input_grad, res]}

    te.lang.cce.cce_build_code(auto_sch, config)
