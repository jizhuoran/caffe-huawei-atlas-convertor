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

sqrt_grad
"""

import operator
from functools import reduce as reduce_ins

import te.lang.cce
from te import platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# General limitation of the reduce size for input shape: 2**30
SHAPE_LIMIT = 1 << 30


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("sqrt_grad")
def sqrt_grad_compute(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_compute
    output_grad = input_grad/(2*input)

    Parameters
    ----------
    x: a tensor of input data

    dx : a tensor of grad

    Returns
    -------
    output data

    """

    dtype = x.dtype.lower()
    mul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        raise RuntimeError(
            "Input dtype only support float16 while input dtype is float32")
    const_val_2 = tvm.const(2.0, dtype)
    mul_val = te.lang.cce.vmuls(x, const_val_2)
    res = te.lang.cce.vdiv(dx, mul_val)
    return res


@util.check_input_type(dict, dict, dict, str)
def sqrt_grad(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_cce

    Parameters
    ----------
    x : dict of data: dict

    dx : dict of data_grad: dict

    out : dict of output: dict

    kernel_name : cce kernel name, default value is "sqrt_grad": str

    Returns
    -------
    None

    """
    util.check_kernel_name(kernel_name)
    shape_x = x.get("shape")
    shape_dx = dx.get("shape")
    dtype_x = x.get("dtype").lower()
    dtype_dx = dx.get("dtype").lower()
    if not operator.eq(list(shape_x), list(shape_dx)):
        raise RuntimeError("Input shapes must be equal")
    if not dtype_x == dtype_dx:
        raise RuntimeError("Input dtype must be same")
    util.check_shape_rule(shape_x)
    util.check_shape_size(shape_x, SHAPE_LIMIT)
    util.check_dtype_rule(dtype_x, ("float16", "float32"))

    shape_x = [reduce_ins(lambda x, y: x * y, shape_x[:])]
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
    data_dx = tvm.placeholder(shape_x, name="data_dx", dtype=dtype_x)
    with tvm.target.cce():
        res = sqrt_grad_compute(data_x, data_dx, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (data_x, data_dx, res)}

    te.lang.cce.cce_build_code(sch, config)