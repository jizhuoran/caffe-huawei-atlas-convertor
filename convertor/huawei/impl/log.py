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

log
"""
from functools import reduce as reduceIns
import math
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util


def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    Return True if the values a and b are close to each other and False otherwise
    See math.isclose for further understanding.
    Parameters
    ----------
    valuex : value x
    valuey : value y
    rel_tol : relative tolerance
    abs_tol : absolute tolerance
    Returns
    -------
    bool
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)


# pylint: disable=too-many-arguments,unused-argument
@fusion_manager.register("log")
def log_compute(input_x, output_y, base=-1.0, scale=1.0,
                shift=0.0, kernel_name="log"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "log"

    Returns
    -------
    output tensor
    """
    log_base = 1.0 if isclose(base, -1.0) else math.log(base)
    base_scale = 1.0 / log_base

    dtype = input_x.dtype
    f322f16_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.cast_to", "f322f16")
    if dtype == "float32" and not f322f16_support:
        raise RuntimeError(
            "Input dtype only support float16 while input dtype is float32")

    if isclose(scale, 1.0) and isclose(shift, 0.0):
        x_log = te.lang.cce.vlog(input_x)
    else:
        x_scale_and_shift = input_x
        if not isclose(scale, 1.0):
            x_scale_and_shift = te.lang.cce.vmuls(input_x,
                                                  tvm.const(scale, dtype=dtype))

        if not isclose(shift, 0.0):
            x_scale_and_shift = te.lang.cce.vadds(x_scale_and_shift,
                                                  tvm.const(shift, dtype=dtype))

        x_log = te.lang.cce.vlog(x_scale_and_shift)

    if not isclose(base_scale, 1.0):
        res = te.lang.cce.vmuls(x_log, tvm.const(base_scale, dtype=dtype))
        return res
    return x_log


@util.check_input_type(dict, dict, float, float, float, str)
def log(input_x, output_y, base=-1.0, scale=1.0, shift=0.0, kernel_name="log"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "log"

    Returns
    -------
    None
    """

    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()

    # input_x' shape check
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)

    # kernel_name check
    util.check_kernel_name(kernel_name)

    # input_x' dtype check, only supports fp16 and fp32
    check_list = ("float16", "float32")
    util.check_dtype_rule(input_dtype, check_list)

    if base <= 0 and (not isclose(base, -1.0)):
        raise RuntimeError("base must be strictly positive or -1.")

    fused_shape = [reduceIns(lambda x, y: x * y, shape[:])]
    data_input = tvm.placeholder(fused_shape, name="data_input",
                                 dtype=input_dtype)

    res = log_compute(data_input, output_y, base, scale, shift, kernel_name)

    # auto schedule
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    # operator build
    config = {"print_ir": False,
              "name": kernel_name,
              "need_build": True,
              "tensor_list": (data_input, res)}

    te.lang.cce.cce_build_code(sch, config)
