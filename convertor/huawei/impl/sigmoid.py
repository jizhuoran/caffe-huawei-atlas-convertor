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

sigmoid
"""

from functools import reduce as reduceIns
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
# General limitation of the reduce size for input shape: 2**30
SHAPE_SIZE_LIMIT = 2**30


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("sigmoid")
def sigmoid_compute(x, y, kernel_name="sigmoid"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    output tensor
    """
    data_input = x
    dtype = x.dtype
    exp_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vexp", "float32")
    mul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        raise RuntimeError(
            "Input dtype only support float16 while input dtype is float32")

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = te.lang.cce.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = te.lang.cce.cast_to(tmp_negative, "float16")
    tmp_exp = te.lang.cce.vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = te.lang.cce.cast_to(tmp_exp, "float32")
    tmp_sum = te.lang.cce.vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = te.lang.cce.broadcast(tvm.const(1, dtype), inp_shape)
        tmp_rec = te.lang.cce.vdiv(tensor_one, tmp_sum)
    else:
        tmp_rec = te.lang.cce.vrec(tmp_sum)
    return tmp_rec


@util.check_input_type(dict, dict, str)
def sigmoid(x, y, kernel_name="sigmoid"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        dict of x, include keys(shape and dtype)
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    input_dtype = dtype.lower()
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)

    fused_shape = [reduceIns(lambda a, b: a * b, shape[:])]
    data_input = tvm.placeholder(fused_shape, name="data_input",
                                 dtype=input_dtype)

    res = sigmoid_compute(data_input, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
