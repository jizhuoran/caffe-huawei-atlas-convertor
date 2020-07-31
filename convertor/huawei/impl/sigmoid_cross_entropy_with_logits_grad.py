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

sigmoid_cross_entropy_with_logits_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from functools import reduce as functools_reduce

# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = -1
SCALAR_NEGTIVE_ONE = -1


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("sigmoid_cross_entropy_with_logits_grad")
def sigmoid_cross_entropy_with_logits_grad_compute(
        predict,
        target,
        dout,
        gradient,
        kernel_name):
    """
    calculating sigmoid_cross_entropy_with_logits_grad_compute

    Parameters
    ----------
    predict : TVM tensor
        the output of previous layer
    target : TVM tensor
        label
    dout : TVM tensor
        last gradient
    gradient : TVM tensor
        result after compute
    Returns
    -------
    output tensor
    """
    dtype = predict.dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        predict = te.lang.cce.cast_to(predict, "float32")
        target = te.lang.cce.cast_to(target, "float32")
        dout = te.lang.cce.cast_to(dout, "float32")

    # e^x
    val1 = te.lang.cce.vexp(predict)
    # 1 + e^x
    val2 = te.lang.cce.vadds(val1, tvm.const(SCALAR_ONE, dtype="float32"))

    val3 = te.lang.cce.vdiv(val1, val2)
    # -target
    val4 = te.lang.cce.vmuls(target,
                             tvm.const(SCALAR_NEGTIVE_ONE, dtype="float32"))

    val5 = te.lang.cce.vadd(val3, val4)

    result = te.lang.cce.vmul(val5, dout)

    if dtype == "float16":
        result = te.lang.cce.cast_to(result, dtype)
    return result


@util.check_input_type(dict, dict, dict, dict, str)
def sigmoid_cross_entropy_with_logits_grad(
        predict,
        target,
        dout,
        gradient,
        kernel_name="sigmoid_cross_entropy_with_logits_grad"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        the output of previous layer
    target : dict
        label
    dout : dict
        last gradient
    gradient : dict
        result after compute
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_grad"

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype")
    gradient_dtype = gradient.get("dtype").lower()
    predict_dtype_lower = predict_dtype.lower()
    util.check_dtype_rule(gradient_dtype, check_list)
    util.check_dtype_rule(predict_dtype_lower, check_list)

    util.check_shape_rule(predict_shape)
    util.check_tensor_shape_size(predict_shape)
    util.check_kernel_name(kernel_name)

    target_shape = target.get("shape")
    target_dtype = target.get("dtype")
    target_dtype_lower = target_dtype.lower()
    util.check_dtype_rule(target_dtype_lower, check_list)

    util.check_shape_rule(target_shape)
    util.check_tensor_shape_size(target_shape)

    dout_shape = dout.get("shape")
    dout_dtype = dout.get("dtype")
    dout_dtype_lower = dout_dtype.lower()
    util.check_dtype_rule(dout_dtype_lower, check_list)

    util.check_shape_rule(dout_shape)
    util.check_tensor_shape_size(dout_shape)
    util.compare_tensor_dict_key(predict, target, "shape")
    util.compare_tensor_dict_key(predict, dout, "shape")
    shape = (functools_reduce(lambda x, y: x * y, predict_shape[:]),)
    predict_data_input = tvm.placeholder(
        shape, name="predict_data_input", dtype=predict_dtype_lower)
    target_data_input = tvm.placeholder(
        shape, name="target_data_input", dtype=target_dtype_lower)
    dout_data_input = tvm.placeholder(
        shape, name="dout_data_input", dtype=dout_dtype_lower)

    res = sigmoid_cross_entropy_with_logits_grad_compute(
        predict_data_input, target_data_input, dout_data_input, gradient,
        kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name":
            kernel_name,
        "tensor_list": [
            predict_data_input, target_data_input, dout_data_input, res
        ]
    }

    te.lang.cce.cce_build_code(sch, config)
