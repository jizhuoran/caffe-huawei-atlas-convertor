#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights losserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

sigmoid_cross_entropy_with_logits
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import refine_shapes_for_broadcast

# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = 0
SCALAR_ZREO = 0


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("sigmoid_cross_entropy_with_logits")
def sigmoid_cross_entropy_with_logits_compute(predict,
                                              target,
                                              loss,
                                              kernel_name):
    """
    calculating data

    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    target : TVM tensor
        the placeholder of target
    loss : dict
        dict of loss, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    output tensor
    """
    predict_dtype = predict.dtype
    target_dtype = target.dtype
    if predict_dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vsub", "float32"):
        predict = te.lang.cce.cast_to(predict, "float32")
    if target_dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        target = te.lang.cce.cast_to(target, "float32")

    dtype_predict = predict.dtype
    shape_predict = te.lang.cce.util.shape_to_list(predict.shape)

    const_zero = tvm.const(SCALAR_ZREO, dtype=dtype_predict)
    max_predict_zero = te.lang.cce.vmaxs(predict, const_zero)

    abs_predict = te.lang.cce.vabs(predict)
    const_zero_broadcast = te.lang.cce.broadcast(const_zero, shape_predict)
    reverse_abs_predict = te.lang.cce.vsub(const_zero_broadcast, abs_predict)
    vexp_predict = te.lang.cce.vexp(reverse_abs_predict)
    const_one = tvm.const(SCALAR_ONE, dtype=dtype_predict)
    vadds_res = te.lang.cce.vadds(vexp_predict, const_one)
    vlog_res = te.lang.cce.vlog(vadds_res, priority_flag=1)
    vmul_res = te.lang.cce.vmul(predict, target)
    res = te.lang.cce.vsub(vlog_res, vmul_res)
    loss = te.lang.cce.vadd(res, max_predict_zero)

    if predict_dtype == "float16":
        loss = te.lang.cce.cast_to(loss, "float16")

    return loss


@util.check_input_type(dict, dict, dict, str)
def sigmoid_cross_entropy_with_logits(
        predict, target, loss,
        kernel_name="sigmoid_cross_entropy_with_logits"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    loss : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    None
    """
    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype")
    input_dtype_predict = dtype_predict.lower()
    util.check_shape_rule(shape_predict)
    util.check_tensor_shape_size(shape_predict)

    shape_target = target.get("shape")
    dtype_target = target.get("dtype")
    input_dtype_target = dtype_target.lower()
    util.check_shape_rule(shape_target)
    util.check_tensor_shape_size(shape_target)

    util.check_kernel_name(kernel_name)

    check_list = ("float16", "float32")
    util.check_dtype_rule(input_dtype_predict, check_list)
    util.check_dtype_rule(input_dtype_target, check_list)
    shape_predict, shape_target = \
        refine_shapes_for_broadcast(shape_predict, shape_target)
    data_predict = tvm.placeholder(shape_predict,
                                   name="data_predict",
                                   dtype=input_dtype_predict)
    data_target = tvm.placeholder(shape_target,
                                  name="data_target",
                                  dtype=input_dtype_target)
    loss = sigmoid_cross_entropy_with_logits_compute(data_predict,
                                                     data_target,
                                                     loss,
                                                     kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(loss)

    config = {"name": kernel_name,
              "tensor_list": [data_predict, data_target, loss]}

    te.lang.cce.cce_build_code(sch, config)
