#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not
use this file except in compliance with the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
softmaxgrad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("softmax_grad")
def softmax_grad_compute(softmax, grad_softmax, grad_x,
                         kernel_name="softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: TVM tensor
        the placeholder of first input data
    grad_softmax: TVM tensor
        the placeholder of second input data
    grad_x: dict
        the dict of output data
    kernel_name: str
        cce kernel name, default value is "softmax_grad"

    Returns
    -------
    res: TVM tensor
        the result of softmax_grad_compute
    """
    dtype = softmax.dtype
    shape_input1 = te.lang.cce.util.shape_to_list(softmax.shape)
    shape_input2 = te.lang.cce.util.shape_to_list(grad_softmax.shape)
    has_improve_precision = False
    if list(shape_input1) != list(shape_input2):
        shape_input1, shape_input2, shape = util.produce_shapes(shape_input1,
                                                                shape_input2)
        softmax = te.lang.cce.broadcast(softmax, shape, dtype)
        grad_softmax = te.lang.cce.broadcast(grad_softmax, shape, dtype)

    data_vmul = te.lang.cce.vmul(softmax, grad_softmax)
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        data_vmul = te.lang.cce.cast_to(data_vmul, "float32")
        has_improve_precision = True
    data_sum = te.lang.cce.sum(data_vmul, axis=-1, keepdims=True)
    if list(shape_input1) != list(shape_input2):
        data_sum_tmp = te.lang.cce.broadcast(data_sum, shape)
    else:
        data_sum_tmp = te.lang.cce.broadcast(data_sum, shape_input2)
    data_sub = te.lang.cce.vsub(grad_softmax, data_sum_tmp)
    res = te.lang.cce.vmul(softmax, data_sub)
    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, dict, str)
def softmax_grad(softmax, grad_softmax, grad_x, kernel_name="softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: dict
        shape and dtype of first input, only support float16, float32
    grad_softmax: dict
        shape and dtype of second input, only support float16, float32
    grad_x: dict
        shape and dtype of output data, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "softmax_grad"

    Returns
    -------
    None
    """
    shape_softmax = softmax.get("shape")
    shape_grad_softmax = grad_softmax.get("shape")
    dtype_softmax = softmax.get("dtype")

    util.compare_tensor_dict_key(softmax, grad_softmax, "dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_softmax)
    util.check_shape_rule(shape_grad_softmax)
    util.check_tensor_shape_size(shape_softmax)
    util.check_tensor_shape_size(shape_grad_softmax)

    check_list = ("float16", "float32")
    input_dtype = dtype_softmax.lower()

    util.check_dtype_rule(input_dtype, check_list)
    if list(shape_softmax) != list(shape_grad_softmax):
        shape_softmax, shape_grad_softmax, shape_max = \
            util.produce_shapes(shape_softmax, shape_grad_softmax)
        util.check_tensor_shape_size(shape_max)

    softmax = tvm.placeholder(shape_softmax, name="softmax", dtype=input_dtype)
    grad_softmaxgrad = tvm.placeholder(shape_grad_softmax,
                                       name="grad_softmaxgrad",
                                       dtype=input_dtype)

    res = softmax_grad_compute(softmax, grad_softmaxgrad, grad_x,
                               kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [softmax, grad_softmaxgrad, res]}
    te.lang.cce.cce_build_code(sch, config)
