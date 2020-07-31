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
confusion_softmax_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.util import util_frac_z as fz


# pylint: disable=locally-disabled,unused-argument,invalid-name
def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = te.lang.cce.util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = te.lang.cce.broadcast(tensor, temp_shape)
    tensor = te.lang.cce.broadcast(tensor, shape)
    return tensor


@fusion_manager.register("confusion_softmax_grad")
def confusion_softmax_grad_compute(grad_dict, grad, x, y,
                                   kernel_name="confusion_softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    y = grad - sum(grad * x)

    Parameters
    ----------
    grad_dict: dict
        the dict of first input data
    grad: TVM tensor
        the placeholder of first input data
    x: TVM tensor
        the placeholder of second input data
    y: dict
        the dict of output data
    kernel_name: str
        cce kernel name, default value is "confusion_softmax_grad"

    Returns
    -------
    res: TVM tensor
        the result of confusion_softmax_grad_compute
    """
    dtype = grad.dtype
    shape_input1 = te.lang.cce.util.shape_to_list(grad.shape)
    shape_input2 = te.lang.cce.util.shape_to_list(x.shape)
    if list(shape_input1) != list(shape_input2):
        shape_input1, shape_input2, shape = util.produce_shapes(shape_input1,
                                                                shape_input2)
        grad = _broadcast_nz(grad, shape)
        x = _broadcast_nz(x, shape)

    data_vmul = te.lang.cce.vmul(grad, x)
    if dtype == "float16":
        data_vmul = te.lang.cce.cast_to(data_vmul, "float32")

    if fz.is_frac_z(grad_dict):
        data_sum = te.lang.cce.sum(data_vmul, axis=[-1, -4], keepdims=True)
    else:
        data_sum = te.lang.cce.sum(data_vmul, axis=-1, keepdims=True)

    if dtype == "float16":
        data_sum = te.lang.cce.cast_to(data_sum, "float16")

    if list(shape_input1) != list(shape_input2):
        data_sum_tmp = _broadcast_nz(data_sum, shape)
    else:
        data_sum_tmp = _broadcast_nz(data_sum, shape_input2)

    res = te.lang.cce.vsub(grad, data_sum_tmp)

    return res


@util.check_input_type(dict, dict, dict, str)
def confusion_softmax_grad(grad, x, y, kernel_name="confusion_softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    y = grad - sum(grad * x)

    Parameters
    ----------
    grad: dict
        shape and dtype of first input, only support float16, float32
    x: dict
        shape and dtype of second input, only support float16, float32
    y: dict
        shape and dtype of output data, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "confusion_softmax_grad"

    Returns
    -------
    None
    """
    shape_grad = grad.get("shape")
    shape_x = x.get("shape")
    dtype_grad = grad.get("dtype")

    util.compare_tensor_dict_key(grad, x, "dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_grad)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_grad)
    util.check_tensor_shape_size(shape_x)

    check_list = ("float16", "float32")
    input_dtype = dtype_grad.lower()

    util.check_dtype_rule(input_dtype, check_list)
    if list(shape_grad) != list(shape_x):
        shape_grad, shape_x, shape_max = \
            util.produce_shapes(shape_grad, shape_x)
        util.check_tensor_shape_size(shape_max)

    data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=input_dtype)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_dtype)

    res = confusion_softmax_grad_compute(grad, data_grad, data_x, y,
                                         kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_grad, data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
