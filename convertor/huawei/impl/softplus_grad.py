#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

softplus_grad
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util

# define a scalar, value = 1
SCALAR_ONE = 1


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("softplus_grad")
def softplus_grad_compute(input_gradients, input_features, output_backprops,
                          kernel_name="softplus_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy * exp(x) / (1 + exp(x))".

    Parameters
    ----------
    input_gradients: TVM tensor
        The backpropagated gradients to the corresponding softplus operation.
    input_features: TVM tensor
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32", "int32", "int8", "uint8".
    output_backprops: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softplus_grad".

    Returns
    -------
    res: TVM tensor
        output tensor. has the same type as "input_gradients".
    """
    shape_dy = te.lang.cce.util.shape_to_list(input_gradients.shape)
    shape_x = te.lang.cce.util.shape_to_list(input_features.shape)
    dtype = input_gradients.dtype

    if list(shape_dy) != list(shape_x):
        shape_dy, shape_x, shape_max = util.produce_shapes(shape_dy, shape_x)
        input_gradients = te.lang.cce.broadcast(input_gradients, shape_max, dtype)
        input_features = te.lang.cce.broadcast(input_features, shape_max, dtype)
    else:
        shape_max = shape_dy

    if dtype != "float32":
        input_gradients = te.lang.cce.cast_to(input_gradients, "float32")
        input_features = te.lang.cce.cast_to(input_features, "float32")

    data_exp_tmp = te.lang.cce.vexp(input_features)
    data_add_tmp = te.lang.cce.vadds(data_exp_tmp, SCALAR_ONE)
    data_div_tmp = te.lang.cce.vdiv(data_exp_tmp, data_add_tmp)
    res_tmp = te.lang.cce.vmul(input_gradients, data_div_tmp)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res_tmp, "float16")
    elif dtype in ("int32", "int8", "uint8"):
        data_zero = te.lang.cce.broadcast(tvm.const(0, "float16"),
                                          shape_max, "float16")
        res_min = te.lang.cce.vmin(res_tmp, data_zero)
        res_max = te.lang.cce.vmax(res_tmp, data_zero)
        res_max_int = te.lang.cce.floor(res_max)
        res_min_int = te.lang.cce.ceil(res_min)
        res = te.lang.cce.vadd(res_max_int, res_min_int)
    else:
        res = res_tmp

    if dtype == "int8":
        res = te.lang.cce.cast_to(res, "int8")
    elif dtype == "uint8":
        res = te.lang.cce.cast_to(res, "uint8")

    return res


@util.check_input_type(dict, dict, dict, str)
def softplus_grad(input_gradients, input_features, output_backprops,
                  kernel_name="softplus_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy * exp(x) / (1 + exp(x))".

    Parameters
    ----------
    input_gradients: dict
        The backpropagated gradients to the corresponding softplus operation.
    input_features: dict
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32", "int32", "int8", "uint8".
    output_backprops: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softplus_grad".

    Returns
    -------
    None
    """
    shape_dy = input_gradients.get("shape")
    dtype_dy = input_gradients.get("dtype")
    shape_x = input_features.get("shape")
    dtype_x = input_features.get("dtype")

    if dtype_dy.lower() != dtype_x.lower():
        raise RuntimeError(
            "type of dy and type of x must be same, \
             while the types are different")
    dtype = dtype_dy

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_dy)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_dy)
    util.check_tensor_shape_size(shape_x)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    input_dtype = dtype.lower()
    util.check_dtype_rule(input_dtype, check_list)
    shape_dy, shape_x, shape_max = util.produce_shapes(shape_dy, shape_x)
    util.check_tensor_shape_size(shape_max)
    reshape_dy, reshape_x = refine_shapes_for_broadcast(shape_dy, shape_x)

    data_dy = tvm.placeholder(reshape_dy, name="data_dy", dtype=input_dtype)
    data_x = tvm.placeholder(reshape_x, name="data_x", dtype=input_dtype)

    res = softplus_grad_compute(data_dy, data_x, output_backprops,
                                kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_dy, data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
