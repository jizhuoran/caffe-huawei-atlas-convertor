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

tanh_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from functools import reduce as reduceIns
from topi import generic
from topi.cce import util

# shape size limit for aicore is 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("tanh_grad")
def tanh_grad_compute(y, dy, z, kernel_name="tanh_grad"):
    """
    do element-wise tanh_grad operation between two input tensors

    Parameters
    ----------
    y: TVM tensor
        the placeholder of y input data
    dy: TVM tensor
        the placeholder of dy input data
    z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh_grad

    Returns
    -------
    res : tvm.tensor
        the result of tanh_grad
    """
    dtype = y.dtype

    if dtype == "float16":
        y = te.lang.cce.cast_to(y, "float32")
        dy = te.lang.cce.cast_to(dy, "float32")

    data1_square = te.lang.cce.vmul(y, y)
    data_mul = te.lang.cce.vmuls(data1_square, tvm.const(-1, dtype=dtype))
    anuminate = te.lang.cce.vadds(data_mul, tvm.const(1, dtype=dtype))
    res = te.lang.cce.vmul(anuminate, dy)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, dict, str)
def tanh_grad(y, dy, z, kernel_name="tanh_grad"):
    """
    do element-wise tanh_grad operation between two input tensors

    Parameters
    ----------
    y : dict
        shape and dtype of y input, only support float16, float32
    dy : dict
        shape and dtype of dy input, only support float16, float32
    z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is tanh_grad

    Returns
    -------
    None
    """
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(shape_y)
    util.check_shape_rule(shape_dy)

    check_list = ("float16", "float32")
    dtype = y.get("dtype").lower()
    util.check_dtype_rule(dtype, check_list)
    if list(shape_y) != list(shape_dy):
        raise RuntimeError(
            "tanh_grad only support input shape"
            "while input_shape1 equals to input_shape2")
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_y)
    data_y = tvm.placeholder(fuseshape, dtype=dtype, name="data1")
    data_dy = tvm.placeholder(fuseshape, dtype=dtype, name="data2")
    res = tanh_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
