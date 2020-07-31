#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except
in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

abs_grad

  Op_description :
    Computes gradients for abs operation

    # abs_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_abs_grad")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : 'y' and 'dy' must have the same type and shape.
    [2] All : shape size limit is 2147483648.
"""

import operator

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
from topi import generic
from topi.cce import util

NUM_MAX = 32768
NUM_MIN = 2**(-15)

# pylint: disable=unused-argument,too-many-locals,invalid-name
@fusion_manager.register("abs_grad")
def abs_grad_compute(y, dy, z, kernel_name="abs_grad"):
    """
    do abs_grad compute
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "abs_grad"
    return: data_dy * sign(data_y)
    ----------------
    """

    data_y = y
    dtype = y.dtype

    data1_mulmax = te.lang.cce.vmuls(data_y, tvm.const(NUM_MAX, dtype=dtype))
    data1_res = te.lang.cce.vabs(data1_mulmax)
    data1_res = te.lang.cce.vadds(data1_res, tvm.const(NUM_MIN, dtype=dtype))
    data1_res = te.lang.cce.vdiv(data1_mulmax, data1_res)
    data1_res = te.lang.cce.round(data1_res)
    data1_res = te.lang.cce.vmul(data1_res, dy)

    return data1_res

# pylint: disable=invalid-name
@util.check_input_type(dict, dict, dict, str)
def abs_grad(y, dy, z, kernel_name="abs_grad"):
    """
    do element-wise abs_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of z, include shape and dtype, dtype support float16, float32

    kernel_name : cce kernel name, default value is "abs_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype_y = y.get("dtype")
    dtype_dy = dy.get("dtype")

    util.check_kernel_name(kernel_name)
    check_shape(shape_y)
    check_shape(shape_dy)
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    check_list = ("float16", "float32")
    check_dtype(dtype_y, check_list)
    check_dtype(dtype_dy, check_list)
    dtype_y = dtype_y.lower()
    dtype_dy = dtype_dy.lower()
    if not operator.eq(shape_y, shape_dy):
        raise RuntimeError(
            "abs_grad only support input shape while input_shape1 equals to input_shape2")
    if dtype_y != dtype_dy:
        raise RuntimeError(
            "abs_grad only support dtype while input_dtype1 equals to input_dtype2")
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data1")
    data_dy = tvm.placeholder(shape_dy, dtype=dtype_dy, name="data2")
    res = abs_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
