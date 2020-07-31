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

acos_grad

  Op_description :
    Computes gradients for Acos operation

    # acos_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="acos_grad")

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
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
from topi import generic
from topi.cce import util

# newton eqation is x1 = x0(3-a*(x0^2))/2
NUM_MINUS_ONE = -1
NUM_ONE = 1


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("acos_grad")
def acos_grad_compute(y, dy, z, kernel_name="acos_grad"):
    """
    do acos_grad compute with sqrt and div
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "acos_grad"
    return: dy * (- 1 / (1 - data_y^2)^1/2)
    ----------------
    """

    dtype = y.dtype
    dtype_1 = dtype
    if dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        y = te.lang.cce.cast_to(y, "float32")
        dy = te.lang.cce.cast_to(dy, "float32")
        dtype = "float32"

    data1_square = te.lang.cce.vmul(y, y)
    data1_square = te.lang.cce.vmuls(data1_square, tvm.const(NUM_MINUS_ONE,
                                                             dtype=dtype))
    data1_square = te.lang.cce.vadds(data1_square, tvm.const(NUM_ONE,
                                                             dtype=dtype))

    data1_reciprocal = te.lang.cce.vsqrt(data1_square, 1)
    data1_reciprocal = te.lang.cce.vdiv(dy, data1_reciprocal)
    res = te.lang.cce.vmuls(data1_reciprocal, tvm.const(NUM_MINUS_ONE,
                                                        dtype=dtype))

    if dtype_1 == "float16":
        res = te.lang.cce.cast_to(res, "float16")
    return res


@util.check_input_type(dict, dict, dict, str)
def acos_grad(y, dy, z, kernel_name="acos_grad"):
    """
    do element-wise acos_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of z, include shape and dtype, dtype support float16, float32

    kernel_name : cce kernel name, default value is "acos_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype = y.get("dtype")
    dtype1 = dy.get("dtype")

    util.check_kernel_name(kernel_name)
    check_shape(shape_y)
    check_shape(shape_dy)
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    # raise runtimeerror if the input paras are invalid
    check_list = ("float16", "float32")
    check_dtype(dtype, check_list)
    check_dtype(dtype1, check_list)
    dtype = dtype.lower()
    dtype1 = dtype1.lower()
    if not operator.eq(shape_y, shape_dy):
        raise RuntimeError(
            "acos_grad only support input shape while input_shape1 equals"
            " to input_shape2")
    if dtype != dtype1:
        raise RuntimeError(
            "acos_grad only support dtype while input_dtype1 equals"
            " to input_dtype2")
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    data_y = tvm.placeholder(shape_y, dtype=dtype, name="data1")
    data_dy = tvm.placeholder(shape_dy, dtype=dtype, name="data2")

    res = acos_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_y, data_dy, res)}
    te.lang.cce.cce_build_code(sch, config)
