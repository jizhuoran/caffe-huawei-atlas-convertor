#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

asin_grad

  Op_description :
    Computes gradients for Asin operation

    # asin_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_asin_grad")

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

# scalar in asin_grad and Newton's equation
NUM_MINUS_ONE = -1
NUM_ONE = 1


# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("asin_grad")
def asin_grad_compute(y, dy, z, kernel_name="asin_grad"):
    """
    do element-wise asin_grad compute

    Parameters:
    ----------
    y : the placeholders of input y

    dy : the placeholders of input dy

    z : output dict

    kernel_name : cce kernel name, default value is "cce_asin_grad"

    return : dy * (1 / sqrt(1 - y^2))
    -------
    """

    dtype = y.dtype
    if dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        y = te.lang.cce.cast_to(y, "float32")
        dy = te.lang.cce.cast_to(dy, "float32")

    # step 1: calculate num_to_vrsqrt = 1 - y^2
    data = te.lang.cce.vmul(y, y)
    data = te.lang.cce.vmuls(data, tvm.const(NUM_MINUS_ONE, y.dtype))
    num_to_vrsqrt = te.lang.cce.vadds(data, tvm.const(NUM_ONE, y.dtype))

    # step 2: calculate dy * (1 / sqrt(1 - y^2))
    vsqrt_res = te.lang.cce.vsqrt(num_to_vrsqrt, 1)
    res = te.lang.cce.vdiv(dy, vsqrt_res)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, dict, str)
def asin_grad(y, dy, z, kernel_name="asin_grad"):
    """
    do element-wise asin_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of output

    kernel_name : cce kernel name, default value is "asin_grad"

    Returns
    -------
    None
    """

    # get the shape and dtype
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype_y = y.get("dtype")
    dtype_dy = dy.get("dtype")

    # kernel name check: should be unique
    util.check_kernel_name(kernel_name)

    # check whether the shape is right
    check_shape(shape_y)
    check_shape(shape_dy)
    if not operator.eq(shape_y, shape_dy):
        raise RuntimeError("all input shape must be the same")
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32")
    check_dtype(dtype_y, check_list)
    check_dtype(dtype_dy, check_list)
    dtype_y = dtype_y.lower()
    if dtype_y != dtype_dy.lower():
        raise RuntimeError("all input dtype must be same")

    # get 2 input tensors: data_y, data_dy
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
    data_dy = tvm.placeholder(shape_y, name="data_dy", dtype=dtype_y)
    res = asin_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
