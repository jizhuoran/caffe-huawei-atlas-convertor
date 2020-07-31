#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

atan_grad

  Op_description :
    Computes gradients for Atan operation

    # atan_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_atan_grad")

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

CONST_ONE = 1


# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("atan_grad")
def atan_grad_compute(y, dy, z, kernel_name="atan_grad"):
    """
    Calculation for backward gradient

    Parameters:
    ----------
    y: the placeholder of input data
    dy: the placeholder of input dy
    output_z : dict of output
    kernel_name : cce kernel name, default value is atan_grad

    Algorithm :
    ----------
        res = 1/(1+y^2)*dy

    Returns
    ----------
    result res
    """

    scalar_one = tvm.const(CONST_ONE, "float32")
    dtype = y.dtype

    if dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        y = te.lang.cce.cast_to(y, "float32")
        dy = te.lang.cce.cast_to(dy, "float32")

    data_square = te.lang.cce.vmul(y, y)
    sum_tmp = te.lang.cce.vadds(data_square, scalar_one)
    res = te.lang.cce.vdiv(dy, sum_tmp)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, dict, str)
def atan_grad(y, dy, z, kernel_name="atan_grad"):
    """
    Gradient calculation for atan(x)

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32
    dy : dict of dy, include shape and dtype, dtype support float16, float32
    z : dict of output, include shape and dtype
    kernel_name : cce kernel name, default value is atan_grad

    Algorithm :
    ----------
    forward :
        y = atan(x)
    backward gradient :
        de/dx = dy/dx*de/dy = 1/(1+x^2)*grad

    Returns
    ----------
    None
    """

    # get the shape and dtype
    shape = y.get("shape")
    shape_grad = dy.get("shape")
    dtype = y.get("dtype")
    dtype_grad = dy.get("dtype")

    # check whether kernel name is unique
    util.check_kernel_name(kernel_name)

    # check whether the shape is right
    check_shape(shape)
    check_shape(shape_grad)
    if not operator.eq(shape, shape_grad):
        raise RuntimeError("all input shape must be the same")
    shape, _ = refine_shape_axes(shape, [])

    # check whether dtypes are fp16,fp32 and whether they are the same
    check_list = ("float16", "float32")
    check_dtype(dtype, check_list)
    check_dtype(dtype_grad, check_list)
    dtype = dtype.lower()
    if dtype != dtype_grad.lower():
        raise RuntimeError("all input dtype must be same")

    # get 2 input placeholders: data_input, grad
    data_input = tvm.placeholder(shape, name="input_data", dtype=dtype)
    grad = tvm.placeholder(shape, name="input_grad", dtype=dtype)

    # compute the backward gradient
    res = atan_grad_compute(data_input, grad, z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, grad, res]}
    te.lang.cce.cce_build_code(sch, config)
