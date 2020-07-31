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

acosh

  Op_description :
    Computes inverse hyperbolic cosine of x element-wise

    # acosh(
    #   input_data,
    #   output_res,
    #   kernel_name="cce_acosh")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.
"""
from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
import topi
from topi.cce import util

CONST_NEG_ONE = -1.0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("acosh")
def acosh_compute(input_data, output_res, kernel_name="acosh"):
    """
    do element-wise acosh compute
    f(x) = log(x+sqrt(x^2-1)),  for all inputs

    Parameters:
    ----------
    input_data: the placeholder of data input

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "acosh"

    Returns : A Tensor. Has the same type as input_data.
    -------
    """
    data = input_data

    input_dtype = data.dtype.lower()
    if input_dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        data = te.lang.cce.cast_to(data, "float32")

    res = te.lang.cce.vmul(data, data)
    res = te.lang.cce.vadds(res, tvm.const(CONST_NEG_ONE, data.dtype))
    res = te.lang.cce.vsqrt(res, 1)
    res = te.lang.cce.vadd(res, data)
    res = te.lang.cce.vlog(res, 1)

    if input_dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, str)
def acosh(input_data, output_res, kernel_name="acosh"):
    """
    calculating data's acosh,y= log(x+sqrt(x^(2)-1))

    Parameters
    ----------
    input_data: the dict of input, only support float16, float32

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "cce_acosh"

    Returns
    -------
    None

    """

    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype")
    util.check_kernel_name(kernel_name)
    check_shape(shape_input)

    check_list = ("float16", "float32")
    check_dtype(dtype_input, check_list)
    shape_input, _ = refine_shape_axes(shape_input, [])

    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_input, dtype=input_dtype, name="data_input")

    res = acosh_compute(data, output_res, kernel_name)

    with tvm.target.cce():
        sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data, res),
              "bool_storage_as_1bit": False}

    te.lang.cce.cce_build_code(sch, config)
