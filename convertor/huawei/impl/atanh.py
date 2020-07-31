#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

atanh

  Op_description :
    Computes inverse hyperbolic tangent of x element-wise

    # atanh(
    #   x,
    #   y,
    #   kernel_name="atanh_cce")

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
from topi import generic
from topi.cce import util


# const value
CONST_HALF = 0.5
CONST_ONE = 1
CONST_NEG_ONE = -1


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("atanh")
def atanh_compute(x, y, kernel_name="atanh"):
    """
    Algrithm : atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) < 1

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of atanh
    """

    inp_dtype = x.dtype
    shape = x.shape

    if inp_dtype == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        x = te.lang.cce.cast_to(x, "float32")

    data_res = _compute(x, shape)

    if inp_dtype == "float16":
        data_res = te.lang.cce.cast_to(data_res, "float16")
    else:
        data_res = te.lang.cce.cast_to(data_res, "float32")

    return data_res


def _compute(data_input, shape):
    """
    Algrithm: atanh(x) = 0.5*log((1+x)/(1-x))

    Parameters
    ----------
    data_input: the placeholder of data input

    shape: the shape of data_input

    Returns
    -------
    data_res :  return of atanh
    """

    data_1_sum_x = te.lang.cce.vadds(data_input, tvm.const(CONST_ONE,
                                                           data_input.dtype))
    data_sub_x = te.lang.cce.vmuls(data_input, tvm.const(CONST_NEG_ONE,
                                                         data_input.dtype))
    data_1_sub_x = te.lang.cce.vadds(data_sub_x, tvm.const(CONST_ONE,
                                                           data_input.dtype))
    data_x_mul = te.lang.cce.vdiv(data_1_sum_x, data_1_sub_x)
    data_x_log = te.lang.cce.vlog(data_x_mul, 1)
    data_res = te.lang.cce.vmuls(data_x_log, tvm.const(CONST_HALF,
                                                       data_input.dtype))

    return data_res


@util.check_input_type(dict, dict, str)
def atanh(x, y, kernel_name="atanh"):
    """
    Algrithm: atanh(x) = atanh

    Parameters
    ----------
    Algorithm: atanh

    Parameters:

    x: the dict of input data, only support float16, float32.

    y: the dict of output

    kernel_name: cce kernel name, default value is "atanh".

    Returns
    -------
    None
    """

    shape = x.get("shape")
    dtype = x.get("dtype")

    util.check_kernel_name(kernel_name)
    check_shape(shape)
    shape, _ = refine_shape_axes(shape, [])

    check_list = ("float16", "float32")
    check_dtype(dtype, check_list)

    dtype = dtype.lower()
    input_data = tvm.placeholder(shape, dtype, "input_data")

    with tvm.target.cce():
        res = atanh_compute(input_data, y, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data, res],
              "print_ir": False,
              "bool_storage_as_1bit": False
             }

    te.lang.cce.cce_build_code(sch, config)
