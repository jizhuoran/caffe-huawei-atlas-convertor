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

rsqrt

  Op_description :
    Computes reciprocal of square root of x element-wise

    # rsqrt(
    #   x,
    #   y,
    #   kernel_name="rsqrt_cce")

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
CONST_ONE = 1.0


# pylint: disable=locally-disabled,too-many-arguments,
# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("rsqrt")
def rsqrt_compute(x, y, kernel_name="rsqrt_cce"):
    """
    Algrithm : rsqrt(x) = 1 / sqrt(x)  where x > 0

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of rsqrt
    """

    inp_dtype = x.dtype

    if inp_dtype == "float16" and api_check_support("te.lang.cce.vadd",
                                                    "float32"):
        x = te.lang.cce.cast_to(x, "float32")

    data_res = _compute(x)

    if inp_dtype == "float16":
        data_res = te.lang.cce.cast_to(data_res, "float16")

    return data_res


def _compute(data_input):
    """
    Algrithm: rsqrt(x) = 1 / sqrt(x)

    Parameters
    ----------
    data_input: the placeholder of data input

    Returns
    -------
    data_res :  return of rsqrt
    """

    inp_shape = data_input.shape
    data_sqrt = te.lang.cce.vsqrt(data_input, 1)
    tesor_one = te.lang.cce.broadcast(tvm.const(CONST_ONE, data_input.dtype),
                                      inp_shape)
    result = te.lang.cce.vdiv(tesor_one, data_sqrt)

    return result


@util.check_input_type(dict, dict, str)
def rsqrt(x, y, kernel_name="rsqrt_cce"):
    """
    Algrithm: rsqrt(x) = 1 / sqrt(x)  where x > 0

    Parameters
    ----------
    Algorithm: rsqrt

    Parameters:

    x: the dict of input data, support float16, float32

    y: the dict of output

    kernel_name: cce kernel name, default value is "rsqrt_cce".

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
        res = rsqrt_compute(input_data, y, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data, res],
              "print_ir": False,
             }

    te.lang.cce.cce_build_code(sch, config)
