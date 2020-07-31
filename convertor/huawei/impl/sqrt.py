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

sqrt
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te import platform as tbe_platform

# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("sqrt")
def sqrt_compute(input_data, output_data, kernel_name="sqrt"):
    """
    calculating data sqrt,y= x**0.5,mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    result: TVM tensor
        the result of sqrt
    """
    dtype = input_data.dtype
    has_improve_precision = False
    if dtype == "float16" and\
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt",
                                                    "float32"):
        input_data = te.lang.cce.cast_to(input_data, "float32")
        has_improve_precision = True
    result = te.lang.cce.vsqrt(input_data)

    if has_improve_precision:
        result = te.lang.cce.cast_to(result, "float16")

    return result


@util.check_input_type(dict, dict, str)
def sqrt(input_x, output_y, kernel_name="sqrt"):
    """
    algorithm: sqrt
    calculating data sqrt,y= x**0.5, mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sqrt

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)
    util.check_dtype_rule(input_dtype, ("float16", "float32"))

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, input_shape)
    input_data = tvm.placeholder(fuseshape, name="input_data",
                                 dtype=input_dtype)
    result = sqrt_compute(input_data, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input_data, result]}

    te.lang.cce.cce_build_code(sch, config)
