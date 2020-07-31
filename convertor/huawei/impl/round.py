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

round
"""
from __future__ import absolute_import

from functools import reduce as functools_reduce

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("round")
def round_compute(x, y, kernel_name="round"):
    """
    calculating data round, round to the nearst,tie to the even

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is round

    Returns
    -------
    result: TVM tensor
        the result of round
    """
    dtype = x.dtype
    if dtype == "int32":
        input_data_one = te.lang.cce.broadcast(tvm.const(0, dtype),
                                               x.shape, dtype)
        result = te.lang.cce.vadd(x, input_data_one)
        return result

    result = te.lang.cce.round(x)
    result = te.lang.cce.cast_to(result, dtype)

    return result


# pylint: disable=locally-disabled,redefined-builtin
@util.check_input_type(dict, dict, str)
def round(x, y, kernel_name="round"):
    """
    algorithm: round
    calculating data round, round to the nearst,tie to the even

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16,float32,int32
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is round

    Returns
    -------
    None
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)
    util.check_dtype_rule(input_dtype, ("float16", "float32", "int32"))

    up_shape = [1]
    up_shape[0] = functools_reduce(lambda x, y: x * y, input_shape[:])

    input_data = tvm.placeholder(up_shape, name="input_data",
                                 dtype=input_dtype)
    result = round_compute(input_data, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"name": kernel_name,
              "tensor_list": [input_data, result]}

    te.lang.cce.cce_build_code(sch, config)
