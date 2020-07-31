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

invert
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("invert")
def invert_compute(input_x, output_y, kernel_name="invert"):
    """Flips all bits elementwise.

    Parameters
    ----------
    input_x: TVM tensor
        input tensor.
    output_y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "invert".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    res = te.lang.cce.vnot(input_x)

    return res


@util.check_input_type(dict, dict, str)
def invert(input_x, output_y, kernel_name="invert"):
    """Flips all bits elementwise.

    Parameters
    ----------
    input_x: dict
        the dict of input tensor.
        Must be one of the following types: `int16`, `uint16`.
    output_y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "invert".

    Returns
    -------
    None.
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    dtype_x_lower = dtype_x.lower()
    check_list = ("int16", "uint16")

    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    util.check_dtype_rule(dtype_x_lower, check_list)
    util.check_kernel_name(kernel_name)

    shape_x = (functools_reduce(lambda x, y: x * y, shape_x[:]),)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x_lower)
    res = invert_compute(data_x, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
