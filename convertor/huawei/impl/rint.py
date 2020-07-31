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

rint
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from functools import reduce as reduceIns
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("rint")
def rint_compute(input_x, output_y, kernel_name="rint"):
    """
    rint compute
    calculating rint(x):
    returns the integer nearest to x by element-wise
    If the result is between two representable values,
     the even number should be used.
    For example:
    x :    [0.9, 2.5, 2.3, 1.5, -4.5]
    res : [ 1.0, 2.0, 2.0, 2.0, -4.0 ]

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    output_y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "rint"

    Returns
    -------
    res: TVM tensor
        the result of rint compute
    """
    res = te.lang.cce.round(input_x)
    res = te.lang.cce.cast_to(res, input_x.dtype)

    return res


@util.check_input_type(dict, dict, str)
def rint(input_x, output_y, kernel_name="rint"):
    """
    algorithm: rint
    calculating rint(x):
    returns the integer nearest to x by element-wise
    If the result is between two representable values,
     the even number should be used.
    For example:
    x :    [0.9, 2.5, 2.3, 1.5, -4.5]
    res : [ 1.0, 2.0, 2.0, 2.0, -4.0 ]

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    output_y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "rint"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)

    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype.lower(), check_list)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_x)
    data_x = tvm.placeholder(fuseshape, dtype=dtype.lower(), name="data")
    res = rint_compute(data_x, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
