#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
div
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,too-many-locals,unused-argument
@fusion_manager.register("div")
def div_compute(input_x, input_y, output_div, kernel_name="div"):
    """
    div compute
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    res: TVM tensor
        the result of div compute
    """
    input_data1 = te.lang.cce.util.shape_to_list(input_x.shape)
    input_data2 = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_list = util.produce_shapes(input_data1, input_data2)
    util.check_tensor_shape_size(shape_list[2])
    dtype = input_x.dtype
    int_list = ("int8", "uint8", "int32")
    int_flag = dtype in int_list
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")
    data_x_broad = te.lang.cce.broadcast(input_x, shape_list[2])
    data_y_broad = te.lang.cce.broadcast(input_y, shape_list[2])
    res = te.lang.cce.vdiv(data_x_broad, data_y_broad)

    if int_flag:
        res = te.lang.cce.floor(res)

    res = te.lang.cce.cast_to(res, dtype)

    return res


@util.check_input_type(dict, dict, dict, str)
def div(input_x, input_y, output_div, kernel_name="div"):
    """
    algorithm: div
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")
    shape_y = input_y.get("shape")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)
    shape_list = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_list[2])
    input_dtype = dtype.lower()
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    util.check_dtype_rule(input_dtype, check_list)

    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_list[0],
                                                       shape_list[1])
    data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")

    res = div_compute(data_x, data_y, output_div, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
