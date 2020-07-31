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

truncate_div
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
@fusion_manager.register("truncate_div")
def truncate_div_compute(input_x, input_y, output_x,
                         kernel_name="truncate_div"):
    """
    compute truncate_div
    calculating data's truncate_div, res = floor(x / y) if x/y>0 else ceil(x/y)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data x
    input_y: TVM tensor
        the placeholder of input data y
    output_x: dict
        not used yet
    kernel_name: str
        kernel name

    Returns
    -------
    res: TVM tensor
        the result of truncate_div_compute
    """
    shape_list = util.produce_shapes(
        te.lang.cce.util.shape_to_list(input_x.shape),
        te.lang.cce.util.shape_to_list(input_y.shape))
    util.check_tensor_shape_size(shape_list[2])
    int_list = ("int32", "int8", "uint8")
    input_dtype = input_x.dtype

    if input_dtype in int_list:
        data_zero = te.lang.cce.broadcast(tvm.const(0, 'float32'),
                                          shape_list[2], 'float32')
        data_x_broad = te.lang.cce.cast_to(input_x, 'float32')
        data_y_broad = te.lang.cce.cast_to(input_y, 'float32')
        data_x_broad = te.lang.cce.broadcast(data_x_broad, shape_list[2])
        data_y_broad = te.lang.cce.broadcast(data_y_broad, shape_list[2])
        res_div = te.lang.cce.vdiv(data_x_broad, data_y_broad)
        res_min_int = te.lang.cce.ceil(te.lang.cce.vmin(res_div, data_zero))
        res_max_int = te.lang.cce.floor(te.lang.cce.vmax(res_div, data_zero))
        res_trunc = te.lang.cce.vadd(res_min_int, res_max_int)
    else:
        if tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                   "float32"):
            input_x = te.lang.cce.cast_to(input_x, 'float32')
            input_y = te.lang.cce.cast_to(input_y, 'float32')
        data_x_broad = te.lang.cce.broadcast(input_x, shape_list[2])
        data_y_broad = te.lang.cce.broadcast(input_y, shape_list[2])
        res_trunc = te.lang.cce.vdiv(data_x_broad, data_y_broad)

    res = te.lang.cce.cast_to(res_trunc, input_dtype)

    return res


@util.check_input_type(dict, dict, dict, str)
def truncate_div(input_x, input_y, output_x, kernel_name="truncate_div"):
    """
    algorithm: truncate_div
    calculating data's truncate_div, res = floor(x / y) if x/y>0 else ceil(x/y)

    Parameters
    ----------
    input_x: dict with keys(shape and dtype)
        only support {float16, float32, int8, uint8(on mini)},
        {float16, float32(on cloud)}
    input_y: dict with keys(shape and dtype)
        dict info of input_y
    output_x: dict with keys(shape and dtype)
        dict info of output_x
    kernel_name: str
        kernel name, default value is "truncate_div"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_x.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)

    input_dtype = dtype.lower()
    check_list = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(input_dtype, check_list)

    shape_list = util.produce_shapes(shape_x, shape_y)
    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_list[0],
                                                       shape_list[1])
    data1 = tvm.placeholder(reshape_x, dtype=input_dtype, name="data1")
    data2 = tvm.placeholder(reshape_y, dtype=input_dtype, name="data2")
    res = truncate_div_compute(data1, data2, output_x, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data1, data2, res]}
    te.lang.cce.cce_build_code(sch, config)
