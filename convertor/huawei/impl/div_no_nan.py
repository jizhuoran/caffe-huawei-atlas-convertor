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

div_no_nan
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("div_no_nan")
def div_no_nan_compute(input_x, input_y, output_z, kernel_name="div_no_nan"):
    """
    div_no_nan_compute
    Returns 0 if the denominator is zero, else, like Div.
    ----------
    input_x: TVM tensor
        the placeholder of input tensor x
    input_y: TVM tensor
        the placeholder of input tensor y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name

    Returns
    -------
    res: TVM tensor
        the result of div_no_nan_compute
    """
    dtype = input_x.dtype
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_max)

    int_list = ("int32", "int8", "uint8")
    if dtype in int_list:
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")

    if dtype in ("float16",):
        help_min = tvm.const(2**(-24), "float16")
        help_rec_one = tvm.const(2**12, "float16")
        help_rec_sec = tvm.const(2**12, "float16")
    else:
        help_min = tvm.const(2**(-126), "float32")
        help_rec_one = tvm.const(2**38, "float32")
        help_rec_sec = tvm.const(2**44, "float32")

    cmp_help = te.lang.cce.broadcast(help_min, shape_y)
    y_cmp = te.lang.cce.vmul(input_y, input_y)
    y_index_help_1 = te.lang.cce.vmin(y_cmp, cmp_help)
    y_index_help_2 = te.lang.cce.vmuls(y_index_help_1, help_rec_one)
    y_index = te.lang.cce.vmuls(y_index_help_2, help_rec_sec)
    if dtype not in ("float16",):
        y_index = te.lang.cce.vmuls(y_index, help_rec_sec)

    data_x_broadcast = te.lang.cce.broadcast(input_x, shape_max)
    data_y_broadcast = te.lang.cce.broadcast(input_y, shape_max)
    index_y_broadcast = te.lang.cce.broadcast(y_index, shape_max)
    res_vdiv = te.lang.cce.vdiv(data_x_broadcast, data_y_broadcast)
    res = te.lang.cce.vmul(res_vdiv, index_y_broadcast)

    if dtype in int_list:
        res = te.lang.cce.floor(res)
        res = te.lang.cce.cast_to(res, dtype)

    return res


@util.check_input_type(dict, dict, dict, str)
def div_no_nan(input_x, input_y, output_z, kernel_name="div_no_nan"):
    """
    algorithm: div_no_nan_cce
    Returns 0 if the denominator is zero, else, like Div.
    Supports broadcasting.

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "div_no_nan"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_x.get("dtype")

    for shape in (shape_x, shape_y):
        util.check_shape_rule(shape)
        util.check_tensor_shape_size(shape)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_max)
    input_dtype = dtype.lower()
    util.check_dtype_rule(input_dtype, ("float16", "float32",
                                        "int32", "int8", "uint8"))
    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(reshape_x, name="data_x", dtype=input_dtype)
    data_y = tvm.placeholder(reshape_y, name="data_y", dtype=input_dtype)

    res = div_no_nan_compute(data_x, data_y, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
