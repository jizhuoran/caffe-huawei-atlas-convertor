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

truncate_mod
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast

# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("truncate_mod")
def truncate_mod_compute(input_x, input_y, output_z,
                         kernel_name="truncate_mod"):
    """
    truncate_mod compute
    calculating data's truncatemod, res = x - truncate(x/y)*y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "truncate_mod"

    Returns
    -------
    res: TVM tensor
        the result of truncate_mod(input_x,input_y)
    """
    input_data_x = te.lang.cce.util.shape_to_list(input_x.shape)
    input_data_y = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_list = util.produce_shapes(input_data_x, input_data_y)
    util.check_tensor_shape_size(shape_list[2])
    dtype = input_x.dtype
    tran_x = te.lang.cce.cast_to(input_x, "float32")
    tran_y = te.lang.cce.cast_to(input_y, "float32")
    data_x_broad = te.lang.cce.broadcast(tran_x, shape_list[2])
    data_y_broad = te.lang.cce.broadcast(tran_y, shape_list[2])

    vdiv_data = te.lang.cce.vdiv(data_x_broad, data_y_broad)
    truncate_data = te.lang.cce.cast_to(vdiv_data, "int32")
    cast_data = te.lang.cce.cast_to(truncate_data, "float32")
    mul_data = te.lang.cce.vmul(cast_data, data_y_broad)
    sub_data = te.lang.cce.vsub(data_x_broad, mul_data)
    res = te.lang.cce.cast_to(sub_data, dtype)

    return res


@util.check_input_type(dict, dict, dict, str)
def truncate_mod(input_x, input_y, output_z, kernel_name="truncate_mod"):
    """
    algorithm: truncatemod
    calculating data's truncate, res = x - truncate(x/y)*y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "truncatemod"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)

    shape_list = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_list[2])
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    util.check_dtype_rule(dtype_x, check_list)
    util.check_dtype_rule(dtype_y, check_list)

    shape_x, shape_y = refine_shapes_for_broadcast(shape_list[0],
                                                   shape_list[1])
    data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_y")
    res = truncate_mod_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
