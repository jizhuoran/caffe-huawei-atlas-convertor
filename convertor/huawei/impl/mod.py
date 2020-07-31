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

mod
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util

# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("mod")
def mod_compute(input_x, input_y, output_z, kernel_name="mod"):
    """
    Returns element-wise remainder of division.
    the result here is consistent with a truncating divide.
    'truncate_mod(x, y) = x - truncate_div(x, y) * y'.

    Parameters
    ----------
    input_x: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32", "int8", "uint8", "int32".
    input_y: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_x'.
    output_z: dict
        data of output.
        Must have the same type as 'input_x'.
    kernel_name: str
        kernel name, default value is "mod"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_x".
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    dtype = input_x.dtype.lower()

    has_improve_precision = False
    if dtype != "float32" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")
        has_improve_precision = True

    if list(shape_x) != list(shape_y):
        shape_x, shape_y, shape_broadcast = util.produce_shapes(shape_x,
                                                                shape_y)
        input_x = te.lang.cce.broadcast(input_x, shape_broadcast, "float32")
        input_y = te.lang.cce.broadcast(input_y, shape_broadcast, "float32")
    else:
        shape_broadcast = shape_x

    data_div = te.lang.cce.vdiv(input_x, input_y)
    data_zero = te.lang.cce.broadcast(tvm.const(0, "float32"), shape_broadcast,
                                      "float32")
    data_div_min = te.lang.cce.vmin(data_div, data_zero)
    data_div_max = te.lang.cce.vmax(data_div, data_zero)
    data_div_max_floor = te.lang.cce.floor(data_div_max)
    data_div_min_ceil = te.lang.cce.ceil(data_div_min)

    if dtype != "int32" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul", "float32"):
        data_div_max_floor = te.lang.cce.cast_to(data_div_max_floor, "float32")
        data_div_min_ceil = te.lang.cce.cast_to(data_div_min_ceil, "float32")

    data_div_res = te.lang.cce.vadd(data_div_max_floor, data_div_min_ceil)
    data_mul = te.lang.cce.vmul(data_div_res, input_y)
    res = te.lang.cce.vsub(input_x, data_mul)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, dtype)

    return res


@util.check_input_type(dict, dict, dict, str)
def mod(input_x, input_y, output_z, kernel_name="mod"):
    """
    Returns element-wise remainder of division.

    Parameters
    ----------
    input_x: dict
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32", "int8", "uint8", "int32".
    input_y: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_x'.
    output_z: dict
        data of output.
        Must have the same type as 'input_x'.
    kernel_name: str
        kernel name, default value is "mod"

    Returns:
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")

    util.compare_tensor_dict_key(input_x, input_y, "dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)

    check_list = ("float16", "float32", "int8", "uint8", "int32")
    input_dtype = input_x.get("dtype").lower()
    util.check_dtype_rule(input_dtype, check_list)
    shape_x, shape_y, shape_broadcast = util.produce_shapes(shape_x, shape_y)

    util.check_tensor_shape_size(shape_broadcast)

    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")
    res = mod_compute(data_x, data_y, output_z, kernel_name="mod")

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}

    te.lang.cce.cce_build_code(sch, config)
