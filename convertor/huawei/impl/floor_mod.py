#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

floor_mod
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform

# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@fusion_manager.register("floor_mod")
def floor_mod_compute(x1, x2, y, kernel_name="floor_mod"):
    """
    Compute remainder of division
    res = x1 -floor(input_data_x / input_data_y)* input_data_y

    Parameters
    ----------
    x1: TVM tensor
        input tensor has shape and dtype attributes
    x2: TVM tensor
        input tensor has shape and dtype attributes
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """
    # calculate result, using float32 for better precision
    dtype = x1.dtype
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    shape_x, shape_y, shape = util.produce_shapes(shape_x, shape_y)

    has_improve_precision = False
    input_x_fp32 = x1
    input_y_fp32 = x2
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        input_x_fp32 = te.lang.cce.cast_to(x1, "float32")
        input_y_fp32 = te.lang.cce.cast_to(x2, "float32")
        has_improve_precision = True

    input_x_fp32 = te.lang.cce.broadcast(input_x_fp32, shape)
    input_y_fp32 = te.lang.cce.broadcast(input_y_fp32, shape)

    res = te.lang.cce.vdiv(input_x_fp32, input_y_fp32)

    res = te.lang.cce.floor(res)

    if dtype != "int32":
        if has_improve_precision:
            res = te.lang.cce.cast_to(res, "float32")
        else:
            res = te.lang.cce.cast_to(res, "float16")
        res = te.lang.cce.vmul(res, input_y_fp32)
        res = te.lang.cce.vsub(input_x_fp32, res)
        if has_improve_precision:
            res = te.lang.cce.cast_to(res, dtype)
    else:
        x2_broad = te.lang.cce.broadcast(x2, shape)
        x1_broad = te.lang.cce.broadcast(x1, shape)
        res = te.lang.cce.vmul(res, x2_broad)
        res = te.lang.cce.vsub(x1_broad, res)

    return res


@util.check_input_type(dict, dict, dict, str)
def floor_mod(x1, x2, y, kernel_name="floor_mod"):
    """
    calculate the remainder of division, support fp16,fp32,int32
    res = x1 -floor(input_data_x / input_data_y)* input_data_y

    Parameters
    ----------
    x1: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32
    x2: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32
    y: dict, reserved field
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    None
    """
    # get dtype and shape attributes
    dtype_x = x1.get("dtype").lower()
    shape_x = x1.get("shape")
    dtype_y = x2.get("dtype").lower()
    shape_y = x2.get("shape")

    # check_kernel_name & shape
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)

    # check input tensor data_type
    check_list = ("float16", "float32", "int32")
    util.check_dtype_rule(dtype_x, check_list)
    util.check_dtype_rule(dtype_y, check_list)

    if dtype_x != dtype_y:
        raise RuntimeError("the type of dtype in two dict is not the same")

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_max)
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)

    input_data_x = tvm.placeholder(shape_x, name="input_data_x", dtype=dtype_x)
    input_data_y = tvm.placeholder(shape_y, name="input_data_y", dtype=dtype_y)
    res = floor_mod_compute(input_data_x, input_data_y, y, kernel_name)
    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, input_data_y, res]}
    te.lang.cce.cce_build_code(auto_sch, config)
