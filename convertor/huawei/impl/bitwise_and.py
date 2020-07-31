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

bitwise_and
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("bitwise_and")
def bitwise_and_compute(x1, x2, y, kernel_name="bitwise_and"):
    """
    calculating data's bitwise and
    res = x & y

    Parameters
    ----------
    x1 : tvm tensor
              input data x1
    x2 : tvm tensor
              input data x2
    y : dict
               the shape and dtype of the tensor y
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    res : output of the data's bitwise and
    """
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)

    data_x = te.lang.cce.broadcast(x1, shape_max)
    data_y = te.lang.cce.broadcast(x2, shape_max)

    res = te.lang.cce.vand(data_x, data_y)

    return res


def _check_parameters(x1, x2, y, kernel_name):
    """
    check the input parameters
    return the shape and data type of x1 and x2
    """
    util.check_kernel_name(kernel_name)

    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    dtype_z = y.get("dtype").lower()

    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)

    check_tuple = ("int16", "uint16")
    util.check_dtype_rule(dtype_x, check_tuple)
    util.check_dtype_rule(dtype_y, check_tuple)
    util.check_dtype_rule(dtype_z, check_tuple)
    if dtype_x != dtype_y:
        raise RuntimeError(
            "two input type must be the same")

    return shape_x, shape_y, dtype_x


@util.check_input_type(dict, dict, dict, str)
def bitwise_and(x1, x2, y, kernel_name="bitwise_and"):
    """
    algorithm: bitwise_and
    computes the bitwise and of `x1` and `x2`

    Parameters
    ----------
    x1 : dict
              the shape and dtype of the tensor x1, only support int16,uint16
    x2 : dict
              the shape and dtype of the tensor x2, only support int16,uint16
    y : dict
              the shape and dtype of the tensor y, only support int16,uint16
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    None
    """
    shape_x, shape_y, dtype = _check_parameters(x1, x2, y, kernel_name)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_max)
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype)
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype)

    res = bitwise_and_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": (data_x, data_y, res)}
    te.lang.cce.cce_build_code(schedule, config)
