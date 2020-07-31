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

bitwise_xor
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,too-many-locals
@fusion_manager.register("bitwise_xor")
def bitwise_xor_compute(x1, x2, y, kernel_name="bitwise_xor"):
    """
    calculating data's bitwise xor
    (x&y)|!(x|y)

    Parameters
    ----------
    x1 : tvm tensor
              input data x
    x2 : tvm tensor
              input data y
    y : dict
               the shape and dtype of the tensor
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    result : y of the data's bitwise xor
    """
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)

    data_x = te.lang.cce.broadcast(x1, shape_max)
    data_y = te.lang.cce.broadcast(x2, shape_max)

    data_and = te.lang.cce.vand(data_x, data_y)
    data_not = te.lang.cce.vnot(data_and)
    data_or = te.lang.cce.vor(data_x, data_y)
    result = te.lang.cce.vand(data_or, data_not)

    return result


@util.check_input_type(dict, dict, dict, str)
def bitwise_xor(x1, x2, y, kernel_name="bitwise_xor"):
    """
    algorithm: bitwise_xor
    calculating: gradient of bitwise_xor

    Parameters
    ----------
    x1 : dict
              the shape and dtype of the tensor x1
    x2 : dict
              the shape and dtype of the tensor x2
    y :  dict
              the shape and dtype of the tensor y
    kernel_name : string
                  cce kernel name, default value is "bitwise_xor"
    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    shape_z = y.get("shape")
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    dtype_z = y.get("dtype").lower()

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_shape_rule(shape_z)
    util.check_tensor_shape_size(shape_x,)
    util.check_tensor_shape_size(shape_y,)
    util.check_tensor_shape_size(shape_max)

    check_tuple = ("int16", "uint16")
    util.check_dtype_rule(dtype_x, check_tuple)
    util.check_dtype_rule(dtype_y, check_tuple)
    util.check_dtype_rule(dtype_z, check_tuple)
    if dtype_x != dtype_y:
        raise RuntimeError(
            "two input type must be the same")
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)

    data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_y")

    result = bitwise_xor_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {
        "name": kernel_name,
        "tensor_list": [data_x, data_y, result]}
    te.lang.cce.cce_build_code(sch, config)
