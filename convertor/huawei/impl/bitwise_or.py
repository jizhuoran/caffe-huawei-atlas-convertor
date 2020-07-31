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

bitwise_or
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,invalid-name
@fusion_manager.register("bitwise_or")
def bitwise_or_compute(placeholders, shape_x, shape_y):
    """
    calculating data's element_or, c = a | b

    Parameters
    ----------
    placeholders : tuple of data
    shape_x: list of int
            shape of input_x
    shape_y: list of int
            shape of input_y

    Returns
    -------
    res : z of the data's bitwise_or
    """
    data_x = placeholders[0]
    data_y = placeholders[1]
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    data_x_broadcast = te.lang.cce.broadcast(data_x, shape_max)
    data_y_broadcast = te.lang.cce.broadcast(data_y, shape_max)
    res = te.lang.cce.vor(data_x_broadcast, data_y_broadcast)

    return res


@util.check_input_type(dict, dict, dict, str, bool, bool)
def bitwise_or(x1, x2, y, kernel_name="bitwise_or",):
    """
    algorithm: bitwise_or
    calculating data's bitwise_or, c = a | b

    Parameters
    ----------
    x1: dict
              shape and dtype of data_1
    x2: dict
              shape and dtype of data_2
    y: dict
              shape and dtype of y
    kernel_name : string
                  cce kernel name, default value is "bitwise_or"

    Returns
    -------
    None
    """
    input_x = x1.get("shape")
    input_y = x2.get("shape")
    dtype_x = x1.get("dtype")
    dtype_y = x2.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_x)
    util.check_shape_rule(input_y)
    util.check_tensor_shape_size(input_x)
    util.check_tensor_shape_size(input_y)

    check_tuple = ("int16", "uint16")
    input_data_type = dtype_x.lower()
    util.check_dtype_rule(input_data_type, check_tuple)

    if dtype_x != dtype_y:
        raise RuntimeError("The type of input must be the same")

    shape_x, shape_y, shape_max = util.produce_shapes(input_x, input_y)
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)

    data_x = tvm.placeholder(shape_x, dtype=input_data_type, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=input_data_type, name="data_y")
    res = bitwise_or_compute((data_x, data_y), shape_x, shape_y)
    y = {'shape': res.shape, 'dtype': input_data_type}

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_x, data_y, res)}

    te.lang.cce.cce_build_code(schedule, config)
