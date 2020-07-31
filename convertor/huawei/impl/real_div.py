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

real_div
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("real_div")
def real_div_compute(x1, x2, y, kernel_name="real_div"):
    """
    calculating data's realdiv, c = a / b

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is real_div

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x = te.lang.cce.broadcast(x1, shape_max)
    data_y = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vdiv(data_x, data_y)

    return res


@util.check_input_type(dict, dict, dict, str)
def real_div(x1, x2, y, kernel_name="real_div"):
    """
    algorithm: real_div
    calculating data's real_div, c = a / b

    Parameters
    ----------
    x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is real_div

    Returns
    -------
    None
    """
    shape_x = util.scalar2tensor_one(x1.get("shape"))
    shape_y = util.scalar2tensor_one(x2.get("shape"))
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    check_tuple = ("float16", "float32")
    input_data_type = x1.get("dtype").lower()
    util.check_dtype_rule(input_data_type, check_tuple)
    input_data_type_x2 = x2.get("dtype").lower()
    util.check_dtype_rule(input_data_type_x2, check_tuple)

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_data_type)
    data_y = tvm.placeholder(shape_y, name="data_y", dtype=input_data_type)

    res = real_div_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": (data_x, data_y, res)}

    te.lang.cce.cce_build_code(schedule, config)
