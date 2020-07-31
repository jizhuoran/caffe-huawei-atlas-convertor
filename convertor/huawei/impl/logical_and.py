#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

logical_and
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,invalid-name
@fusion_manager.register("logical_and")
def logical_and_compute(x1, x2, y, kernel_name="logical_and"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    x2: TVM tensor
        the placeholder of x2
    y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "logical_and"

    Returns
    -------
    output tensor
    """
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    _, _, shape_max = util.produce_shapes(shape_x, shape_y)

    x1 = te.lang.cce.cast_to(x1, "float16")
    x2 = te.lang.cce.cast_to(x2, "float16")

    data_x = te.lang.cce.broadcast(x1, shape_max)
    data_y = te.lang.cce.broadcast(x2, shape_max)

    res = te.lang.cce.vmul(data_x, data_y)

    res = te.lang.cce.cast_to(res, "int8", True)

    return res


@util.check_input_type(dict, dict, dict, str)
def logical_and(x1, x2, y, kernel_name="logical_and"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input, only support float16, float32
    x2 : dict
        shape and dtype of input, only support float16, float32
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "logical_and"

    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    dtype_x = x1.get("dtype")
    dtype_y = x2.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)

    if dtype_x != dtype_y:
        raise RuntimeError("The type of input must be the same")

    input_data_type = dtype_x.lower()
    check_tuple = ("int8",)
    util.check_dtype_rule(input_data_type, check_tuple)

    shape_x, shape_y, _ = util.produce_shapes(shape_x, shape_y)
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_y")

    res = logical_and_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_x, data_y, res)}

    te.lang.cce.cce_build_code(sch, config)
