#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

squared_difference
"""

import te.lang.cce
from te import tvm
from topi import generic

from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable=locally-disabled,too-many-locals,invalid-name
@util.check_input_type(dict, dict, dict, str)
def squared_difference(x1, x2, y, kernel_name="squared_difference"):
    """
    algorithm: squared_difference

    calculating data's tf_squared_difference,y= (x - y) * (x - y)

    Parameters
    ----------
    x2 : dict
        shape and dtype of y input, only support float16, float32
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    output_x: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is squared_difference

    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32", "int32"]
    dtype = x1.get("dtype").lower()

    if not dtype in check_list:
        raise RuntimeError(
            "tf_squared_difference_cce only support float16, float32, int32")

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, dtype=dtype, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype, name="data_y")

    with tvm.target.cce():
        shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
        data_x_tmp = te.lang.cce.broadcast(data_x, shape_max)
        data_y_tmp = te.lang.cce.broadcast(data_y, shape_max)
        data_sub = te.lang.cce.vsub(data_x_tmp, data_y_tmp)
        res = te.lang.cce.vmul(data_sub, data_sub)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, data_y, res]}

    te.lang.cce.cce_build_code(sch, config)
