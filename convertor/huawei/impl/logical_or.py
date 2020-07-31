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

logical_or
"""
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from topi import generic
from topi.cce import util

# pylint: disable=unused-argument,invalid-name
# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@fusion_manager.register("logical_or")
def logical_or_compute(x1, x2, y, kernel_name="logical_or"):
    """
    algorithm : logical_or_compute
    calculating the value of x1 OR x2 element-wise

    Parameters
    ----------
    x1 : the placeholders of x1

    x2 : the placeholders of x2

    y : the dict of y

    kernel_name : string, cce kernel name, default value is "logical_or"

    Returns
    -------
    result res
    """
    _, _, shape_max = util.produce_shapes(te.lang.cce.util.shape_to_list(
        x1.shape), te.lang.cce.util.shape_to_list(x2.shape))
    x1 = te.lang.cce.cast_to(x1, "float16")
    x2 = te.lang.cce.cast_to(x2, "float16")
    x1 = te.lang.cce.broadcast(x1, shape_max)
    x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vmax(x1, x2)
    res = te.lang.cce.cast_to(res, "int8")

    return res


@util.check_input_type(dict, dict, dict, str)
def logical_or(x1, x2, y, kernel_name="logical_or"):
    """
    algorithm : logical_or
    calculating the value of x1 OR x2 element-wise

    Parameters
    ----------
    x1 : the dict of x1,
         include shape and dtype,
         dtype support int8, the value only support 0, 1

    x2 : the dict of x2,
         include shape and dtype,
         dtype support int8, the value only support 0, 1

    y : the dict of y, include shape and dtype

    kernel_name : string, cce kernel name, default value is "logical_or"

    Returns
    -------
    None
    """

    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    dtype_x1 = x1.get("dtype")
    dtype_x2 = x2.get("dtype")
    if dtype_x1 == "bool" or dtype_x2 == "bool":
        dtype_x1 = "int8"
        dtype_x2 = "int8"

    util.check_kernel_name(kernel_name)
    check_shape(shape_x1)
    check_shape(shape_x2)

    check_tuple = ("int8",)
    check_dtype(dtype_x1, check_tuple)
    check_dtype(dtype_x2, check_tuple)

    shape_x1, shape_x2, shape_max = broadcast_shapes(shape_x1, shape_x2)
    dtype = dtype_x1.lower()
    check_shape(shape_max)
    data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype)
    data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype)

    res = logical_or_compute(data_x1, data_x2, y, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": (data_x1, data_x2, res)}
    te.lang.cce.cce_build_code(schedule, config)
