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

diag_d
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

# pylint: disable = locally-disabled,invalid-name,unused-argument,no-member
@fusion_manager.register("diag_d")
def diag_d_compute(x, assit, y, kernel_name="diag_d"):
    """
    diag_d compute
    calculating diag_d(x,help):
    returns a diagonal tensor with a given x values.
    If the shape of x is [D1,...,Dk],the shape of diagonal tensor is
    [D1,...,Dk,D1,...,Dk]
    For example:
    x :    [1, 2, 3]
    res :  [[1, 0, 0]
            [0, 2, 0]
            [0, 0, 3]]

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x
    assit: TVM tensor
        the placeholder of assit
    y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "diag_d"

    Returns
    -------
    res: TVM tensor
        the result of diag compute
    """
    list_shape = te.lang.cce.util.shape_to_list(assit.shape)
    x_broad = te.lang.cce.broadcast(x, list_shape)
    res = te.lang.cce.vmul(x_broad, assit)

    return res

# pylint: disable =too-many-locals
@util.check_input_type(dict, dict, dict, str)
def diag_d(x, assist, y, kernel_name="diag_d"):
    """
    algorithm: diag_d
    calculating diag_d(x,help):
    returns a diagonal tensor with a given x values.
    If the shape of x is [D1,...,Dk],the shape of diagonal tensor is
    [D1,...,Dk,D1,...,Dk]
    For example:
    x :    [1, 2, 3]
    res :  [[1, 0, 0]
            [0, 2, 0]
            [0, 0, 3]]

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of x
    assist: dict
        dict with keys(shape and dtype) of assist
    y: dict
        dict with keys(shape and dtype) of y
    kernel_name: str
        kernel name, default value is "diag"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype = x.get("dtype")

    if len(shape_x) > 4:
        raise RuntimeError("the length of x.shape "
                           "should be less than 5")

    shape_help = assist.get("shape")
    dtype_help = assist.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    util.check_shape_rule(shape_help)
    util.check_tensor_shape_size(shape_help)

    check_list = ("float16", "float32", "int32")
    util.check_dtype_rule(dtype.lower(), check_list)
    util.check_dtype_rule(dtype_help.lower(), check_list)

    shape_list = util.produce_shapes(shape_x, shape_help)
    util.check_tensor_shape_size(shape_list[2])
    for i, element in enumerate(shape_x):
        if element != shape_help[i] or \
                element != shape_help[i + len(shape_x)] or \
                len(shape_help) != 2 * len(shape_x):
            raise RuntimeError(
                "shape mismatch of x and assist : "
                "the correct shapes should be "
                "x.shape = [D1,...,Dn],"
                "assist.shape = [D1,...,Dn,D1,...Dn]")
    shape_x, shape_y = refine_shapes_for_broadcast(shape_list[0], shape_list[1])
    data_x = tvm.placeholder(shape_x, dtype=dtype.lower(), name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_help.lower(),
                             name="data_y")
    res = diag_d_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
