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

diag_part_d
"""
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

# define a VALUE, value = 2
VALUE_TWO = 2

# pylint: disable=locally-disabled,unused-argument,invalid-name,no-member
@fusion_manager.register("diag_part_d")
def diag_part_d_compute(x, assist, y, kernel_name="diag_part_d"):
    """
    compute for diag_part_d

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input diagonal
    assist: TVM tensor
        the placeholder of input help
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "diag_part_d"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    dtype_x = x.dtype

    res_vmul = te.lang.cce.vmul(x, assist)
    sum_dims = []
    len_output = len(shape_x) // VALUE_TWO
    for dims in range(len_output):
        sum_dims.append(dims + len_output)

    has_improve_precision = False
    if dtype_x == "int32" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.sum",
                                                    "float32"):
        res_vmul = te.lang.cce.cast_to(res_vmul, "float32")
        has_improve_precision = True

    res = te.lang.cce.sum(res_vmul, sum_dims)
    if has_improve_precision:
        res = te.lang.cce.cast_to_round(res, "int32")
    return res

# pylint: disable=too-many-locals
@util.check_input_type(dict, dict, dict, str)
def diag_part_d(x, assist, y, kernel_name="diag_part_d"):
    """
    Returns the batched diagonal part of a batched tensor

    Parameters
    ----------
    x: dict
        dict of x, include keys(shape and dtype)
    assist: dict
        dict of help Matrix, Its Diagonal Line value is 1 else value is 0
    y: dict
        dict of output
    kernel_name: str
        cce kernel name, default value is "diag_part_d"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    shape_assist = assist.get("shape")
    dtype_assist = assist.get("dtype")
    shape_y = y.get("shape")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_assist)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_assist)

    if len(shape_x) not in (2, 4, 6, 8):
        raise RuntimeError("Input tensors of rank 2,4,6,8 are supported!")
    if list(shape_x) != list(shape_assist):
        raise RuntimeError("the shape of data must be equal!")
    len_shape_out = len(shape_x) // VALUE_TWO
    for i in range(len_shape_out):
        if shape_x[i] != shape_x[i + len_shape_out]:
            raise RuntimeError("the shape of input is not supported!")
    if list(shape_x) != list(shape_y + shape_y):
        raise RuntimeError("the shape of output is not supported!")
    if list(shape_x) != list(shape_assist):
        raise RuntimeError("the shape of data must be equal!")

    check_list = ("float16", "float32", "int32")
    dtype_x = dtype_x.lower()
    util.check_dtype_rule(dtype_x, check_list)
    dtype_assist = dtype_assist.lower()
    util.check_dtype_rule(dtype_assist, check_list)
    if dtype_assist != dtype_x:
        raise RuntimeError("the dtype of data must be equal!")

    data_x = tvm.placeholder(shape_x, name="data_x",
                             dtype=dtype_x)
    data_assist = tvm.placeholder(shape_assist, name="data_assist",
                                  dtype=dtype_assist)

    res = diag_part_d_compute(data_x, data_assist, y,
                              kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_assist, res]}
    te.lang.cce.cce_build_code(sch, config)
