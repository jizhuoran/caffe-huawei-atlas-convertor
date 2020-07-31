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

tf reduce prod
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

NoneType = type(None)

SHAPE_SIZE_LIMIT = 100000000  # shape limit

# pylint: disable=locally-disabled, unused-argument
@fusion_manager.register("reduce_prod_d")
def reduce_prod_d_compute(data_input, output_y, axes,
                          keepdims, kernel_name="reduce_prod_d"):
    """
    Reduce a tensor on a certain axes based on product.

    Parameters:
    ----------
    data_input : dict
        shape and dtype of input
    output_y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keep_dims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_prod_d

    Returns
    -------
    res
    """
    shape = te.lang.cce.util.shape_to_list(data_input.shape)
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)

    inp_dtype = data_input.dtype

    res = te.lang.cce.reduce_prod(data_input, axes, keepdims)
    res = te.lang.cce.cast_to(res, inp_dtype, f1628IntegerFlag=True)
    return res

# pylint: disable=invalid-name
@util.check_input_type(dict, dict, (int, list, tuple, NoneType),
                       (bool, NoneType), str)
def reduce_prod_d(x, y, axes, keep_dims=None, kernel_name="reduce_prod_d"):
    """
    Reduce a tensor on a certain axes based on product.

    Parameters:
    ----------
    x : dict
        shape and dtype of input
    y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keep_dims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_prod_d
    Returns
    -------
    None
    """
    shape = x.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)

    inp_dtype = x.get("dtype").lower()
    check_list = ["float16", "float32", "int8", "uint8"]
    util.check_dtype_rule(inp_dtype, check_list)

    shape_len = len(shape)

    if not axes:
        axes = range(shape_len)

    if hasattr(axes, 'index'):
        axes = list(axes)

    axes = util.axis_check(shape_len, axes)
    util.check_reduce_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    shape, axes = util.shape_refine(list(shape), axes)
    shape, axes = util.simplify_axis_shape(shape, axes)

    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)
    with tvm.target.cce():
        res = reduce_prod_d_compute(data_input, y, axes, keep_dims, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
