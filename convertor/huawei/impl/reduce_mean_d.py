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

tf reduce mean
"""

from collections import Iterable

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 100000000  # shape limit for tf_reduce_mean

NoneType = type(None)


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("reduce_mean_d")
def reduce_mean_d_compute(x,
                          y,
                          axes,
                          keepdims,
                          kernel_name="reduce_mean_d",
                          is_5hdc=False):
    """reduce_mean_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axes: int, list, tuple or NoneType
        the axes for reduce.
    keepdims: bool or NoneType
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_mean_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = util.axis_check(shape_len, axes)

    reduce_elts = 1.0
    if isinstance(axes, Iterable):
        for i in axes:
            reduce_elts *= shape[i]
    else:
        reduce_elts = shape[axes]
    cof = reduce_elts**(-1)

    dtype = x.dtype
    if dtype in ("int8", "uint8"):
        data_input_tmp = te.lang.cce.vmuls(x, 1.0)
        data_input_tmp = te.lang.cce.cast_to(data_input_tmp, "float16")
    else:
        data_input_tmp = x

    has_improve_precision = False
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if cce_product not in ("Ascend310",) and dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support(
                "te.lang.cce.sum", "float32") and not is_5hdc:
        data_input_tmp = te.lang.cce.cast_to(data_input_tmp, "float32")
        has_improve_precision = True

    data_input_tmp = te.lang.cce.vmuls(data_input_tmp, cof)
    res = te.lang.cce.sum(data_input_tmp, axis=axes, keepdims=keepdims)

    if dtype in ("int8", "uint8"):
        res = te.lang.cce.cast_to(res, dtype, False)
    if has_improve_precision:
        res = te.lang.cce.cast_to(res, dtype)

    return res

@util.check_input_type(dict, dict, (int, list, tuple, NoneType),
                       (bool, NoneType), str)
def reduce_mean_d(input_x, output_y, axes,
                  keepdims=None, kernel_name="reduce_mean_d"):
    """
    Reduce a tensor on a certa in axes based on mean.

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input
    output_y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keepdims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_mean_d

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    check_list = ["float16", "float32", "int8", "uint8"]
    shape_len = len(shape)

    if not axes:
        axes = range(shape_len)

    if hasattr(axes, 'index'):
        axes = list(axes)

    inp_dtype = input_x.get("dtype").lower()
    util.check_dtype_rule(inp_dtype, check_list)

    axes = util.axis_check(shape_len, axes)
    util.check_tensor_shape_size(shape)

    # Shape should not be modified in 5HD mode
    # 5HD Special param for 5hd schedule
    is_5hdc = util.check_and_init_5hdc_reduce_support(input_x, axes, kernel_name)
    if not is_5hdc:
        shape, axes = util.shape_refine(list(shape), axes)
        shape, axes = util.simplify_axis_shape(shape, axes)

    data_input = tvm.placeholder(shape, name="data_input_" + kernel_name, dtype=inp_dtype)
    res = reduce_mean_d_compute(data_input, output_y, axes,
                                keepdims, is_5hdc=is_5hdc)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
