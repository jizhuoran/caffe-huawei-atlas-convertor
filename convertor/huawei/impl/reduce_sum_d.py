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

reduce_sum_d
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
# define the type of None
NONETYPE = type(None)
# define the limit of shape dim
MAX_SHAPE_NUM = 10000000


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("reduce_sum_d")
def reduce_sum_d_compute(x,
                         y,
                         axis,
                         keepdims,
                         kernel_name="reduce_sum_d",
                         is_5hdc=False):
    """redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    shape_len = len(shape)
    if not axis:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)

    dtype = x.dtype
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if cce_product not in ("Ascend310",) and dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support(
                "te.lang.cce.sum", "float32") and not is_5hdc:
        x = te.lang.cce.cast_to(x, "float32")
    res_sum = te.lang.cce.sum(x, axis=axis, keepdims=keepdims)
    res = te.lang.cce.cast_to(res_sum, dtype)

    return res


# pylint: disable=locally-disabled,too-many-locals
@util.check_input_type(dict, dict, (int, list, tuple, NONETYPE),
                       (bool, NONETYPE), str)
def reduce_sum_d(x, y, axis, keepdims=None, kernel_name="reduce_sum_d"):
    """reduce a tensor on a certain axis based on sum.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")

    util.check_shape_rule(shape, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape)
    util.check_dtype_rule(dtype_lower, check_list)
    util.check_kernel_name(kernel_name)

    axis_d = []
    shape_len = len(shape)
    if not axis:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    else:
        axis_d = list(axis)
    axis_d = util.axis_check(shape_len, axis_d)

    data_input = tvm.placeholder(shape, name="data_input_" + kernel_name,
                                 dtype=dtype_lower)
    # 5HD Special param for 5hd schedule
    is_5hdc = util.check_and_init_5hdc_reduce_support(x, axis, kernel_name)
    res = reduce_sum_d_compute(data_input, y, axis_d, keepdims,
                               is_5hdc=is_5hdc)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
