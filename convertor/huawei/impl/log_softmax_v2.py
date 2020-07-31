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

log_softmax_v2
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform

# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable = locally-disabled,unused-argument
@fusion_manager.register("log_softmax_v2")
def log_softmax_v2_compute(input_x, output_y, axis=-1, kernel_name="log_softmax_v2"):
    """
    process of calculating data's log_softmax, x - log(sum(exp(x)))
    this x is x - xmax

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output: dict
        shape and dtype of output, should be same shape and type as input
    axis: int, list or tuple
        the data's axis, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    result: TVM tensor.
    """
    inp_dtype = input_x.dtype
    shape = te.lang.cce.util.shape_to_list(input_x.shape)

    data_max = te.lang.cce.reduce_max(input_x, axis=axis, keepdims=True)
    data_max_broadcast = te.lang.cce.broadcast(data_max, shape)
    data_sub = te.lang.cce.vsub(input_x, data_max_broadcast)

    # increase accuracy
    has_improve_precision = False
    if inp_dtype == "float16" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                "float32"):
        data_sub = te.lang.cce.cast_to(data_sub, "float32")
        has_improve_precision = True

    data_exp = te.lang.cce.vexp(data_sub)
    data_sum = te.lang.cce.sum(data_exp, axis=axis, keepdims=True)
    data_log = te.lang.cce.vlog(data_sum)
    data_log_broadcast = te.lang.cce.broadcast(data_log, shape)
    res = te.lang.cce.vsub(data_sub, data_log_broadcast)

    # cast output type same as input type
    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, (int, list, tuple), str)
def log_softmax_v2(input_x, output_y, axis=-1, kernel_name="log_softmax_v2"):
    """
    algorithm: log_softmax
    calculating data's log_softmax, x - log(sum(exp(x)))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis: int, list or tuple
        the data's axis, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")
    shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    shape_len = len(shape)
    shape_list = list(shape)

    if not isinstance(axis, int):
        axis = list(axis)

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_dtype_rule(input_dtype, check_list)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    axis = util.axis_check(shape_len, axis)

    if not isinstance(axis, int):
        for i in axis:
            if shape_list[i] == 1:
                raise RuntimeError("Cannot reduce on an axis with dimension 1")
    else:
        if shape_list[axis] == 1:
            raise RuntimeError("Cannot reduce on an axis with dimension 1")

    shape, axis = util.shape_refine(list(shape), axis)
    shape, axis = util.simplify_axis_shape(shape, axis)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    result = log_softmax_v2_compute(data_input, output_y, axis=axis,
                                    kernel_name=kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input, result]}

    te.lang.cce.cce_build_code(sch, config)
