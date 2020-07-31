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

reduce_max
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.reduce_max_d_tik import reduce_max_d_tik

NoneType = type(None)


# pylint: disable=unused-argument,invalid-name,unexpected-keyword-arg
def reduce_max_d_compute(x, y, axis, keepdims, kernel_name="reduce_max_d"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input_x
    y : dict
        dict of output_y, include keys(shape and dtype)
    axis: list
        list axis to reduce
    keepdims: bool
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        kernel name, default value is "reduce_max_d"

    Returns
    -------
    output tensor
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    shape_len = len(shape)
    if not axis:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)

    inp_dtype = x.dtype

    res_tmp = te.lang.cce.reduce_max(x, axis=axis,
                                     keepdims=keepdims,
                                     priority_flag=True)
    res = te.lang.cce.cast_to(res_tmp, inp_dtype, f1628IntegerFlag=True)
    return res


# pylint: disable=invalid-name
@util.check_input_type(dict, dict, (int, list, tuple, NoneType),
                       (bool, NoneType), str)
def reduce_max_d(x, y, axis, keepdims=False, kernel_name="reduce_max_d"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    axis: list
        the first axis to reduce,may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
    keepdims: bool
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        kernel name, default value is "reduce_max_d"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    input_dtype = dtype.lower()

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)

    check_list = ["float16", "float32", "int8", "uint8", "int32"]
    util.check_dtype_rule(input_dtype, check_list)

    shape_len = len(shape)

    if not axis:
        axis = range(shape_len)

    if hasattr(axis, 'index'):
        axis = list(axis)

    axis = util.axis_check(shape_len, axis)

    util.check_tensor_shape_size(shape)

    # Shape should not be modified in 5HD mode
    # 5HD Special param for 5hd schedule
    is_5hdc = util.check_and_init_5hdc_reduce_support(x, axis, kernel_name)
    if not is_5hdc:
        shape, axis = util.shape_refine(list(shape), axis)
        shape, axis = util.simplify_axis_shape(shape, axis)
    shape_len = len(shape)
    x["shape"] = shape
    if input_dtype in ("float32", "int32") and len(axis) == 1 \
            and ((axis[0] == (shape_len - 1)) or (axis[0] == -1)):
        reduce_max_d_tik(x, y, axis[0], kernel_name)
    else:
        data_input = tvm.placeholder(shape,
                                     name="data_input_" + kernel_name,
                                     dtype=input_dtype)
        res = reduce_max_d_compute(data_input, y, axis,
                                   keepdims, kernel_name)
        with tvm.target.cce():
            sch = generic.auto_schedule(res)

        config = {"name": kernel_name, "tensor_list": [data_input, res]}

        te.lang.cce.cce_build_code(sch, config)
