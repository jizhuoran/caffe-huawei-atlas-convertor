"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

euclidean_norm_d
"""

import te.lang.cce
from te import tvm
from te.utils.op_utils import *
from te.platform.fusion_manager import fusion_manager
from topi import generic


@fusion_manager.register("euclidean_norm_d")
def euclidean_norm_d_compute(x,
                             dtype,
                             y,
                             axes,
                             keepdims,
                             kernel_name="euclidean_norm_d"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input_x
    dtype : string
        the type of input_x
    y : dict
        dict of output_y, include keys(shape and dtype)
    axes: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "euclidean_norm_d"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    if dtype in ("int32"):
        x = te.lang.cce.cast_to(x, "float16")

    res_mul = te.lang.cce.vmul(x, x)
    res_sum = te.lang.cce.sum(res_mul, axis=axes, keepdims=keepdims)
    res = te.lang.cce.vsqrt(res_sum)

    if dtype in ("int32"):
        res = te.lang.cce.cast_to(res, dtype)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_LIST_INT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def euclidean_norm_d(input_data,
                     output_data,
                     axes=None,
                     keepdims=False,
                     kernel_name="euclidean_norm_d"):
    """
    calculating data

    Parameters
    ----------
    input_data : dict
        shape and dtype of input
    output_data : dict
        shape and dtype of output, should be same format and type as input
    axes : int, list ,tuple or None.
        the first axes to reduce, may be negative to index from the end
    keepdims : bool or None .
        if true, retains reduced dimensions with length 1,
        default value is None

    Returns
    -------
    None
    """
    shape = input_data.get("shape")
    dtype = input_data.get("dtype")
    input_dtype = dtype.lower()

    check_list = ["float16", "float32", "int32"]
    check_dtype(input_dtype, check_list)

    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    wrap_axes_to_positive(axes, shape_len)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = euclidean_norm_d_compute(data_input, input_dtype, output_data, axes,
                                   keepdims, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)
