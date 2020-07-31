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

neg
"""
from __future__ import absolute_import

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce

# define a scaler, value = -1
SCALER_NEGATIVE_ONE = -1
# define a float scaler, value = -1.0
SCALER_NEGATIVE_ONE_FLOAT = -1.0


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("neg")
def neg_compute(input_x, output_y, kernel_name="neg"):
    """
    compute neg

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "neg"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype.lower()
    shape = te.lang.cce.util.shape_to_list(input_x.shape)

    if dtype == "int32":
        data_tmp = te.lang.cce.broadcast(SCALER_NEGATIVE_ONE, shape)
        res = te.lang.cce.vmul(input_x, data_tmp)
    else:
        res = te.lang.cce.vmuls(input_x, SCALER_NEGATIVE_ONE_FLOAT)

    if dtype == "int8":
        res = te.lang.cce.cast_to(res, dtype, f1628IntegerFlag=True)

    return res


@util.check_input_type(dict, dict, str)
def neg(input_x, output_y, kernel_name="neg"):
    """
    Computes numerical negative value element-wise, y = -x.

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same type as input
    kernel_name: str
        kernel name, default value is "neg"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input)
    util.check_tensor_shape_size(shape_input)

    dtype_input = dtype_input.lower()
    check_list = ("float16", "float32", "int32", "int8")
    util.check_dtype_rule(dtype_input, check_list)

    shape_input = util.shape_refine(shape_input)
    shape_input = (functools_reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(shape_input, name="data_input", dtype=dtype_input)

    res = neg_compute(data_input, output_y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
