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

sinh
"""
from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

# define a scaler , value = -1
SCALER_NEGATIVE_ONE = -1
# define a scaler , value = 0.5
SCALER_ZERO_POINT_FIVE = 0.5
# define a scaler , value = 2
SCALAR_TWO = 2

# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("sinh")
def sinh_compute(input_data, output_data, kernel_name="sinh"):
    """
    algorithm: sinh
    calculating data's sinh = (exp(x) - exp(-x)) / 2

    Parameters
    ----------
    input_data: TVM tensor
        data of input.
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "sinh"

    Returns
    -------
    res: TVM tensor
        the res of sinh
    """

    dtype = input_data.dtype
    shape = input_data.shape

    # in order to get the precise calcuate result
    has_improve_precision = False
    if dtype.lower() == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                    "float32"):
        input_data = te.lang.cce.cast_to(input_data, "float32")
        dtype = "float32"
        has_improve_precision = True

    data_mul = te.lang.cce.vmuls(input_data,
                                 tvm.const(SCALER_NEGATIVE_ONE, dtype))
    data_exp = te.lang.cce.vexp(data_mul)
    data_exp_x = te.lang.cce.vmuls(data_exp,
                                   tvm.const(SCALER_ZERO_POINT_FIVE,
                                             dtype))

    tensor_two = te.lang.cce.broadcast(tvm.const(SCALAR_TWO, dtype), shape)
    data_ln2 = te.lang.cce.vlog(tensor_two)
    data_neg_ln2 = te.lang.cce.vmuls(data_ln2, tvm.const(SCALER_NEGATIVE_ONE,
                                                         dtype))
    data_x = te.lang.cce.vadd(input_data, data_neg_ln2)
    data_exp_data = te.lang.cce.vexp(data_x)

    res = te.lang.cce.vsub(data_exp_data, data_exp_x)

    # cast the dtype to float16
    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, str)
def sinh(input_data, output_data, kernel_name="sinh"):
    """
    algorithm: sinh
    calculating data's sinh = (exp(x) - exp(-x)) / 2

    Parameters
    ----------
    input_data: dict
        shape and dtype of input, only support float16, float32
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "sinh"

    Returns
    -------
    None
    """
    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype")

    util.check_shape_rule(shape_input)
    util.check_tensor_shape_size(shape_input)
    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    input_dtype = dtype_input.lower()
    util.check_dtype_rule(input_dtype, check_list)

    reshape_input = (functools_reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(reshape_input,
                                 name="data_input", dtype=dtype_input)
    res = sinh_compute(data_input, output_data, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
