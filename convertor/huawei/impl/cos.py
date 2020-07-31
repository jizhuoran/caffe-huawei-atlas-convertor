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

cos
"""
from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

# 2pi, the cycle of cosin
TWO_PI = 2*3.14159265358979

# pylint: disable=locally-disabled, unused-argument
@fusion_manager.register("cos")
def cos_compute(input_x, output_y, kernel_name="cos"):
    """
    algorithm: cos
    calculating data's cos x = 1 - x^2/2! + x^4/4! + ... + (-1)^k*x^2k/(2k)!

    Parameters
    ----------
    input_x : TVM tensor
              data of input
    output_y: dict
              shape and dtype of output, should be same shape and type as input
    kernel_name: str
              kernel name, default value is "cos"

    Returns
    -------
    res : TVM tensor
          the result of cos
    """

    dtype = input_x.dtype
    shape = te.lang.cce.util.shape_to_list(input_x.shape)

    # cast to type float32 when type is float16
    has_improve_precision = False
    if dtype.lower() == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul",
                                                    "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    # round the input
    round_fp16 = te.lang.cce.round(te.lang.cce.vmuls(input_x, 1.0/TWO_PI))
    round_fp32 = te.lang.cce.cast_to(round_fp16, dtype)
    input_x_round = te.lang.cce.vsub(input_x,
                                     te.lang.cce.vmuls(round_fp32, TWO_PI))

    # the initial value one
    const_res = tvm.const(1.0, dtype=dtype)
    res = te.lang.cce.broadcast(const_res, shape)
    # compute the rank 2
    input_x_power = te.lang.cce.vmul(input_x_round, input_x_round)
    iter_value = te.lang.cce.vmuls(input_x_power, -1.0/2.0)
    res = te.lang.cce.vadd(res, iter_value)
    # compute the rank 4~14
    iter_list = (4, 6, 8, 10, 12, 14)
    for i in iter_list:
        iter_value = te.lang.cce.vmuls(te.lang.cce.vmul(input_x_power,
                                                        iter_value),
                                       -1.0/(i*(i-1)))
        res = te.lang.cce.vadd(res, iter_value)

    # cast the dtype to float16
    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, str)
def cos(input_x, output_y, kernel_name="cos"):
    """
    algorithm: cos
    calculating data's cos x = 1 - x^2/2! + x^4/4! + ... + (-1)^k*x^2k/(2k)!

    Parameters
    ----------
    input_x : dict
              shape and dtype of input, only support float16, float32
    output_y: dict
              shape and dtype of output, should be same shape and type as input
    kernel_name : str
              kernel name, default value is "cos"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype").lower()

    util.check_shape_rule(shape_input)
    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype_input, check_list)
    util.check_tensor_shape_size(shape_input)

    reshape_input = (functools_reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(reshape_input,
                                 name="data_input", dtype=dtype_input)
    res = cos_compute(data_input, output_y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
