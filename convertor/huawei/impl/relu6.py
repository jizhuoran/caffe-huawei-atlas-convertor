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

relu6
f(x) = min(max(0,x), 6)
"""

from functools import reduce as reduce_ins
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
import topi
from topi.cce import util
from te import platform as tbe_platform

SHAPE_SIZE_LIMIT = 2**30  # shape limit


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("relu6")
def relu6_compute(input_x, output_y, kernel_name="relu6"):
    """
    compute of relu6

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of first input data
    output_y: dict
        shape and dtype of output,should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "relu6"

    Returns
    -------
    compute result of relu6
    """
    tmp_res = te.lang.cce.vmaxs(input_x, tvm.const(0, input_x.dtype))
    final_res = te.lang.cce.vmins(tmp_res, tvm.const(6, input_x.dtype))

    return final_res


@util.check_input_type(dict, dict, str)
def relu6(input_x, output_y, kernel_name="relu6"):
    """
       f(x)= 6(x >= 6)
       f(x)= 0(x <= 0)
       f(x)= x(0<x<6)

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output_y, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "relu6"

    Returns
    ------
    None
    """
    input_shape = util.scalar2tensor_one(input_x.get("shape"))
    input_dtype = input_x.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)

    vmaxs_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmaxs", "float32")
    if input_dtype == "float32" and not vmaxs_support:
        raise RuntimeError(
            "Input dtype is float32, but do not support on the platform")

    # check input tensor data_type
    check_list = ("int32", "float16", "float32")
    if not input_dtype in check_list:
        raise RuntimeError("relu6 only support int32 float16 float32.")

    input_shape = [reduce_ins(lambda x, y: x * y, input_shape[:])]
    input_data = tvm.placeholder(input_shape,
                                 name="input_data",
                                 dtype=input_dtype)
    final_res = relu6_compute(input_data,
                              output_y,
                              kernel_name="relu6")

    with tvm.target.cce():
        auto_sch = topi.generic.auto_schedule(final_res)

    config = {"name": kernel_name, "tensor_list": (input_data, final_res)}
    te.lang.cce.cce_build_code(auto_sch, config)
