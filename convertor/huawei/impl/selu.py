#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

selu
if < 0:`scale * alpha * (exp(features) - 1)`
otherwise:`scale * features`
"""
from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# define selu oprator's required constants
ALPHA = 1.67326324235
SCALE = 1.05070098736
# define product of scale and alpha
SCALE_ALPHA_PRODUCT = 1.75809934085
# define a scalar, value = -1, the calculation of exp need minus one
SCALAR_NEGATIVE_ONE = -1


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("selu")
def selu_compute(input_x, y, kernel_name="selu"):
    """
    Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`
    if < 0, `scale * features` otherwise.
    alpha =  1.6732632423543772848170429916717
    scale =  1.0507009873554804934193349852946

    Parameters
    ----------
    input_x: TVM tensor
        input tensor has shape and dtype attributes
    y: TVM tensor
        outputtensor has shape and dtype attributes
    kernel_name : str
        cce kernel name, default value is "selu"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """
    # if input_dtype is float16,convert it to float32
    input_data = input_x
    dtype = input_data.dtype
    if dtype in ("float16", "float32"):
        input_data = te.lang.cce.cast_to(input_data, "float32")
        type_tmp = "float32"
    else:
        input_data = te.lang.cce.cast_to(input_data, "float16")
        type_tmp = "float16"

    # generate tensor_zero to be compared
    tensor_zero = te.lang.cce.vmuls(input_data, tvm.const(0, dtype=type_tmp))
    # generate negative_res and positive_res to compute
    # When the element value is greater than 0 and less than 0
    negative_res = te.lang.cce.vmin(input_data, tensor_zero)
    positive_res = te.lang.cce.vmax(input_data, tensor_zero)
    exp_res = te.lang.cce.vexp(negative_res)
    sub_res = te.lang.cce.vadds(exp_res, tvm.const(SCALAR_NEGATIVE_ONE,
                                                   dtype=type_tmp))
    negative_muls_res = te.lang.cce.vmuls(sub_res,
                                          tvm.const(SCALE_ALPHA_PRODUCT,
                                                    dtype=type_tmp))
    if dtype == "int8":
        negative_muls_res = te.lang.cce.ceil(negative_muls_res)

    positive_muls_res = te.lang.cce.vmuls(positive_res,
                                          tvm.const(SCALE, dtype=type_tmp))
    res = te.lang.cce.vadd(negative_muls_res, positive_muls_res)
    # if input_dtype is float16, has converted to float32,
    # output should convert back
    if dtype in ("float16", "int8", "int32"):
        res = te.lang.cce.cast_to(res, dtype)

    return res


@util.check_input_type(dict, dict, str)
def selu(x, y, kernel_name="selu"):
    """
    Generate selu_cce operator use selu_compute

    Parameters
    ----------
    x: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data, assume src_shape equals dst_shape,
        the data type, src_dtype equals dst_dtype,
         support fp16, fp32, int8, int32
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "selu"

    Returns
    ------
    None
    """
    # get dtype and shape attributes
    dtype = x.get("dtype")
    shape = x.get("shape")
    # check_kernel_name & shape
    input_dtype = dtype.lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    # check input tensor data_type
    check_list = ("float16", "float32", "int8", "int32")
    util.check_dtype_rule(input_dtype, check_list)

    reshape_input = (functools_reduce(lambda x, y: x * y, shape[:]),)
    input_data = tvm.placeholder(reshape_input, name="input_data",
                                 dtype=input_dtype)
    res = selu_compute(input_data, y, kernel_name)
    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data, res]}
    te.lang.cce.cce_build_code(auto_sch, config)
