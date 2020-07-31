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

zeros_like
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce

# pylint: disable=locally-disabled,invalid-name,unused-argument
@fusion_manager.register("zeros_like")
def zeros_like_compute(x, y, kernel_name="zeros_like"):
    """
    Enter a tensor, output a tensor of all zero,
    you can specify the output data type

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    y: TVM tensor
        the placeholder of output data
    kernel_name : str
        cce kernel name, default value is "zeros_like"

    Returns
    -------
    res: TVM tensor
        the result of zeros_like_compute
    """
    src_dtype = x.dtype.lower()
    dst_type = src_dtype
    src_type_list = ("int8", "uint8")
    dst_type_list = ("int8", "uint8")
    if src_dtype in src_type_list:
        src_dtype = "float16"

    zero = tvm.const(0, dtype=src_dtype)

    zero_src = te.lang.cce.broadcast(zero, x.shape)
    if src_dtype in dst_type_list:
        zero_src = te.lang.cce.cast_to(zero_src, dst_type,
                                       f1628IntegerFlag=True)
    else:
        zero_src = te.lang.cce.cast_to(zero_src, dst_type)
    with tvm.tag_scope("elewise_binary_phony"):
        res = te.tvm.compute(x.shape,
                             lambda *indices: zero_src[indices] + x[indices],
                             name="elewise_binary_phony_output")

    return res


@util.check_input_type(dict, dict, str)
def zeros_like(x, y, kernel_name="zeros_like"):
    """
    output a tensor of all zero, you can specify the output type

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32,
        int32,int8,uint8
    y: dict
        shape and dtype of output data
    kernel_name: str
        cce kernel name, default value is "zeros_like"

    Returns
    ------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)

    check_list_src = ("float16", "float32", "int32", "int8", "uint8")
    src_dtype = dtype_x.lower()
    util.check_dtype_rule(src_dtype, check_list_src)
    shape_x = (functools_reduce(lambda x, y: x * y, shape_x[:]),)
    x_input = tvm.placeholder(shape_x, name="x_input", dtype=src_dtype)
    res = zeros_like_compute(x_input, y, kernel_name=kernel_name)

    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [x_input, res]}
    te.lang.cce.cce_build_code(auto_sch, config)
