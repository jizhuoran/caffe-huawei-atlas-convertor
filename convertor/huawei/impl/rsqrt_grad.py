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

rsqrt_grad
"""
from __future__ import absolute_import

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util

# define a scalar, value = -0.5
SCALAR = -0.5

# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("rsqrt_grad")
def rsqrt_grad_compute(input_y, input_dy, output_z, kernel_name="rsqrt_grad"):
    """
    compute for rsqrt_grad

    Parameters
    ----------
    input_y: TVM tensor
        the placeholder of input_y
    input_dy: TVM tensor
        the placeholder of input_dy
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "rsqrt_grad"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_input_y = input_y.dtype
    rsqrt_const = tvm.const(SCALAR, dtype=dtype_input_y)
    if dtype_input_y in ("int8", "float16"):
        rsqrt_const = tvm.const(SCALAR, dtype="float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")
        input_dy = te.lang.cce.cast_to(input_dy, "float32")
    res_vmul = te.lang.cce.vmul(input_y, input_y)
    res_vmul1 = te.lang.cce.vmul(res_vmul, input_y)
    res_vmul2 = te.lang.cce.vmul(res_vmul1, input_dy)
    res = te.lang.cce.vmuls(res_vmul2, rsqrt_const)
    if dtype_input_y in ("int8", "int32", "float16"):
        res = te.lang.cce.cast_to(res, dtype_input_y, f1628IntegerFlag=True)
    return res


@util.check_input_type(dict, dict, dict, str)
def rsqrt_grad(input_y, input_dy, output_z, kernel_name="rsqrt_grad"):
    """
    calculate the backpropagation of rsqrt operation
    rsqrt: y = 1 / sqrt（x）
    rsqrt_grad: -1/2 * y**3 *dy

    Parameters
    ----------
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    input_dy: dict
        dict of input_dy, include keys(shape and dtype)
    output_z: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "rsqrt_grad"

    Returns
    -------
    None
    """
    shape_input_y = input_y.get("shape")
    dtype_input_y = input_y.get("dtype")
    shape_input_dy = input_dy.get("shape")
    dtype_input_dy = input_dy.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input_y)
    util.check_shape_rule(shape_input_dy)
    util.check_tensor_shape_size(shape_input_y)
    util.check_tensor_shape_size(shape_input_dy)
    util.compare_tensor_dict_key(input_y, input_dy, "shape")

    check_list = ("float16", "float32", "int32", "int8")
    dtype_input_y = dtype_input_y.lower()
    util.check_dtype_rule(dtype_input_y, check_list)
    dtype_input_dy = dtype_input_dy.lower()
    util.check_dtype_rule(dtype_input_dy, check_list)
    util.compare_tensor_dict_key(input_y, input_dy, "dtype")
    reshape_y, reshape_dy = refine_shapes_for_broadcast(shape_input_y,
                                                        shape_input_dy)

    data_input_y = tvm.placeholder(reshape_y,
                                   name="data_input_y",
                                   dtype=dtype_input_y)
    data_input_dy = tvm.placeholder(reshape_dy,
                                    name="data_input_dy",
                                    dtype=dtype_input_dy)

    res = rsqrt_grad_compute(data_input_y, data_input_dy, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_y, data_input_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
