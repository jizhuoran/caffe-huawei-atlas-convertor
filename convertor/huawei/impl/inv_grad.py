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

inv_grad
"""
from __future__ import absolute_import

from te import tvm
import te.lang.cce
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util

# define a scalar , value = -1
SCALAR_NEGATIVE_ONE = -1


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("inv_grad")
def inv_grad_compute(input_y, input_dy, output_z, kernel_name="inv_grad"):
    """
    compute inv_grad

    Parameters
    ----------
    input_y: TVM tensor
        the placeholder of input y
    input_dy: TVM tensor
        the placeholder of input dy
    output_z: TVM tensor
        shape and dtype of output
    kernel_name: str
        kernel name, default value is "inv_grad"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    dtype = input_y.dtype

    inv_const = tvm.const(SCALAR_NEGATIVE_ONE, dtype=dtype)
    has_improve_precision = False
    if dtype in ("float16", "int8"):
        if tbe_platform.cce_conf.api_check_support("te.lang.cce.vmuls",
                                                   "float32"):
            inv_const = tvm.const(SCALAR_NEGATIVE_ONE, dtype="float32")
            input_y = te.lang.cce.cast_to(input_y, "float32")
            input_dy = te.lang.cce.cast_to(input_dy, "float32")
            has_improve_precision = True
        const_res = te.lang.cce.vmuls(input_y, inv_const)
    elif dtype in ("int32",):
        inv_const = te.lang.cce.broadcast(inv_const, shape_y, "int32")
        const_res = te.lang.cce.vmul(inv_const, input_y)
    else:
        const_res = te.lang.cce.vmuls(input_y, inv_const)
    vmul_res = te.lang.cce.vmul(const_res, input_y)
    res = te.lang.cce.vmul(vmul_res, input_dy)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, dtype, f1628IntegerFlag=True)

    return res


@util.check_input_type(dict, dict, dict, str)
def inv_grad(input_y, input_dy, output_z, kernel_name="inv_grad"):
    """
    algorithm: inv_grad
    calculating data's reciprocal grad,dx = -1*dy*y*y, where `y = 1/x`, and `dy`
    is the corresponding input gradient.

    Parameters
    ----------
    input_y: dict
        shape and dtype of input_y, only support float16, float32, int32, int8
    input_dy: dict
        shape and dtype of input_dy, should be same shape and type as input_y
    output_z: dict
        shape and dtype of output, should be same shape and type as input_y
    kernel_name: str
        kernel name, default value is "inv_grad"

    Returns
    -------
    None
    """
    shape_input_y = input_y.get("shape")
    shape_input_dy = input_dy.get("shape")
    dtype_input_y = input_y.get("dtype")
    dtype_input_dy = input_dy.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input_y)
    util.check_shape_rule(shape_input_dy)
    util.check_tensor_shape_size(shape_input_y)
    util.check_tensor_shape_size(shape_input_dy)

    shape_input_y = util.shape_refine(shape_input_y)
    shape_input_dy = util.shape_refine(shape_input_dy)

    if list(shape_input_y) != list(shape_input_dy):
        raise RuntimeError("the shape of input must be equal!")

    dtype_input_y = dtype_input_y.lower()
    dtype_input_dy = dtype_input_dy.lower()

    if dtype_input_dy != dtype_input_y:
        raise RuntimeError("the dtype of input must be equal!")

    check_list = ("float16", "float32", "int32", "int8")
    util.check_dtype_rule(dtype_input_y, check_list)

    shape_input_dy, shape_input_y = refine_shapes_for_broadcast(shape_input_dy,
                                                                shape_input_y)
    data_dy = tvm.placeholder(shape_input_dy, name="data_dy",
                              dtype=dtype_input_dy)
    data_y = tvm.placeholder(shape_input_y, name="data_y", dtype=dtype_input_y)

    res = inv_grad_compute(data_y, data_dy, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
