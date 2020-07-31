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

reciprocal_grad
"""
from te import tvm
from te import platform as tbe_platform
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

# define a scaler , value = -1
SCALER_NEGATIVE_ONE = -1


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("reciprocal_grad")
def reciprocal_grad_compute(input_y, input_dy, output_data,
                            kernel_name="reciprocal_grad"):
    """
    compute reciprocal_grad

    Parameters
    ----------
    input_y: TVM tensor
        the placeholder of input y
    input_dy: TVM tensor
        the placeholder of input dy
    output_data: TVM tensor
        shape and dtype of output
    kernel_name: str
        kernel name, default value is "reciprocal_grad"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    dtype = input_y.dtype

    reciprocal_const = tvm.const(SCALER_NEGATIVE_ONE, dtype=dtype)
    is_cast = False

    if dtype in ("int32",):
        reciprocal_const = te.lang.cce.broadcast(reciprocal_const,
                                                 shape_y, "int32")
        const_res = te.lang.cce.vmul(reciprocal_const, input_y)
    if dtype == "float32" and tbe_platform.cce_conf.\
            api_check_support("te.lang.cce.vmuls", "float32"):
        const_res = te.lang.cce.vmuls(input_y, reciprocal_const)
    if dtype in ("float16", "int8") and tbe_platform.cce_conf.\
            api_check_support("te.lang.cce.vmuls", "float32"):
        is_cast = True
        reciprocal_const = tvm.const(SCALER_NEGATIVE_ONE, dtype="float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")
        input_dy = te.lang.cce.cast_to(input_dy, "float32")
        const_res = te.lang.cce.vmuls(input_y, reciprocal_const)
    if dtype != "float32" and not tbe_platform.cce_conf.\
            api_check_support("te.lang.cce.vmuls", "float32"):
        const_res = te.lang.cce.vmuls(input_y, reciprocal_const)
    vmul_res = te.lang.cce.vmul(const_res, input_y)
    res = te.lang.cce.vmul(vmul_res, input_dy)

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype, f1628IntegerFlag=True)

    return res


@util.check_input_type(dict, dict, dict, str)
def reciprocal_grad(input_y, input_dy, output_data,
                    kernel_name="reciprocal_grad"):
    """
    algorithm: reciprocal_grad
    calculating data's reciprocal grad,dx = -1*dy*y*y,
    where `y = 1/x`, and `dy`
    is the corresponding input gradient.

    Parameters
    ----------
    input_y: dict
        shape and dtype of input_y, only support float16, float32, int32, int8
    input_dy: dict
        shape and dtype of input_dy, should be same shape and type as input_y
    output_data: dict
        shape and dtype of output, should be same shape and type as input_y
    kernel_name: str
        kernel name, default value is "reciprocal_grad"

    Returns
    -------
    None
    """
    shape_y = input_y.get("shape")
    shape_dy = input_dy.get("shape")
    dtype_y = input_y.get("dtype").lower()
    dtype_dy = input_dy.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_y)
    util.check_shape_rule(shape_dy)
    util.check_tensor_shape_size(shape_y)
    util.check_tensor_shape_size(shape_dy)

    shape_y = util.shape_refine(shape_y)
    shape_dy = util.shape_refine(shape_dy)

    util.compare_tensor_dict_key(input_y, input_dy, "shape")
    util.compare_tensor_dict_key(input_y, input_dy, "dtype")

    check_list = ("float16", "float32", "int32", "int8")
    util.check_dtype_rule(dtype_y, check_list)

    reshape_y, reshape_dy = refine_shapes_for_broadcast(shape_y, shape_dy)
    data_dy = tvm.placeholder(reshape_dy, name="data_dy", dtype=dtype_dy)
    data_y = tvm.placeholder(reshape_y, name="data_y", dtype=dtype_y)

    res = reciprocal_grad_compute(data_y, data_dy, output_data, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
