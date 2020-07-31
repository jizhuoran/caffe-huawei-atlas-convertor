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

approximate_equal
"""
import operator
from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
import topi
from topi.cce import util

NUM_ONE = 1.0
NUM_ZERO = 0.0

__all__ = ["approximate_equal"]

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("approximate_equal")
def approximate_equal_compute(input_x, input_y, output_z, tolerance,
                              kernel_name="approximate_equal"):
    """
    algorithm: approximate_equal

    calculating abs(x-y) <= tolerance

    Parameters
    ----------
    input_x : the placeholders of input data
    input_y : the placeholders of input data
    tolerance: default 1e-5
    output_z: shape and dtype of output
    kernel_name: cce kernel name, default value is "approximate_equal"
    Returns
    -------
    the function of _approximate_equal_compute
    """

    input_dtype = input_x.dtype
    if input_dtype == "float16" and api_check_support("te.lang.cce.vadd",
                                                      "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")

    res_vsub = te.lang.cce.vsub(input_x, input_y)
    res_vabs = te.lang.cce.vabs(res_vsub)

    res_vabs = te.lang.cce.cast_to(res_vabs, input_x.dtype)
    tol_tensor = te.lang.cce.broadcast(tvm.const(tolerance, input_x.dtype),
                                       input_x.shape)

    res_cmp = te.lang.cce.vcmp(res_vabs, tol_tensor, 'le')
    zero_rb_tensor = te.lang.cce.broadcast(tvm.const(NUM_ZERO, "float16"),
                                           input_x.shape)
    one_rb_tensor = te.lang.cce.broadcast(tvm.const(NUM_ONE, "float16"),
                                          input_x.shape)
    res = te.lang.cce.vsel(res_cmp, one_rb_tensor, zero_rb_tensor)

    res = te.lang.cce.cast_to(res, "int8")

    return res


@util.check_input_type(dict, dict, dict, float, str)
def approximate_equal(input_x, input_y, output_z, tolerance=1e-5,
                      kernel_name="approximate_equal"):
    """
    abs(x-y) <= tolerance
    Parameters
    ----------
    input_x : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    input_y : dict, include shape and dtype, support fp16 and fp32
        shape of tensors, assume src_shape equals dst_shape

    output_z : dict, include shape and dtype, reserve

    tolerance: default 1e-5

    kernel_name : str
        cce kernel name, default value is "approximate_equal"

    Returns
    ------
    None
    """

    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    in_dtype = input_x.get("dtype")
    in_y_dtype = input_y.get("dtype")

    if tolerance < 0:
        raise RuntimeError("tolerance should >= 0")

    util.check_kernel_name(kernel_name)

    # check shape
    if not operator.eq(shape_x, shape_y):
        raise RuntimeError("all input shape must same")
    check_shape(shape_x)
    shape_x, _ = refine_shape_axes(shape_x, [])
    shape_y, _ = refine_shape_axes(shape_y, [])

    # check input tensor data_type
    check_list = ("float16", "float32")
    check_dtype(in_dtype, check_list)
    check_dtype(in_y_dtype, check_list)
    in_dtype = input_x.get("dtype").lower()
    in_y_dtype = input_y.get("dtype").lower()
    if not operator.eq(in_dtype, in_y_dtype):
        raise RuntimeError("all input type must same.")

    in_data_x = tvm.placeholder(shape_x, name="shape_x", dtype=in_dtype)
    in_data_y = tvm.placeholder(shape_y, name="shape_y", dtype=in_dtype)
    res = approximate_equal_compute(in_data_x, in_data_y, output_z,
                                    tolerance, kernel_name)

    with tvm.target.cce():
        auto_sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [in_data_x, in_data_y, res],
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(auto_sch, config)
