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

less_equal
"""
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils.op_utils import refine_shapes_for_broadcast
from topi.cce import util

# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2**(-126)
# define a scalar, value = 2**(50)
SCALAR_MUL_FP32 = 2**(50)
# define a scalar, value = 2**(26)
SCALAR_MUL2_FP32 = 2**(26)
# define a scalar, value = 2**(-24), minimun num of float16 2**(-24)
SCALAR_MIN_FP16 = 2**(-24)
# define a scalar, value = 2**(12)
SCALAR_MUL_FP16 = 2**(12)
# define a scalar, value = 1
SCALAR_ONE = 1

# limit of input shape
MAX_SHAPE_NUM = 10000000

# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("less_equal")
def less_equal_compute(input_x, input_y, output_z, kernel_name="less_equal"):
    """
    compute for less_equal

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "less_equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype_x = input_x.dtype
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_broadcast = util.produce_shapes(shape_x, shape_y)

    if dtype_x == "float32":
        tensor_min = te.lang.cce.broadcast(tvm.const(SCALAR_MIN_FP32,
                                                     dtype="float32"),
                                           shape_broadcast)
        tensor_mul = te.lang.cce.broadcast(tvm.const(SCALAR_MUL_FP32,
                                                     dtype="float32"),
                                           shape_broadcast)
        tensor_mul1 = te.lang.cce.broadcast(tvm.const(SCALAR_MUL2_FP32,
                                                      dtype="float32"),
                                            shape_broadcast)
        tensor_one = te.lang.cce.broadcast(tvm.const(SCALAR_ONE,
                                                     dtype="float32"),
                                           shape_broadcast)
    else:
        tensor_min = te.lang.cce.broadcast(tvm.const(SCALAR_MIN_FP16,
                                                     dtype="float16"),
                                           shape_broadcast)
        tensor_mul = te.lang.cce.broadcast(tvm.const(SCALAR_MUL_FP16,
                                                     dtype="float16"),
                                           shape_broadcast)
        tensor_one = te.lang.cce.broadcast(tvm.const(SCALAR_ONE,
                                                     dtype="float16"),
                                           shape_broadcast)

    if dtype_x in ("int8", "uint8"):
        input_x = te.lang.cce.cast_to(input_x, "float16")
        input_y = te.lang.cce.cast_to(input_y, "float16")

    input_x = te.lang.cce.broadcast(input_x, shape_broadcast)
    input_y = te.lang.cce.broadcast(input_y, shape_broadcast)
    res_max = te.lang.cce.vmax(input_x, input_y)
    res_vsub = te.lang.cce.vsub(input_y, res_max)
    res_vabs = te.lang.cce.vabs(res_vsub)
    res_min = te.lang.cce.vmin(res_vabs, tensor_min)
    res_vmul = te.lang.cce.vmul(res_min, tensor_mul)
    res_vmul1 = te.lang.cce.vmul(res_vmul, tensor_mul)

    if dtype_x == "float32":
        res_vmul2 = te.lang.cce.vmul(res_vmul1, tensor_mul1)
        res_vsub1 = te.lang.cce.vsub(res_vmul2, tensor_one)
        res_vabs1 = te.lang.cce.vabs(res_vsub1)
    else:
        res_vsub1 = te.lang.cce.vsub(res_vmul1, tensor_one)
        res_vabs1 = te.lang.cce.vabs(res_vsub1)

    res = te.lang.cce.cast_to(res_vabs1, "int8", True)

    return res


@util.check_input_type(dict, dict, dict, str)
def less_equal(input_x, input_y, output_z, kernel_name="less_equal"):
    """
    Returns the truth value of (x <= y) element-wise

    Parameters
    ----------
    input_x: dict
        dict of input_x, include keys(shape and dtype)
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    output_z: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "less_equal"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")
    shape_x, shape_y, shape_broadcast = util.produce_shapes(shape_x, shape_y)

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x, max_shape_num=MAX_SHAPE_NUM)
    util.check_shape_rule(shape_y, max_shape_num=MAX_SHAPE_NUM)
    util.check_shape_rule(shape_broadcast, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)
    util.check_tensor_shape_size(shape_broadcast)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    dtype_x = dtype_x.lower()
    util.check_dtype_rule(dtype_x, check_list)
    dtype_y = dtype_y.lower()
    util.check_dtype_rule(dtype_y, check_list)
    util.compare_tensor_dict_key(input_x, input_y, "dtype")

    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=dtype_x)
    data_input_y = tvm.placeholder(shape_y, name="data_input_y", dtype=dtype_y)

    res = less_equal_compute(data_input_x, data_input_y, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_x, data_input_y, res]}
    te.lang.cce.cce_build_code(sch, config)
