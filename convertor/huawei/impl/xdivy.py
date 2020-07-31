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

xdivy
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
# define a scalar , value = 1
SCALAR_ONE = 1
# minimun num of float32 2**(-126)
MININUM_NUM_FLOAT = 2**(-126)
# minimun num of float16 2**(-24)
MININUM_NUM_HALF = 2**(-24)
# max num of float32 is 2**126, but cce can only support 2**62,
# so use 62/62/2 to adaptor 149
MAX_ONE_CONST_FLOAT = 2**62
MAX_TWO_CONST_FLOAT = 2**2
# max num of float16 is 2**24, but cce can only support 2**12,
# so use 12/12 to adaptor 24
MAX_CONST_HALF = 2**12

# pylint: disable=locally-disabled,too-many-locals,unused-argument
@fusion_manager.register("xdivy")
def xdivy_compute(input_x, input_y, output_z, kernel_name="xdivy"):
    """
    xdivy compute
    calculating data's xdivy,return 0 if x==0 and x/y otherwise, elementwise

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel name, default value is "xdivy"

    Returns
    -------
    res: TVM tensor
        the result of xdivy compute
    """
    input_data1 = te.lang.cce.util.shape_to_list(input_x.shape)
    input_data2 = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_list = util.produce_shapes(input_data1, input_data2)
    util.check_tensor_shape_size(shape_list[2])
    dtype = input_x.dtype

    broadcast_x = te.lang.cce.broadcast(input_x, shape_list[2])
    broadcast_y = te.lang.cce.broadcast(input_y, shape_list[2])
    broadcast_one = te.lang.cce.broadcast(tvm.const(SCALAR_ONE, dtype),
                                          shape_list[2], dtype)

    abs_x = te.lang.cce.vabs(broadcast_x)
    abs_y = te.lang.cce.vabs(broadcast_y)
    add_x_y = te.lang.cce.vadd(abs_x, abs_y)

    if dtype == "float32":
        data_min = te.lang.cce.broadcast(tvm.const(MININUM_NUM_FLOAT,
                                                   dtype=dtype),
                                         shape_list[2], dtype)
    elif dtype == "float16":
        data_min = te.lang.cce.broadcast(tvm.const(MININUM_NUM_HALF,
                                                   dtype=dtype),
                                         shape_list[2], dtype)

    zero_x_y = te.lang.cce.vmin(add_x_y, data_min)

    if dtype == "float32":
        data_mul1 = te.lang.cce.vmuls(zero_x_y, tvm.const(MAX_ONE_CONST_FLOAT,
                                                          dtype=dtype))
        data_mul2 = te.lang.cce.vmuls(data_mul1, tvm.const(MAX_ONE_CONST_FLOAT,
                                                           dtype=dtype))
        mul_data = te.lang.cce.vmuls(data_mul2, tvm.const(MAX_TWO_CONST_FLOAT,
                                                          dtype=dtype))
    elif dtype == "float16":
        data_mul1 = te.lang.cce.vmuls(zero_x_y, tvm.const(MAX_CONST_HALF,
                                                          dtype=dtype))
        mul_data = te.lang.cce.vmuls(data_mul1, tvm.const(MAX_CONST_HALF,
                                                          dtype=dtype))

    sub_x_y_zero = te.lang.cce.vsub(mul_data, broadcast_one)
    abs_x_y_zero = te.lang.cce.vabs(sub_x_y_zero)
    input_y_revised = te.lang.cce.vadd(broadcast_y, abs_x_y_zero)

    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        broadcast_x = te.lang.cce.cast_to(broadcast_x, "float32")
        input_y_revised = te.lang.cce.cast_to(input_y_revised, "float32")
        has_improve_precision = True

    res = te.lang.cce.vdiv(broadcast_x, input_y_revised)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, dtype)

    return res


@util.check_input_type(dict, dict, dict, str)
def xdivy(input_x, input_y, output_z, kernel_name="xdivy"):
    """
    algorithm: xdivy
    calculating data's xdivy,return 0 if x==0 and x/y otherwise, elementwise

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel name, default value is "xdivy"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")

    util.compare_tensor_dict_key(input_x, input_y, "dtype")
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_kernel_name(kernel_name)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)
    shape_list = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_list[2])
    input_dtype = dtype.lower()
    input_dtype_y = dtype_y.lower()
    check_list = ("float16", "float32")
    util.check_dtype_rule(input_dtype, check_list)
    util.check_dtype_rule(input_dtype_y, check_list)

    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_list[0],
                                                       shape_list[1])
    data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")

    res = xdivy_compute(data_x, data_y, output_z, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
