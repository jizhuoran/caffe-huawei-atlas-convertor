#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

pow
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform

def _less_compute(input_x, input_y):
    """
    if x is less than y, then return 1, else return 0.
    """
    dtype = input_x.dtype
    shape = input_x.shape

    data_min = te.lang.cce.broadcast(tvm.const(2 ** (-126), dtype=dtype),
                                     shape, dtype)
    data_zero = te.lang.cce.broadcast(tvm.const(0, dtype), shape, dtype)
    res_sub = te.lang.cce.vsub(input_y, input_x)
    res_min = te.lang.cce.vmin(res_sub, data_min)
    res_max = te.lang.cce.vmax(res_min, data_zero)

    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 62/62/2 to adaptor 126
    res_mul_val = te.lang.cce.vmuls(res_max, tvm.const(2 ** 62, dtype=dtype))
    res_mul = te.lang.cce.vmuls(res_mul_val, tvm.const(2 ** 62, dtype=dtype))
    res = te.lang.cce.vmuls(res_mul, tvm.const(2 ** 2, dtype=dtype))

    return res


def _less_compute_fp16(input_x, input_y):
    """
    if x is less than y, then return 1, else return 0.
    """
    dtype = input_x.dtype
    shape = input_x.shape

    data_min = te.lang.cce.broadcast(tvm.const(2 ** (-24), dtype=dtype),
                                     shape, dtype)
    data_zero = te.lang.cce.broadcast(tvm.const(0, dtype), shape, dtype)
    res_sub = te.lang.cce.vsub(input_y, input_x)
    res_min = te.lang.cce.vmin(res_sub, data_min)
    res_max = te.lang.cce.vmax(res_min, data_zero)

    # max num of float32 is 2**24
    # but cce can only support 2**24, so use 12/12 to adaptor 24
    res_mul_val = te.lang.cce.vmuls(res_max, tvm.const(2 ** 12, dtype=dtype))
    res_mul = te.lang.cce.vmuls(res_mul_val, tvm.const(2 ** 12, dtype=dtype))
    res = te.lang.cce.vmuls(res_mul, tvm.const(1, dtype=dtype))

    return res


def _positive_compute(input_x, input_y):
    """
    compute result of pow when data_x is more than 0,
    use exp(y * ln(x)).
    """
    input_x = te.lang.cce.vabs(input_x)
    log_value = te.lang.cce.vlog(input_x)
    mul_value = te.lang.cce.vmul(input_y, log_value)
    res = te.lang.cce.vexp(mul_value)

    return res


def _negative_compute(input_x, input_y):
    """
    compute result of pow when data_x is less than 0,
    use [-2 * (|y| % 2) + 1] * exp(y * ln|x|)
    """
    dtype = input_x.dtype
    shape = input_x.shape
    abs_value = te.lang.cce.vabs(input_y)

    if not tbe_platform.cce_conf.api_check_support("te.lang.cce.vmod",
                                                   "float32"):
        dtype = "float16"
        abs_value = te.lang.cce.cast_to(abs_value, "float16")

    data_two = te.lang.cce.broadcast(tvm.const(2, dtype), shape, dtype)
    mod_value = te.lang.cce.vmod(abs_value, data_two)
    mul_value = te.lang.cce.vmuls(mod_value, tvm.const(-2, dtype))
    add_value = te.lang.cce.vadds(mul_value, tvm.const(1, dtype))

    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        add_value = te.lang.cce.cast_to(add_value, "float32")

    abs_data_x = te.lang.cce.vabs(input_x)
    log_value = te.lang.cce.vlog(abs_data_x)
    mul_value = te.lang.cce.vmul(input_y, log_value)
    exp_value = te.lang.cce.vexp(mul_value)
    res = te.lang.cce.vmul(add_value, exp_value)

    return res


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("pow")
def pow_compute(input_x, input_y, output_z, kernel_name="pow"):
    """
    pow compute
    calculating data pow, res =x ^ y,
    x > 0: use exp(y*ln(x))
    x < 0: use [-2*(|y|%2)+1]*exp(y*ln|x|)
    x = 0: 0^0=1 & 0^y=0

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "pow"

    Returns
    -------
    res: TVM tensor
        the result of pow compute
    """
    input_dtype = input_x.dtype.lower()
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_list = util.produce_shapes(shape_x, shape_y)

    has_improve_precision = False
    data_x_cast = input_x
    data_y_cast = input_y
    cast_dtype = "float16"
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        data_x_cast = te.lang.cce.cast_to(input_x, "float32")
        data_y_cast = te.lang.cce.cast_to(input_y, "float32")
        has_improve_precision = True
        cast_dtype = "float32"

    data_x = te.lang.cce.broadcast(data_x_cast, shape_list[2])
    data_y = te.lang.cce.broadcast(data_y_cast, shape_list[2])

    data_zero = te.lang.cce.broadcast(tvm.const(0, cast_dtype),
                                      shape_list[2], cast_dtype)
    if has_improve_precision:
        data_x_negative = _less_compute(data_x, data_zero)
    else:
        data_x_negative = _less_compute_fp16(data_x, data_zero)

    # compute result of pow when data_x is more than 0
    res_val_positive = _positive_compute(data_x, data_y)
    data_one = te.lang.cce.broadcast(tvm.const(1, cast_dtype),
                                     shape_list[2], cast_dtype)
    sub_one_val = te.lang.cce.vsub(data_x_negative, data_one)
    abs_val = te.lang.cce.vabs(sub_one_val)
    res_positive = te.lang.cce.vmul(res_val_positive, abs_val)

    # compute result of pow when data_x is less than 0
    res_val_negative = _negative_compute(data_x, data_y)
    res_negative = te.lang.cce.vmul(res_val_negative, data_x_negative)

    res = te.lang.cce.vadd(res_positive, res_negative)
    if input_dtype == "int32":
        res = te.lang.cce.round(res)
    else:
        res = te.lang.cce.cast_to(res, input_dtype)

    return res

# pylint: disable=locally-disabled,redefined-builtin
@util.check_input_type(dict, dict, dict, str)
def pow(input_x, input_y, output_z, kernel_name="pow"):
    """
    algorithm: pow
    calculating data pow, res =x ** y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_z: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "pow"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    if len(shape_x) == 0:
        shape_x = (1,)
    if len(shape_y) == 0:
        shape_y = (1,)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)
    shape_list = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_list[2])
    util.check_kernel_name(kernel_name)

    input_x_dtype = input_x.get("dtype").lower()
    input_y_dtype = input_y.get("dtype").lower()
    if input_x_dtype != input_y_dtype:
        raise RuntimeError("Dtype of input_x and input_y must be the same.")
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    util.check_dtype_rule(input_x_dtype, check_list)

    shape_x, shape_y = refine_shapes_for_broadcast(shape_list[0],
                                                   shape_list[1])
    data_x = tvm.placeholder(shape_x, dtype=input_x_dtype, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=input_y_dtype, name="data_y")
    res = pow_compute(data_x, data_y, output_z, kernel_name="pow")

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res],
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(sch, config)
