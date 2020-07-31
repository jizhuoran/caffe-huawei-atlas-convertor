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

asinh

  Op_description :
    Computes inverse hyperbolic sine of x element-wise

    # asinh(
    #   input_x,
    #   output_y,
    #   kernel_name="cce_asinh")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : shape size limit is 2147483648.

"""

from functools import reduce as functool_reduce
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shape_axes
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from impl.util.util_compute import sign

# shape limit
SHAPE_SIZE_LIMIT = 2147483648

# Newton Factor
CONST_NEWTON_FACTOR = 0.5
CONST_NEWTON_FACTOR_NEG = -0.5

# Log threshold
CONST_LOG_THRESHOLD_1 = 0.6666666666666667
CONST_LOG_THRESHOLD_2 = 0.3333333333333333

# Log value
LOG_FOUR_THREE = 0.28768207245178085
LOG_FIVE_THREE = 0.5108256237659907
LOG_FIVE_TWO = 0.916290731874155

# const value
CONST_NEG_ONE = -1
CONST_ZERO = 0
CONST_ONE = 1
CONST_TWO = 2
CONST_ONE_THREE = 0.3333333333333333
CONST_THREE_FOUR = 0.75
CONST_ONE_FIVE = 0.2
CONST_ONE_FOUR_NEG = -0.25
CONST_FIVE_TWO = 0.4
CONST_DOT_SIX = 0.6
FLOAT_16_MAX = 32768
# min float16 value
MIN_FP16 = 2**(-24)


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("asinh")
def asinh_compute_mini(input_x, output_y, kernel_name="asinh"):
    """
    algrithm: asinh(x) = log(x + sqrt(x^2 + 1))

    Parameters
    ----------
    input_x: the placeholder of data input

    output_y : the dict of output

    kernel_name : cce kernel name, default value is "asinh"

    Returns
    -------
    res : result of asinh

    """

    inp_dtype = input_x.dtype.lower()
    shape = input_x.shape
    const_zero_tensor = te.lang.cce.broadcast(tvm.const(CONST_ZERO, inp_dtype),
                                              shape)
    compare_one = te.lang.cce.vcmp(input_x, const_zero_tensor, "le")
    has_improve_precision = False
    if inp_dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vrec",
                                                    "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        has_improve_precision = True

    input_x = te.lang.cce.vabs(input_x)
    # to fix bug for input data is 0.0
    input_x = te.lang.cce.vadds(input_x, MIN_FP16)
    data_1_x = te.lang.cce.vrec(input_x)
    data_1_x_square = te.lang.cce.vmul(data_1_x, data_1_x)
    data_1_x_square = te.lang.cce.vadds(data_1_x_square, tvm.const(CONST_ONE,
                                                                   "float32"))
    data_s_1_sqrt = _newton_sqrt(data_1_x_square, inp_dtype)
    data_res = te.lang.cce.vmul(data_s_1_sqrt, input_x)
    data_res = te.lang.cce.vadd(input_x, data_res)
    result = _log_taylor(data_res, shape)

    res_neg = te.lang.cce.vmuls(result, tvm.const(CONST_NEG_ONE, inp_dtype))
    res = te.lang.cce.vsel(compare_one, res_neg, result)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")
    else:
        res = te.lang.cce.cast_to(res, "float32")

    return res


def asinh_compute_cloud(input_x, output_y, kernel_name="asinh"):
    """
    algrithm: asinh(x) = log(x + sqrt(x^2 + 1))

    Parameters
    ----------
    input_x: the placeholder of data input

    output_y : the dict of output

    kernel_name : cce kernel name, default value is "asinh"

    Returns
    -------
    res : result of asinh

    """

    inp_dtype = input_x.dtype.lower()
    has_improve_precision = False
    if inp_dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vlog",
                                                    "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        has_improve_precision = True
        inp_dtype = "float32"

    data_abs = te.lang.cce.vabs(input_x)
    data_x_square = te.lang.cce.vmul(data_abs, data_abs)
    data_add = te.lang.cce.vadds(data_x_square,
                                 tvm.const(CONST_ONE, inp_dtype))
    data_s_1_sqrt = te.lang.cce.vsqrt(data_add)
    data_res = te.lang.cce.vadd(data_s_1_sqrt, data_abs)
    result = te.lang.cce.vlog(data_res)
    res = te.lang.cce.vmul(result, sign(input_x))

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


def _newton_iter(data, data_x0, dtype):
    """
    algrithm: x(n+1) = 1/2 ( x(n) + a/x(n))

    Parameters
    ----------
    data: input tensor that we want to calculate sqrt

    data_x0 : input tensor of an approximate value of sqrt

    dtype : the type of tensor

    Returns
    -------
    data_newton : result of newton iter

    """
    # Newton begin:
    data_newton = te.lang.cce.vrec(data)
    data_newton = te.lang.cce.vmul(data_x0, data_newton)
    data_newton = te.lang.cce.vadd(data_newton, data)
    data_newton = te.lang.cce.vmuls(data_newton,
                                    tvm.const(CONST_NEWTON_FACTOR, dtype))
    # Newton end
    return data_newton


def _newton_sqrt(data, dtype):
    """
    use three times to calculate sqrt

    Parameters
    ----------
    data: input tensor that we want to calculate sqrt

    dtype : the type of tensor

    Returns
    -------
    data_sqrt : return of sqrt

    """
    data_sqrt = te.lang.cce.vrsqrt(data)
    data_sqrt = te.lang.cce.vrec(data_sqrt)
    data_sqrt = _newton_iter(data_sqrt, data, dtype)
    data_sqrt = _newton_iter(data_sqrt, data, dtype)
    data_sqrt = _newton_iter(data_sqrt, data, dtype)
    return data_sqrt


def _log_taylor(data_x, shape):
    """
    use taylor expansion to calculate log

    Parameters
    ----------
    data_x: input tensor that we want to calculate sqrt

    dtype : the type of tensor

    Returns
    -------
    res :  return of log

    """
    data = te.lang.cce.vadds(data_x, tvm.const(CONST_NEG_ONE, "float32"))
    threshold_1 = te.lang.cce.broadcast(
        tvm.const(CONST_LOG_THRESHOLD_1, "float32"), shape)
    index_1 = te.lang.cce.vcmp(data, threshold_1, 'ge')
    data_1 = te.lang.cce.vadds(
        data,
        tvm.const(CONST_NEG_ONE*CONST_LOG_THRESHOLD_1, "float32"))
    data_sel = te.lang.cce.vsel(
        index_1,
        te.lang.cce.vmuls(data_1, tvm.const(CONST_DOT_SIX, "float32")),
        data)
    data_sel = te.lang.cce.cast_to(data_sel, "float32")

    threshold_2 = te.lang.cce.broadcast(
        tvm.const(CONST_LOG_THRESHOLD_2, "float32"), shape)
    index_2 = te.lang.cce.vcmp(data_sel, threshold_2, 'ge')
    data_2 = te.lang.cce.vadds(
        data_sel,
        tvm.const(CONST_NEG_ONE*CONST_LOG_THRESHOLD_2, "float32"))
    data_vmuls = te.lang.cce.vmuls(
        data_2,
        tvm.const(CONST_THREE_FOUR, "float32"))
    data_sel = te.lang.cce.vsel(index_2, data_vmuls, data_sel)
    data_sel = te.lang.cce.cast_to(data_sel, "float32")

    taylor = _taylor_compute(data_sel)

    # add log(4/3)
    res = te.lang.cce.vsel(
        index_2,
        te.lang.cce.vadds(taylor, tvm.const(LOG_FOUR_THREE, "float32")),
        taylor)
    res = te.lang.cce.cast_to(res, "float32")
    # add log(5/3)
    res = te.lang.cce.vsel(
        index_1,
        te.lang.cce.vadds(taylor, tvm.const(LOG_FIVE_THREE, "float32")),
        res)
    res = te.lang.cce.cast_to(res, "float32")
    # d: vlog:
    res = _log_compute(data_x, res, shape)

    return res


def _taylor_compute(data):
    """
    algrithm: log(x) = ((((0.2x - 0.25)x + 0.33333)x - 0.5)x + 1)x

    Parameters
    ----------
    data: input tensor that we want to calculate log

    Returns
    -------
    None

    """
    # 0.2x - 0.25
    taylor_five = te.lang.cce.vmuls(data, tvm.const(CONST_ONE_FIVE, "float32"))
    taylor_four_1 = te.lang.cce.vadds(taylor_five,
                                      tvm.const(CONST_ONE_FOUR_NEG, "float32"))
    # (0.2x - 0.25)x + 0.33333
    taylor_four_2 = te.lang.cce.vmul(taylor_four_1, data)
    taylor_three_1 = te.lang.cce.vadds(taylor_four_2,
                                       tvm.const(CONST_ONE_THREE, "float32"))
    # ((0.2x - 0.25)x + 0.33333)x - 0.5
    taylor_three_2 = te.lang.cce.vmul(taylor_three_1, data)
    taylor_two_1 = te.lang.cce.vadds(
        taylor_three_2,
        tvm.const(CONST_NEWTON_FACTOR_NEG, "float32"))
    # (((0.2x - 0.25)x + 0.33333)x - 0.5)x+1
    taylor_two_2 = te.lang.cce.vmul(taylor_two_1, data)
    taylor_one = te.lang.cce.vadds(taylor_two_2,
                                   tvm.const(CONST_ONE, "float32"))
    # ((((0.2x - 0.25)x + 0.33333)x - 0.5)x + 1)x
    taylor = te.lang.cce.vmul(taylor_one, data)

    return taylor


def _log_compute(data_x, res, shape):
    """
    when data > 2, use vlog directly
    when data > 32768, float16 will overflow, use log(x/2.5)+log(2.5)

    Parameters
    ----------
    data: input tensor that we want to calculate log

    Returns
    -------
    res : return of log

    """
    # if data > 2, use vlog
    threshold_3 = te.lang.cce.broadcast(tvm.const(CONST_TWO, "float32"), shape)
    index_3 = te.lang.cce.vcmp(data_x, threshold_3, 'ge')
    res = te.lang.cce.vsel(index_3, te.lang.cce.vlog(data_x), res)
    # if data > 32768, use log(x/2.5)+log(2.5)
    float_16_max_tensor = te.lang.cce.broadcast(
        tvm.const(FLOAT_16_MAX, "float32"), shape)
    index_4 = te.lang.cce.vcmp(data_x, float_16_max_tensor, 'ge')
    overflow_value = te.lang.cce.vmuls(data_x, CONST_FIVE_TWO)
    res_overflow = te.lang.cce.vadds(
        te.lang.cce.vlog(overflow_value), LOG_FIVE_TWO)
    res = te.lang.cce.vsel(index_4, res_overflow, res)
    res = te.lang.cce.cast_to(res, "float32")

    return res


@util.check_input_type(dict, dict, str)
def asinh(input_x, output_y, kernel_name="asinh"):
    """
    algrithm: asinh(x) = log(x + sqrt(x^2 + 1))

    Parameters
    ----------
    input_x: the dict of input_x, only support float16, float32

    output_y : the dict of output_y

    kernel_name : cce kernel name, default value is "asinh"

    Returns
    -------
    None

    """

    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input)
    util.check_shape_size(shape_input, SHAPE_SIZE_LIMIT)
    shape_input, _ = refine_shape_axes(shape_input, [])

    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype_input, check_list)

    inp_dtype = dtype_input.lower()
    shape_input = (functool_reduce(lambda x, y: x * y, shape_input),)
    data_input = tvm.placeholder(shape_input,
                                 dtype=inp_dtype, name="data_input")

    with tvm.target.cce():
        if tbe_platform.cce_conf.api_check_support("te.lang.cce.vlog",
                                                   "float32") or not \
                tbe_platform.cce_conf.api_check_support("te.lang.cce.vrec",
                                                        "float32"):
            res = asinh_compute_cloud(data_input, output_y, kernel_name)
        else:
            res = asinh_compute_mini(data_input, output_y, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res],
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(sch, config)
