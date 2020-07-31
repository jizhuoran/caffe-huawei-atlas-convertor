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

bn_training_update_v3
"""

from __future__ import absolute_import
from __future__ import division
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
# pylint: disable=locally-disabled,too-many-arguments,redefined-builtin
def _check_shape(shape_x, shape_sum, shape_square_sum,
                 shape_scale, shape_offset):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_sum: list or tuple
        sum's data shape
    shape_square_sum: list or tuple
        square_sum's data shape
    shape_scale: list or tuple
        scale's data shape
    shape_offset: list or tuple
        offset's data shape

    Returns
    -------
    None
    """
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)

    util.check_shape_rule(shape_sum)
    util.check_tensor_shape_size(shape_sum)

    util.check_shape_rule(shape_square_sum)
    util.check_tensor_shape_size(shape_square_sum)

    util.check_shape_rule(shape_scale)
    util.check_tensor_shape_size(shape_scale)

    util.check_shape_rule(shape_offset)
    util.check_tensor_shape_size(shape_offset)

    if len(shape_x) != 5 or len(shape_sum) != 5 \
            or len(shape_square_sum) != 5 or len(shape_scale) != 5 \
            or len(shape_offset) != 5:
        raise RuntimeError(
            "The data format is 5HD, "
            "but some input's shape length is not 5")

    dim_c1 = shape_x[1]
    dim_c0 = shape_x[4]

    if shape_sum[1] != dim_c1 or shape_sum[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal, but %s and %s"
            % (str(shape_x), str(shape_sum)))
    if shape_square_sum[1] != dim_c1 or shape_square_sum[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal, but %s and %s"
            % (str(shape_x), str(shape_square_sum)))
    if shape_scale[1] != dim_c1 or shape_scale[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal, but %s and %s"
            % (str(shape_x), str(shape_scale)))
    if shape_offset[1] != dim_c1 or shape_offset[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal, but %s and %s"
            % (str(shape_x), str(shape_offset)))


# pylint: disable=locally-disabled,too-many-arguments
def _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_sum: str
        sum's data type
    dtype_square_sum: str
        square_sum's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type

    Returns
    -------
    None
    """
    util.check_dtype_rule(dtype_x.lower(), ("float16", "float32"))
    util.check_dtype_rule(dtype_sum.lower(), ("float32",))
    util.check_dtype_rule(dtype_square_sum.lower(), ("float32",))
    util.check_dtype_rule(dtype_scale.lower(), ("float32",))
    util.check_dtype_rule(dtype_offset.lower(), ("float32",))


@fusion_manager.register("bn_training_update_v3")
def bn_training_update_v3_compute(x, sum, square_sum, scale, offset,
                                  y, batch_mean, batch_variance,
                                  reserve_1, reserve_2, epsilon,
                                  kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v3 compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    num = shape_x[0] * shape_x[2] * shape_x[3]
    num_rec = 1.0/num

    # compute the saved mean of x
    save_mean_reduce = te.lang.cce.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = te.lang.cce.vmuls(square_sum, num_rec)
    variance_square = te.lang.cce.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = te.lang.cce.vsub(variance_div, variance_square)

    # compute the oefficient of y
    multiplier_add = te.lang.cce.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = te.lang.cce.vsqrt(multiplier_add)
    multiplier_div = te.lang.cce.vdiv(scale, multiplier_sqrt)
    multiplier = te.lang.cce.broadcast(multiplier_div, shape_x)

    addend_mul = te.lang.cce.vmul(multiplier_div, save_mean_reduce)
    addend_sub = te.lang.cce.vsub(offset, addend_mul)
    addend = te.lang.cce.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16":
        is_cast = True
        x = te.lang.cce.cast_to(x, "float32")

    res_y = te.lang.cce.vadd(te.lang.cce.vmul(multiplier, x), addend)
    if is_cast:
        res_y = te.lang.cce.cast_to(res_y, "float16")

    # compute batch_mean and batch_var
    res_batch_mean = te.lang.cce.vmuls(sum, num_rec)
    if num == 1:
        batch_var_scaler = 0.0
    else:
        batch_var_scaler = float(num)/(num - 1)
    res_batch_var = te.lang.cce.vmuls(save_variance_reduce, batch_var_scaler)

    res = [res_y, res_batch_mean, res_batch_var,
           save_mean_reduce, save_variance_reduce]

    return res


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@util.check_input_type(dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, float, str)
def bn_training_update_v3(x, sum, square_sum, scale, offset,
                          y, batch_mean, batch_variance,
                          reserve_1, reserve_2, epsilon,
                          kernel_name="bn_training_update_v3"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A 5HD Tensor for mean.
    offset: dict
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")

    dtype_x = x.get("dtype")
    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")

    _check_shape(shape_x, shape_sum, shape_square_sum,
                 shape_scale, shape_offset)

    _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input",
                                dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum,
                                       name="square_sum_input",
                                       dtype=dtype_square_sum.lower())
    scale_input = tvm.placeholder(shape_sum, name="scale_input",
                                  dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_sum, name="offset_input",
                                   dtype=dtype_offset.lower())

    res = bn_training_update_v3_compute(x_input, sum_input, square_sum_input,
                                        scale_input, offset_input, y,
                                        batch_mean, batch_variance,
                                        reserve_1, reserve_2,
                                        epsilon, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [x_input, sum_input, square_sum_input,
                   scale_input, offset_input, ] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
