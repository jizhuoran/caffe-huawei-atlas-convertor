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

bn_training_update
"""
from __future__ import division

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,too-many-arguments,redefined-builtin
# pylint: disable=locally-disabled,invalid-name,too-many-locals,unused-argument
def _check_shape(shape_x, shape_sum, shape_square_sum, shape_scale,
                 shape_offset, shape_mean, shape_variance):
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
    shape_mean: list or tuple
        mean's data shape
    shape_variance: list or tuple
        variance's data shape

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

    util.check_shape_rule(shape_mean)
    util.check_tensor_shape_size(shape_mean)

    util.check_shape_rule(shape_variance)
    util.check_tensor_shape_size(shape_variance)

    if len(shape_x) != 5 or len(shape_sum) != 5 \
            or len(shape_square_sum) != 5 or len(shape_scale) != 5:
        raise RuntimeError(
            "This operator can only support 5HD, "
            "but some input's shape length is not 5")
    if len(shape_offset) != 5 or len(shape_mean) != 5 \
            or len(shape_variance) != 5:
        raise RuntimeError(
            "This operator can only support 5HD, "
            "but some input's shape length is not 5")

    dim_c1 = shape_x[1]
    dim_c0 = shape_x[4]

    if shape_sum[1] != dim_c1 or shape_sum[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and sum must be equal")
    if shape_square_sum[1] != dim_c1 or shape_square_sum[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and square_sum must be equal")
    if shape_scale[1] != dim_c1 or shape_scale[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and scale must be equal")
    if shape_offset[1] != dim_c1 or shape_offset[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and offset must be equal")
    if shape_mean[1] != dim_c1 or shape_mean[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and mean must be equal")
    if shape_variance[1] != dim_c1 or shape_variance[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and variance must be equal")


def _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale,
                 dtype_offset, dtype_mean, dtype_variance):
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
    dtype_mean: str
        mean's data type
    dtype_variance: str
        variance's data type

    Returns
    -------
    None
    """
    util.check_dtype_rule(dtype_x.lower(), ("float16", "float32"))
    util.check_dtype_rule(dtype_sum.lower(), ("float32",))
    util.check_dtype_rule(dtype_square_sum.lower(), ("float32",))
    util.check_dtype_rule(dtype_scale.lower(), ("float32",))
    util.check_dtype_rule(dtype_offset.lower(), ("float32",))
    util.check_dtype_rule(dtype_mean.lower(), ("float32",))
    util.check_dtype_rule(dtype_variance.lower(), ("float32",))


@fusion_manager.register("bn_training_update")
def bn_training_update_compute(x, sum, square_sum,
                               scale, offset, mean,
                               variance, y, mean_out,
                               variance_out, batch_mean,
                               batch_variance, factor, epsilon,
                               kernel_name="bn_training_update"):
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
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    mean_out: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    factor: float
        A ratio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    num = shape_x[0]*shape_x[2]*shape_x[3]
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

    if num == 1:
        batch_var_scaler = 0.0
    else:
        batch_var_scaler = float(num)/(num - 1)
    batch_variance = te.lang.cce.vmuls(save_variance_reduce, batch_var_scaler)

    factor_reverse = 1.0 - factor
    mean_mul = te.lang.cce.vmuls(save_mean_reduce, factor)
    mean_mul_rev = te.lang.cce.vmuls(mean, factor_reverse)
    mean = te.lang.cce.vadd(mean_mul, mean_mul_rev)

    var_mul = te.lang.cce.vmuls(batch_variance, factor)
    var_mul_rev = te.lang.cce.vmuls(variance, factor_reverse)
    variance = te.lang.cce.vadd(var_mul, var_mul_rev)

    res = [res_y, mean, variance, save_mean_reduce, save_variance_reduce]

    return res


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, float, float, str)
def bn_training_update(x, sum, square_sum,
                       scale, offset, mean,
                       variance, y, mean_out,
                       variance_out, batch_mean,
                       batch_variance, factor, epsilon,
                       kernel_name="bn_training_update"):
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
    mean: dict
        dict of mean, A 5HD Tensor for mean.
    variance: dict
        dict of variance, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    mean_out: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    factor: float
        A retio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")

    dtype_x = x.get("dtype")
    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")

    _check_shape(shape_x, shape_sum, shape_square_sum, shape_scale,
                 shape_offset, shape_mean, shape_variance)
    _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale,
                 dtype_offset, dtype_mean, dtype_variance)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input",
                                dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum,
                                       name="square_sum_input",
                                       dtype=dtype_square_sum.lower())
    scale_input = tvm.placeholder(shape_scale, name="scale_input",
                                  dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_offset, name="offset_input",
                                   dtype=dtype_offset.lower())
    mean_input = tvm.placeholder(shape_mean, name="mean_input",
                                 dtype=dtype_mean.lower())
    variance_input = tvm.placeholder(shape_variance, name="variance_input",
                                     dtype=dtype_variance.lower())

    res = bn_training_update_compute(x_input, sum_input, square_sum_input,
                                     scale_input, offset_input, mean_input,
                                     variance_input, y, mean_out,
                                     variance_out, batch_mean,
                                     batch_variance, factor,
                                     epsilon, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [x_input, sum_input, square_sum_input, scale_input,
                   offset_input, mean_input, variance_input] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
