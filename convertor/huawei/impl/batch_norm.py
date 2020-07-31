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

batch_norm
"""

from __future__ import absolute_import
from __future__ import division

import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

NONETYPE = type(None)


def _format_check(arg_input, data_format):
    """
    Function to check if the data_format is in line with norms.

    Parameters
    ----------
    input: dict
        dict of input
    data_format: str
        format of input data

    Returns
    -------
    None
    """
    format_data = arg_input.get("format")
    if format_data not in ("NHWC", "NCHW", "NC1HWC0"):
        raise RuntimeError(
            "Format of input only support 4D and 5HD")
    if data_format not in ("NHWC", "NCHW"):
        raise RuntimeError(
            "The data_format only support 'NHWC' and 'NCHW'")


def _check_dims_equal(shape_x, shape, data_format):
    """
    Function to check the dimension C to be equal.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape: list or tuple
        data shape of test input
    data_format: str
        format of input data

    Returns
    -------
    None
    """
    if data_format == "NC1HWC0":
        if shape_x[1] != shape[1] or shape_x[4] != shape[4]:
            raise RuntimeError(
                "The dimensions C1 C0 of shape_x and shape must be equal")
        if shape[0] != 1 or shape[2] != 1 or shape[3] != 1:
            raise RuntimeError(
                "Dimension N,H,W must be 1")
    elif data_format == "NCHW":
        if shape_x[1] != shape[0]:
            raise RuntimeError(
                "Dimensions must be equal")
    else:
        if shape_x[3] != shape[0]:
            raise RuntimeError(
                "Dimensions must be equal")



def _check_shape_dims(shape, data_format, is_x=False):
    """
    Function to check input tensors must be 5D ones.

    Parameters
    ----------
    shape: list or tuple
        data shape of test input
    data_format: str
        format of input data
    is_x: bool
        data to check is input_x or not

    Returns
    -------
    None
    """
    if data_format == "NC1HWC0":
        if len(shape) != 5:
            raise RuntimeError(
                "The input shape only support 5D Tensor")
    elif is_x:
        if len(shape) != 4:
            raise RuntimeError(
                "The input shape only support 4D Tensor")
    else:
        if len(shape) != 1:
            raise RuntimeError(
                "The input shape only support 1D Tensor")


# pylint: disable=locally-disabled,too-many-arguments
def _shape_check(shape_x, shape_scale, shape_offset,
                 mean, variance, is_training, data_format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        shape_scale's data shape
    shape_offset: list or tuple
        shape_offset's data shape
    shape_mean: list or tuple
        shape_mean's data shape
    shape_variance: list or tuple
        shape_variance's data shape
    is_training: bool
        A bool value to indicate the operation is for training or inference.

    Returns
    -------
    None
    """
    _check_shape_dims(shape_x, data_format, True)
    _check_shape_dims(shape_scale, data_format)
    _check_shape_dims(shape_offset, data_format)

    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_scale)
    util.check_shape_rule(shape_offset)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_scale)
    util.check_tensor_shape_size(shape_offset)
    _check_dims_equal(shape_x, shape_scale, data_format)
    _check_dims_equal(shape_x, shape_offset, data_format)

    if not is_training:
        shape_mean = mean.get("shape")
        shape_variance = variance.get("shape")
        util.check_shape_rule(shape_mean)
        util.check_shape_rule(shape_variance)
        util.check_tensor_shape_size(shape_mean)
        util.check_tensor_shape_size(shape_variance)
        _check_shape_dims(shape_mean, data_format)
        _check_shape_dims(shape_variance, data_format)
        _check_dims_equal(shape_x, shape_mean, data_format)
        _check_dims_equal(shape_x, shape_variance, data_format)
    elif mean is not None or variance is not None:
        raise RuntimeError(
            "Estimated_mean or estimated_variance must be empty for training")


# pylint: disable=locally-disabled,too-many-arguments,invalid-name
def _output_data_y_compute(x, mean, variance, scale, offset, epsilon):
    """
    Function to calculate the y, which is a public function

    Parameters
    ----------
    x: TVM tensor
        contains x data
    mean: TVM tensor
        contains mean data.
    variance: TVM tensor
        contains variance data.
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    epsilon: float
        A small float number added to the variance of x.

    Returns
    -------
    res: TVM tensor
        the y of batch_norm_ext2 compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    y_add = te.lang.cce.vadds(variance, epsilon)
    y_sqrt = te.lang.cce.vsqrt(y_add)
    var_sub = te.lang.cce.vsub(x, mean)
    y_norm = te.lang.cce.vdiv(var_sub, y_sqrt)
    scale_broad = te.lang.cce.broadcast(scale, shape_x)
    offset_broad = te.lang.cce.broadcast(offset, shape_x)
    res = te.lang.cce.vadd(te.lang.cce.vmul(scale_broad, y_norm), offset_broad)

    return res


# pylint: disable=locally-disabled,too-many-locals
def _fused_batch_norm_inf_compute(x, scale, offset, mean, variance,
                                  epsilon, is_py, format_data):
    """
    Function to calculate the output of batch_norm when is_training is False.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data. Used for inference only.
    variance: TVM tensor
        contains variance data. Used for inference only.
    epsilon: float
        A small float number added to the variance of x.

    Returns
    -------
    res: TVM tensor list
        the result of batch_norm_ext2 inference compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    is_cast = False

    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        is_cast = True
        x = te.lang.cce.cast_to(x, "float32")

    mean_broadcast = te.lang.cce.broadcast(mean, shape_x)
    var_broadcast = te.lang.cce.broadcast(variance, shape_x)

    res_y = _output_data_y_compute(x, mean_broadcast, var_broadcast,
                                   scale, offset, epsilon)
    if is_cast:
        res_y = te.lang.cce.cast_to(res_y, "float16")

    if format_data == "NHWC":
        axis = [0, 1, 2]
    else:
        axis = [0, 2, 3]

    scaler_zero = 0.0
    res_batch_mean = te.lang.cce.vadds(mean, scaler_zero)
    res_batch_var = te.lang.cce.vadds(variance, scaler_zero)
    if format_data != "NC1HWC0":
        res_batch_mean = te.lang.cce.sum(res_batch_mean, axis, False)
        res_batch_var = te.lang.cce.sum(res_batch_var, axis, False)
    res = [res_y, res_batch_mean, res_batch_var]

    if not is_py:
        res_reserve_space_1 = te.lang.cce.vadds(mean, scaler_zero)
        res_reserve_space_2 = te.lang.cce.vadds(variance, scaler_zero)
        if format_data != "NC1HWC0":
            res_reserve_space_1 = te.lang.cce.sum(res_reserve_space_1,
                                                  axis, False)
            res_reserve_space_2 = te.lang.cce.sum(res_reserve_space_2,
                                                  axis, False)
        res = res + [res_reserve_space_1, res_reserve_space_2]

    return res


def _fused_batch_norm_train_compute(x, scale, offset, epsilon,
                                    is_py, format_data):
    """
    Function to calculate the output of batch_norm when is_training is True.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    epsilon: float
        A small float number added to the variance of x.

    Returns
    -------
    res: TVM tensor list
        the result of batch_norm_ext2 training compute
    """
    is_cast = False
    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        is_cast = True
        x = te.lang.cce.cast_to(x, "float32")

    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    if format_data == "NHWC":
        axis = [0, 1, 2]
        num = shape_x[0]*shape_x[1]*shape_x[2]
    else:
        axis = [0, 2, 3]
        num = shape_x[0]*shape_x[2]*shape_x[3]
    num_rec = 1.0/num

    # compute saved mean according to dimension C of x
    mean_sum = te.lang.cce.sum(x, axis, True)
    mean_muls = te.lang.cce.vmuls(mean_sum, num_rec)
    mean_broad = te.lang.cce.broadcast(mean_muls, shape_x)

    # compute saved var according to dimension C of x
    var_sub = te.lang.cce.vsub(x, mean_broad)
    var_mul = te.lang.cce.vmul(var_sub, var_sub)
    var_sum = te.lang.cce.sum(var_mul, axis, True)
    var_muls = te.lang.cce.vmuls(var_sum, num_rec)
    var = te.lang.cce.broadcast(var_muls, shape_x)

    res_y = _output_data_y_compute(x, mean_broad, var, scale, offset, epsilon)
    if is_cast:
        res_y = te.lang.cce.cast_to(res_y, "float16")

    res_batch_mean = te.lang.cce.vmuls(mean_sum, num_rec)
    if format_data != "NC1HWC0":
        res_batch_mean = te.lang.cce.sum(res_batch_mean, axis, False)

    if num == 1:
        batch_var_scaler = 0.0
    else:
        batch_var_scaler = float(num)/(num - 1)
    res_batch_var = te.lang.cce.vmuls(var_muls, batch_var_scaler)
    if format_data != "NC1HWC0":
        res_batch_var = te.lang.cce.sum(res_batch_var, axis, False)

    res = [res_y, res_batch_mean, res_batch_var]
    if not is_py:
        res_reserve_space_1 = te.lang.cce.vmuls(mean_sum, num_rec)
        res_reserve_space_2 = te.lang.cce.vmuls(var_sum, num_rec)
        if format_data != "NC1HWC0":
            res_reserve_space_1 = te.lang.cce.sum(res_reserve_space_1,
                                                  axis, False)
            res_reserve_space_2 = te.lang.cce.sum(res_reserve_space_2,
                                                  axis, False)
        res = res + [res_reserve_space_1, res_reserve_space_2]

    return res


# pylint: disable=locally-disabled,unused-argument,too-many-statements
@fusion_manager.register("batch_norm")
def batch_norm_compute(x, scale, offset, mean, variance, y,
                       batch_mean, batch_variance, reserve_space_1,
                       reserve_space_2, epsilon=0.001,
                       data_format="NHWC", is_training=True,
                       kernel_name="batch_norm"):
    """
    algorithm: fused_batch_norm
    Description of calculating process with TE api,
    the computational formula is as follows.
    x = (x - mean)/(var + epsilon)**0.5
    y = scale*x + offset

    Parameters
    ----------
    x: TVM tensor
        contains x data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data. Used for inference only, must be empty for training.
    variance: TVM tensor
        contains variance data.
        Used for inference only, must be empty for training.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`. Has the same type as `mean`.
    batch_variance: dict
        dict of batch_var, A `Tensor`. Has the same type as `variance`.
    reserve_space_1: dict
        dict of reserve_space_1, A `Tensor`.
    reserve_space_2: dict
        dict of reserve_space_2, A `Tensor`.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.001`.
    data_format: str
        The data format for x and y. Support "NC1HWC0" only.
    is_training: bool
        A bool value indicates the operation for train (default) or inference.
    kernel_name: str
        kernel name, default value is "batch_norm"

    Returns
    -------
    res: TVM tensor list
        the result of batch_norm_ext2 compute
    """
    format_data = y.get("format")
    is_py = reserve_space_1 is None

    if is_training:
        res = _fused_batch_norm_train_compute(x, scale, offset,
                                              epsilon, is_py, format_data)
    else:
        res = _fused_batch_norm_inf_compute(x, scale, offset, mean, variance,
                                            epsilon, is_py, format_data)

    return res


@util.check_input_type(dict, dict, dict, (dict, NONETYPE), (dict, NONETYPE),
                       dict, dict, dict, (dict, NONETYPE), (dict, NONETYPE),
                       float, str, bool, str)
def batch_norm(x, scale, offset, mean, variance, y, batch_mean,
               batch_variance, reserve_space_1, reserve_space_2,
               epsilon=0.0001, data_format="NHWC",
               is_training=True, kernel_name="batch_norm"):
    """
    algorithm: fused_batch_norm
    Batch normalization.
    Note that the size of 5D Tensors are defined by "NC1HWC0".
    The input tensor's dimension C should be equal.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    scale: dict
        dict of scale,
        A Tensor for scaling factor, to scale the normalized x.
    offset: dict
        dict of offset, A Tensor for offset, to shift to the normalized x.
    mean: dict
        dict of mean, A Tensor for population mean.
        Used for inference only, must be empty for training.
    variance: dict
        dict of variance, A Tensor for population variance.
        Used for inference only, must be empty for training.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`. Has the same type as `mean`.
    batch_variance: dict
        dict of batch_var, A `Tensor`. Has the same type as `variance`.
    reserve_space_1: dict
        dict of reserve_space_1, A `Tensor`.
    reserve_space_2: dict
        dict of reserve_space_2, A `Tensor`.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.001`.
    data_format: str
        The data format for x and y. Support "NC1HWC0" only.
    is_training: bool
        A bool value indicates the operation for train (default) or inference.
    kernel_name: str
        kernel name, default value is "batch_norm"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    if len(shape_x) == 2:
        shape_x = list(shape_x) + [1, 1]
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")
    if not is_training:
        shape_mean = mean.get("shape")
        shape_variance = variance.get("shape")

    dtype_x = x.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")
    if not is_training:
        dtype_mean = mean.get("dtype")
        dtype_variance = variance.get("dtype")
        util.check_dtype_rule(dtype_mean.lower(), ("float32", "float16"))
        util.check_dtype_rule(dtype_variance.lower(), ("float32", "float16"))

    _format_check(x, data_format)
    format_data = x.get("format")

    _shape_check(shape_x, shape_scale, shape_offset, mean,
                 variance, is_training, format_data)

    util.check_kernel_name(kernel_name)

    util.check_dtype_rule(dtype_x.lower(), ("float16", "float32"))
    util.check_dtype_rule(dtype_scale.lower(), ("float32", "float16"))
    util.check_dtype_rule(dtype_offset.lower(), ("float32", "float16"))

    if format_data == "NHWC":
        shape_scale = [1, 1, 1] + list(shape_scale)
        shape_offset = [1, 1, 1] + list(shape_offset)
        if not is_training:
            shape_mean = [1, 1, 1] + list(shape_mean)
            shape_variance = [1, 1, 1] + list(shape_variance)
    elif format_data == "NCHW":
        shape_scale = [1] + list(shape_scale) + [1, 1]
        shape_offset = [1] + list(shape_offset) + [1, 1]
        if not is_training:
            shape_mean = [1] + list(shape_mean) + [1, 1]
            shape_variance = [1] + list(shape_variance) + [1, 1]

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    scale_input = tvm.placeholder(shape_scale, name="scale_input",
                                  dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_offset, name="offset_input",
                                   dtype=dtype_offset.lower())

    if is_training:
        mean_input, variance_input = [], []
    else:
        mean_input = tvm.placeholder(shape_mean, name="mean_input",
                                     dtype=dtype_mean.lower())
        variance_input = tvm.placeholder(shape_variance, name="variance_input",
                                         dtype=dtype_variance.lower())

    res = batch_norm_compute(x_input, scale_input, offset_input, mean_input,
                             variance_input, y, batch_mean,
                             batch_variance, reserve_space_1,
                             reserve_space_2,
                             epsilon, data_format,
                             is_training, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    if is_training:
        tensor_list = [x_input, scale_input, offset_input] + list(res)
    else:
        tensor_list = [x_input, scale_input, offset_input,
                       mean_input, variance_input] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
