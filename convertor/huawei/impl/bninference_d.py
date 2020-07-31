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
from te import platform as cceconf
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


NONETYPE = type(None)
CONST_NEWTON_FACTOR1 = 3.0
CONST_ONE = 1
CONST_HALF = 0.5
CONST_NEG_ONE = -1.0


# pylint: disable=locally-disabled,too-few-public-methods,no-init
def _format_check(arg_input):
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
    if format_data not in ("ND", "NC1HWC0", "NCHW"):
        raise RuntimeError(
            "format is invalid, format only support ND, 4D and 5HD")


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

    if data_format in ("ND", "NCHW"):

        if len(shape_x) == 1:
            index_c = 0
        else:
            index_c = 1
        if shape_x[index_c] != shape[0]:
            raise RuntimeError(
                "Dimensions must be equal")


def _check_shape_dims(shape, data_format):
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
                "shape is invalid, which only support 5D Tensor")

# pylint: disable=locally-disabled,too-many-arguments
def _shape_check(shape_x, shape_mean, shape_variance, format_x):
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

    util.check_shape_rule(shape_x)

    util.check_tensor_shape_size(shape_x)

    _check_shape_dims(shape_x, format_x)
    if len(shape_mean) != 1:
        raise RuntimeError("shape mean is invalid, which only support 1D Tensor")
    if len(shape_variance) != 1:
        raise RuntimeError("shape var is invalid, which only support 1D Tensor")

    _check_dims_equal(shape_x, shape_mean, format_x)
    _check_dims_equal(shape_x, shape_variance, format_x)


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name,protected-access
def bninference_d_compute(x, mean, variance, scale, b, y, epsilon, format_x):
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
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    mean_broadcast = te.lang.cce.broadcast(mean, shape_x)
    var_broadcast = te.lang.cce.broadcast(variance, shape_x)
    mean_add = te.lang.cce.vadd(x, mean_broadcast)
    res_y = te.lang.cce.vmul(var_broadcast, mean_add)
    res = [res_y]
    return res


def _dtype_check(x, mean, variance):
    dtype_x = x.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")

    checklist = ["float32", "float16"]
    util.check_dtype_rule(dtype_mean.lower(), checklist)
    util.check_dtype_rule(dtype_variance.lower(), checklist)
    util.check_dtype_rule(dtype_x.lower(), checklist)


def para_shape_check(x, mean, variance, format_x):
    """
    :param x:input tensor
    :param mean:mean tensor
    :param variance: var tensor
    :param format_x: format tensor
    :return: None
    """
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")
    shape_x = x.get("shape")
    util.check_shape_rule(shape_mean)
    util.check_shape_rule(shape_variance)
    util.check_tensor_shape_size(shape_mean)
    util.check_tensor_shape_size(shape_variance)
    _shape_check(shape_x, shape_mean, shape_variance, format_x)


def para_check(x, mean, variance, use_global_stats, kernel_name):
    """
    :param x:input tensor
    :param mean: mean tensor
    :param variance: var tensor
    :param use_global_stats: inference type
    :param kernel_name: kernel_name
    :return: none
    """
    format_x = x.get("format")
    _format_check(x)
    _dtype_check(x, mean, variance)
    if not use_global_stats:
        raise RuntimeError("use_global_stats can only be true in BN inference")
    para_shape_check(x, mean, variance, format_x)
    util.check_kernel_name(kernel_name)


def gen_tensor(x, mean, variance):
    """
    :param x:x tensor
    :param mean: mean tensor
    :param variance:var tensor
    :return:
    x_input:x
    mean_input:mean
    var_input:var
    scale:scale,not use
    b:not use
    """
    shape_x = x.get("shape")
    format_x = x.get("format")
    dtype_x = x.get("dtype")
    if format_x in ("ND", "NCHW"):
        if len(shape_x) == 1:
            index_C = 0
        else:
            index_C = 1
    else:
        c1 = shape_x[1]
        c0 = shape_x[4]
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")

    if format_x in ("ND", "NCHW"):
        shape_mean = [1]*len(shape_x[:index_C]) + list(shape_mean) \
                     + [1]*len(shape_x[index_C+1:])
        shape_variance = [1]*len(shape_x[:index_C]) + list(shape_variance) \
                         + [1]*len(shape_x[index_C+1:])
    else:
        shape_mean = [1]+[c1]+[1, 1]+[c0]
        shape_variance = [1]+[c1]+[1, 1]+[c0]
    x_input = tvm.placeholder(shape_x, name="x", dtype=dtype_x.lower())
    mean_input = tvm.placeholder(shape_mean, name="mean",
                                 dtype=dtype_x.lower())
    variance_input = tvm.placeholder(shape_variance, name="variance",
                                     dtype=dtype_x.lower())

    scale = []
    offect = []
    return x_input, mean_input, variance_input, scale, offect

# pylint: disable=locally-disabled,no-member
@util.check_input_type(dict, dict, dict, (dict, NONETYPE), (dict, NONETYPE),
                       dict, float, float, bool, int, str)
def bninference_d(x, mean, variance, scale, offect, y, momentum, epsilon,
                  use_global_stats, mode, kernel_name="bninference"):
    """
    algorithm: fused_batch_norm
    Batch normalization.
    Note that the size of 5D Tensors are defined by "NC1HWC0".
    The input tensor's dimension C should be equal.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.

    mean: dict
        dict of mean, A Tensor for population mean.
        Used for inference only, must be empty for training.
    variance: dict
        dict of variance, A Tensor for population variance.
        Used for inference only, must be empty for training.
    scale: dict
        dict of scale,
        A Tensor for scaling factor, to scale the normalized x.
    offset: dict
        dict of offset, A Tensor for offset, to shift to the normalized x.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    momentum:float
        max or min num
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.001`.
    mode:int
      format not use
    use_global_stats: bool
        A bool value indicates the operation for train (default) or inference.default true
    kernel_name: str
        kernel name, default value is "batch_norm"

    Returns
    -------
    None
    """
    para_check(x, mean, variance, use_global_stats, kernel_name)
    x_input, mean_input, variance_input, scale, offect = gen_tensor(x, mean, variance)
    format_x = x.get("foramt")
    res = bninference_d_compute(x_input, mean_input,
                                variance_input, scale, offect, y, epsilon, format_x)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    tensor_list = [x_input, mean_input, variance_input] + list(res)
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
