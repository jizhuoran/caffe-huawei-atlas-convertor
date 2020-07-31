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

bn_infer_grad
"""

from __future__ import absolute_import
from __future__ import division

from te import tvm
from te import platform as tbe_platform
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.platform import cce_policy
from topi import generic
from topi.cce import util
cce_policy.disableL2()

MAX_SHAPE_NUM = 10000000
SCALAR_ONE = 1


def _check_shape(shape_x, shape_scale):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape

    Returns
    -------
    None
    """
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)

    util.check_shape_rule(shape_scale)
    util.check_tensor_shape_size(shape_scale)

    if len(shape_x) != 5 or len(shape_scale) != 5:
        raise RuntimeError(
            "The data format is 5HD, "
            "but some input's shape length is not 5")

    dim_c1 = shape_x[1]
    dim_c0 = shape_x[4]

    if shape_scale[1] != dim_c1 or shape_scale[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal")


def _check_dtype(dtype_x, dtype_scale, dtype_offset,
                 dtype_mean, dtype_variance):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
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
    util.check_dtype_rule(dtype_scale.lower(), ("float32", "float16"))
    util.check_dtype_rule(dtype_offset.lower(), ("float32", "float16"))
    util.check_dtype_rule(dtype_mean.lower(), ("float32", "float16"))
    util.check_dtype_rule(dtype_variance.lower(), ("float32", "float16"))


# pylint: disable=locally-disabled,invalid-name,too-many-arguments
# pylint: disable=locally-disabled,too-many-locals,unused-argument
@fusion_manager.register("bn_infer")
def bn_infer_compute(x, scale, offset, mean, variance,
                     y, epsilon, kernel_name="bn_inf"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization for inference

    Parameters
    ----------
    x: TVM tensor
        contains x data
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
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    res: TVM tensor
        the result of bn_training_update_v2 compute for inference
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    # compute the oefficient of y
    multiplier_add = te.lang.cce.vadds(variance, epsilon)
    multiplier_sqrt = te.lang.cce.vsqrt(multiplier_add)
    multiplier_div = te.lang.cce.vdiv(scale, multiplier_sqrt)
    multiplier = te.lang.cce.broadcast(multiplier_div, shape_x)

    addend_mul = te.lang.cce.vmul(multiplier_div, mean)
    addend_sub = te.lang.cce.vsub(offset, addend_mul)
    addend = te.lang.cce.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul",
                                                    "float32"):
        is_cast = True
        x = te.lang.cce.cast_to(x, "float32")

    res = te.lang.cce.vadd(te.lang.cce.vmul(multiplier, x), addend)
    if is_cast:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, dict, dict,
                       dict, dict, float, str)
def bn_infer(x, scale, offset, mean, variance, y,
             epsilon, kernel_name="bn_infer"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization for inference

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    scale: dict
        dict of scale, A 5HD Tensor for scale.
    offset: dict
        dict of offset, A 5HD Tensor for offset.
    mean: dict
        dict of mean, A `Tensor`.
        dict of scale, A 5HD Tensor for mean.
    variance: dict
        dict of batch_variance, A `Tensor`.
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_infer"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_scale = scale.get("shape")

    dtype_x = x.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")

    data_format = x.get("format").upper()

    if data_format not in ("NC1HWC0",):
        raise RuntimeError(
            "Format only support 5HD")

    _check_shape(shape_x, shape_scale)
    util.compare_tensor_dict_key(scale, offset, "shape")
    util.compare_tensor_dict_key(scale, mean, "shape")
    util.compare_tensor_dict_key(scale, variance, "shape")

    _check_dtype(dtype_x, dtype_scale, dtype_offset, dtype_mean, dtype_variance)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    scale_input = tvm.placeholder(shape_scale, name="scale_input",
                                  dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_scale, name="offset_input",
                                   dtype=dtype_offset.lower())
    mean_input = tvm.placeholder(shape_scale, name="mean_input",
                                 dtype=dtype_mean.lower())
    variance_input = tvm.placeholder(shape_scale, name="variance_input",
                                     dtype=dtype_variance.lower())

    res = bn_infer_compute(x_input, scale_input, offset_input,
                           mean_input, variance_input, y,
                           epsilon, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [x_input, scale_input, offset_input,
                   mean_input, variance_input, res]

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
