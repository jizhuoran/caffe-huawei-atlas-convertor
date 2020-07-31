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
from topi import generic
from topi.cce import util

MAX_SHAPE_NUM = 10000000
SCALAR_ONE = 1


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=locally-disabled,too-many-arguments
@fusion_manager.register("bn_infer_grad")
def bn_infer_grad_compute(grads, scale, batch_variance, x_backprop,
                          epsilon, kernel_name="bn_infer_grad"):
    """
    Compute for bn_infer_grad_compute
    x_norm:(x-input_reserve_space_1)*
            np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_scale:np.sum(y*(x-input_reserve_space_1)*
                         np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_offset: np.sum(y)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads. Must be one of the following
        type: `float16`, `float32`.
    scale: TVM tensor 5D
        the placeholder of x. Must be one of the following
        type: `float32`.
    batch_variance: TVM tensor 5D
        the placeholder of batch_variance. Must be one of the following
        type: `float32`.
    x_backprop: dict
        dict of x_norm, A 5D Tensor for output x_norm.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_infer_grad"

    Returns
    -------
    res: x_backprop
   """
    shape_x = te.lang.cce.util.shape_to_list(grads.shape)

    is_cast = False
    if grads.dtype == "float16" and \
           tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        is_cast = True
        grads = te.lang.cce.cast_to(grads, "float32")

    data_adds = te.lang.cce.vadds(batch_variance, epsilon)
    data_rsqrt = te.lang.cce.vsqrt(data_adds)
    shape_var = te.lang.cce.util.shape_to_list(batch_variance.shape)
    data_cast = te.lang.cce.broadcast(tvm.const(SCALAR_ONE, "float32"),
                                      shape_var)
    data_rsqrts = te.lang.cce.vdiv(data_cast, data_rsqrt)

    scale_mul = te.lang.cce.vmul(scale, data_rsqrts)
    scale_mul_broadcast = te.lang.cce.broadcast(scale_mul, shape_x)
    res = te.lang.cce.vmul(scale_mul_broadcast, grads)
    if is_cast:
        res = te.lang.cce.cast_to(res, "float16")

    return res


def _check_shape(shape_grads, shape_batch_variance):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_grads: list or tuple
        input grads's data shape
    shape_batch_variance: list or tuple
        input batch_variance's data shape
    Returns
    -------
    None
    """
    util.check_shape_rule(shape_grads, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_grads)

    util.check_shape_rule(shape_batch_variance, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_batch_variance)

    dim_c1 = shape_grads[1]
    dim_c0 = shape_grads[4]

    if len(shape_grads) != 5:
        raise RuntimeError(
            "This operator can only support 5D")
    if dim_c0 != 16:
        raise RuntimeError("shape_grads last dim must be 16")
    if len(shape_batch_variance) != 5:
        raise RuntimeError(
            "This operator can only support 5D")

    if shape_batch_variance[0] != 1 or shape_batch_variance[2] != 1\
            or shape_batch_variance[3] != 1:
        raise RuntimeError(
            "Dimensions except Dimension C must be one for shape_batch_mean")
    if shape_batch_variance[1] != dim_c1 or shape_batch_variance[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal")


@util.check_input_type(dict, dict, dict, dict, float, str)
def bn_infer_grad(grads, scale, batch_variance,
                  x_backprop, epsilon=0.0001,
                  kernel_name="bn_infer_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_infer_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
    scale: dict
        dict of scale, A 5D Tensor for input scale.
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
    x_backprop: dict
        dict of x_backprop, A 5D Tensor for output x_backprop.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_infer_grad"

    Returns
    -------
    None
    """

    shape_grads = grads.get("shape")
    shape_scale = scale.get("shape")
    shape_batch_variance = batch_variance.get("shape")

    input_grads_dtype = grads.get("dtype").lower()
    input_scale_dtype = scale.get("dtype").lower()
    batch_variance_dtype = batch_variance.get("dtype").lower()

    util.check_dtype_rule(input_grads_dtype, ("float32", "float16"))
    util.check_dtype_rule(input_scale_dtype, ("float32",))
    util.check_dtype_rule(batch_variance_dtype, ("float32",))

    _check_shape(shape_grads, shape_batch_variance)
    util.compare_tensor_dict_key(scale, batch_variance, "shape")

    grads_input = tvm.placeholder(shape_grads, name="grads_input",
                                  dtype=input_grads_dtype)
    scale_input = tvm.placeholder(shape_scale, name="x_input",
                                  dtype=input_scale_dtype)
    batch_variance_input = tvm.placeholder(shape_batch_variance,
                                           name="batch_variance_input",
                                           dtype=batch_variance_dtype)

    res = bn_infer_grad_compute(grads_input, scale_input,
                                batch_variance_input,
                                x_backprop, epsilon,
                                kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    tensor_list = [grads_input, scale_input, batch_variance_input, res]
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
