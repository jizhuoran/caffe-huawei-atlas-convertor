"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lars_v2_update op
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_build import build_config
from te.platform.cce_build import build_config_update
from topi import generic
from topi.cce import util
from te import platform as tbe_platform

from functools import reduce as functools_reduce
import operator
# General limitation of the reduce size for input shape: 2**30
SHAPE_SIZE_LIMIT = 2**30


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("lars_v2_update")
def lars_v2_update_compute(inputs_data,
                           hyperparam,
                           epsilon,
                           use_clip,
                           out,
                           kernel_name="lars"):

    """
    lars_update compute

    Parameters:
    ----------
    inputs_data: list
        the placeholders of input data
    hyperparam: float
        default value is 0.001
    epsilon: float
        default value is 1e-5
    use_clip: bool
        default value is "False".
    out: dict
        output contains shape and dtype attributes.
    kernel_name : str
        kernel name, default value is "lars_update"

    Returns:
    None
    """
    weight, grad, weight_s, grad_s, weight_decay, learning_rate = inputs_data

    weight_norm = te.lang.cce.vsqrt(weight_s)
    grad_norm = te.lang.cce.vsqrt(grad_s)

    coeff_weight_norm = te.lang.cce.vmuls(weight_norm, hyperparam)
    weight_norm_decay = te.lang.cce.vmul(weight_norm, weight_decay)
    weight_grad_norm = te.lang.cce.vadd(weight_norm_decay, grad_norm)
    norm_res = te.lang.cce.vadds(weight_grad_norm, epsilon)
    coeff = te.lang.cce.vdiv(coeff_weight_norm, norm_res)

    if use_clip:
        coeff_clip = te.lang.cce.vdiv(coeff, learning_rate)
        coff_max = te.lang.cce.vmins(coeff_clip, tvm.const(1, dtype=weight.dtype))
        clip_coff = te.lang.cce.vmaxs(coff_max, tvm.const(0, dtype=weight.dtype))
        coeff_broadcast = te.lang.cce.broadcast(clip_coff, weight.shape)
    else:
        coeff_broadcast = te.lang.cce.broadcast(coeff, weight.shape)

    weight_decay_broadcast = te.lang.cce.broadcast(weight_decay, weight.shape)
    weight_weight_decay = te.lang.cce.vmul(weight, weight_decay_broadcast)
    weight_grad = te.lang.cce.vadd(weight_weight_decay, grad)

    out = te.lang.cce.vmul(weight_grad, coeff_broadcast)

    return out


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, float,
                       float, bool, str)
def lars_v2_update(weight, grad, weight_s, grad_s, weight_decay, learning_rate,
                   out, hyperparam=0.001, epsilon=1e-5, use_clip=False,
                   kernel_name="lars_update"):
    """
    the opreator's compute
    hyper_weight_norm = hyperparam * sqrt(weight_s)
    grad_weight_norm = sqrt(grad_s) + weight_decay*sqrt(weight_s) + epsilon
    grad_weight = grad + weight_decay * weight

    if use_clip == True:
        coeff = hyper_weight_norm / grad_weight_norm
        coeff = min(coeff / learning_rate, 1)
        coeff = max(coeff, 0)
    else:
        coeff = hyper_weight_norm / grad_weight_norm

    grad_new = coeff * grad_weight
    Parameters:
    ----------
    weight: dict
        input tensor contains shape and dtype attributes.
        only support float32.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype and shape as 'weight'.
    weight_s: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    grad_s: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    weight_decay: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    learning_rate: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    out: dict
        output tensor contains shape and dtype attributes.
        Must have the same dtype and shape  as 'weight'.
    hyperparam: float
        default value is 0.001
    epsilon: float
        default value is 1e-5
    use_clip: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "lars_update"

    Returns:
    None
    """


    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    inputs = [weight, grad, weight_s, grad_s, weight_decay, learning_rate]

    weight_shape = weight.get("shape")
    grad_shape = grad.get("shape")
    weight_dtype = weight.get("dtype")
    grad_dtype = grad.get("dtype")
    if list(weight_shape) != list(grad_shape):
        raise RuntimeError("weight and grad must be the same shape")

    if grad_dtype != weight_dtype:
        raise RuntimeError("wight and grad must be the same dtype")

    vdiv_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vdiv", "float32")
    if weight_dtype == "float32" and not vdiv_support:
        raise RuntimeError(
            "Input dtype is float32, but do not support on the platform")

    input_place_holders = []
    for i, input_val in enumerate(inputs):
        input_dtype = input_val.get("dtype").lower()
        input_shape = input_val.get("shape")
        util.check_shape_rule(input_shape)
        util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)
        util.check_dtype_rule(input_dtype, check_list)
        shape_one_dim = (functools_reduce(operator.mul, input_shape),)
        input_place_holders.append(tvm.placeholder(shape_one_dim,
                                                   name="input_data_%d" % i,
                                                   dtype=input_dtype))

    res = lars_v2_update_compute(input_place_holders,
                               hyperparam,
                               epsilon,
                               use_clip,
                               out,
                               kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    data = input_place_holders
    data.append(res)

    new_config = build_config_update(build_config, "dummy_placeholder", True)
    with new_config:
        tvm.build(schedule, data, "cce", name=kernel_name)


