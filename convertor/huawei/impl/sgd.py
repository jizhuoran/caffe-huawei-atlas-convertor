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

sgd
"""

import te.lang.cce
from impl.util.util_apply_op_schedule import (ApplyOpConfig,
                                              common_apply_op_process)
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi.cce import util

NUM_ZERO = 0.0

# pylint: disable=too-many-arguments,unused-argument,invalid-name,too-many-locals
@fusion_manager.register("sgd")
def sgd_compute(parameters, gradient, learning_rate, accum, momentum, stat,
                update, dampening, weight_decay, nesterov, kernel_name="sgd"):
    """
    Update '*parameters' according to the SGD algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        parameters -= grad * lr + accum * momentum * lr
    else:
        parameters -= accum * lr

    Parameters:
    ----------
    parameters : mutable tensor parameters.

    gradient : tensor grad.

    learning_rate : scalar lr.

    accum: mutable tensor accum.

    momentum : scalar momentum.

    stat : mutable tensor stat.

    update: out dict.

    dampening: (float, optional): dampening for momentum (default: 0)

    weight_decay: weight decay (L2 penalty) (default: 0)

    nesterov: bool. If true, use nesterov computing grad,
    default value is False.

    kernel_name : cce kernel name, default value is "sgd" (optional).

    Returns:
    -------
    outs
    """
    dtype = parameters.dtype
    support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmuls", "float32")
    if not support:
        raise RuntimeError(
            "only support float16 while need to cast to float32")
    if dtype == "float16":
        parameters = te.lang.cce.cast_to(parameters, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        learning_rate = te.lang.cce.cast_to(learning_rate, "float32")
        gradient = te.lang.cce.cast_to(gradient, "float32")
        momentum = te.lang.cce.cast_to(momentum, "float32")
        stat = te.lang.cce.cast_to(stat, "float32")

    # if weight_decay != 0.0
    if weight_decay != 0.0:
        parameters = te.lang.cce.vmuls(parameters, tvm.const(1.0, 'float32'))
        grad_delta = te.lang.cce.vmuls(parameters, weight_decay)
        gradient = te.lang.cce.vadd(gradient, grad_delta)

    stat_mid = te.lang.cce.vmuls(stat, tvm.const(-1, "float32"))
    stat_act = te.lang.cce.vadds(stat_mid, tvm.const(1, "float32"))

    dampening_t = te.lang.cce.vmuls(stat_act, dampening)

    # update accum
    accum_delta = tvm.compute(accum.shape,
                              lambda *indice: accum(*indice) * momentum[0],
                              tag='elewise_single_VS_mul')

    gradient_damp = te.lang.cce.vmul(gradient, dampening_t)
    accum_t = te.lang.cce.vadd(accum_delta, gradient)
    if dampening != 0.0:
        accum_t = te.lang.cce.vsub(accum_t, gradient_damp)

    # update parameters
    if nesterov:
        parameters_delta = tvm.compute(gradient.shape,
                                       lambda *indice: gradient(*indice) *
                                       learning_rate[0],
                                       tag='elewise_single_VS_mul')
        parameters_delta_2 = tvm.compute(
            accum_t.shape,
            lambda *indice: accum_t(*indice) * momentum[0],
            tag='elewise_single_VS_mul')
        parameters_delta_2 = tvm.compute(parameters_delta_2.shape,
                                         lambda *indice: parameters_delta_2(
                                             *indice) * learning_rate[0],
                                         tag='elewise_single_VS_mul')
        parameters_delta = te.lang.cce.vadd(parameters_delta,
                                            parameters_delta_2)
        parameters_t = te.lang.cce.vsub(parameters, parameters_delta)
    else:
        parameters_delta = tvm.compute(accum_t.shape,
                                       lambda *indice: accum_t(*indice) *
                                       learning_rate[0],
                                       tag='elewise_single_VS_mul')
        parameters_t = te.lang.cce.vsub(parameters, parameters_delta)

    # update stat
    stat_t = te.lang.cce.vmuls(stat_act, tvm.const(NUM_ZERO, 'float32'))

    if dtype == "float16":
        parameters_t = te.lang.cce.cast_to(parameters_t, "float16")
        accum_t = te.lang.cce.cast_to(accum_t, "float16")
        stat_t = te.lang.cce.cast_to(stat_t, "float16")

    def _compute(*index):
        return accum_t(*index), parameters_t(*index), stat_t(
            *index), parameters_t(*index)

    return tvm.compute(parameters.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, float, float,
                       bool, str)
def sgd(parameters, gradient, learning_rate, accum, momentum, stat, update,
        dampening, weight_decay, nesterov, kernel_name="sgd"):
    """
    Update '*parameters' according to the SGD algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        parameters -= grad * lr + accum * momentum * lr
    else:
        parameters -= accum * lr

    Parameters:
    ----------
    parameters : mutable tensor parameters.

    gradient : tensor grad.

    learning_rate : scalar lr.

    accum: mutable tensor accum.

    momentum : scalar momentum.

    stat : mutable tensor stat.

    update: out dict.

    dampening: (float, optional): dampening for momentum (default: 0)

    weight_decay: weight decay (L2 penalty) (default: 0)

    nesterov: bool. If true, use nesterov computing grad,
    default value is False.

    kernel_name : cce kernel name, default value is "sgd" (optional).

    Returns:
    -------
    None
    """
    if nesterov and dampening != 0:
        raise RuntimeError("Nesterov requires zero dampening!")
    if weight_decay < 0:
        raise RuntimeError("weight_decay must >=0.")

    input_dict = (parameters, gradient, learning_rate, accum, momentum, stat)
    args = ApplyOpConfig.TensorArgs(
        input_dict,
        sgd_compute,
        update,
        17 if nesterov else 9, )

    name = ApplyOpConfig.TensorName(all=(
        'parameters', 'gradient', 'learning_rate', 'accum', 'momentum', 'stat'),
                                    scalar=('learning_rate', 'momentum'),
                                    reuse=('accum', 'parameters', 'stat'))
    options = ApplyOpConfig.TensorOptions(
        attrs=[dampening, weight_decay, nesterov])
    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
