#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

apply_proximal_adagrad
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig
from impl.util.util_compute import sign
from te import platform as tbe_platform

CONST_ZERO = 0
CONST_ONE = 1


def _check_shape_is_same(var, accum, grad):
    """
    Check whether var.shape accum.shape and grad.shape is same or not.

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.

    Returns:
    None
    """
    shape_var = var.get("shape")
    shape_accum = accum.get("shape")
    shape_grad = grad.get("shape")
    if shape_var != shape_accum or shape_var != shape_grad:
        raise RuntimeError("var.shape accum.shape and grad.shape must be same")

# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-locals,unused-argument,invalid-name
@fusion_manager.register("apply_proximal_adagrad_d")
def apply_proximal_adagrad_d_compute(var, accum, lr, l1, l2, grad, var_out,
                                     accum_out, use_locking=False,
                                     kernel_name="apply_proximal_adagrad"):
    """
    the operator's compute
    accum += grad * grad
    learning_rate = lr_broad * rsqrt(accum)
    prox_v = var - grad * learning_rate
    if l1 > 0 :
        var = sign(prox_v)/(1+learning_rate*l2)*max{|prox_v|-learning_rate*l1,0}
    else:
        var = prox_v / (1+l2*learning_rate)

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "apply_proximal_adagrad_d"

    Returns:
        the value of out_var, accum_out, out_data
    """
    dtype = var.dtype
    has_improve_precision = False
    if dtype == "float16" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt",
                                                "float32"):
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        l1 = te.lang.cce.cast_to(l1, "float32")
        l2 = te.lang.cce.cast_to(l2, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")
        has_improve_precision = True

    lr_broad = te.lang.cce.broadcast(lr, var.shape)
    l1_broad = te.lang.cce.broadcast(l1, var.shape)
    l2_broad = te.lang.cce.broadcast(l2, var.shape)

    grad_2 = te.lang.cce.vmul(grad, grad)
    accum_out = te.lang.cce.vadd(accum, grad_2)
    accum_sqrt = te.lang.cce.vsqrt(accum_out)
    learning_rate = te.lang.cce.vdiv(lr_broad, accum_sqrt)
    learning_rate_grad = te.lang.cce.vmul(grad, learning_rate)
    prox_v = te.lang.cce.vsub(var, learning_rate_grad)
    l2_lr = te.lang.cce.vmul(l2_broad, learning_rate)
    l2_lr_1 = te.lang.cce.vadds(l2_lr, tvm.const(CONST_ONE, "float32"))
    prox_v_abs = te.lang.cce.vabs(prox_v)
    prox_v_sign = sign(prox_v)
    learning_rate_l1 = te.lang.cce.vmul(learning_rate, l1_broad)
    prox_v_l1 = te.lang.cce.vsub(prox_v_abs, learning_rate_l1)
    max_value = te.lang.cce.vmax(prox_v_l1, te.lang.cce.broadcast(
        tvm.const(CONST_ZERO, "float32"), prox_v.shape))
    var_res = te.lang.cce.vmul(prox_v_sign, max_value)
    var_new = te.lang.cce.vdiv(var_res, l2_lr_1)
    output_data = te.lang.cce.vadds(var_new, tvm.const(CONST_ZERO, "float32"))
    output_accum_data = te.lang.cce.vadds(accum_out, tvm.const(CONST_ZERO, "float32"))

    if has_improve_precision:
        var_new = te.lang.cce.cast_to(var_new, "float16")
        accum_out = te.lang.cce.cast_to(accum_out, "float16")
        output_data = te.lang.cce.cast_to(output_data, "float16")
        output_accum_data = te.lang.cce.cast_to(output_accum_data, "float16")

    # this compute is for muti output
    def _compute(*index):
        return var_new(*index), accum_out(*index), output_data(*index), output_accum_data(*index)

    return tvm.compute(var.shape, _compute, name="outputs")

@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, bool, str)
def apply_proximal_adagrad_d(var, accum, lr, l1, l2, grad, var_out,
                             accum_out, use_locking=False,
                             kernel_name="apply_proximal_adagrad_d"):
    """
    Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "apply_proximal_adagrad_d"

    Returns:
    None
    """
    _check_shape_is_same(var, accum, grad)

    input_dict = (var, accum, lr, l1, l2, grad)
    args = ApplyOpConfig.TensorArgs(input_dict, apply_proximal_adagrad_d_compute,
                                    [var_out, accum_out], 15)
    name = ApplyOpConfig.TensorName(all=('var', 'accum', 'lr',
                                         'l1', 'l2', 'grad'),
                                    scalar=('lr', 'l1', 'l2'),
                                    reuse=('var', 'accum'))
    common_apply_op_process(ApplyOpConfig(args, name), kernel_name)
