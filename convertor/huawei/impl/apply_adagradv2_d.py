#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

apply_adagradv2_d
"""

import operator
from functools import reduce as functools_reduce

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2 << 30

# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("apply_adagradv2_d")
def apply_adagradv2_d_compute(var,
                              accum,
                              lr,
                              grad,
                              out_var,
                              out_accum,
                              epsilon,
                              update_slots,
                              kernel_name="apply_adagradv2_d"):
    """
    ApplyAdagradv2 algorithm:

    if update_slots
        accum += grad ** 2
    var -= lr * grad / (accum.sqrt() + epsilon)

    Parameters:
    ----------
    var: placeholder, input tensor with dtype float16 or float32

    accum: placeholder, has same shape and dtype as var

    lr: placeholder, has same dtype as var

    grad: placeholder, has same shape and dtype as var

    out_var: output var, has same shape and dtype as var

    out_accum: output accum, has same shape and dtype as var

    epsilon: scalar, has same dtype as var

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagradv2_d".

    Returns:
    -------
    out_var, out_accum
    """
    input_dtype = var.dtype
    if input_dtype == "float16":
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")

    if update_slots is True:
        grad_square = te.lang.cce.vmul(grad, grad)
        out_accum = te.lang.cce.vadd(accum, grad_square)
    else:
        out_accum = accum

    lr_brc = te.lang.cce.broadcast(lr, grad.shape)
    lr_grad = te.lang.cce.vmul(grad, lr_brc)
    sqrt_accum = te.lang.cce.vsqrt(out_accum)
    sqrt_accum_epsilon = te.lang.cce.vadds(sqrt_accum,
                                           tvm.const(epsilon, "float32"))
    update = te.lang.cce.vdiv(lr_grad, sqrt_accum_epsilon)
    out_var = te.lang.cce.vsub(var, update)

    if input_dtype == "float16":
        out_var = te.lang.cce.cast_to(out_var, "float16")
        out_accum = te.lang.cce.cast_to(out_accum, "float16")

    return out_var, out_accum


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    var_shape = []
    for var, name in zip(dict_list, name_list):
        shape = var.get('shape')
        dtype = var.get('dtype').lower()
        if name == 'var':
            var_shape = list(shape)
        if name != 'lr' and var_shape != list(shape):
            raise RuntimeError("The shape of var, accum, grad must be equal.")
        if name == 'lr' and shape[0] != 1:
            raise RuntimeError("The shape of lr must be scalar.")

        util.check_dtype_rule(dtype, ('float32', 'float16'))
        util.check_shape_rule(shape, max_shape_num=SHAPE_SIZE_LIMIT)
        util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
        shape_refine = (functools_reduce(operator.mul, shape), )
        list_placeholder.append(
            tvm.placeholder(shape=shape_refine, name=name, dtype=dtype))
    return list_placeholder

# pylint: disable=unbalanced-tuple-unpacking,invalid-name
@util.check_input_type(dict, dict, dict, dict, dict, dict, float, bool, str)
def apply_adagradv2_d(var,
                      accum,
                      lr,
                      grad,
                      out_var,
                      out_accum,
                      epsilon,
                      update_slots=True,
                      kernel_name="apply_adagradv2_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    if update_slots
        accum += grad ** 2
    var -= lr * grad / (accum.sqrt() + epsilon)

    Parameters:
    ----------
    var: the dict of var, only support float16, float32

    accum: the dict of accum, only support float16, float32

    lr: the dict of lr, only support float16, float32

    grad: the dict of grad, only support float16, float32

    out_var: the dict of output, only support float16, float32

    out_accum: the dict of output, only support float16, float32

    epsilon: scalar, only support float16, float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagradv2_d".

    Returns
    -------
    None
    """
    input_name_list = ['var', 'accum', 'lr', 'grad']
    util.check_kernel_name(kernel_name)
    var, accum, lr, grad = _get_placeholder([var, accum, lr, grad],
                                            input_name_list)

    out_var, out_accum = apply_adagradv2_d_compute(var, accum, lr, grad,
                                                   out_var, out_accum, epsilon,
                                                   update_slots)
    outs = [out_var, out_accum]
    build_list = [var, accum, lr, grad, out_var, out_accum]
    with tvm.target.cce():
        sch = generic.auto_schedule(outs)
    config = {"name": kernel_name, "tensor_list": build_list}

    te.lang.cce.cce_build_code(sch, config)
