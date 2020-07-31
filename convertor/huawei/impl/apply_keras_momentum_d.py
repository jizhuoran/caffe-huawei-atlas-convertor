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

apply_keras_momentum
"""
from functools import reduce as functools_reduce
import operator
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util


# pylint: disable=too-many-arguments,invalid-name,too-many-locals
# pylint: disable=unused-argument
@fusion_manager.register("apply_keras_momentum_d")
def apply_keras_momentum_d_compute(var, accum, lr, grad, momentum,
                                   out_var, out_accum,
                                   use_nesterov,
                                   kernel_name="apply_keras_momentum_d"):
    """
    the operator's compute
    :param var: weight, placeholder
    :param accum: accum, placeholder
    :param lr: learning rate, placeholder
    :param grad: gradient, placeholder
    :param momentum: nesterov momentum, placeholder
    :param out_var: updated of var
    :param out_accum: updated of accum
    :param use_nesterov: bool
    :return: out_var, out_accum
    """
    inp_dtype = var.dtype
    # check the instruction supports or not
    vmul_support = tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul",
                                                           "float32")
    if inp_dtype == "float32" and not vmul_support:
        raise RuntimeError(
            "Input dtype is float32, but do not support on the platform")

    # cast to float32 for higher accuracy on different platforms
    if inp_dtype == "float16" and vmul_support:
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")
        momentum = te.lang.cce.cast_to(momentum, "float32")

    # update var and accum according to the momentum scheme
    # accum = accum * momentum - grad * lr
    lr_brc = te.lang.cce.broadcast(lr, var.shape)
    momentum_brc = te.lang.cce.broadcast(momentum, var.shape)
    accum_momen = te.lang.cce.vmul(accum, momentum_brc)
    grad_lr = te.lang.cce.vmul(grad, lr_brc)
    out_accum = te.lang.cce.vsub(accum_momen, grad_lr)

    # var = var + accum * momentum - grad * lr
    if use_nesterov is True:
        accum_momen2 = te.lang.cce.vmul(
            out_accum, momentum_brc)
        add_var_am = te.lang.cce.vadd(var, accum_momen2)
        out_var = te.lang.cce.vsub(add_var_am, grad_lr)
    # var = var + accum
    else:
        out_var = te.lang.cce.vadd(var, out_accum)

    if inp_dtype == "float16" and vmul_support:
        out_accum = te.lang.cce.cast_to(out_accum, "float16")
        out_var = te.lang.cce.cast_to(out_var, "float16")

    return out_var, out_accum


def _check_para_and_get_place(check_list, scalar_input, tensor_input, input_dict):
    var_shape = input_dict["var"].get("shape")
    var_dtype = input_dict["var"].get("dtype")
    list_placeholder = []
    for key, value in input_dict.items():
        shape = util.scalar2tensor_one(value.get("shape"))
        util.check_shape_rule(shape)
        util.check_tensor_shape_size(shape)
        if value in scalar_input:
            if not util.is_scalar(shape):
                raise RuntimeError("The shape of ", key, " must be scalar")
        if value in tensor_input:
            if shape != var_shape:
                raise RuntimeError("The shape of", key,
                                   "must be the same as the var")

        dtype = value.get("dtype").lower()
        util.check_dtype_rule(dtype, check_list)
        if dtype != var_dtype:
            raise RuntimeError("The dtype of", key,
                               "must be the same as the var")

        shape_refine = (functools_reduce(operator.mul, shape), )
        list_placeholder.append(tvm.placeholder(
            shape=shape_refine, name=key, dtype=dtype))
    return list_placeholder


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, bool, bool, str)
# pylint: disable=too-many-arguments,unused-argument,unbalanced-tuple-unpacking
def apply_keras_momentum_d(var, accum, lr, grad, momentum,
                           out_var, out_accum,
                           use_locking=False, use_nesterov=False,
                           kernel_name="apply_keras_momentum_d"):
    """
    Update '*var' according to the momentum scheme.

    accum = accum * momentum - grad * lr
    if use_nesterov is True:
        var = var + accum * momentum - grad * lr
    else:
        var = var + accum

    Parameters
    ----------
    var : dict of tensor var, include shape and dtype.

    accum : dict of tensor accum, include shape and dtype.

    lr: dict of scalar lr(learning rate), include shape and dtype.

    grad: dict of tensor grad, include shape and dtype.

    momentum: dict of scala, include shape and dtype.

    out_var: dict of updated var.

    out_accum: dict of updated accum.

    use_locking: bool, default value is "False",
                 if "True", var will be updated by using Nesterov momentum.

    use_nesterov: bool, default value is "False".

    kernel_name :  kernel name, default value is "apply_keras_momentum_d"

    Returns
    -------
    None
    """
    # Check the input paras
    util.check_kernel_name(kernel_name)
    input_dict = {"var": var, "accum": accum, "lr": lr, "grad": grad, "momentum": momentum}
    scalar_input = (lr, momentum)
    tensor_input = (var, accum, grad)

    check_list = ("float16", "float32")
    data_var, data_accum, data_lr, data_grad, data_momentum = _check_para_and_get_place(
        check_list, scalar_input, tensor_input, input_dict)

    out_var, out_accum = apply_keras_momentum_d_compute(data_var, data_accum,
                                                        data_lr, data_grad, data_momentum,
                                                        out_var, out_accum,
                                                        use_nesterov,
                                                        kernel_name)

    inputlist = (data_var, data_accum, data_lr, data_grad, data_momentum, out_var, out_accum)
    with tvm.target.cce():
        sch = generic.auto_schedule([out_var, out_accum])
    config = {"name": kernel_name,
              "tensor_list": inputlist}

    te.lang.cce.cce_build_code(sch, config)
