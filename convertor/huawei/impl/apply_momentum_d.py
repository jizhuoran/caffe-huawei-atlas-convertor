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

apply_momentum

  Op_description :
    Update '*var' according to the ApplyMomentum algorithm.

    # apply_momentum_d(var,
    #   accum,
    #   lr,
    #   grad,
    #   momentum,
    #   var_out,
    #   accum_out,
    #   use_nesterov,
    #   kernel_name='apply_momentum')

  Supportive_dtype_format :
    ['int32', 'int8', 'uint8', 'float32', 'float16']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : the input tensors must have the same shape and type.
    [2] All : shape size limit is 2147483648.
"""


import te.lang.cce
from te import tvm
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig

# scalar in apply_momentum
NUM_ZERO = 0.0


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name,too-many-locals
@fusion_manager.register("apply_momentum")
def apply_momentum_compute_d(var,
                             accum,
                             lr,
                             grad,
                             momentum,
                             var_out,
                             accum_out,
                             use_nesterov,
                             kernel_name='apply_momentum_d'):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        var -= grad * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    lr : scalar lr.

    grad : tensor grad.

    momentum : scalar momentum.

    var_out : the dict of output var.

    accum_out : the dict of output accum.

    use_nesterov: bool. If true, use nesterov computing grad,
        default value is False.

    kernel_name : cce kernel name, default value is "apply_momentum_d".

    Returns:
    -------
    None
    """

    # cast to float32 for higher accuracy
    dtype = var.dtype
    if dtype == "float16" and api_check_support("te.lang.cce.vadd", "float32"):
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")
        momentum = te.lang.cce.cast_to(momentum, "float32")

    # update accum
    accum_delta = tvm.compute(accum.shape,
                              lambda *indice: accum(*indice) * momentum[0],
                              tag='elewise_single_VS_mul')
    accum_t = te.lang.cce.vadd(accum_delta, grad)

    # update var
    if use_nesterov:
        var_delta = tvm.compute(grad.shape,
                                lambda *indice: grad(*indice) * lr[0],
                                tag='elewise_single_VS_mul')
        var_delta_2 = tvm.compute(
            accum_t.shape,
            lambda *indice: accum_t(*indice) * momentum[0],
            tag='elewise_single_VS_mul')
        var_delta_2 = tvm.compute(var_delta_2.shape,
                                  lambda *indice: var_delta_2(*indice) * lr[0],
                                  tag='elewise_single_VS_mul')
        var_delta = te.lang.cce.vadd(var_delta, var_delta_2)
        var_t = te.lang.cce.vsub(var, var_delta)
    else:
        var_delta = tvm.compute(accum_t.shape,
                                lambda *indice: accum_t(*indice) * lr[0],
                                tag='elewise_single_VS_mul')
        var_t = te.lang.cce.vsub(var, var_delta)

    if dtype == "float16":
        var_t = te.lang.cce.cast_to(var_t, "float16")
        accum_t = te.lang.cce.cast_to(accum_t, "float16")

    var_out_data = te.lang.cce.vadds(
        var_t, tvm.const(NUM_ZERO, var_t.dtype))
    accum_out_data = te.lang.cce.vadds(
        accum_t, tvm.const(NUM_ZERO, accum_t.dtype))

    def _compute(*index):
        return accum_t(*index), var_t(*index), var_out_data(*index), \
               accum_out_data(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, bool, str)
def apply_momentum_d(var,
                     accum,
                     lr,
                     grad,
                     momentum,
                     var_out,
                     accum_out,
                     use_nesterov=False,
                     kernel_name="apply_momentum_d"):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        var -= gard * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32.

    accum : the dict of mutable tensor accum.
        Must have the same data type as `var`.

    lr : the dict of scalar lr. Must have the same data type as `var`.

    grad : the dict of tensor grad. Must have the same data type as `var`.

    momentum : the dict of scalar momentum.
        Must have the same data type as `var`.

    var_out : the dict of output var.

    accum_out : the dict of output accum.

    use_nesterov: bool. If true, use nesterov computing grad,
        default value is False.

    kernel_name : cce kernel name, default value is "apply_momentum_d".

    Returns
    -------
    None
    """

    input_dict = (var, accum, lr, grad, momentum)

    args = ApplyOpConfig.TensorArgs(
        input_dict,
        apply_momentum_compute_d,
        [var_out, accum_out],
        8 if use_nesterov else 6,
    )
    name = ApplyOpConfig.TensorName(all=('var', 'accum', 'lr', 'grad',
                                         'momentum'),
                                    scalar=('lr', 'momentum'),
                                    reuse=('accum', 'var'))
    options = ApplyOpConfig.TensorOptions(attrs=use_nesterov)

    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
