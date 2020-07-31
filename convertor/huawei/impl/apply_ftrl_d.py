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

apply_ftrl_d
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig
from impl.util.util_build import set_bool_storage_config
from impl.util.util_compute import sign
from te import platform as tbe_platform

# scalar in apply_ftrl_d
NUM_ONE = 1.0
NUM_TWO = 2.0
NUM_ZERO = 0.0
NUM_M_ONE = -1.0


def _pow(data, index, bound):
    """
    Calculate power result for non-negative data.

    res = data^index
        = exp(index * ln(data)) if data > 0
        = 1                    if data = 0, index = 0
        = 0                   if data = 0, index is not 0

    Parameters:
    ----------
    data : base value of power operation.

    index: index value of power operation.

    bound : computation bound for data.

    Returns:
    -------
    power result of data^index
    """

    log_value = te.lang.cce.vlog(data)
    base = te.lang.cce.vmul(log_value, index)
    res = te.lang.cce.vexp(base)

    return res


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name,too-many-locals
@fusion_manager.register("apply_ftrl_d")
def apply_ftrl_d_compute(var,
                         accum,
                         linear,
                         grad,
                         lr,
                         l1,
                         l2,
                         lr_power,
                         var_out,
                         accum_out,
                         linear_out,
                         kernel_name='apply_ftrl_d'):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    accum_new = accum + grad * grad
    linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
    x = l1 * linear.sign - linear
    y = accum_new^(-lr_power) / lr + 2 * l2
    var = x / y if |linear| > l1 else 0.0
    accum = accum_new

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    linear : mutable tensor linear.

    grad : tensor grad.

    lr : scalar lr.

    l1 : scalar l1.

    l2 : scalar l2.

    lr_power : scalar lr_power.

    var_out : the dict of var output.

    accum_out : the dict of accum output.

    linear_out : the dict of linear output.

    kernel_name : cce kernel name, default value is "apply_ftrl_d" (optional).

    Returns:
    -------
    None
    """

    # cast to float32 for higher accuracy
    dtype = var.dtype
    has_improve_precision = False
    if dtype == "float16" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                "float32"):
        var_tmp = te.lang.cce.cast_to(var, "float32")
        accum_tmp = te.lang.cce.cast_to(accum, "float32")
        linear_tmp = te.lang.cce.cast_to(linear, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        l1 = te.lang.cce.cast_to(l1, "float32")
        l2 = te.lang.cce.cast_to(l2, "float32")
        lr_power = te.lang.cce.cast_to(lr_power, "float32")
        has_improve_precision = True
    else:
        var_tmp = te.lang.cce.vadds(var, tvm.const(NUM_ZERO, dtype))
        accum_tmp = te.lang.cce.vadds(accum, tvm.const(NUM_ZERO, dtype))
        linear_tmp = te.lang.cce.vadds(linear, tvm.const(NUM_ZERO, dtype))

    # broadcast scalar to appropriate shape
    zero_tensor = te.lang.cce.broadcast(tvm.const(NUM_ZERO, var_tmp.dtype),
                                        var.shape)
    lr = te.lang.cce.broadcast(lr, var.shape)
    l1 = te.lang.cce.broadcast(l1, var.shape)
    l2 = te.lang.cce.broadcast(l2, var.shape)
    lr_power = te.lang.cce.broadcast(lr_power, var.shape)

    # 1.accum_new = accum + grad^2
    gs = te.lang.cce.vmul(grad, grad)
    accum_new = te.lang.cce.vadd(accum_tmp, gs)

    # 2.linear += grad - (accum_new^(-lr_power)-accum^(-lr_power))/lr*var
    lr_power = te.lang.cce.vmuls(lr_power, tvm.const(NUM_M_ONE, var_tmp.dtype))
    accum_new_p = _pow(accum_new, lr_power, zero_tensor)
    accum_p = _pow(accum_tmp, lr_power, zero_tensor)
    accum_p = te.lang.cce.vsub(accum_new_p, accum_p)

    accum_p = te.lang.cce.vdiv(accum_p, lr)
    accum_p = te.lang.cce.vmul(accum_p, var_tmp)
    accum_p = te.lang.cce.vsub(grad, accum_p)
    linear_t = te.lang.cce.vadd(linear_tmp, accum_p)

    # 3.x_res = l1*linear.sign()-linear
    x_res = sign(linear_t)
    x_res = te.lang.cce.vmul(x_res, l1)
    x_res = te.lang.cce.vsub(x_res, linear_t)

    # 4.y_res = accum_new^(-lr_power)/lr + 2*l2
    l2 = te.lang.cce.vmuls(l2, tvm.const(NUM_TWO, var_tmp.dtype))
    y_res = te.lang.cce.vdiv(accum_new_p, lr)
    y_res = te.lang.cce.vadd(y_res, l2)

    # 5.var = x_res / y_res if linear.abs > l1, else var = 0
    x_res = te.lang.cce.vdiv(x_res, y_res)
    linear_abs = te.lang.cce.vabs(linear_t)
    var_sel = te.lang.cce.vcmp(linear_abs, l1, 'gt')
    var_t = te.lang.cce.vsel(var_sel, x_res, zero_tensor)

    # result of vsel is fp16, should cast to fp32
    var_t = te.lang.cce.cast_to(var_t, "float32")

    if has_improve_precision:
        var_t = te.lang.cce.cast_to(var_t, "float16")
        accum_new = te.lang.cce.cast_to(accum_new, "float16")
        linear_t = te.lang.cce.cast_to(linear_t, "float16")

    # 8.var_output_data = var_t
    var_output_data = te.lang.cce.vadds(var_t, tvm.const(NUM_ZERO, var_t.dtype))
    accum_output_data = te.lang.cce.vadds(accum_new, tvm.const(NUM_ZERO, accum_new.dtype))
    linear_output_data = te.lang.cce.vadds(linear_t, tvm.const(NUM_ZERO, linear_t.dtype))

    def _compute(*index):
        return var_t(*index), accum_new(*index), linear_t(*index), var_output_data(
            *index), accum_output_data(*index), linear_output_data(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, dict, str)
def apply_ftrl_d(var,
                 accum,
                 linear,
                 grad,
                 lr,
                 l1,
                 l2,
                 lr_power,
                 var_out,
                 accum_out,
                 linear_out,
                 kernel_name="apply_ftrl_d"):
    """
    Update '*var' according to the Ftrl-proximal algorithm.
    accum_new = accum + grad * grad
    linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
    quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
    var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
    accum = accum_new

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32

    accum : the dict of mutable tensor accum.
        Must have the same data type as `var`.

    linear : the dict of mutable tensor linear.
        Must have the same data type as `var`.

    grad : the dict of tensor grad. Must have the same data type as `var`.

    lr : the dict of scalar lr. Must have the same data type as `var`.

    l1 : the dict of scalar l1. Must have the same data type as `var`.

    l2 : the dict of scalar l2. Must have the same data type as `var`.

    lr_power : the dict of scalar lr_power.
        Must have the same data type as `var`.

    var_out: the dict of var output data.

    accum_out: the dict of accum output data.

    linear_out: the dict of linear output data

    kernel_name : cce kernel name, default value is "apply_ftrl_d".

    Returns
    -------
    None
    """

    input_dict = (var, accum, linear, grad, lr, l1, l2, lr_power)
    out = [var_out, accum_out, linear_out]
    args = ApplyOpConfig.TensorArgs(input_dict, apply_ftrl_d_compute, out, 15)
    name = ApplyOpConfig.TensorName(all=('var', 'accum', 'linear', 'grad',
                                         'lr', 'l1', 'l2', 'lr_power'),
                                    scalar=('lr', 'l1', 'l2', 'lr_power'),
                                    reuse=('var', 'accum', 'linear'))
    options = ApplyOpConfig.TensorOptions(
        build=set_bool_storage_config())

    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
