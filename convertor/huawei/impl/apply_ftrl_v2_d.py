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

apply_ftrl_v2_d
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

# scalar in apply_ftrl_v2
NUM_ONE = 1.0
NUM_TWO = 2.0
NUM_ZERO = 0.0
NUM_M_ONE = -1.0


def _pow(data_x, data_y):
    """
    Calculate power result for non-negative data.

    res = data_x^data_y
        = exp(data_y * ln(data_x)) if data_x >= 0

    Parameters:
    ----------
    data_x : base value of power operation.

    data_y: index value of power operation.

    Returns:
    -------
    power result of data_x^data_y
    """

    log_value = te.lang.cce.vlog(data_x, priority_flag=1.0)
    mul_value = te.lang.cce.vmul(data_y, log_value)
    res = te.lang.cce.vexp(mul_value)

    return res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,too-many-locals, too-many-statements
@fusion_manager.register("apply_ftrl_v2_d")
def apply_ftrl_v2_d_compute(var, accum, linear, grad, lr, l1, l2, l2_shrinkage,
                            lr_power, var_out, accum_out, linear_out,
                            kernel_name='apply_ftrl_v2_d'):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    grad_with_shrinkage = grad + 2 * l2_shrinkage * var
    accum_new = accum + grad * grad
    linear += grad_with_shrinkage -
        (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
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

    l2_shrinkage: scalar l2_shrinkage.

    lr_power : scalar lr_power.

    var_out : the dict of output var.

    accum_out : the dict of output accum.

    linear_out : the dict of output linear.

    kernel_name : cce kernel name, default value is "apply_ftrl_v2_d".

    Returns:
    -------
    the value of var_new, accum_new, linear_new, output_data
    """
    dtype = var.dtype
    # cast to float32 for higher accuracy
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
        l2_shrinkage = te.lang.cce.cast_to(l2_shrinkage, "float32")
        lr_power = te.lang.cce.cast_to(lr_power, "float32")
        has_improve_precision = True
    else:
        var_tmp = te.lang.cce.vadds(var, tvm.const(NUM_ZERO, "float32"))
        accum_tmp = te.lang.cce.vadds(accum, tvm.const(NUM_ZERO, "float32"))
        linear_tmp = te.lang.cce.vadds(linear, tvm.const(NUM_ZERO, "float32"))

    # 1.grad_with_shrinkage = grad + 2 * l2_shrinkage * var
    mul_value = te.lang.cce.vmuls(l2_shrinkage, tvm.const(NUM_TWO, "float32"))
    mul_value = te.lang.cce.broadcast(mul_value, var_tmp.shape)
    mul_value2 = te.lang.cce.vmul(mul_value, var_tmp)
    grad_with_shrinkage = te.lang.cce.vadd(grad, mul_value2)

    # 2.accum_new = accum + grad^2
    gs = te.lang.cce.vmul(grad, grad)
    accum_new = te.lang.cce.vadd(accum_tmp, gs)

    # 3.accum_pow_sub = accum_new^(-lr_power)-accum^(-lr_power)
    lr_power = te.lang.cce.vmuls(lr_power, tvm.const(NUM_M_ONE, "float32"))
    lr_power = te.lang.cce.broadcast(lr_power, var_tmp.shape)
    accum_new_pow = _pow(accum_new, lr_power)
    accum_pow = _pow(accum_tmp, lr_power)
    accum_pow_sub = te.lang.cce.vsub(accum_new_pow, accum_pow)

    # 4.linear += grad_with_shrinkage - accum_pow_sub / lr * var
    lr = te.lang.cce.broadcast(lr, var_tmp.shape)
    accum_pow_div = te.lang.cce.vdiv(accum_pow_sub, lr)
    accum_pow_mul = te.lang.cce.vmul(accum_pow_div, var_tmp)
    accum_pow = te.lang.cce.vsub(grad_with_shrinkage, accum_pow_mul)
    linear_new = te.lang.cce.vadd(linear_tmp, accum_pow)

    # 5.x_res = l1*linear.sign()-linear
    l1 = te.lang.cce.broadcast(l1, var_tmp.shape)
    x_res = sign(linear_new)
    x_res = te.lang.cce.vmul(x_res, l1)
    x_res = te.lang.cce.vsub(x_res, linear_new)

    # 6.y_res = accum_new^(-lr_power)/lr + 2*l2
    l2 = te.lang.cce.vmuls(l2, tvm.const(NUM_TWO, "float32"))
    l2 = te.lang.cce.broadcast(l2, var_tmp.shape)
    y_res = te.lang.cce.vdiv(accum_new_pow, lr)
    y_res = te.lang.cce.vadd(y_res, l2)

    # 7.var = x_res / y_res if linear.abs > l1, else var = 0
    x_res = te.lang.cce.vdiv(x_res, y_res)
    linear_abs = te.lang.cce.vabs(linear_new)
    zero_tensor = te.lang.cce.broadcast(tvm.const(NUM_ZERO, "float32"),
                                        var_tmp.shape)
    var_sel = te.lang.cce.vcmp(linear_abs, l1, 'gt')
    var_new = te.lang.cce.vsel(var_sel, x_res, zero_tensor)

    # dtype after vsel is float16 at mini
    var_new = te.lang.cce.cast_to(var_new, "float32")

    if has_improve_precision:
        var_new = te.lang.cce.cast_to(var_new, "float16")
        accum_new = te.lang.cce.cast_to(accum_new, "float16")
        linear_new = te.lang.cce.cast_to(linear_new, "float16")

    # 8.output_var = var_new
    output_data = te.lang.cce.vadds(var_new, tvm.const(NUM_ZERO, var_new.dtype))
    accum_out_data = te.lang.cce.vadds(
        accum_new, tvm.const(NUM_ZERO, accum_new.dtype))
    linear_out_data = te.lang.cce.vadds(
        linear_new, tvm.const(NUM_ZERO, linear_new.dtype))

    def _compute(*index):
        return var_new(*index), accum_new(*index), \
               linear_new(*index), output_data(*index), \
               accum_out_data(*index), linear_out_data(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, dict, bool, str)
def apply_ftrl_v2_d(var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power,
                    var_out, accum_out, linear_out, use_locking=False,
                    kernel_name="apply_ftrl_v2_d"):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    grad_with_shrinkage = grad + 2 * l2_shrinkage * var
    accum_new = accum + grad * grad
    linear += grad_with_shrinkage -
        (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
    x = l1 * linear.sign - linear
    y = accum_new^(-lr_power) / lr + 2 * l2
    var = x / y if |linear| > l1 else 0.0
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

    l2_shrinkage: the dict of scalar l2_shrinkage.
        Must have the same data type as `var`.

    lr_power : the dict of scalar lr_power.
        Must have the same data type as `var`.

    var_out : the dict of output var.

    accum_out : the dict of output accum.

    linear_out : the dict of output linear.

    use_locking : optional attr, default value is False.

    kernel_name : cce kernel name, default value is "apply_ftrl_v2_d".

    Returns
    -------
    None
    """
    input_dict = (var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power)

    args = ApplyOpConfig.TensorArgs(input_dict, apply_ftrl_v2_d_compute,
                                    [var_out, accum_out, linear_out], 15)
    name = ApplyOpConfig.TensorName(all=('var', 'accum', 'linear', 'grad',
                                         'lr', 'l1', 'l2',
                                         'l2_shrinkage', 'lr_power'),
                                    scalar=('lr', 'l1', 'l2',
                                            'l2_shrinkage', 'lr_power'),
                                    reuse=('var', 'accum', 'linear'))
    options = ApplyOpConfig.TensorOptions(
        build=set_bool_storage_config())
    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
