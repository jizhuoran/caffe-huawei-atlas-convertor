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

apply_adagrad_da_d
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from te import platform as tbe_platform
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig
from impl.util.util_build import set_bool_storage_config
from impl.util.util_compute import sign

# scalar in apply_adagrad_da_d
NUM_ONE = 1.0
NUM_TWO = 2.0
NUM_ZERO = 0.0
NUM_M_ONE = -1.0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,too-many-locals
@fusion_manager.register("apply_adagrad_da_d")
def apply_adagrad_da_d_compute(var, gradient_accumulator,
                               gradient_squared_accumulator, grad,
                               lr, l1, l2, global_step, var_out,
                               gradient_accumulator_out,
                               gradient_squared_accumulator_out,
                               kernel_name='apply_adagrad_da_d'):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    grad_accum += grad
    grad_squared_accum += grad * grad
    tmp_val=sign(grad_accum) * max⁡{|grad_accum|-l1*global_step, 0}
        if l1>0 else grad_accum
    x_value = -1 * lr * tmp_val
    y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    var = x_value / y_value

    Parameters:
    ----------
    var : mutable tensor var.

    gradient_accumulator: mutable tensor gradient_accumulator.

    gradient_squared_accumulator : mutable tensor gradient_squared_accumulator.

    grad : tensor grad.

    lr : scalar lr.

    l1 : scalar l1.

    l2 : scalar l2.

    global_step : scalar global_step.

    var_out : the dict of output.

    gradient_accumulator_out : the dict of output.

    gradient_squared_accumulator_out : the dict of output.

    kernel_name : cce kernel name, default value is "apply_adagrad_da".

    Returns:
    -------
    None
    """
    # cast to float32 for higher accuracy
    dtype = var.dtype
    has_improve_precision = False
    cast_type = "float16"
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt",
                                                    "float32"):
        cast_type = "float32"
        has_improve_precision = True

    if dtype == "float16":
        if has_improve_precision:
            var_tmp = te.lang.cce.cast_to(var, "float32")
            var_tmp = te.lang.cce.vmuls(var_tmp, tvm.const(NUM_ZERO, "float32"))
            grad_accum_tmp = te.lang.cce.cast_to(gradient_accumulator, "float32")
            grad_sq_accum_tmp = te.lang.cce.cast_to(
                gradient_squared_accumulator, "float32")
            grad = te.lang.cce.cast_to(grad, "float32")
            lr = te.lang.cce.cast_to(lr, "float32")
            l1 = te.lang.cce.cast_to(l1, "float32")
            l2 = te.lang.cce.cast_to(l2, "float32")
        else:
            var_tmp = te.lang.cce.vmuls(var, tvm.const(NUM_ZERO, "float16"))
            grad_accum_tmp = te.lang.cce.vadds(gradient_accumulator,
                                               tvm.const(NUM_ZERO, "float16"))
            grad_sq_accum_tmp = te.lang.cce.vadds(gradient_squared_accumulator,
                                                  tvm.const(NUM_ZERO, "float16"))
    else:
        var_tmp = te.lang.cce.vmuls(var, tvm.const(NUM_ZERO, "float32"))
        grad_accum_tmp = te.lang.cce.vadds(gradient_accumulator,
                                           tvm.const(NUM_ZERO, "float32"))
        grad_sq_accum_tmp = te.lang.cce.vadds(gradient_squared_accumulator,
                                              tvm.const(NUM_ZERO, "float32"))

    global_step = te.lang.cce.cast_to(global_step, cast_type)

    # 1.grad_accum += grad
    gradient_accum_new = te.lang.cce.vadd(grad_accum_tmp, grad)

    # 2.grad_squared_accum += grad * grad
    gs = te.lang.cce.vmul(grad, grad)
    gradient_squared_accum_new = te.lang.cce.vadd(grad_sq_accum_tmp, gs)

    # 3.tmp_val = sign(grad_accum) * max⁡{|grad_accum|-l1*global_step, 0}
    #     if l1>0 else grad_accum
    sign_val = sign(gradient_accum_new)
    abs_val = te.lang.cce.vabs(gradient_accum_new)

    mul_val = te.lang.cce.vmul(global_step, l1)
    mul_val = te.lang.cce.broadcast(mul_val, var_tmp.shape)
    sub_val = te.lang.cce.vsub(abs_val, mul_val)
    zero_tensor = te.lang.cce.broadcast(tvm.const(NUM_ZERO, cast_type),
                                        var_tmp.shape)
    max_val = te.lang.cce.vmax(sub_val, zero_tensor)
    tmp_val = te.lang.cce.vmul(sign_val, max_val)

    l1 = te.lang.cce.broadcast(l1, var_tmp.shape)
    l1_cmp = te.lang.cce.vcmp(l1, zero_tensor, "gt")
    tmp_val = te.lang.cce.vsel(l1_cmp, tmp_val, gradient_accum_new)

    # 4.x_value = -1 * lr * tmp_val
    x_value = te.lang.cce.vmuls(lr, tvm.const(NUM_M_ONE, cast_type))
    x_value = te.lang.cce.broadcast(x_value, var_tmp.shape)
    x_value = te.lang.cce.vmul(x_value, tmp_val)

    # 5.y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    pro_val = te.lang.cce.vmul(l2, global_step)
    pro_val = te.lang.cce.vmul(pro_val, lr)
    pro_val = te.lang.cce.broadcast(pro_val, var_tmp.shape)
    sqrt_val = te.lang.cce.vsqrt(gradient_squared_accum_new, priority_flag=1.0)
    y_value = te.lang.cce.vadd(pro_val, sqrt_val)

    # 6.var = x_value / y_value
    var_t = te.lang.cce.vdiv(x_value, y_value)
    var_new = te.lang.cce.vadd(var_t, var_tmp)

    if dtype == "float16" and has_improve_precision:
        var_new = te.lang.cce.cast_to(var_new, "float16")
        gradient_accum_new = te.lang.cce.cast_to(
            gradient_accum_new, "float16")
        gradient_squared_accum_new = te.lang.cce.cast_to(
            gradient_squared_accum_new, "float16")

    # 7. output_data = var_new
    output_data = te.lang.cce.vadds(var_new, tvm.const(NUM_ZERO, var_new.dtype))
    res1_data = te.lang.cce.vadds(gradient_accum_new, tvm.const(NUM_ZERO,
                                                                var_new.dtype))
    res2_data = te.lang.cce.vadds(gradient_squared_accum_new,
                                  tvm.const(NUM_ZERO, var_new.dtype))

    def _compute(*index):
        return var_new(*index), gradient_accum_new(*index), \
               gradient_squared_accum_new(*index), output_data(*index),\
               res1_data(*index), res2_data(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


# pylint: disable=locally-disabled,unused-argument,too-many-arguments
# pylint: disable=too-many-locals
@util.check_input_type(dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, dict, bool, str)
def apply_adagrad_da_d(var, gradient_accumulator,
                       gradient_squared_accumulator, grad,
                       lr, l1, l2, global_step, var_out,
                       gradient_accumulator_out,
                       gradient_squared_accumulator_out,
                       use_locking=False, kernel_name='apply_adagrad_da_d'):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    grad_accum += grad
    grad_squared_accum += grad * grad
    tmp_val=sign(grad_accum) * max⁡{|grad_accum|-l1*global_step, 0}
        if l1>0 else grad_accum
    x_value = -1 * lr * tmp_val
    y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    var = x_value / y_value

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32

    gradient_accumulator:
        the dict of mutable tensor gradient_accumulator,
        Must have the same data type as `var`.

    gradient_squared_accumulator :
        the dict of mutable tensor gradient_squared_accumulator,
        Must have the same data type as `var`.

    grad : the dict of tensor grad. Must have the same data type as `var`.

    lr : the dict of scalar lr. Must have the same data type as `var`.

    l1 : the dict of scalar l1. Must have the same data type as `var`.

    l2 : the dict of scalar l2. Must have the same data type as `var`.

    global_step : the dict of scalar global_step, only support int32.

    var_out : the dict of output.

    gradient_accumulator_out : the dict of output.

    gradient_squared_accumulator_out : the dict of output.

    use_locking : optional attr, default value is False.

    kernel_name : cce kernel name, default value is "apply_adagrad_da".

    Returns:
    -------
    None
    """
    # check dtype same
    stype_dict = (var, gradient_accumulator, gradient_squared_accumulator, grad,
                  lr, l1, l2)
    normalized_dtype_list = [None] * len(stype_dict)
    for i, d in enumerate(stype_dict):
        dtype = d.get('dtype')
        normalized_dtype_list[i] = dtype.lower()
    if any(elem != normalized_dtype_list[0] for elem in normalized_dtype_list):
        raise RuntimeError("All input data types must be the same")

    # check global_step dtype
    dtype = global_step.get("dtype").lower()
    util.check_dtype_rule(dtype, ("int32",))

    input_dict = (var, gradient_accumulator, gradient_squared_accumulator, grad,
                  lr, l1, l2, global_step)
    args = ApplyOpConfig.TensorArgs(input_dict, apply_adagrad_da_d_compute,
                                    [var_out, gradient_accumulator_out,
                                     gradient_squared_accumulator_out], 15)
    name = ApplyOpConfig.TensorName(all=('var', 'gradient_accumulator',
                                         'gradient_squared_accumulator', 'grad',
                                         'lr', 'l1', 'l2', 'global_step'),
                                    scalar=('lr', 'l1', 'l2', 'global_step'),
                                    reuse=('var', 'gradient_accumulator',
                                           'gradient_squared_accumulator'))
    options = ApplyOpConfig.TensorOptions(
        build=set_bool_storage_config(),
        dtype=('float16', 'float32', 'int32'))
    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name,
                            same_flag=False)
