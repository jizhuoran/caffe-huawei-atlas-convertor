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

apply_adam_with_amsgrad_d
"""

from functools import reduce as functools_reduce
import operator
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

NUM_ONE = 1.0
NUM_N_ONE = -1.0


# pylint: disable=too-many-arguments,invalid-name,too-many-locals
# pylint: disable=unused-argument
@fusion_manager.register("apply_adam_with_amsgrad_d")
def apply_adam_with_amsgrad_d_compute(var, m, v, vhat, beta1_power, beta2_power,
                                      lr, beta1, beta2, epsilon, grad,
                                      kernel_name="apply_adam_with_amsgrad_d"):
    """
    the operator's compute
    :param var: weight, placeholder
    :param m: moment, placeholder
    :param v: moment, placeholder
    :param vhat: vhat, placeholder
    :param beta1_power: beta1_power, placeholder
    :param beta2_power: beta2_power, placeholder
    :param lr: learning rate, const
    :param beta1: beta1, const
    :param beta2: beta2, const
    :param epsilon: epsilon, const
    :param grad: grad, placeholder
    """
    one = tvm.const(NUM_ONE, "float32")
    neg_one = tvm.const(NUM_N_ONE, "float32")

    beta1_power = te.lang.cce.broadcast(beta1_power, var.shape)
    beta2_power = te.lang.cce.broadcast(beta2_power, var.shape)
    lr = te.lang.cce.broadcast(lr, var.shape)

    # update lr
    beta1_power_neg = te.lang.cce.vmuls(beta1_power, neg_one)
    beta2_power_neg = te.lang.cce.vmuls(beta2_power, neg_one)
    beta1_power_tmp = te.lang.cce.vadds(beta1_power_neg, one)
    beta2_power_tmp = te.lang.cce.vadds(beta2_power_neg, one)
    beta_sqrt = te.lang.cce.vsqrt(beta2_power_tmp)
    lr_sqrt = te.lang.cce.vmul(lr, beta_sqrt)
    lr_t = te.lang.cce.vdiv(lr_sqrt, beta1_power_tmp)

    # update m
    m_mul = te.lang.cce.vmuls(m, beta1)
    beta1_negadd = beta1 * neg_one + one
    m_grad = te.lang.cce.vmuls(grad, beta1_negadd)
    m_t = te.lang.cce.vadd(m_mul, m_grad)

    # update v
    beta2_t = te.lang.cce.vmuls(v, beta2)
    beta2_negadd = beta2 * neg_one + one
    grad_pow = te.lang.cce.vmul(grad, grad)
    beta2_grad = te.lang.cce.vmuls(grad_pow, beta2_negadd)
    v_t = te.lang.cce.vadd(beta2_t, beta2_grad)

    # update vhat
    vhat_t = te.lang.cce.vmax(vhat, v_t)

    # update var
    var_m = te.lang.cce.vmul(lr_t, m_t)
    var_sqrt = te.lang.cce.vsqrt(vhat_t)
    var_epsilon = te.lang.cce.vadds(var_sqrt, epsilon)
    var_div = te.lang.cce.vdiv(var_m, var_epsilon)
    var_t = te.lang.cce.vsub(var, var_div)

    return var_t, m_t, v_t, vhat_t


def _check_para_and_getplaceholder(scalar_input, tensor_input, input_dict):
    check_list = ("float32", )
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


# pylint: disable=too-many-arguments,unused-argument,too-many-locals,
# pylint: disable=unbalanced-tuple-unpacking
@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, float, float, float,
                       bool, str)
def apply_adam_with_amsgrad_d(var, m, v, vhat, beta1_power, beta2_power, lr,
                              grad, var_output, m_output, v_output,
                              vhat_output, beta1, beta2,
                              epsilon, use_locking=False,
                              kernel_name="apply_adam_with_amsgrad_d"):
    """
    Update '*var' according to the Adam algorithm.

    lr_t := {learning_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)

    m_t := beta_1 * m_{t-1} + (1 - beta_1) * g

    v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g

    vhat_t := max{vhat_{t-1}, v_t}

    variable := variable - lr_t * m_t / (sqrt{vhat_t} + epsilon)

    Parameters
    ----------
    var : dict of tensor var, include shape and dtype

    m : dict of tensor m, include shape and dtype

    v: dict of tensor v, include shape and dtype

    vhat : dict of tensor vhat, include shape and dtype

    beta1_power: dict of beta1_power, include shape and dtype.

    beta2_power: dict of beta2_power, include shape and dtype.

    lr: dict of lr, include shape and dtype.

    grad: dict of grad, include shape and dtype.

    var_output: dict of update var.

    m_output: dict of update m.

    v_output: dict of update v.

    vhat_output: dict of update vhat.

    beta1: scalar, attr in D. Must have the same dtype as var.

    beta2: scalar, attr in D. Must have the same dtype as var.

    epsilon: scalar, attr in D. Must have the same dtype as var.

    use_locking: An optional `bool`. Defaults to `False`. If `True`,
    updating of the var, m, and v tensors will be protected.

    kernel_name : kernel name, default value is "apply_adam_with_amsgrad_d"

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    input_dict = {"var": var, "m": m, "v": v, "vhat": vhat,
                  "beta1_power": beta1_power, "beta2_power": beta2_power,
                  "lr": lr, "grad": grad}
    scalar_input = (lr, beta1_power, beta2_power, epsilon)
    tensor_input = (var, m, v, vhat, grad)
    var_input, m_input, v_input, vhat_input, \
    beta1_power, beta2_power, lr_input, grad_input = \
        _check_para_and_getplaceholder(scalar_input, tensor_input, input_dict)

    var_output, m_output, \
    v_output, vhat_output = apply_adam_with_amsgrad_d_compute(var_input,
                                                              m_input,
                                                              v_input,
                                                              vhat_input,
                                                              beta1_power,
                                                              beta2_power,
                                                              lr_input,
                                                              beta1,
                                                              beta2,
                                                              epsilon,
                                                              grad_input,
                                                              kernel_name)
    with tvm.target.cce():
        schedule = generic.auto_schedule([var_output, m_output, v_output,
                                          vhat_output])

    config = {"name": kernel_name,
              "tensor_list": [var_input, m_input, v_input, vhat_input,
                              beta1_power, beta2_power, lr_input,
                              grad_input, var_output, m_output, v_output,
                              vhat_output]}

    te.lang.cce.cce_build_code(schedule, config)

