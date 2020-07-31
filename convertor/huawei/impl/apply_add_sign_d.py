#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

this file achieved the apply_add_sign_d which is a optimizer operator
to update weight, this file contains compute and schedule.
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig
from impl.util.util_build import set_bool_storage_config

CONST_ZERO = 0
CONST_ONE = 1
CONST_ONE_NEG = -1


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name,too-many-locals
@fusion_manager.register("apply_add_sign_d")
def apply_add_sign_d_compute(var,
                             m,
                             lr,
                             alpha,
                             sign_decay,
                             beta,
                             grad,
                             var_output,
                             m_output,
                             kernel_name='apply_add_sign_d'):
    """
    Update '*var' according to the AddSign update.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    update <- (alpha + sign_decay * sign(g) *sign(m)) * g
    variable <- variable - lr_t * update

    Parameters:
    ----------
    var: the dict of var, support float16, float32
    m: the dict of m, support float16, float32
    lr: the dict of lr, support float16, float32
    alpha: the dict of alpha, support float16, float32
    sign_decay: the dict of sign_decay, support float16, float32
    beta: the dict of beta, support float16, float32
    grad: the dict of grad, support float16, float32

    Returns
    -------
    the new value of var and m
    the output
    """
    m_out, var_out, output_data, m_output_data = _compute_process(
        var, m, te.lang.cce.broadcast(lr, var.shape),
        te.lang.cce.broadcast(alpha, var.shape),
        te.lang.cce.broadcast(sign_decay, var.shape),
        te.lang.cce.broadcast(beta, var.shape), grad)

    out_var, out_m, out_data, m_out_data = _muti_output(var_out, m_out, output_data,
                                                        m_output_data, var.shape)

    return out_var, out_m, out_data, m_out_data


def _compute_process(var, m, lr_broad, alpha_broad, sign_decay_broad,
                     beta_broad, grad):
    """
    calculate
    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    update <- (alpha + sign_decay * sign(g) *sign(m)) * g
    variable <- variable - lr_t * update

    Parameters:
    ----------
    var: the dict of var, support float16, float32
    m: the dict of m, support float16, float32
    lr: the dict of lr, support float16, float32
    alpha: the dict of alpha, support float16, float32
    sign_decay: the dict of sign_decay, support float16, float32
    beta: the dict of beta, support float16, float32
    grad: the dict of grad, support float16, float32

    Returns
    -------
    the new value of var and out
    the output
    """
    m_out = _update_m(m, beta_broad, grad)
    sign_gm = te.lang.cce.vmul(_sign_compute(grad), _sign_compute(m_out))
    decay_gm = te.lang.cce.vmul(sign_gm, sign_decay_broad)
    var_out = _update_var(decay_gm, alpha_broad, lr_broad, grad, var)

    output_data = te.lang.cce.vadds(var_out, tvm.const(CONST_ZERO, "float32"))
    m_output_data = te.lang.cce.vadds(m_out, tvm.const(CONST_ZERO, "float32"))

    return m_out, var_out, output_data, m_output_data


def _update_m(m_old, beta_broad, grad):
    """
    calculate m = m * beta + grad * (1 - beta)

    Parameters:
    ----------
    m_old: the value of initial m
    beta_broad: the value of beta_broad
    grad: the value of grad

    Returns
    -------
    the new value of m
    """
    m_beta = te.lang.cce.vmul(m_old, beta_broad)
    beta_neg = te.lang.cce.vmuls(beta_broad, tvm.const(CONST_ONE_NEG,
                                                       "float32"))
    beta_1 = te.lang.cce.vadds(beta_neg, tvm.const(CONST_ONE, "float32"))
    grad_beta_gs = te.lang.cce.vmul(grad, beta_1)
    m_out = te.lang.cce.vadd(m_beta, grad_beta_gs)

    return m_out


def _update_var(decay_gm, alpha_broad, lr_broad, grad, var):
    """
    calculate var = var - lr * (alpha + sign_decay * sign_gm) * grad

    Parameters:
    ----------
    decay_gm: the value of decay_gm
    alpha_broad: the dict of alpha_broad
    lr_broad: the dict of lr_broad
    grad: the dict of grad
    var: the value of var

    Returns
    -------
    the new value of var
    """
    decay_gm_alpha = te.lang.cce.vadd(decay_gm, alpha_broad)
    res = te.lang.cce.vmul(decay_gm_alpha, lr_broad)
    res = te.lang.cce.vmul(res, grad)
    res_neg = te.lang.cce.vmuls(res, tvm.const(CONST_ONE_NEG, "float32"))
    var_out = te.lang.cce.vadd(var, res_neg)

    return var_out


def _sign_compute(input_data):
    input_dtype = input_data.dtype
    input_x = te.lang.cce.broadcast(tvm.const(CONST_ONE, input_dtype),
                                    input_data.shape)
    input_y = te.lang.cce.broadcast(tvm.const(CONST_ZERO, input_dtype),
                                    input_data.shape)
    input_z = te.lang.cce.broadcast(tvm.const(CONST_ONE_NEG, input_dtype),
                                    input_data.shape)
    new_data = te.lang.cce.vcmp(input_data,
                                tvm.const(CONST_ZERO, input_dtype),
                                operation="gt", mode="bool")
    res1 = te.lang.cce.vsel(new_data, input_x, input_y)
    new_data_one = te.lang.cce.vcmp(input_data,
                                    tvm.const(CONST_ZERO, input_dtype),
                                    operation="lt", mode="bool")
    res2 = te.lang.cce.vsel(new_data_one, input_z, input_y)
    res = te.lang.cce.vadd(res1, res2)

    return res


def _muti_output(var_out, m_out, output_data, m_output_data, shape):
    """
    this compute is for muti output

    Parameters:
    ----------
    var_out: the value of var_out
    m_out: the value of m_out
    output_data: the dict of output_data
    shape: the shape of var

    Returns
    -------
    the new value of out_var and out_m
    the output
    """

    # this compute is for muti output
    def _compute(*index):
        return var_out(*index), m_out(*index), output_data(*index), m_output_data(*index)

    out_var, out_m, out_data, m_out_data = tvm.compute(shape, _compute, name="outputs")

    return out_var, out_m, out_data, m_out_data


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict, str)
def apply_add_sign_d(var,
                     m,
                     lr,
                     alpha,
                     sign_decay,
                     beta,
                     grad,
                     var_out,
                     m_out,
                     kernel_name="apply_add_sign_d"):
    """
    Update '*var' according to the AddSign update.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    update <- (alpha + sign_decay * sign(g) *sign(m)) * g
    variable <- variable - lr_t * update

    Parameters:
    ----------
    var: the dict of var, support float16, float32
    m: the dict of m, support float16, float32
    lr: the dict of lr, support float16, float32
    alpha: the dict of alpha, support float16, float32
    sign_decay: the dict of sign_decay, support float16, float32
    beta: the dict of beta, support float16, float32
    grad: the dict of grad, support float16, float32
    var_out: the dict of var output data
    m_out: the dict of m output data
    otherwise the behavior is undefined, but may exhibit less contention.
    kernel_name : cce kernel name, default value is "apply_add_sign_d"

    Returns
    -------
    None
    """

    input_dict = (var, m, lr, alpha, sign_decay, beta, grad)
    out = [var_out, m_out]
    args = ApplyOpConfig.TensorArgs(input_dict, apply_add_sign_d_compute, out,
                                    10)
    name = ApplyOpConfig.TensorName(all=('var', 'm', 'lr', 'alpha',
                                         'sign_decay', 'beta', 'grad'),
                                    scalar=('lr', 'alpha', 'sign_decay',
                                            'beta'),
                                    reuse=('var', 'm'))
    options = ApplyOpConfig.TensorOptions(build=set_bool_storage_config())
    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
