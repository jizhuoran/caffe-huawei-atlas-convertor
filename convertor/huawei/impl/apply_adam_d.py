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

apply_adam_d
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig


# pylint: disable=invalid-name, too-many-arguments, too-many-locals
# pylint: disable=unused-argument
@fusion_manager.register("apply_adam_d")
def apply_adam_d_compute(var, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                         epsilon, grad, var_out, m_out, v_out, use_nesterov,
                         kernel_name="apply_adam_d"):
    """
    the opreator's compute
    lr_t = learning_rate*(sqrt(1-beta2_power)) / (1-beta1_power)
    m_t = m + (1-beta1)*(grad-m)
    v_t = v + (1-beta2)*(grad*grad-v)
    if use_nesterov == True:
        var_t = var - lr_t*(m_t*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_t))
    else:
        vat_t = var - lr_t*m_t / (epsilon + sqrt(v_t))
    Parameters:
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    m: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    v: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta1_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta2_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    epsilon: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    m_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'm'.
    v_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'v'.
    use_nesterov: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "apply_adam_d"

    Returns:
        the value of var, m, v, res1, res2, res3
    """
    inp_dtype = var.dtype
    var = te.lang.cce.vadds(var, tvm.const(0.0, dtype=inp_dtype))
    m = te.lang.cce.vadds(m, tvm.const(0.0, dtype=inp_dtype))
    v = te.lang.cce.vadds(v, tvm.const(0.0, dtype=inp_dtype))

    # m.device(d) += (1-beta1)*(grad - m)
    lhs = tvm.compute(beta1.shape,
                      lambda *indices:
                      beta1(*indices) * tvm.const(-1.0, dtype=inp_dtype),
                      name="lhs", tag='elewise_single_VS_mul')
    lhs_beta1 = tvm.compute(lhs.shape,
                            lambda *indices:
                            lhs(*indices) + tvm.const(1.0, dtype=inp_dtype),
                            name="lhs_beta1", tag='elewise_single_VS_add')
    vsub_grad_m = te.lang.cce.vsub(grad, m)
    vmul_m = tvm.compute(vsub_grad_m.shape,
                         lambda *indices: vsub_grad_m(*indices) * lhs_beta1[0],
                         tag="elewise_single_VS_mul")
    m = te.lang.cce.vadd(m, vmul_m)

    # v.device(d) += (1 - beta2)*(grad*grad - v)
    rhs = tvm.compute(beta2.shape,
                      lambda *indices:
                      beta2(*indices) * tvm.const(-1.0, dtype=inp_dtype),
                      name="rhs", tag='elewise_single_VS_mul')
    rhs_beta2 = tvm.compute(rhs.shape,
                            lambda *indices:
                            rhs(*indices) + tvm.const(1.0, dtype=inp_dtype),
                            name="rhs_beta2", tag='elewise_single_VS_add')
    vmul_grad = te.lang.cce.vmul(grad, grad)
    vsub_v = te.lang.cce.vsub(vmul_grad, v)
    vmul_v = tvm.compute(vsub_v.shape,
                         lambda *indices:
                         vsub_v(*indices) * rhs_beta2[0],
                         tag="elewise_single_VS_mul")
    v = te.lang.cce.vadd(v, vmul_v)

    # alpha.device(d) = lr*sqrt(T(1) - beta2_power) / (T(1) - beta1_power)
    lhs1 = tvm.compute(beta2_power.shape,
                       lambda *indices:
                       beta2_power(*indices) * tvm.const(-1.0, dtype=inp_dtype),
                       name="lhs1", tag='elewise_single_VS_mul')
    lhs_power = tvm.compute(lhs1.shape,
                            lambda *indices:
                            lhs1(*indices) + tvm.const(1.0, dtype=inp_dtype),
                            name="lhs_power", tag='elewise_single_VS_add')
    rhs2 = tvm.compute(beta1_power.shape,
                       lambda *indices:
                       beta1_power(*indices) * tvm.const(-1.0, dtype=inp_dtype),
                       name="rhs2", tag='elewise_single_VS_mul')
    rhs_power = tvm.compute(rhs2.shape,
                            lambda *indices:
                            rhs2(*indices) + tvm.const(1.0, dtype=inp_dtype),
                            name="rhs_power", tag='elewise_single_VS_add')

    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ("Ascend910",):
        vlog_t = te.lang.cce.vsqrt(lhs_power, 1)
        alpha_lr = te.lang.cce.vmul(vlog_t, lr)
        alpha = te.lang.cce.vdiv(alpha_lr, rhs_power)
    else:
        vlog_t = tvm.compute(lhs_power.shape,
                             lambda *indices: tvm.sqrt(lhs_power(*indices)),
                             name="vlog_t", tag="elewise_single_sqrt")
        alpha_lr = tvm.compute(lr.shape,
                               lambda *indices: lr(*indices) * vlog_t(*indices),
                               name="alpha_lr", tag="elewise_binary_mul")
        alpha = tvm.compute(alpha_lr.shape,
                            lambda *indices:
                            alpha_lr(*indices) / rhs_power(*indices),
                            name="alpha", tag="elewise_binary_div")

    # cacluate the sqrt of v.
    vexp_v_sqrt = te.lang.cce.vsqrt(v, 1)
    vadd_sqrt = tvm.compute(vexp_v_sqrt.shape,
                            lambda *indices: vexp_v_sqrt(*indices) + epsilon[0],
                            tag="elewise_single_VS_add")

    # according to different use_nesterov, var is cacluate differently
    if use_nesterov is True:
        vmul_m_beta1 = tvm.compute(m.shape,
                                   lambda *indices: m(*indices) * beta1[0],
                                   tag="elewise_single_VS_mul")
        vmul_g_beta1 = tvm.compute(grad.shape,
                                   lambda *indices:
                                   grad(*indices) * lhs_beta1[0],
                                   tag="elewise_single_VS_mul")
        vadd_m_v = te.lang.cce.vadd(vmul_g_beta1, vmul_m_beta1)
        vmul_alpha = tvm.compute(vadd_m_v.shape,
                                 lambda *indices: vadd_m_v(*indices) * alpha[0],
                                 tag="elewise_single_VS_mul")
        vdiv_beta1_sqrt = te.lang.cce.vdiv(vmul_alpha, vadd_sqrt)
        var = te.lang.cce.vsub(var, vdiv_beta1_sqrt)
    else:
        vmul_sqrt = tvm.compute(m.shape,
                                lambda *indices: m(*indices) * alpha[0],
                                tag="elewise_single_VS_mul")
        vdiv_beta1_sqrt = te.lang.cce.vdiv(vmul_sqrt, vadd_sqrt)
        var = te.lang.cce.vsub(var, vdiv_beta1_sqrt)
    res1 = te.lang.cce.vadds(var, tvm.const(0.0, dtype=inp_dtype))
    res2 = te.lang.cce.vadds(m, tvm.const(0.0, dtype=inp_dtype))
    res3 = te.lang.cce.vadds(v, tvm.const(0.0, dtype=inp_dtype))

    def _compute(*index):
        return m(*index), v(*index), var(*index),\
               res1(*index), res2(*index), res3(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, bool, bool, str)
def apply_adam_d(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
                 grad, var_out, m_out, v_out, use_locking=False,
                 use_nesterov=False, kernel_name="apply_adam_d"):
    """
    the opreator's compute
    lr_t = learning_rate*(sqrt(1-beta2_power)) / (1-beta1_power)
    m_t = m + (1-beta1)*(grad-m)
    v_t = v + (1-beta2)*(grad*grad-v)
    if use_nesterov == True:
        var_t = var - lr_t*(m_t*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_t))
    else:
        vat_t = var - lr_t*m_t / (epsilon + sqrt(v_t))
    Parameters:
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    m: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    v: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta1_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta2_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    epsilon: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    m_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'm'.
    v_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'v'.
    use_locking: bool
        default value is "False".
    use_nesterov: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "apply_adam_d"

    Returns:
    None
    """
    input_dict = (var, m, v, beta1_power, beta2_power, lr, beta1,
                  beta2, epsilon, grad)

    args = ApplyOpConfig.TensorArgs(input_dict, apply_adam_d_compute,
                                    [var_out, m_out, v_out], 15)
    name = ApplyOpConfig.TensorName(all=('var', 'm', 'v', 'beta1_power',
                                         'beta2_power', 'lr', 'beta1', 'beta2',
                                         'epsilon', 'grad'),
                                    scalar=('lr', 'beta1_power', 'beta2_power',
                                            'beta1', 'beta2', 'epsilon'),
                                    reuse=('m', 'v', 'var'))
    options = ApplyOpConfig.TensorOptions(attrs=(use_nesterov))
    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
