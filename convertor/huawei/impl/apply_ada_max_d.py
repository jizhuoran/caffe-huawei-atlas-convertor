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

apply_ada_max

  Op_description :
    Update '*var' according to the AdaMax algorithm.

    # apply_ada_max_d(var,
    #   m,
    #   v,
    #   beta1_power,
    #   lr,
    #   beta1,
    #   beta2,
    #   epsilon,
    #   grad,
    #   m_out,
    #   v_out,
    #   kernel_name='apply_ada_max_d')

  Supportive_dtype_format :
    ['int32', 'int8', 'uint8', 'float32', 'float16']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : the input tensors must have the same shape and type.
    [2] All : shape size limit is 2147483648.
"""

from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig


# pylint: disable=locally-disabled, too-many-arguments
# pylint: disable=unused-argument, invalid-name, too-many-locals
@fusion_manager.register("apply_ada_max_d")
def apply_ada_max_d_compute(var,
                            m,
                            v,
                            beta1_power,
                            lr,
                            beta1,
                            beta2,
                            epsilon,
                            grad,
                            var_out,
                            m_out,
                            v_out,
                            kernel_name='apply_ada_max_d'):
    """
    Update '*var' according to the AdaMax algorithm.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- max(beta2 * v_{t-1}, abs(g))
    variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)

    Parameters:
    ----------
    var : mutable tensor var.

    m : mutable tensor m.

    v : mutable tensor v.

    beta1_power : scalar beta1_power.

    lr : scalar lr.

    beta1 : scalar beta1.

    beta2 : scalar beta2.

    epsilon : scalar epsilon.

    grad : tensor grad.

    var_out : the dict of var output.

    m_out : the dict of m output.

    v_out : the dict of v output.

    kernel_name : cce kernel name, default value is "apply_ada_max_d" (optional).

    Returns:
    -------
    None
    """

    # cast to float32 for improved accuracy
    inp_dtype = var.dtype
    if inp_dtype == 'float16' and api_check_support("te.lang.cce.vadd",
                                                    "float32"):
        var = te.lang.cce.cast_to(var, 'float32')
        m = te.lang.cce.cast_to(m, 'float32')
        v = te.lang.cce.cast_to(v, 'float32')
        lr = te.lang.cce.cast_to(lr, 'float32')
        beta1_power = te.lang.cce.cast_to(beta1_power, 'float32')
        beta1 = te.lang.cce.cast_to(beta1, 'float32')
        beta2 = te.lang.cce.cast_to(beta2, 'float32')
        epsilon = te.lang.cce.cast_to(epsilon, 'float32')
        grad = te.lang.cce.cast_to(grad, 'float32')
    else:
        m = te.lang.cce.vmuls(m, tvm.const(1, dtype=inp_dtype))

    rhs = tvm.compute(beta1.shape,
                      lambda *indices: beta1(*indices) * -1,
                      tag='elewise_single_VS_mul')
    rhs = tvm.compute(rhs.shape,
                      lambda *indices: rhs(*indices)
                                       + tvm.const(1.0, dtype=rhs.dtype),
                      tag='elewise_single_VS_add')
    lhs = te.lang.cce.vsub(grad, m)
    rhs = tvm.compute(lhs.shape,
                      lambda *indices: lhs(*indices) * rhs[0],
                      tag='elewise_single_VS_mul')
    m = te.lang.cce.vadd(m, rhs)

    lhs = tvm.compute(v.shape,
                      lambda *indices: v(*indices) * beta2[0],
                      tag='elewise_single_VS_mul')
    rhs = te.lang.cce.vabs(grad)
    v = te.lang.cce.vmax(lhs, rhs)

    rhs = tvm.compute(v.shape,
                      lambda *indices: v(*indices) + epsilon[0],
                      tag='elewise_single_VS_add')
    lhs = tvm.compute(beta1_power.shape,
                      lambda *indices: beta1_power(*indices) * -1,
                      tag='elewise_single_VS_mul')
    lhs = tvm.compute(lhs.shape,
                      lambda *indices: lhs(*indices)
                                       + tvm.const(1.0, dtype=lhs.dtype),
                      tag='elewise_single_VS_add')
    rhs = tvm.compute(rhs.shape,
                      lambda *indices: rhs(*indices) * lhs[0],
                      tag='elewise_single_VS_mul')
    lhs = tvm.compute(m.shape,
                      lambda *indices: m(*indices) * lr[0],
                      tag='elewise_single_VS_mul')
    rhs = te.lang.cce.vdiv(lhs, rhs)
    var = te.lang.cce.vsub(var, rhs)

    res1 = te.lang.cce.vadds(var, tvm.const(0.0, dtype="float32"))
    res2 = te.lang.cce.vadds(m, tvm.const(0.0, dtype="float32"))
    res3 = te.lang.cce.vadds(v, tvm.const(0.0, dtype="float32"))

    if inp_dtype == 'float16':
        res1 = te.lang.cce.cast_to(res1, inp_dtype)
        res2 = te.lang.cce.cast_to(res2, inp_dtype)
        res3 = te.lang.cce.cast_to(res3, inp_dtype)

    def _compute(*index):
        return m(*index), v(*index), var(*index),\
               res1(*index), res2(*index), res3(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type((dict), (dict), (dict), (dict), (dict), (dict), (dict),
                       (dict), (dict), (dict), (dict), (dict), str)
def apply_ada_max_d(var,
                    m,
                    v,
                    beta1_power,
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    grad,
                    var_out,
                    m_out,
                    v_out,
                    kernel_name='apply_ada_max_d'):
    """
    Update '*var' according to the AdaMax algorithm.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- max(beta2 * v_{t-1}, abs(g))
    variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)

    Parameters:
    ----------
    var : the dict of mutable tensor var. Must be one of the following data types:
          `float32`, `float16`.

    m: the dict of mutable tensor m. Must have the same data type as `var`.

    v : the dict of mutable tensor v. Must have the same data type as `var`.

    beta1_power : the dict of scalar beta1_power.
        Must have the same data type as `var`.

    lr : the dict of scalar lr. Must have the same data type as `var`.

    beta1 : the dict of scalar beta1. Must have the same data type as `var`.

    beta2 : the dict of scalar beta2. Must have the same data type as `var`.

    epsilon : the dict of scalar epsilon. Must have the same data type as `var`.

    grad : the dict of tensor grad. Must have the same data type as `var`.

    var_out : the dict of var output.

    m_out : the dict of m output.

    v_out : the dict of v output.

    kernel_name : cce kernel name, default value is "apply_ada_max" (optional).

    Returns:
    -------
    None
    """

    input_dict = (var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad)

    args = ApplyOpConfig.TensorArgs(input_dict, apply_ada_max_d_compute,
                                    [var_out, m_out, v_out], 14)
    name = ApplyOpConfig.TensorName(all=('var', 'm', 'v', 'beta1_power', 'lr',
                                         'beta1', 'beta2', 'epsilon', 'grad'),
                                    scalar=('lr', 'beta1_power', 'beta1',
                                            'beta2', 'epsilon'),
                                    reuse=('m', 'v', 'var'))

    common_apply_op_process(ApplyOpConfig(args, name), kernel_name)
