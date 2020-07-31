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

this file achieved the apply_rms_prop which is a optimizer operator to update
weight, this file contains compute and schedule.
"""

from functools import reduce as functools_reduce
import operator
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from topi import generic

# shape limit for apply_rms_prop
SHAPE_SIZE_LIMIT = 2**30

# pylint: disable=too-many-arguments,invalid-name,too-many-locals
# pylint: disable=unused-argument
@fusion_manager.register('apply_rms_prop_d')
def apply_rms_prop_d_compute(var, ms, mom, lr, grad, out_var, out_ms, out_mom, rho, momentum,
                             epsilon, kernel_name="apply_rms_prop_d"):
    """
    the operator's compute
    :param var: weight, placeholder, float32
    :param ms: mean square, placeholder, float32
    :param mom: momentum, placeholder, float32
    :param lr: learning rate, placeholder, float32
    :param grad: gradient, placeholder, float32
    :param out_var: updated of var
    :param out_ms: updated of ms
    :param out_mom: updated of mom
    :param rho: const, float32
    :param momentum: const, float32
    :param epsilon: const, float32
    :return: out_var, out_ms, out_mom
    """
    grad_square = te.lang.cce.vmul(grad, grad)
    grad_square_ms = te.lang.cce.vsub(grad_square, ms)
    rho_gs = te.lang.cce.vmuls(grad_square_ms, tvm.const(1.0 - rho, grad.dtype))
    ms_t = te.lang.cce.vadd(ms, rho_gs)

    m_mom = te.lang.cce.vmuls(mom, tvm.const(momentum, mom.dtype))

    lr_brc = te.lang.cce.broadcast(lr, grad.shape)
    lr_grad = te.lang.cce.vmul(grad, lr_brc)

    e_ms = te.lang.cce.vadds(ms_t, tvm.const(epsilon, ms.dtype))
    sqrt_ms = te.lang.cce.vsqrt(e_ms)
    tmp_grad = te.lang.cce.vdiv(lr_grad, sqrt_ms)
    mom_t = te.lang.cce.vadd(m_mom, tmp_grad)

    var_t = te.lang.cce.vsub(var, mom_t)

    return var_t, ms_t, mom_t


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    var_shape = []
    for var, name in zip(dict_list, name_list):
        shape = var.get('shape')
        dtype = var.get('dtype').lower()
        if name == 'var':
            var_shape = list(shape)
        if name != 'lr' and var_shape != list(shape):
            raise RuntimeError("The shape of var, ms, mom, grad must be equal.")
        if name == 'lr' and shape[0] != 1:
            raise RuntimeError("The shape of lr must be scalar.")

        util.check_dtype_rule(dtype, ['float32'])
        util.check_shape_rule(shape, max_shape_num=SHAPE_SIZE_LIMIT)
        util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
        shape_refine = (functools_reduce(operator.mul, shape),)
        list_placeholder.append(tvm.placeholder(shape=shape_refine, name=name,
                                                dtype=dtype))
    return list_placeholder


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, float, float, float,
                       str)
# pylint: disable=too-many-arguments,unused-argument,unbalanced-tuple-unpacking
def apply_rms_prop_d(var, ms, mom, lr, grad, out_var, out_ms, out_mom, rho, momentum, epsilon,
                     kernel_name="apply_rms_prop_d"):
    """
    Update '*var' according to the RMSProp algorithm.

    mean_square = decay * mean_square + (1-decay) * gradient ** 2
    Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

    ms_{t} <- rho * ms_{t-1} + (1-rho) * grad * grad
    mom_{t} <- momentum * mom_{t-1} + learning_rate * grad / sqrt(ms_{t} +
            epsilon)
    var_{t} <- var_{t-1} - mom_{t}
    shape of learning_rate is (1,)

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype, dtype support float32.

    ms: dict of tensor ms(mean_square), include shape and dtype, dtype support
        float32.

    mom: dict of tensor mom, include shape and dtype, dtype support float32.

    lr: dict of scalar lr(learning rate), include shape and dtype, dtype
        support float32

    grad: dict of tensor grad, include shape and dtype, dtype support float32.

    out_var: dict of updated var.

    out_ms: dict of updated ms.

    out_mom: dict of updated mom.

    rho: scalar rho(decay rate), attr in d. Must have the same dtype as var.

    momentum: scalar momentum, attr in d. Must have the same dtype as var.

    epsilon: scalar epsilon, attr in d. Must have the same dtype as var.

    kernel_name : cce kernel name, default value is "apply_rms_prop".

    Returns
    -------
    None
    """

    input_name_list = ['var', 'ms', 'mom', 'lr', 'grad']
    util.check_kernel_name(kernel_name)
    var, ms, mom, lr, grad = _get_placeholder([var, ms, mom, lr, grad],
                                              input_name_list)

    # compute
    out_var, out_ms, out_mom = apply_rms_prop_d_compute(var, ms, mom, lr, grad, out_var, out_ms,
                                                        out_mom, rho, momentum, epsilon)

    outs = [out_var, out_ms, out_mom]
    build_list = [var, ms, mom, lr, grad, out_var, out_ms, out_mom]

    with tvm.target.cce():
        sch = generic.auto_schedule(outs)
    config = {"name": kernel_name, "tensor_list": build_list}
    te.lang.cce.cce_build_code(sch, config)
