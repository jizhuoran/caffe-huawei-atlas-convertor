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

This file achieved the apply_centered_rms_prop_d which is a optimizer operator
to update weight.

apply_centered_rms_prop_d

  Op_description :
    Update '*var' according to the centered RMSProp algorithm.
    Update '*mg' according to the centered RMSProp algorithm.
    Update '*ms' according to the centered RMSProp algorithm.
    Update '*mom' according to the centered RMSProp algorithm.

    # apply_centered_rms_prop_d(var,
    #   mg,
    #   ms,
    #   mom,
    #   lr,
    #   rho,
    #   momentum,
    #   epsilon,
    #   grad,
    #   var_out,
    #   mg_out,
    #   ms_out,
    #   mom_out,
    #   kernel_name='apply_centered_rms_prop_d')

  Supportive_dtype_format :
    ['int32', 'int8', 'uint8', 'float32', 'float16']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : the input tensors must have the same shape and type.
    [2] All : shape size limit is 2147483648.
"""

from te import tvm
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
import te.lang.cce
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig

# const value
NUM_ONE = 1.0
NUM_ZERO = 0.0
NUM_ONE_NA = -1.0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@fusion_manager.register("apply_centered_rms_prop_d")
def apply_centered_rms_prop_d_compute(var,
                                      mg,
                                      ms,
                                      mom,
                                      lr,
                                      rho,
                                      momentum,
                                      epsilon,
                                      grad,
                                      var_out,
                                      mg_out,
                                      ms_out,
                                      mom_out,
                                      kernel_name="apply_centered_rms_prop_d"):
    """
    Update '*var' according to the centered RMSProp algorithm.

    mean_square = decay * mean_square + (1-decay) * gradient ** 2
    mean_grad = decay * mean_grad + (1-decay) * gradient
    Delta = learning_rate*gradient/sqrt(mean_square+epsilon-mean_grad**2)
    mg_{t} <- rho * mg_{t-1} + (1-rho) * grad
    ms_{t} <- rho * ms_{t-1} + (1-rho) * grad * grad
    mom_{t} <- momentum*mom_{t-1}+lr*grad/sqrt(ms_{t}-mg{t}*mg{t}+epsilon)
    var_{t} <- var_{t-1} - mom_{t}

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype,
        dtype support float16 and float32.

    mg: dict of tensor mg(mean_grad), include shape and dtype,
        dtype support float16 and float32.

    ms: dict of tensor ms(mean_square), include shape and dtype,
        dtype support float16 and float32.

    mom: dict of tensor mom, include shape and dtype,
        dtype support float16 and float32.

    lr: dict of scalar lr(learning rate). Must have the same dtype as var.

    rho: dict of scalar rho(decay rate). Must have the same dtype as var.

    momentum: dict of scalar momentum. Must have the same dtype as var.

    epsilon: dict of scalar epsilon. Must have the same dtype as var.

    grad: dict of tensor grad. Must have the same dtype as var.

    out: dict of output out.

    kernel_name : cce kernel name, default value is "apply_centered_rms_prop_d".

    Returns
    -------
    None
    """

    inp_dtype = var.dtype
    if inp_dtype == "float16" and api_check_support("te.lang.cce.vadd",
                                                    "float32"):
        var = te.lang.cce.cast_to(var, "float32")
        mg = te.lang.cce.cast_to(mg, "float32")
        ms = te.lang.cce.cast_to(ms, "float32")
        mom = te.lang.cce.cast_to(mom, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        rho = te.lang.cce.cast_to(rho, "float32")
        momentum = te.lang.cce.cast_to(momentum, "float32")
        epsilon = te.lang.cce.cast_to(epsilon, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")

    tensor_one_rho = tvm.compute(rho.shape,
                                 lambda *indices: rho(*indices)
                                     * tvm.const(NUM_ONE_NA, rho.dtype),
                                 tag='elewise_single_VS_mul')
    tensor_one_rho = tvm.compute(
        tensor_one_rho.shape,
        lambda *indices: tensor_one_rho(*indices)
                         + tvm.const(NUM_ONE, tensor_one_rho.dtype),
        tag='elewise_single_VS_add')

    mg_rho = tvm.compute(mg.shape,
                         lambda *indices: mg(*indices) * rho[0],
                         tag='elewise_single_VS_mul')
    rhs = tvm.compute(grad.shape,
                      lambda *indices: grad(*indices) * tensor_one_rho[0],
                      tag='elewise_single_VS_mul')
    out_mg = te.lang.cce.vadd(mg_rho, rhs)

    ms_rho = tvm.compute(ms.shape,
                         lambda *indices: ms(*indices) * rho[0],
                         tag='elewise_single_VS_mul')
    rhs = te.lang.cce.vmul(grad, grad)
    rhs = tvm.compute(rhs.shape,
                      lambda *indices: rhs(*indices) * tensor_one_rho[0],
                      tag='elewise_single_VS_mul')
    out_ms = te.lang.cce.vadd(ms_rho, rhs)

    lhs_mom = tvm.compute(mom.shape,
                          lambda *indices: mom(*indices) * momentum[0],
                          tag='elewise_single_VS_mul')
    lr_grad = tvm.compute(grad.shape,
                          lambda *indices: grad(*indices) * lr[0],
                          tag='elewise_single_VS_mul')
    rhs = te.lang.cce.vmul(out_mg, out_mg)
    rhs = te.lang.cce.vsub(out_ms, rhs)
    rhs_eps = tvm.compute(rhs.shape,
                          lambda *indices: rhs(*indices) + epsilon[0],
                          tag='elewise_single_VS_add')
    rhs_eps = te.lang.cce.vsqrt(rhs_eps)
    rhs_eps = te.lang.cce.vdiv(lr_grad, rhs_eps)
    out_mom = te.lang.cce.vadd(lhs_mom, rhs_eps)

    out_var = te.lang.cce.vsub(var, out_mom)

    if inp_dtype == "float16":
        out_var = te.lang.cce.cast_to(out_var, "float16")
        out_mg = te.lang.cce.cast_to(out_mg, "float16")
        out_ms = te.lang.cce.cast_to(out_ms, "float16")
        out_mom = te.lang.cce.cast_to(out_mom, "float16")

    mg_output_data = te.lang.cce.vadds(out_mg, NUM_ZERO)
    ms_output_data = te.lang.cce.vadds(out_ms, NUM_ZERO)
    mom_output_data = te.lang.cce.vadds(out_mom, NUM_ZERO)

    # this compute is for multi output
    def _compute(*index):
        return out_mg(*index), out_ms(*index), out_mom(*index), out_var(
            *index), out_var(*index), mg_output_data(*index), ms_output_data(
            *index), mom_output_data(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, str)
def apply_centered_rms_prop_d(var,
                              mg,
                              ms,
                              mom,
                              lr,
                              rho,
                              momentum,
                              epsilon,
                              grad,
                              var_out,
                              mg_out,
                              ms_out,
                              mom_out,
                              kernel_name="apply_centered_rms_prop_d"):
    """
    Update '*var' according to the centered RMSProp algorithm.

    mean_square = decay * mean_square + (1-decay) * gradient ** 2
    mean_grad = decay * mean_grad + (1-decay) * gradient
    Delta = learning_rate*gradient/sqrt(mean_square+epsilon-mean_grad**2)
    mg_{t} <- rho * mg_{t-1} + (1-rho) * grad
    ms_{t} <- rho * ms_{t-1} + (1-rho) * grad * grad
    mom_{t} <- momentum*mom_{t-1}+lr*grad/sqrt(ms_{t}-mg{t}*mg{t}+epsilon)
    var_{t} <- var_{t-1} - mom_{t}

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype,
        dtype support float16 and float32.

    mg: dict of tensor mg(mean_grad), include shape and dtype,
        dtype support float16 and float32.

    ms: dict of tensor ms(mean_square), include shape and dtype,
        dtype support float16 and float32.

    mom: dict of tensor mom, include shape and dtype,
        dtype support float16 and float32.

    lr: dict of scalar lr(learning rate). Must have the same dtype as var.

    rho: dict of scalar rho(decay rate). Must have the same dtype as var.

    momentum: dict of scalar momentum. Must have the same dtype as var.

    epsilon: dict of scalar epsilon. Must have the same dtype as var.

    grad: dict of tensor grad. Must have the same dtype as var.

    var_out: the dict of var output, only support float16, float32

    mg_out: the dict of mg output, only support float16, float32

    ms_out: the dict of ms output, only support float16, float32

    mom_out: the dict of mom output, only support float16, float32

    kernel_name : cce kernel name, default value is "apply_centered_rms_prop_d".

    Returns
    -------
    None
    """

    input_dict = (var, mg, ms, mom, lr, rho, momentum, epsilon, grad)
    out = [var_out, mg_out, ms_out, mom_out]
    check_list = ('float16', 'float32')
    dtype = var.get('dtype')
    check_dtype(dtype, check_list)
    dtype = dtype.lower()

    args = ApplyOpConfig.TensorArgs(input_dict,
                                    apply_centered_rms_prop_d_compute, out,
                                    6 if dtype == "float32" else 12)
    name = ApplyOpConfig.TensorName(all=('var', 'mg', 'ms', 'mom', 'lr', 'rho',
                                         'momentum', 'epsilon', 'grad'),
                                    scalar=('lr', 'rho', 'momentum',
                                            'epsilon'),
                                    reuse=('mg', 'ms', 'mom', 'var'))

    common_apply_op_process(ApplyOpConfig(args, name), kernel_name)
