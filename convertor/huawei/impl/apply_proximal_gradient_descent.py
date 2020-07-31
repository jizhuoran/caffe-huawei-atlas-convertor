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

this file achieved the apply_proximal_gradient_descent which is
a optimizer operator to update weight, this file contains compute and schedule.

apply_proximal_gradient_descent

  Op_description :
    Update '*var' as FOBOS algorithm with fixed learning rate.

    # apply_proximal_gradient_descent(var,
    #   alpha,
    #   l1,
    #   l2,
    #   delta,
    #   out,
    #   kernel_name='apply_proximal_gradient_descent')

  Supportive_dtype_format :
    ['int32', 'int8', 'uint8', 'float32', 'float16']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : the input tensors must have the same shape and type.
    [2] All : shape size limit is 2147483648.
"""


from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig
from impl.util.util_build import set_bool_storage_config
from impl.util.util_compute import sign

CONST_ZERO = 0
CONST_ONE = 1
CONST_ONE_NEG = -1


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("apply_proximal_gradient_descent")
def apply_proximal_gradient_descent_compute(
        var,
        alpha,
        l1,
        l2,
        delta,
        out,
        kernel_name="apply_proximal_gradient_descent"):
    """
    the operator's compute
    prox_v = var - alpha * delta
    if l1 > 0 :
        var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
    else:
        var = prox_v / (var + l2 * delta)

    Parameters:
    ----------
    var: the dict of var, only support float16, float32
    alpha: the dict of alpha, only support float16, float32
    l1: the dict of l1, only support float16, float32
    l2: the dict of l2, only support float16, float32
    delta: the dict of delta, only support float16, float32
    out: the dict of output, only support float16, float32

    Returns
    the value of out_var
    output_data
    """
    dtype = var.dtype

    if dtype == "float16":
        var = te.lang.cce.cast_to(var, "float32")
        alpha = te.lang.cce.cast_to(alpha, "float32")
        l1 = te.lang.cce.cast_to(l1, "float32")
        l2 = te.lang.cce.cast_to(l2, "float32")
        delta = te.lang.cce.cast_to(delta, "float32")

    alpha_broad = te.lang.cce.broadcast(alpha, var.shape)
    l1_broad = te.lang.cce.broadcast(l1, var.shape)
    l2_broad = te.lang.cce.broadcast(l2, var.shape)

    var_out = _compute_process(var, alpha_broad, l1_broad, l2_broad, delta)

    if dtype == "float16":
        var_out = te.lang.cce.cast_to(var_out, "float16")
    else:
        var_out = te.lang.cce.cast_to(var_out, "float32")

    # this compute is for muti output
    def _compute(*index):
        return var_out(*index), var_out(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


def _compute_process(var, alpha_broad, l1_broad, l2_broad, delta):
    """
    the operator's compute
    prox_v = var - alpha * delta
    if l1 > 0 :
        var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
    else:
        var = prox_v / (var + l2 * delta)

    Parameters:
    ----------
    var: the value of var
    alpha_broad: the value of alpha_broad
    l1_broad: the value of l1_broad
    l2_broad: the value of l2_broad
    delta: the value of delta

    Returns
    the value of out_var
    output_data
    """
    # var - alpha * delta
    alpha_delta = te.lang.cce.vmul(alpha_broad, delta)
    alpha_delta = te.lang.cce.vmuls(alpha_delta,
                                    tvm.const(CONST_ONE_NEG, "float32"))
    prox_v = te.lang.cce.vadd(var, alpha_delta)

    const_zero_tensor = te.lang.cce.broadcast(
        tvm.const(CONST_ZERO, var.dtype.lower()), delta.shape)
    situation = te.lang.cce.vcmp(l1_broad, const_zero_tensor, 'gt')

    var_res = _compute_positive(prox_v, alpha_broad, l1_broad, l2_broad)

    # prox_var / 1 + l2 * alpha
    l2_lr = te.lang.cce.vmul(l2_broad, alpha_broad)
    l2_lr_1 = te.lang.cce.vadds(l2_lr, tvm.const(CONST_ONE, "float32"))
    var_t_neg = te.lang.cce.vdiv(prox_v, l2_lr_1)

    var_out = te.lang.cce.vsel(situation, var_res, var_t_neg)

    return var_out


def _compute_positive(prox_v, alpha_broad, l1_broad, l2_broad):
    """
    the operator's compute
    var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

    Parameters:
    ----------
    prox_v: the value of prox_v
    alpha_broad: the value of alpha_broad
    l1_broad: the value of l1_broad
    l2_broad: the value of l2_broad

    Returns
    the value of var_res
    """
    prox_v_abs = te.lang.cce.vabs(prox_v)
    prox_v_sign = sign(prox_v)
    # 1+alpha*l2
    alpha_l2 = te.lang.cce.vmul(alpha_broad, l2_broad)
    alpha_l2_1 = te.lang.cce.vadds(alpha_l2, tvm.const(CONST_ONE, "float32"))
    # max{|prox_v|-alpha*l1,0}
    alpha_l1 = te.lang.cce.vmul(alpha_broad, l1_broad)
    alpha_l1_neg = te.lang.cce.vmuls(alpha_l1,
                                     tvm.const(CONST_ONE_NEG, "float32"))
    prox_v_l1 = te.lang.cce.vadd(prox_v_abs, alpha_l1_neg)
    max_value = te.lang.cce.vmax(
        prox_v_l1,
        te.lang.cce.broadcast(tvm.const(CONST_ZERO, "float32"), prox_v.shape))
    # sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
    res = te.lang.cce.vdiv(prox_v_sign, alpha_l2_1)
    var_res = te.lang.cce.vmul(res, max_value)

    return var_res


@util.check_input_type(dict, dict, dict, dict, dict, dict, str)
def apply_proximal_gradient_descent(
        var,
        alpha,
        l1,
        l2,
        delta,
        out,
        kernel_name="apply_proximal_gradient_descent"):
    """
    Update '*var' as FOBOS algorithm with fixed learning rate..

    prox_v = var - alpha * delta
    var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

    Parameters:
    ----------
    var: the dict of var, only support float16, float32
    alpha: the dict of alpha, only support float16, float32
    l1: the dict of l1, only support float16, float32
    l2: the dict of l2, only support float16, float32
    delta: the dict of delta, only support float16, float32
    out: the dict of output, only support float16, float32

    kernel_name : cce kernel name, default value is
        "apply_proximal_gradient_descent"

    Returns
    -------
    None
    """

    check_list = ('float16', 'float32')
    dtype = var.get('dtype')
    util.check_dtype_rule(dtype, check_list)
    dtype = dtype.lower()

    input_dict = (var, alpha, l1, l2, delta)

    args = ApplyOpConfig.TensorArgs(input_dict,
                                    apply_proximal_gradient_descent_compute,
                                    out, 5 if dtype == 'float32' else 10)
    name = ApplyOpConfig.TensorName(all=('var', 'alpha', 'l1', 'l2', 'delta'),
                                    scalar=('alpha', 'l1', 'l2'),
                                    reuse=('var', ))
    options = ApplyOpConfig.TensorOptions(
        build=set_bool_storage_config())

    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
