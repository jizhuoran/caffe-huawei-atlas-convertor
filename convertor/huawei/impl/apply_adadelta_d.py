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

this file achieved the apply_adadelta_d which is a optimizer operator
to update weight
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
from te import platform as tbe_platform
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig

NUM_ONE = 1.0
NUM_ZERO = 0.0


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,invalid-name,too-many-locals
@fusion_manager.register("apply_adadelta_d")
def apply_adadelta_d_compute(var,
                             accum,
                             accum_update,
                             lr,
                             rho,
                             epsilon,
                             grad,
                             var_out,
                             accum_out,
                             accum_update_out,
                             kernel_name='apply_adadelta_d'):
    """
    Update '*var' according to the adadelta scheme.

    accum = rho * accum + (1 - rho) * grad ** 2
    update = (update_accum + epsilon).sqrt() * (accum + epsilon).rsqrt()*grad
    update_accum = rho * update_accum + (1 - rho) * update.square();
    var -= update * lr;

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    accum_update : mutable tensor accum_update.

    lr : scalar lr.

    rho : scalar rho.

    epsilon : scalar epsilon.

    grad : tensor grad.

    var_out : the dict of var output.

    accum_out : the dict of accum output.

    accum_update_out : the dict of accum_update output.

    kernel_name : cce kernel name, default value is "apply_adadelta_d".

    Returns:
    -------
    None
    """

    dtype = var.dtype
    has_improve_precision = False
    cast_type = "float16"
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt",
                                                    "float32"):
        cast_type = "float32"
        has_improve_precision = True

    if dtype == "float16" and has_improve_precision:
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        accum_update = te.lang.cce.cast_to(accum_update, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        rho = te.lang.cce.cast_to(rho, "float32")
        epsilon = te.lang.cce.cast_to(epsilon, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")

    tensor_one = te.lang.cce.broadcast(tvm.const(NUM_ONE, cast_type),
                                       var.shape)
    tensor_rho = te.lang.cce.broadcast(rho, var.shape)
    tensor_rho_gs = te.lang.cce.vsub(tensor_one, tensor_rho)
    tensor_epsilon = te.lang.cce.broadcast(epsilon, var.shape)

    # step1: update accum
    rhs = te.lang.cce.vmul(accum, tensor_rho)
    lhs = te.lang.cce.vmul(grad, grad)
    lhs = te.lang.cce.vmul(lhs, tensor_rho_gs)
    accum_res = te.lang.cce.vadd(lhs, rhs)

    accum_update_orig = te.lang.cce.vadds(accum_update, NUM_ZERO)
    # step2
    rhs = te.lang.cce.vadd(accum_update_orig, tensor_epsilon)
    rhs = te.lang.cce.vsqrt(rhs)
    lhs = te.lang.cce.vadd(accum_res, tensor_epsilon)
    lhs = te.lang.cce.vsqrt(lhs)
    lhs = te.lang.cce.vdiv(grad, lhs)
    update = te.lang.cce.vmul(lhs, rhs)

    # step3: update var
    var_res = te.lang.cce.broadcast(lr, var.shape)
    var_res = te.lang.cce.vmul(update, var_res)
    var_res = te.lang.cce.vsub(var, var_res)

    # step4: update accum_update
    rhs = te.lang.cce.vmul(accum_update_orig, tensor_rho)
    lhs = te.lang.cce.vmul(update, update)
    lhs = te.lang.cce.vmul(lhs, tensor_rho_gs)
    accum_update_res = te.lang.cce.vadd(lhs, rhs)

    # out
    output_data = te.lang.cce.vadds(var_res, NUM_ZERO)
    accum_output_data = te.lang.cce.vadds(accum_res, NUM_ZERO)
    accum_update_output_data = te.lang.cce.vadds(accum_update_res, NUM_ZERO)

    if dtype == "float16" and has_improve_precision:
        var_res = te.lang.cce.cast_to(var_res, "float16")
        accum_res = te.lang.cce.cast_to(accum_res, "float16")
        accum_update_res = te.lang.cce.cast_to(accum_update_res, "float16")
        output_data = te.lang.cce.cast_to(output_data, "float16")
        accum_output_data = te.lang.cce.cast_to(accum_output_data, "float16")
        accum_update_output_data = \
            te.lang.cce.cast_to(accum_update_output_data, "float16")

    # this compute is for muti output
    def _compute(*index):
        return var_res(*index), accum_res(*index), accum_update_res(*index), \
               output_data(*index), accum_output_data(*index), \
               accum_update_output_data(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, str)
def apply_adadelta_d(var,
                     accum,
                     accum_update,
                     lr,
                     rho,
                     epsilon,
                     grad,
                     var_out,
                     accum_out,
                     accum_update_out,
                     kernel_name="apply_adadelta_d"):
    """
    Update '*var' according to the adadelta scheme.

    accum = rho * accum + (1 - rho) * grad ** 2
    update = (update_accum + epsilon).sqrt() * (accum + epsilon).rsqrt() * grad
    update_accum = rho * update_accum + (1 - rho) * update.square();
    var -= update * lr;

    Parameters:
    ----------
    var: the dict of input, only support float16, float32

    accum: the dict of accum, only support float16, float32

    accum_update: the dict of accum_update, only support float16, float32

    lr: the dict of lr, only support float16, float32

    rho: the dict of rho, only support float16, float32

    epsilon: the dict of epsilon, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output data

    accum_out: the dict of accum output data

    accum_update_out: the dict of accum_update output data

    kernel_name : cce kernel name, default value is "apply_adadelta_d"

    Returns
    -------
    None
    """
    input_dict = (var, accum, accum_update, lr, rho, epsilon, grad)

    args = ApplyOpConfig.TensorArgs(input_dict, apply_adadelta_d_compute,
                                    [var_out, accum_out, accum_update_out], 16)
    name = ApplyOpConfig.TensorName(all=('var', 'accum', 'accum_update', 'lr',
                                         'rho', 'epsilon', 'grad'),
                                    scalar=('lr', 'rho', 'epsilon'),
                                    reuse=('var', 'accum', 'accum_update'))

    common_apply_op_process(ApplyOpConfig(args, name), kernel_name)
