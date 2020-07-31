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

this file achieved the apply_adagrad_d which is a optimizer operator
to update weight, this file contains compute and schedule.

apply_adagrad_d

  Op_description :
    Update '*var' according to the Adagrad algorithm.

    # apply_adagrad_d(var,
    #   accum,
    #   lr,
    #   grad,
    #   var_out,
    #   accum_out,
    #   update_slots,
    #   kernel_name='apply_adagrad_d')

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
from te.utils.op_utils import check_dtype
from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig

NUM_ZERO = 0.0


# pylint: disable=locally-disabled, too-many-arguments, unused-argument
# pylint: disable=too-many-locals, invalid-name
@fusion_manager.register("apply_adagrad_d")
def apply_adagrad_d_compute(var,
                            accum,
                            lr,
                            grad,
                            var_out,
                            accum_out,
                            update_slots,
                            kernel_name="apply_adagrad_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    accum += grad ** 2
    var -= lr * grad / accum.sqrt()

    Parameters:
    ----------
    var: the dict of var, only support float16, float32

    accum: the dict of accum, only support float16, float32

    lr: the dict of lr, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output, only support float16, float32

    accum_out: the dict of accum output, only support float16, float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagrad".

    Returns
    -------
    None
    """

    input_dtype = var.dtype
    if input_dtype == "float16" and api_check_support("te.lang.cce.vadd",
                                                      "float32"):
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")

    if update_slots is True:
        grad_square = te.lang.cce.vmul(grad, grad)
        accum = te.lang.cce.vadd(accum, grad_square)
    elif input_dtype == 'float32':
        accum = te.lang.cce.vadds(accum, tvm.const(NUM_ZERO, "float32"))

    lr_grad = tvm.compute(grad.shape,
                          lambda *indices: grad(*indices) * lr[0],
                          tag='elewise_single_VS_mul')
    sqrt_accum = te.lang.cce.vsqrt(accum)

    update = te.lang.cce.vdiv(lr_grad, sqrt_accum)
    var = te.lang.cce.vsub(var, update)

    res1 = te.lang.cce.vadds(var, tvm.const(0.0, dtype="float32"))
    res2 = te.lang.cce.vadds(accum, tvm.const(0.0, dtype="float32"))

    if input_dtype == "float16":
        res1 = te.lang.cce.cast_to(res1, "float16")
        res2 = te.lang.cce.cast_to(res2, "float16")

    # this compute is for muti output
    def _compute(*index):
        return accum(*index), var(*index), res1(*index),\
               res2(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, dict, dict, bool, str)
def apply_adagrad_d(var,
                    accum,
                    lr,
                    grad,
                    var_out,
                    accum_out,
                    update_slots=True,
                    kernel_name="apply_adagrad_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    accum += grad ** 2
    var -= lr * grad / accum.sqrt()

    Parameters:
    ----------
    var: the dict of var, only support float16, float32

    accum: the dict of accum, only support float16, float32

    lr: the dict of lr, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output, only support float16, float32

    accum_out: the dict of accum output, only support float16, float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagrad".

    Returns
    -------
    None
    """

    check_list = ('float16', 'float32')
    dtype = var.get('dtype')
    check_dtype(dtype, check_list)
    dtype = dtype.lower()

    input_dict = (var, accum, lr, grad)

    args = ApplyOpConfig.TensorArgs(input_dict, apply_adagrad_d_compute,
                                    [var_out, accum_out],
                                    5 if dtype == 'float32' else 7)
    name = ApplyOpConfig.TensorName(all=('var', 'accum', 'lr', 'grad'),
                                    scalar=('lr', ),
                                    reuse=('accum', 'var'))
    options = ApplyOpConfig.TensorOptions(attrs=update_slots)

    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
