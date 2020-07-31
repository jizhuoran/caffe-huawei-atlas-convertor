#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

apply_gradient_descent

  Op_description :
    Update var by subtracting alpha * delta from it.

    # apply_gradient_descent(var,
    #   alpha,
    #   delta,
    #   out,
    #   kernel_name='apply_gradient_descent')

  Supportive_dtype_format :
    ['int32', 'int8', 'uint8', 'float32', 'float16']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : the input tensors must have the same shape and type.
    [2] All : shape size limit is 2147483648.
"""


import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager

from topi.cce import util
from impl.util.util_apply_op_schedule import common_apply_op_process
from impl.util.util_apply_op_schedule import ApplyOpConfig


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("apply_gradient_descent")
def apply_gradient_descent_compute(var,
                                   alpha,
                                   delta,
                                   out,
                                   kernel_name="apply_gradient_descent"):
    """
    compute out_var = var - alpha * delta

    Parameters:
    ----------
    var: the placeholder of var.

    alpha : the placeholder of alpha.

    delta : the placeholder of delta.

    out : the dict of output.

    kernel_name :  cce kernel name, default value is "apply_gradient_descent".

    Returns
    -------
    out
    """

    # step 1: calculate delta * alpha
    var_change = tvm.compute(delta.shape,
                             lambda *indices: delta(*indices) * alpha[0],
                             tag='elewise_single_VS_mul')
    # step 2: calculate var - delta * alpha
    reuse_var = te.lang.cce.vsub(var, var_change)

    def _compute(*index):
        return reuse_var(*index), reuse_var(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@util.check_input_type(dict, dict, dict, dict, str)
def apply_gradient_descent(var,
                           alpha,
                           delta,
                           out,
                           kernel_name="apply_gradient_descent"):
    """
    Update var by subtracting alpha * delta from it.

    var_{t} = var_{t-1} - alpha * delta

    Parameters:
    ----------
    var: dict of input_var, include shape and dtype,
        dtype support float16, float32.

    alpha : dict of input_alpha, include shape and dtype,
        dtype support float16, float32.
        Must have the same type as 'var', Must have the shape(1,).

    delta : dict of input_delta, include shape and dtype,
        dtype support float16, float32.
        Must have the same shape and dtype as input_var.

    out : dict of output, include shape and dtype.

    kernel_name : cce kernel name, default value is "apply_gradient_descent".

    Returns
    -------
    None
    """

    input_dict = (var, alpha, delta)

    args = ApplyOpConfig.TensorArgs(input_dict,
                                    apply_gradient_descent_compute, out, 1.5)
    name = ApplyOpConfig.TensorName(all=("var", "alpha", "delta"),
                                    scalar=("alpha", ),
                                    reuse=("var", ))

    common_apply_op_process(ApplyOpConfig(args, name), kernel_name)
