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

fused_mul_apply_momentum

  Op_description :
    Update '*var' according to the ApplyMomentum algorithm.
"""
import operator
from functools import reduce as functools_reduce

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2 << 30


# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("fused_mul_apply_momentum")
def _fused_mul_apply_momentum_compute(var,
                                      accum,
                                      lr,
                                      x1,
                                      momentum,
                                      x2,
                                      out_var,
                                      out_accum,
                                      use_nesterov,
                                      kernel_name='fused_mul_apply_momentum'):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + x1 * x2
    if use_nesterov is True:
        var -= x1 * x2 * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    lr : scalar lr.

    x1 : tensor x1.

    momentum : scalar momentum.

    x2 : scalar x2.

    out_var : the var output.

    out_accum : the accum output

    use_nesterov: bool. If true, use nesterov computing grad,
                  default value is False.

    kernel_name : cce kernel name, default value is
                 "cce_fused_mul_apply_momentum" (optional).

    Returns:
    -------
    out_var, out_accum
    """

    # cast to float32 for higher accuracy
    dtype = var.dtype

    if dtype == "float16":
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        x1 = te.lang.cce.cast_to(x1, "float32")
        x2 = te.lang.cce.cast_to(x2, "float32")
        momentum = te.lang.cce.cast_to(momentum, "float32")

    # calc grad
    x2_brc = te.lang.cce.broadcast(x2, x1.shape)
    grad = te.lang.cce.vmul(x1, x2_brc)
    # update accum
    momentum_brc = te.lang.cce.broadcast(momentum, accum.shape)
    accum_delta = te.lang.cce.vmul(accum, momentum_brc)
    accum_t = te.lang.cce.vadd(accum_delta, grad)

    # update var
    lr_brc = te.lang.cce.broadcast(lr, accum.shape)
    if use_nesterov:
        var_delta = te.lang.cce.vmul(grad, lr_brc)
        var_delta_2 = te.lang.cce.vmul(accum_t, momentum_brc)
        var_delta_2 = te.lang.cce.vmul(var_delta_2, lr_brc)
        var_delta = te.lang.cce.vadd(var_delta, var_delta_2)
        var_t = te.lang.cce.vsub(var, var_delta)
    else:
        var_delta = te.lang.cce.vmul(accum_t, lr_brc)
        var_t = te.lang.cce.vsub(var, var_delta)

    if dtype == "float16":
        var_t = te.lang.cce.cast_to(var_t, "float16")
        accum_t = te.lang.cce.cast_to(accum_t, "float16")

    return var_t, accum_t


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    var_shape = []
    for var, name in zip(dict_list, name_list):
        shape = var.get('shape')
        dtype = var.get('dtype').lower()
        if name == 'var':
            var_shape = list(shape)
        if name != 'lr' and name != 'momentum' and name != 'x2' \
                and var_shape != list(shape):
            raise RuntimeError(
                "The shapes of var, accum and x1 must be equal.")
        if (name == 'lr' or name == 'momentum'
            or name == 'x2') and shape[0] != 1:
            raise RuntimeError(
                "The shapes of lr, momentum and x2 must be scalar.")

        util.check_dtype_rule(dtype, ('float32', 'float16'))
        util.check_shape_rule(shape, max_shape_num=SHAPE_SIZE_LIMIT)
        util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
        shape_refine = (functools_reduce(operator.mul, shape), )
        list_placeholder.append(
            tvm.placeholder(shape=shape_refine, name=name, dtype=dtype))
    return list_placeholder


# pylint: disable=unbalanced-tuple-unpacking
@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, bool,
                       str)
def fused_mul_apply_momentum(var,
                             accum,
                             lr,
                             x1,
                             momentum,
                             x2,
                             out_var,
                             out_accum,
                             use_nesterov=False,
                             kernel_name="fused_mul_apply_momentum"):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + x1 * x2
    if use_nesterov is True:
        var -= gard * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32.

    accum: the dict of mutable tensor accum. Must have the same dtype as `var`.

    lr : the dict of scalar lr. Must have the same dtype as `var`.

    x1 : the dict of tensor grad. Must have the same dtype as `var`.

    momentum : the dict of scalar momentum. Must have the same dtype as `var`.

    x2 : the dict of scalar grad. Must have the same dtype as `var`.

    out_var : the dict of var output.

    out_accum : the dict of accum output

    use_nesterov: bool. If true, use nesterov computing grad,
                 default value is False.

    kernel_name : cce kernel name, default value is "fused_mul_apply_momentum".

    Returns
    -------
    None
    """

    input_name_list = ['var', 'accum', 'lr', 'x1', 'momentum', 'x2']
    util.check_kernel_name(kernel_name)
    var, accum, lr, x1, momentum, x2 = _get_placeholder(
        [var, accum, lr, x1, momentum, x2], input_name_list)
    out_var, out_accum = _fused_mul_apply_momentum_compute(
        var, accum, lr, x1, momentum, x2, out_var, out_accum, use_nesterov)
    outs = [out_var, out_accum]
    build_list = [var, accum, lr, x1, momentum, x2, out_var, out_accum]

    with tvm.target.cce():
        sch = generic.auto_schedule(outs)
    config = {"name": kernel_name, "tensor_list": build_list}
    te.lang.cce.cce_build_code(sch, config)

