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

elu_grad

  Op_description :
    do element-wise elu operation.

    # elu_grad(
    #   grads,
    #   activations,
    #   y,
    #   kernel_name='cce_elu_grad')

  Supportive_dtype_format :
    ["float16", "float32"]
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : `grads` and `activations` must have the same shape and type.
    [2] All : shape size limit is 2147483648.
"""
import operator

from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from te.utils.op_utils import refine_shape_axes
import topi
from topi.cce import util

NUM_ZERO = 0.0
NUM_ONE = 1.0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("elu_grad")
def elu_grad_compute(grads, activations, y, kernel_name="elu_grad"):
    """
    elu_grad_compute
    f(x) = vmul(add(min(activation, 0), 1), gradient)

    Parameters:
    ----------
    data_gradient : the placeholder of gradient data

    data_activation : the placeholder of activation data

    data_output : the dict of output

    kernel_name : cce kernel name, default value is "elu_grad"

    Returns : A Tensor. Has the same type as data_gradient.
    -------
    """

    dtype = grads.dtype
    shape = grads.shape

    if dtype.lower() == "float16" and \
       api_check_support("te.lang.cce.vadd", "float32"):
        grads = te.lang.cce.cast_to(grads, "float32")
        activations = te.lang.cce.cast_to(activations, "float32")

    input_border = tvm.const(NUM_ZERO, grads.dtype)
    scalar_param_one = tvm.const(NUM_ONE, grads.dtype)
    tensor_input_border = te.lang.cce.broadcast(input_border, shape)
    tensor_scalar_param_one = te.lang.cce.broadcast(scalar_param_one, shape)

    min_res = te.lang.cce.vmin(activations, tensor_input_border)
    add_res = te.lang.cce.vadd(min_res, tensor_scalar_param_one)
    res = te.lang.cce.vmul(add_res, grads)

    if dtype.lower() == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res

# pylint: disable=invalid-name
@util.check_input_type(dict, dict, dict, str)
def elu_grad(grads, activations, y, kernel_name="elu_grad"):
    """
    do element-wise elu_grad operation

    Parameters:
    ----------
    grads: the dict of gradient input, only support float16, float32

    activations: the dict of activation input, only support float16, float32

    y : the dict of output

    kernel_name : cce kernel name, default value is "cce_elu_grad"

    Returns
    -------
    None
    """

    shape_gradient = grads.get("shape")
    shape_activation = activations.get("shape")
    dtype_gradient = grads.get("dtype")
    dtype_activation = activations.get("dtype")

    util.check_kernel_name(kernel_name)

    check_shape(shape_gradient)
    check_shape(shape_activation)
    if not operator.eq(shape_gradient, shape_activation):
        raise RuntimeError("all input shape must be equal")
    shape_gradient, _ = refine_shape_axes(shape_gradient, [])
    shape_activation, _ = refine_shape_axes(shape_activation, [])

    check_list = ("float16", "float32")
    check_dtype(dtype_gradient, check_list)
    check_dtype(dtype_activation, check_list)
    if dtype_gradient.lower() != dtype_activation.lower():
        raise RuntimeError("all input dtype must be same")

    dtype = dtype_gradient.lower()
    data_gradient = tvm.placeholder(shape_gradient, dtype=dtype, name="data_gradient")
    data_activation = tvm.placeholder(shape_activation, dtype=dtype, name="data_activation")
    res = elu_grad_compute(data_gradient, data_activation, y, kernel_name)

    with tvm.target.cce():
        auto_sch = topi.generic.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": [data_gradient, data_activation, res]}
    te.lang.cce.cce_build_code(auto_sch, config)
