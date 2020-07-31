"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fused_mul_addn_l2loss
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util


@fusion_manager.register("fused_mul_addn_l2loss")
def fused_mul_addn_l2loss_compute(weight, const_input, weight_grad):
    """
    calculating data

    Parameters
    ----------
    weight : TVM tensor
        the placeholder of input_x
    const_input : TVM tensor
        the placeholder of input_x
    weight_grad : TVM tensor
        the placeholder of input_y
    kernel_name : str
        kernel name, default value is "fused_mul_addn_l2loss"

    Returns
    -------
    output tensor
    """

    # cal vmul and addn
    const_input = te.lang.cce.broadcast(const_input, weight.shape)
    data_mul = te.lang.cce.vmul(weight, const_input)
    data_addn = te.lang.cce.vadd(data_mul, weight_grad)

    axis = [i for i in range(len(weight.shape))]
    # cal l2 loss
    coeff_sqrt = tvm.const(1.0 / (2**(0.5)), dtype=weight.dtype)
    l2_loss_vmuls = te.lang.cce.vmuls(weight, coeff_sqrt)
    l2_loss_sqr = te.lang.cce.vmul(l2_loss_vmuls, l2_loss_vmuls)
    l2_loss = te.lang.cce.sum(l2_loss_sqr, axis)

    return data_addn, l2_loss

# pylint: disable=too-many-locals,too-many-arguments,unused-argument
@util.check_input_type(dict, dict, dict, dict, dict, str)
def fused_mul_addn_l2loss(input_x,
                          input_y,
                          input_z,
                          output_x,
                          output_y,
                          kernel_name="fused_mul_addn_l2loss"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    input_y : dict
        shape and dtype of input_y
    input_z : dict
        shape and dtype of input_z
    output_x : dict
        shape and dtype of first output, which should have shape (1,) and dtype
        as input
    output_y : dict
        shape and dtype of second output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "fused_mul_addn_l2loss"

    Returns
    -------
    None
    """

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()

    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype").lower()

    shape_z = [1 for _ in range(len(shape_x))]
    dtype_z = input_z.get("dtype").lower()

    check_list = ("float16", "float32")
    # check input x attr
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    util.check_dtype_rule(dtype_x, check_list)

    # check input y attr
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_y)
    util.check_dtype_rule(dtype_y, check_list)

    # check input z attr
    util.check_shape_rule(shape_z)
    util.check_tensor_shape_size(shape_z)
    util.check_dtype_rule(dtype_z, check_list)

    if dtype_x != dtype_y or dtype_x != dtype_z or dtype_y != dtype_z:
        raise RuntimeError(
            " Three input dtype must keep the same")

    if dtype_x == "float32":
        if not tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul", "float32"):
            raise RuntimeError(
                "Input dtype only support float16 while input dtype is float32")

        if not tbe_platform.cce_conf.api_check_support("te.lang.cce.vmuls", "float32"):
            raise RuntimeError(
                "Input dtype only support float16 while input dtype is float32")

        if not tbe_platform.cce_conf.api_check_support("te.lang.cce.sum", "float32"):
            raise RuntimeError(
                "Input dtype only support float16 while input dtype is float32")

        if not tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd", "float32"):
            raise RuntimeError(
                "Input dtype only support float16 while input dtype is float32")


    weight = tvm.placeholder(shape_x, name="weight", dtype=dtype_x)
    weight_grad = tvm.placeholder(shape_y, name="weight_grad", dtype=dtype_y)
    const_input = tvm.placeholder(shape_z, name="const_input", dtype=dtype_z)

    res1, res2 = fused_mul_addn_l2loss_compute(weight, const_input, weight_grad)
    res_list = [res1, res2]
    with tvm.target.cce():
        sch = generic.auto_schedule(res_list)


    config = {"name": kernel_name,
              "tensor_list": [weight, weight_grad, const_input] + res_list}

    te.lang.cce.cce_build_code(sch, config)
