#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

smooth_l1_loss
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("smooth_l1_loss")
def smooth_l1_loss_compute(input_predict,
                           input_label,
                           output_loss,
                           sigma,
                           kernel_name="smooth_l1_loss"):
    """
    calculating data

    Parameters
    ----------
    input_predict : TVM tensor
        the placeholder of input_predict
    input_label : TVM tensor
        the placeholder of input_label
    output_loss : dict
        dict of output_loss, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "smooth_l1_loss"

    Returns
    -------
    output tensor
    """

    input_dtype = input_predict.dtype
    half_const = tvm.const(0.5, dtype=input_dtype)
    half_const_tensor = te.lang.cce.broadcast(half_const, input_predict.shape)
    one_const = tvm.const(1.0, dtype=input_dtype)
    one_const_tensor = te.lang.cce.broadcast(one_const, input_predict.shape)

    sigma_scalar = tvm.const(sigma, dtype=input_dtype)

    input_sub_res = te.lang.cce.vsub(input_predict, input_label)

    method_one_res = te.lang.cce.vmul(
        te.lang.cce.vmuls(input_sub_res, half_const), input_sub_res)
    method_one_res = te.lang.cce.vmuls(method_one_res, 1 / sigma_scalar)
    predict_label_sub_abs = te.lang.cce.vabs(input_sub_res)
    method_two_res = te.lang.cce.vsub(predict_label_sub_abs,
                                      te.lang.cce.vmuls(half_const_tensor,
                                                        sigma_scalar))

    is_method_one_res = te.lang.cce.vcmpsel(predict_label_sub_abs,
                                            sigma_scalar,
                                            'lt', 1.0, 0.0)
    is_method_two_res = te.lang.cce.vsub(one_const_tensor, is_method_one_res)
    method_one_get_res = te.lang.cce.vmul(method_one_res, is_method_one_res)
    method_two_get_res = te.lang.cce.vmul(method_two_res, is_method_two_res)
    res = te.lang.cce.vadd(method_one_get_res, method_two_get_res)
    return res


@util.check_input_type(dict, dict, dict, float, str)
def smooth_l1_loss(predict,
                   label,
                   loss,
                   sigma=1.0,
                   kernel_name="smooth_l1_loss"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of input
    loss : dict
        shape and dtype of output,
        should be same shape and type as input
    sigma: float
        sigma,default value is 1
    kernel_name : str
        kernel name, default value is "smooth_l1_loss"

    Returns
    -------
    None
    """

    check_list = ("float16", "float32")
    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype")
    input_predict_dtype = dtype_predict.lower()
    util.check_dtype_rule(input_predict_dtype, check_list)

    shape_label = label.get("shape")
    dtype_label = label.get("dtype")
    input_label_dtype = dtype_label.lower()
    dtype_loss = loss.get("dtype").lower()
    util.check_dtype_rule(input_label_dtype, check_list)
    util.check_dtype_rule(dtype_loss, check_list)

    util.compare_tensor_dict_key(predict, label, "shape")
    util.check_shape_rule(shape_predict)
    util.check_tensor_shape_size(shape_predict)
    util.check_shape_rule(shape_label)
    util.check_tensor_shape_size(shape_label)
    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(input_predict_dtype, check_list)
    shape_predict, shape_label = \
        refine_shapes_for_broadcast(shape_predict, shape_label)
    input_predict = tvm.placeholder(
        shape_predict, name="predict", dtype=input_predict_dtype)
    input_label = tvm.placeholder(
        shape_label, name="label", dtype=input_label_dtype)
    res = smooth_l1_loss_compute(input_predict, input_label, loss, sigma,
                                 kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [input_predict, input_label, res]
    }

    te.lang.cce.cce_build_code(sch, config)
