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

smooth_l1_loss_grad
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce


# pylint: disable=unused-argument,too-many-arguments
@fusion_manager.register("smooth_l1_loss_grad")
def smooth_l1_loss_grad_compute(predict, label, dout, gradient, sigma,
                                kernel_name):
    """
    calculating data

    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    label : TVM tensor
        the placeholder of label
    dout : TVM tensor
        the placeholder of dout
    gradient : dict
        dict of gradient, include keys(shape and dtype)
    sigma : float
        sigma
    kernel_name : str
        kernel name, default value is "smooth_l1_loss_grad"

    Returns
    -------
    output tensor
    """
    dtype = predict.dtype
    shape_input_predict = te.lang.cce.util.shape_to_list(predict.shape)
    shape_input_label = te.lang.cce.util.shape_to_list(label.shape)

    if list(shape_input_predict) != list(shape_input_label):
        shape_input_predict, shape_input_label, shape = \
            util.produce_shapes(shape_input_predict, shape_input_label)
        predict = te.lang.cce.broadcast(predict, shape, dtype)
        label = te.lang.cce.broadcast(label, shape, dtype)
    out_sub = te.lang.cce.vsub(predict, label)
    # out = sigma if out_sub > sigma
    out_sub_one = te.lang.cce.vmins(out_sub, sigma)
    # out = -sigma if out_sub < -sigma
    out_sub_one_neg_one = te.lang.cce.vmaxs(out_sub_one, -sigma)
    out_sub_one_neg_one_sigma = te.lang.cce.vmuls(out_sub_one_neg_one,
                                                  1 / float(sigma))
    res = te.lang.cce.vmul(out_sub_one_neg_one_sigma, dout)

    return res


# pylint: disable=too-many-arguments,too-many-locals
@util.check_input_type(dict, dict, dict, dict, float, str)
def smooth_l1_loss_grad(predict,
                        label,
                        dout,
                        gradient,
                        sigma=1.0,
                        kernel_name="smooth_l1_loss_grad"):
    """
    calculating data
    smooth = x/sigma        if -sigma < x < sigma
             1              if x > sigma
             -1             if x < -sigma
    out = smooth * dout

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of output, should be same shape and type as predict
    gradient : dict
        shape and dtype of output, should be same shape and type as predict
    dout : dict
        shape and dtype of output, should be same shape and type as predict
    sigma : float
        sigma
    kernel_name : str
        kernel name, default value is "smooth_l1_loss_grad"

    Returns
    -------
    None
    """

    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype")
    label_shape = label.get("shape")
    dout_shape = dout.get("shape")
    input_dtype = predict_dtype.lower()
    label_dtype = label.get("dtype").lower()
    dout_dtype = dout.get("dtype").lower()

    util.compare_tensor_dict_key(predict, label, "shape")
    util.compare_tensor_dict_key(predict, dout, "shape")
    util.compare_tensor_dict_key(predict, label, "dtype")
    util.compare_tensor_dict_key(predict, dout, "dtype")
    check_list = ("float16", "float32")
    util.check_dtype_rule(input_dtype, check_list)
    util.check_dtype_rule(label_dtype, check_list)
    util.check_dtype_rule(dout_dtype, check_list)

    util.check_shape_rule(predict_shape)
    util.check_tensor_shape_size(predict_shape)
    util.check_shape_rule(label_shape)
    util.check_tensor_shape_size(label_shape)
    util.check_shape_rule(dout_shape)
    util.check_tensor_shape_size(dout_shape)
    util.check_kernel_name(kernel_name)
    shape = (functools_reduce(lambda x, y: x * y, predict_shape[:]),)
    predict_input = tvm.placeholder(
        shape, name="predict_input", dtype=input_dtype)
    label_input = tvm.placeholder(
        shape, name="label_input", dtype=input_dtype)
    dout_input = tvm.placeholder(
        shape, name="dout_input", dtype=input_dtype)
    res = smooth_l1_loss_grad_compute(predict_input, label_input, dout_input,
                                      gradient, sigma, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [predict_input, label_input, dout_input, res]
    }

    te.lang.cce.cce_build_code(sch, config)
