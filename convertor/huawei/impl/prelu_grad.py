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

prelu_grad
"""

import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from topi import generic
from topi.cce import util


def check_inputs_shape(features_shape, weights_shape, input_format):
    """
    check input para

    Parameters
    ----------
    features_shape : list
        shape of feature_map
    weights_shape : list
        shape of weights
    input_format : str
        str of input

    Returns
    -------
    None
    """
    features_shape = list(features_shape)
    weights_shape = list(weights_shape)
    features_dim = len(features_shape)
    weights_dim = len(weights_shape)

    if features_dim == 1:
        raise RuntimeError(
            "weight dim only support two values: 1,"
            " or the number of channels at input, "
            "feature don't support 1D shape,"
            " while feature shape is",
            features_shape)
    if input_format == "NC1HWC0" and features_dim == 5 and weights_dim == 5 \
            and (features_shape[1] != weights_shape[1]
                 or features_shape[4] != weights_shape[4]):
        raise RuntimeError(
            "weight dim only support two values: 1,"
            "or the number of channels at input, "
            "when feature_dim and weight_dim are 5(NC1HWC0), "
            "channel(C1/C0) dim for features and "
            "weights must be matched, while feature "
            "[C1, C0]:[%d, %d], weight [C1, C0]:[%d, %d]" %
            (features_shape[1], features_shape[4], weights_shape[1],
             weights_shape[4]))
    if input_format == "NC1HWC0" and features_dim == 5 and weights_dim == 1 \
            and features_shape[1]*features_shape[4] != weights_shape[0] \
            and weights_shape[0] != 1:
        raise RuntimeError(
            "weight dim only support two values: 1,"
            " or the number of channels at input, "
            "when feature_dim is 5(NC1HWC0), and weight_dim is 1, "
            "weight value must be 1 or the number of channel(C1*C0),"
            " while feature shape is",
            features_shape, ",weight shape is", weights_shape)
    if input_format == "NC1HWC0" and features_dim == 5 and weights_dim != 5 \
            and weights_dim != 1:
        raise RuntimeError(
            "weight dim only support two values: 1,"
            " or the number of channels at input, "
            "when feature_dim is 5(NC1HWC0),"
            " weight_dim must be equal to 5(1, C1, 1, 1, C0) "
            "or 1(1 or C1*C0),"
            "while weight shape is", weights_shape)
    if features_dim == 4 and weights_dim == 1 and \
            features_shape[1] != weights_shape[0] and weights_shape[0] != 1:
        raise RuntimeError(
            "weight dim only support two values: 1,"
            " or the number of channels at input, "
            "when feature_dim is 4, weight dim must"
            " be 1(weight shape is a vector), "
            " while feature shape is", features_shape,
            ",weight shape is", weights_shape)
    if features_dim == 4 and weights_dim != 1:
        raise RuntimeError(
            "weight dim only support two values: 1,"
            " or the number of channels at input, "
            "when feature_dim is 4, weight dim must"
            " be 1(weight shape is a vector),"
            "while feature shape is", features_shape,
            ",weight shape is", weights_shape)
    if input_format == "ND" and features_dim != 1 and weights_dim == 1\
            and features_shape[1] != weights_shape[0] \
            and weights_shape[0] != 1:
        raise RuntimeError(
            "weight dim only support two values: 1, "
            "or the number of channels at input, "
            "When feature_dim is ND(except 1D), weight "
            "dim must be 1(weight shape is a vector),"
            "channel dim for features and weights' must be matched,"
            "while feature shape is", features_shape,
            ",weight shape is", weights_shape)
    if input_format == "ND" and features_dim != 1 and weights_dim != 1:
        raise RuntimeError(
            "weight dim only support two values: 1, "
            "or the number of channels at input, "
            "When feature_dim is ND(except 1D), "
            "weight dim must be 1(weight shape is a vector),"
            "channel dim for features and weights' must be matched,"
            "while feature shape is", features_shape,
            ",weight shape is", weights_shape)


# pylint: disable=too-many-locals
def compare_zero_and_select(input_features, input_weights, shape, dtype):
    """
    compare_zero_and_select comput

    Parameters
    ----------
    input_features : TVM tensor
        the tensor of input_features
    input_weights : TVM tensor
        the tensor of input_weights
    shape: list
        list shape
    dtype : str
        dtype of input

    Returns
    -------
    result_dx : TVM tensor
        the tensor of result_dx
    result_da : TVM tensor
        the tensor of result_da
    """
    shape_input_features = \
        te.lang.cce.util.shape_to_list(input_features.shape)
    shape_input_weights = \
        te.lang.cce.util.shape_to_list(input_weights.shape)
    help_min = tvm.const(2**(-126), "float32")
    help_rec_one = tvm.const(2**38, "float32")
    help_rec_sec = tvm.const(2**44, "float32")

    if list(shape_input_features) != list(shape_input_weights):
        input_weights = te.lang.cce.broadcast(input_weights, shape, dtype)

    tmp_min = te.lang.cce.vmins(input_features, help_min)
    tmp_max = te.lang.cce.vmaxs(tmp_min, tvm.const(0, dtype))
    tmp_result = te.lang.cce.vmuls(tmp_max, help_rec_one)
    if dtype == "float32":
        tmp_result = te.lang.cce.vmuls(tmp_result, help_rec_sec)
    tmp_result = te.lang.cce.vmuls(tmp_result, help_rec_sec)
    tmp_neg_result = te.lang.cce.vadds(tmp_result, tvm.const(-1, dtype))
    tmp_neg_result = te.lang.cce.vabs(tmp_neg_result)

    result_dx_pos = tmp_result
    result_dx_neg = te.lang.cce.vmul(tmp_neg_result, input_weights)

    result_da_neg = te.lang.cce.vmul(tmp_neg_result, input_features)
    result_dx = te.lang.cce.vadd(result_dx_pos, result_dx_neg)
    result_da = result_da_neg

    return result_dx, result_da


# pylint: disable=too-many-arguments,unused-argument,too-many-branches
# pylint: disable=too-many-statements
def prelu_grad_compute(input_gradients,
                       input_features,
                       input_weights,
                       output_backprops_dx,
                       output_backprops_da,
                       input_format,
                       kernel_name="prelu_grad"):
    """
    calculating the backpropagation of prelu operation
    prelu equivalent function: prelu(x) = max(0, input_features)
    + input_weights * min(0, input_features)
    so prelu_grad output_backprops:
        output_backprops_dx = input_features > 0
                ? input_gradients : input_weights * input_gradients
        output_backprops_da = input_features > 0
                ? 0 : input_features * input_gradients

    Parameters
    ----------
    input_gradients : TVM tensor
        input tensor of grad
    input_features : TVM tensor
        input tensor of prelu output
    input_weights : TVM tensor
        input tensor of prelu output
    output_backprops_dx : dict
        dx output dict of prelu_grad
    output_backprops_da : dict
        da output dict of prelu_grad
    input_format : str
        input format of grad
    kernel_name : str
        kernel name, default value is "prelu_grad"

    Returns
    -------
    output tensor
    """
    dtype = input_gradients.dtype
    trans_type = dtype
    shape_input_gradients = te.lang.cce.util.shape_to_list(
        input_gradients.shape)
    shape_input_features = \
        te.lang.cce.util.shape_to_list(input_features.shape)
    shape_input_weights = \
        te.lang.cce.util.shape_to_list(input_weights.shape)
    shape = shape_input_gradients
    weight_share = False
    if input_format == "NC1HWC0":
        if shape_input_weights[4] == 1:
            weight_share = True
    else:
        if shape_input_weights[1] == 1:
            weight_share = True

    # need cast float16 to float32
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.broadcast", "float32"):
        input_gradients = te.lang.cce.cast_to(input_gradients, "float32")
        input_features = te.lang.cce.cast_to(
            input_features,
            "float32")
        input_weights = te.lang.cce.cast_to(input_weights, "float32")
        trans_type = "float32"

    # broadcast in case the input shapes are not same
    if list(shape_input_gradients) != list(shape_input_features):
        shape_input_gradients, shape_input_features, shape = \
            util.produce_shapes(shape_input_gradients, shape_input_features)
        input_gradients = te.lang.cce.broadcast(input_gradients, shape,
                                                trans_type)
        input_features = te.lang.cce.broadcast(input_features, shape,
                                               trans_type)

    # custom vcmpsel start
    res_dx, res_da = compare_zero_and_select(input_features, input_weights,
                                             shape, trans_type)
    output_backprops_dx = te.lang.cce.vmul(res_dx, input_gradients)
    output_backprops_da = te.lang.cce.vmul(res_da, input_gradients)
    # custom vcmpsel end

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmuls", "float32"):
        output_backprops_dx = te.lang.cce.cast_to(output_backprops_dx, dtype)
        output_backprops_dx_tmp = te.lang.cce.cast_to(output_backprops_dx,
                                                      "float32")
    else:
        output_backprops_dx_tmp = output_backprops_dx

    output_backprops_dx_zero = te.lang.cce.vmuls(output_backprops_dx_tmp,
                                                 tvm.const(0, trans_type))
    output_backprops_da = te.lang.cce.vadd(output_backprops_da,
                                           output_backprops_dx_zero)

    shape_input_da = te.lang.cce.util.shape_to_list(output_backprops_da.shape)
    axis = list(range(len(shape_input_da)))

    if len(shape_input_features) == 4:
        if not weight_share:
            output_backprops_da = te.lang.cce.sum(
                output_backprops_da, axis=[0, 2, 3], keepdims=False)
        else:
            output_backprops_da = te.lang.cce.sum(
                output_backprops_da, axis=axis, keepdims=False)
    elif len(shape_input_features) == 5 and input_format == "NC1HWC0":
        if not weight_share:
            output_backprops_da = te.lang.cce.sum(
                output_backprops_da, axis=[0, 2, 3], keepdims=True)
        else:
            output_backprops_da = te.lang.cce.sum(
                output_backprops_da, axis=axis, keepdims=False)
    else:
        if not weight_share:
            axis_nd = axis[0:1] + axis[2:]
            output_backprops_da = te.lang.cce.sum(
                output_backprops_da, axis=axis_nd, keepdims=False)
        else:
            output_backprops_da = te.lang.cce.sum(
                output_backprops_da, axis=axis, keepdims=False)

    if dtype == "float16":
        output_backprops_da = te.lang.cce.cast_to(output_backprops_da, dtype)

    return output_backprops_dx, output_backprops_da


@util.check_input_type(dict, dict, dict, dict, dict, str)
def prelu_grad(input_gradients,
               input_features,
               input_weights,
               output_backprops_dx,
               output_backprops_da,
               kernel_name="prelu_grad"):
    """
    calculating the backpropagation of prelu operation
    prelu equivalent function: prelu(x) =
    max(0, input_features) + input_weights * min(0, input_features)

    so prelu_grad output_backprops:
        output_backprops_dx = input_features > 0
            ? input_gradients : input_weights * input_gradients
        output_backprops_da = input_features > 0
            ? 0 : input_features * input_gradients

    support dtype:float16, float32

    Parameters
    ----------
    input_gradients : dict
        shape and dtype of grad, not support 1D
    input_features : dict
        shape and dtype of input tensor, not support 1D
    input_weights : dict
        shape and dtype of input learning weight
    output_backprops_dx : dict
        shape and dtype of output, should be same shape
         and type as input_features
    output_backprops_da : dict
        shape and dtype of output, should be same shape
         and type as input_features
    kernel_name : str
        kernel name, default value is "prelu_grad"

    Returns
    -------
    None
    """
    shape_input_gradients = input_gradients.get("shape")
    dtype_input_gradients = input_gradients.get("dtype")
    input_gradients_dtype = dtype_input_gradients.lower()
    input_format = input_gradients.get("format")

    shape_input_features = input_features.get("shape")
    dtype_input_features = input_features.get("dtype")
    input_features_dtype = dtype_input_features.lower()

    shape_input_weights = input_weights.get("shape")
    dtype_input_weights = input_weights.get("dtype")
    input_weights_dtype = dtype_input_weights.lower()

    # check dtype
    check_list = ("float16", "float32")
    util.compare_tensor_dict_key(input_gradients, input_features, "dtype")
    util.compare_tensor_dict_key(input_gradients, input_weights, "dtype")
    util.check_dtype_rule(dtype_input_gradients, check_list)
    util.check_dtype_rule(dtype_input_features, check_list)
    util.check_dtype_rule(dtype_input_weights, check_list)
    # check shape
    util.check_shape_rule(shape_input_gradients)
    util.check_shape_rule(shape_input_features)
    util.check_shape_rule(shape_input_weights)
    util.check_tensor_shape_size(shape_input_gradients)
    util.check_tensor_shape_size(shape_input_features)
    util.check_tensor_shape_size(shape_input_weights)
    if list(shape_input_gradients) != list(shape_input_features):
        shape_input_gradients, shape_input_features, shape_max = \
            util.produce_shapes(shape_input_gradients, shape_input_features)
        util.check_tensor_shape_size(shape_max)
    # check kernelname
    util.check_kernel_name(kernel_name)
    check_inputs_shape(shape_input_features, shape_input_weights, input_format)

    if len(shape_input_features) == 4:
        shape_input_weights = [1, shape_input_weights[0], 1, 1]
    elif input_format == "NC1HWC0" and len(shape_input_weights) == 5:
        pass
    elif input_format == "NC1HWC0" and len(shape_input_weights) == 1 \
            and shape_input_weights[0] != 1:
        weights_c1 = (shape_input_weights[0] + 15) // 16
        shape_input_weights = [1, weights_c1, 1, 1, 16]
    else:
        weights_shape = [1 for _ in range(len(shape_input_features))]
        weights_shape[1] = shape_input_weights[0]
        shape_input_weights = weights_shape
    data_input_gradients = tvm.placeholder(
        shape_input_gradients,
        name="data_input_gradients",
        dtype=input_gradients_dtype)
    data_input_features = tvm.placeholder(
        shape_input_features,
        name="data_input_features",
        dtype=input_features_dtype)
    data_input_weights = tvm.placeholder(
        shape_input_weights,
        name="data_input_weights",
        dtype=input_weights_dtype)
    res_dx, res_da = prelu_grad_compute(
        data_input_gradients, data_input_features, data_input_weights,
        output_backprops_dx, output_backprops_da, input_format, kernel_name)
    res = [res_dx, res_da]
    tensor_list = [
        data_input_gradients, data_input_features, data_input_weights
    ] + list(res)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)

