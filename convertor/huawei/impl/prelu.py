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

prelu
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=unused-argument
@fusion_manager.register("prelu")
def prelu_compute(input_x, weight_input, output_y, kernel_name="prelu"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    weight_input : TVM tensor
        the placeholder of weight_input
    kernel_name : str
        kernel name, default value is "prelu"

    Returns
    -------
    output tensor
    """
    if input_x.dtype == "float16":
        scalar_zero = tvm.const(0, dtype="float16")
    else:
        scalar_zero = tvm.const(0, dtype="float32")
    val_max = te.lang.cce.vmaxs(input_x, scalar_zero)
    val_min = te.lang.cce.vmins(input_x, scalar_zero)
    val_prod = te.lang.cce.vmul(val_min, weight_input)
    res = te.lang.cce.vadd(val_max, val_prod)
    return res


# pylint: disable=too-many-locals,invalid-name
@util.check_input_type(dict, dict, dict, str)
def prelu(input_x, input_A, output_y, kernel_name="prelu"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    input_A : dict
        shape and dtype of input_A, should be same type as input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input_x
    kernel_name : str
        kernel name, default value is "prelu"
    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_format = input_x.get("format")
    input_dtype = dtype.lower()

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")
    util.check_dtype_rule(input_dtype, check_list)

    weight_shape = input_A.get("shape")
    weight_dtype = input_A.get("dtype").lower()

    util.check_shape_rule(weight_shape)
    util.check_tensor_shape_size(weight_shape)
    util.check_dtype_rule(weight_dtype, check_list)
    if weight_dtype != input_dtype:
        raise RuntimeError("dtype of input_x and input_A should be same")

    weight_dim = len(weight_shape)
    feature_dim = len(shape)

    if input_format == "NC1HWC0":
        if weight_dim == 5 and (shape[1] != weight_shape[1] or
                                shape[4] != weight_shape[4]):
            raise RuntimeError(
                "weight dim only support two values: 1 or 5, "
                "when feature_dim is 5, channel(C1/C0) dim"
                " for input_x and input_A must be matched, "
                "while feature [C1, C0]:[%d, %d], weight [C1, C0]:[%d, %d]" %
                (shape[1], shape[4], weight_shape[1], weight_shape[4]))
        if weight_dim != 5:
            if weight_dim == 1 and weight_shape[0] == 1:
                weight_shape_new = [1] * 5
            else:
                raise RuntimeError(
                    "weight_dim only support two values: 1 or 5,"
                    " when feature_dim is 5(NC1HWC0) and "
                    "weight_dim is not equal to 5, both weight_shape[0] "
                    "and weight_dim must be 1, "
                    "while weight_shape[0] is {0}, weight_dim is {1}".format(
                        weight_shape[0], weight_dim))
        else:
            weight_shape_new = [1] * feature_dim
            weight_shape_new[1] = weight_shape[1]
            weight_shape_new[-1] = weight_shape[-1]

    elif feature_dim == 1:
        if weight_shape[0] != 1 or weight_dim != 1:
            raise RuntimeError(
                "when feature_dim is 1, both weight_shape[0] "
                "and weight_dim must be 1, "
                "while weight_shape[0] is {0}, weight_dim is {1}".format(
                    weight_shape[0], weight_dim))
        weight_shape_new = [1]
    # input_x:DIM = 2,3,4,5,6,7...
    else:
        if (weight_shape[0] != shape[1] and
                weight_shape[0] != 1) or weight_dim != 1:
            raise RuntimeError(
                "channel dim of input_x and input_A must be matched, "
                "and weight_dim must be 1, "
                "while channel dim of input_A is {0}, ".format(
                    weight_shape[0]), "channel dim of input_x is {0}, ".format(
                        shape[1]), "weight_dim is {0}".format(weight_dim))
        weight_shape_new = [1] * feature_dim
        weight_shape_new[1] = weight_shape[0]

    if len(weight_shape_new) == sum(weight_shape_new):
        weight_shape_new = [1]
        total_calc_num = 1
        for i, _ in enumerate(shape):
            total_calc_num = total_calc_num * shape[i]
        shape_new = [total_calc_num]
        data_input = tvm.placeholder(
            shape_new, name="data_input", dtype=input_dtype)
        weight_input = tvm.placeholder(
            weight_shape_new, name="weight_input", dtype=input_dtype)
        weight_input1 = te.lang.cce.broadcast(
            weight_input, shape_new, output_dtype=input_dtype)
    else:
        data_input = tvm.placeholder(
            shape, name="data_input", dtype=input_dtype)
        weight_input = tvm.placeholder(
            weight_shape_new, name="weight_input", dtype=input_dtype)
        weight_input1 = te.lang.cce.broadcast(
            weight_input, shape, output_dtype=input_dtype)

    res = prelu_compute(data_input, weight_input1, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_input, weight_input, res]
    }

    te.lang.cce.cce_build_code(sch, config)
