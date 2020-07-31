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

clip_by_norm_no_div_sum
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
def select_compute(condition, x1, x2, y, kernel_name="select"):
    """
    compute for select

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of input x1
    x2: TVM tensor
        the placeholder of input x2
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    res = te.lang.cce.vsel(condition, x1, x2)
    return res


def greater_compute(x, y, z, kernel_name="greater"):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    x : Tensor
        input data_x
    y : Tensor
        input data_y
    z : dict
        shape and dtype of output data_z
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    the result
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    shape_y = te.lang.cce.util.shape_to_list(y.shape)
    dtype = x.dtype.lower()
    shape_x, shape_y, shape = util.produce_shapes(shape_x, shape_y)
    data_x = te.lang.cce.broadcast(x, shape)
    data_y = te.lang.cce.broadcast(y, shape)

    res = te.lang.cce.vcmp(data_x, data_y, 'gt', 'bool')

    return res


def maximum_compute(input_x, input_y, output_z, kernel_name="maximum"):
    """
    calculating data maximum

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    result: TVM tensor
        the result of sqrt
    """
    shape1 = te.lang.cce.util.shape_to_list(input_x.shape)
    shape2 = te.lang.cce.util.shape_to_list(input_y.shape)
    shape1 = util.scalar2tensor_one(shape1)

    shape2 = util.scalar2tensor_one(shape2)

    shape1, shape2, shape_max = util.produce_shapes(shape1, shape2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    data1_tmp1 = te.lang.cce.broadcast(input_x, shape_max)
    data2_tmp1 = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vmax(data1_tmp1, data2_tmp1)
    return res


@fusion_manager.register("clip_by_norm_no_div_sum")
def clip_by_norm_no_div_sum_compute(data_input_x,
                                    data_greater_zeros,
                                    data_select_ones,
                                    data_maximum_ones,
                                    y,
                                    kernel_name="clip_by_norm_no_div_sum"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """

    # greater
    greater_result = greater_compute(data_input_x, data_greater_zeros,
                                     {}, kernel_name)

    # select
    select_result = select_compute(greater_result, data_input_x,
                                   data_select_ones, {}, kernel_name)

    # sqrt
    sqrt_result = te.lang.cce.vsqrt(select_result)

    # select1
    select1_result = select_compute(greater_result, sqrt_result,
                                    data_input_x, {}, kernel_name)

    res = maximum_compute(select1_result, data_maximum_ones, {}, kernel_name)

    return res


def clip_by_norm_no_div_sum(x, greater_zeros, select_ones, maximum_ones, y,
                            kernel_name="clip_by_norm_no_div_sum"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    shape_greater_zeros = greater_zeros.get("shape")
    shape_select_ones = select_ones.get("shape")
    shape_maximum_ones = maximum_ones.get("shape")

    shape_x, shape_greater_zeros, shape_greater_max = \
        util.produce_shapes(shape_x, shape_greater_zeros)
    shape_x, shape_select_ones, shape_select_max = \
        util.produce_shapes(shape_x, shape_select_ones)
    shape_x, shape_maximum_ones, shape_maximum_max = \
        util.produce_shapes(shape_x, shape_maximum_ones)

    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    util.check_kernel_name(kernel_name)

    data_input_x = tvm.placeholder(shape_x,
                                   name="data_input_x",
                                   dtype=dtype_x)
    data_greater_zeros = tvm.placeholder(shape_greater_zeros,
                                         name="data_greater_zeros",
                                         dtype=dtype_x)
    data_select_ones = tvm.placeholder(shape_select_ones,
                                       name="data_select_ones",
                                       dtype=dtype_x)
    data_maximum_ones = tvm.placeholder(shape_maximum_ones,
                                        name="data_maximum_ones",
                                        dtype=dtype_x)

    res = clip_by_norm_no_div_sum_compute(data_input_x,
                                          data_greater_zeros,
                                          data_select_ones,
                                          data_maximum_ones,
                                          y,
                                          kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "bool_storage_as_1bit": False,
              "tensor_list": [data_input_x, data_greater_zeros,
                              data_select_ones, data_maximum_ones, res]}

    te.lang.cce.cce_build_code(sch, config)
