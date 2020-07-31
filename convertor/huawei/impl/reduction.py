#!/usr/bin/python3
# -*- coding: UTF-8 -*-
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

Reduction
"""
from __future__ import absolute_import
from functools import reduce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_conf import CceProductParams
from topi import generic
from topi.cce import util
import te.lang.cce
from te import platform as tbe_platform



@fusion_manager.register("reduction")
def reduction_compute(data_info, product_verion, operation, axis, coeff):
    """
    Reduce a tensor on a certain axis, and scale output with coeff
    Parameters
    ----------
    data_info: include TVM tensor,shape and dtype
    product_verion: include mini("1.1"、"1.3"),cloud("1.6"),es("5.10"),1951("2.3")
    operation : can only be one of "1:SUM, 2:ASUM (sum of abs), 3:SUMSQ (sum of sqr), 4:MEAN"
    axis : the axis to reduce
    coeff : scale for output
    Returns
    -------
    output of the data's reduction
    """

    input_data = data_info.get("tensor")
    input_data_shape = data_info.get("shape")
    input_data_dtype = data_info.get("dtype")

    if product_verion not in ("Hi3796CV300ES", "Hi3796CV300CS"):
    #if product_verion in ("1.1", "1.3", "1.6", "2.3"):
        if input_data_dtype == "float16":
            input_data = te.lang.cce.cast_to(input_data, "float32")

    # computational process
    if operation == 2:
        data_tmp_input = te.lang.cce.vabs(input_data)
        tmp = te.lang.cce.vmuls(data_tmp_input, coeff)

    elif operation == 3:
        data_tmp_input = te.lang.cce.vmul(input_data, input_data)
        tmp = te.lang.cce.vmuls(data_tmp_input, coeff)

    elif operation == 4:
        size = input_data_shape[-1]
        cof = float(coeff * (size ** (-0.5)))
        tmp = te.lang.cce.vmuls(input_data, cof)

    elif operation == 1:
        tmp = te.lang.cce.vmuls(input_data, coeff)

    res = te.lang.cce.sum(tmp, axis=axis)

    if operation == 4:
        size = input_data_shape[-1]
        size_reci = float(size ** (-0.5))
        res = te.lang.cce.vmuls(res, size_reci)

    if product_verion not in ("Hi3796CV300ES", "Hi3796CV300CS"):
    #if product_verion in ("1.1", "1.3", "1.6", "2.3"):
        if input_data_dtype == "float16":
            res = te.lang.cce.cast_to(res, "float16")

    return res


# pylint: disable=redefined-outer-name, too-many-arguments, E1101
@util.check_input_type(dict, dict, int, int, float, str)
def reduction(input_x, output_y, operation=1, axis=0, coeff=1.0, kernel_name="reduction"):
    """
    Reduce a tensor on a certain axis, and scale output with coeff
    Parameters
    ----------
    input_x : input tensor
    output_y: output tensor
    operation : can only be one of "1:SUM, 2:ASUM (sum of abs), 3:SUMSQ (sum of sqr), 4:MEAN"
    axis : the first axis to reduce, may be negative to index from the end
            (e.g., -1 for the last axis).If axis == 0, the output Blob always has
            the empty shape (count 1), performing reduction across the entire input.
    coeff : scale for output
    kernel_name : cce kernel name, default value is "cce_reductionLayer"
    Returns
    -------
    None
    """
    #cur_cce_product = CceProductParams().cce_product
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    # input_x's shape check
    shape = input_x.get("shape")
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)

    # input_x' dtype check
    check_list = ("float16", "float32")
    inp_dtype = input_x.get("dtype").lower()
    util.check_dtype_rule(inp_dtype, check_list)
    if cce_product in ("Hi3796CV300ES") and inp_dtype == "float32":
        raise RuntimeError("The current product not support float32.")

    # axis parameter check
    if axis >= len(shape) or axis < -len(shape):
        raise RuntimeError("input axis is out of range, axis value can be from %d to %d"
                           % (-len(shape), len(shape) - 1))

    # operation parameter check
    if operation not in (1, 2, 3, 4):
        raise RuntimeError("operation can only be one of 1, 2, 3, 4")

    # Preprocess
    if axis < 0:
        axis = len(shape) + axis
    shape = list(shape)
    shape = shape[:axis] + [reduce(lambda x, y: x * y, shape[axis:])]

    # define input
    data = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    data_info = {"tensor": data, "shape": shape, "dtype": inp_dtype}
    res = reduction_compute(data_info, cce_product, operation, axis, coeff)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}
    te.lang.cce.cce_build_code(sch, config)
