#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf l2loss
"""

import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2147483648  # shape limit

# pylint: disable=invalid-name
@util.check_input_type(dict, dict, str)
def l2_loss(x, y, kernel_name="l2_loss"):
    """
    Reduce a tensor on a certain axis, and scale output with coeff

    Parameters
    ----------
    shape : shape of data

    dtype : source data type, only support float16, float32

    kernel_name : kernel name, default value is "l2_loss"

    Returns
    -------
    None

    """
    shape = x.get("shape")
    dtype = x.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32"]
    if not dtype.lower() in check_list:
        raise RuntimeError(
            "l2_loss only support float16 float32")

    shape, axis = util.simplify_axis_shape(shape, range(len(shape)))

    inp_dtype = dtype.lower()
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)

    coeff_sqrt = tvm.const(1.0 / (2**(0.5)), dtype=inp_dtype)

    data_mul = te.lang.cce.vmuls(data_input, coeff_sqrt)
    data_sqr = te.lang.cce.vmul(data_mul, data_mul)
    res = te.lang.cce.sum(data_sqr, axis)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
