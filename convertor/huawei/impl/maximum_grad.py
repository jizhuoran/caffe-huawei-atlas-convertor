#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

maximum_grad
"""
from __future__ import absolute_import
from impl import fused_minimum_or_maximum_grad
from topi.cce import util

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@util.check_input_type(dict, dict, dict, dict, dict, bool, bool, str)
def maximum_grad(input_dz, input_x, input_y, output_dx, output_dy,
                 grad_x=True, grad_y=True, kernel_name="maximum_grad"):
    """
    algorithm:
    calculating maximum_grad of the three input data

    Parameters
    ----------
    input_dz : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    input_y : dict
        shape and dtype of y input, only support float16, float32
    output_dx: dict
        shape and dtype of output, should be same shape and type as input
    output_dy: dict
        shape and dtype of output, should be same shape and type as input
    grad_x: bool
        if grad_x is true,output need return dx
    grad_y: bool
        if grad_y is true,output need return dy
    kernel_name : str
        cce kernel name, default value is maximum_grad

    Returns:
    -------
    none.
    """
    shape_dz = input_dz.get("shape")
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_dz.get("dtype").lower()
    fused_minimum_or_maximum_grad.fused_minimum_or_maximum_grad_cce(shape_dz,
                                                                    shape_x,
                                                                    shape_y,
                                                                    grad_x,
                                                                    grad_y,
                                                                    "GE",
                                                                    dtype,
                                                                    kernel_name)
