#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

common function
"""

from te import tvm
import te.lang.cce



def sign(input_data):
    """
    Algrithm:
        sign(x) = 2**(15)/(2**(-15) + 2**(15) *|x|)
    ----------
    Parameters
        input_data: the placeholder of data input
    ----------
    Returns
        A tensor of sign(x)
    -------
    """
    dtype = input_data.dtype

    if dtype == "float16":
        fp_max = tvm.const(2**15, dtype)
        fp_min = tvm.const(2**(-15), dtype)
    elif dtype == "float32":
        fp_max = tvm.const(2**62, dtype)
        fp_min = tvm.const(2**(-62), dtype)
    else:
        raise RuntimeError(
            "The type must be float16 or float32.")
    new_data = te.lang.cce.vmuls(input_data, fp_max)
    abs_data = te.lang.cce.vabs(new_data)
    denominator = te.lang.cce.vadds(abs_data, fp_min)
    res = te.lang.cce.vdiv(new_data, denominator)

    return res
