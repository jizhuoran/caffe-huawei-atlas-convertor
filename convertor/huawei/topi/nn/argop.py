#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

argmin or argmax op util
"""

from te import tvm
import topi
from topi import tag


def compute_argop_cce(data, axis, arg_type):
    """cce argmax arithmetic operator, find the index of max value

    Parameters
    ----------
    data : tvm.Tensor
    
    axis : Int
        axis must in the range [- len(data.shap), len(data.shap) - 1]

    arg_type: string
        "argmax" or "argmin"

    Returns
    -------
    Output : tvm.Tensor
    """
    if not isinstance(axis, int):
        raise RuntimeError("axis must be int")

    real_axis = axis
    if real_axis < 0:
        real_axis += len(data.shape)

    if real_axis >= len(data.shape):
        raise RuntimeError(
            "Axis must in range [- len(data.shap), len(data.shap)-1], while axis is now %d" % real_axis)

    if data.dtype == "float16":
        data_fp16 = data
    else:
        with tvm.tag_scope(tag.BROADCAST):
            data_fp16 = tvm.compute(data.shape, lambda *indice: data(*indice).astype("float16"),
                                    name="data_fp16_cast")

    if arg_type == "argmax":
        res = topi.argmax(data_fp16, axis=axis, keepdims=True)
    elif arg_type == "argmin":
        res = topi.argmin(data_fp16, axis=axis, keepdims=True)
    else:
        raise RuntimeError("arg_tpye must be argmax or argmin!")

    return res
