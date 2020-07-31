#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=invalid-name, unused-variable
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cross
"""
from __future__ import absolute_import as _abs
from te import tvm


@tvm.tag_scope(tag="cross")
def compute_cross_cce(Input1, Input2, need_type_convert=False):
    """cce cross arithmetic operator, a x b = (l, m, n) x (o, p, q) = (mq-np, no-lq, lp-mo)

    Parameters
    ----------
    Input1 : tvm.Tensor
        with shape [3, n]
    
    Input2 : tvm.Tensor
        with shape [3, n]

    need_type_convert : 
        need to convert the input type to fp16

    Returns
    -------
    Output : tvm.Tensor
        with shape [3, n]
    """
    shape = tuple(Input1.shape)
    sh = (shape[-1],)

    if need_type_convert:
        data_a = tvm.compute(shape, lambda i, j: Input1[i, j].astype("float16"), name="data_a")
        data_b = tvm.compute(shape, lambda i, j: Input2[i, j].astype("float16"), name="data_b")
    else:
        data_a = Input1
        data_b = Input2

    a0_x_b1 = tvm.compute(sh, lambda i: data_a[0, i]*data_b[1, i], name="a0_x_b1")
    a0_x_b2 = tvm.compute(sh, lambda i: data_a[0, i]*data_b[2, i], name="a0_x_b2")
    a1_x_b0 = tvm.compute(sh, lambda i: data_a[1, i]*data_b[0, i], name="a1_x_b0")
    a1_x_b2 = tvm.compute(sh, lambda i: data_a[1, i]*data_b[2, i], name="a1_x_b2")
    a2_x_b0 = tvm.compute(sh, lambda i: data_a[2, i]*data_b[0, i], name="a2_x_b0")
    a2_x_b1 = tvm.compute(sh, lambda i: data_a[2, i]*data_b[1, i], name="a2_x_b1")

    res0 = tvm.compute(sh, lambda i: a1_x_b2[i] - a2_x_b1[i], name="res0")
    res1 = tvm.compute(sh, lambda i: a2_x_b0[i] - a0_x_b2[i], name="res1")
    res2 = tvm.compute(sh, lambda i: a0_x_b1[i] - a1_x_b0[i], name="res2")

    res = tvm.compute(shape, lambda i, j: tvm.select(i == 0, res0[j],
                                                     tvm.select(i == 1, res1[j], res2[j])),
                      name='res')

    if need_type_convert:
        Output = tvm.compute(shape, lambda i, j: res[i, j].astype(Input1.dtype), name="Output")
    else:
        Output = res

    return Output
