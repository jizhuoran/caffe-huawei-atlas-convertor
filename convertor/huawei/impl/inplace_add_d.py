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

inplace_add_d
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable = locally-disabled,invalid-name,too-many-arguments
# pylint: disable = unused-argument,no-member
@fusion_manager.register("inplace_add_d")
def inplace_add_d_compute(x, v, y, indices, kernel_name="inplace_add_d"):
    """
    inplace_add_d compute process

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    v : TVM tensor.
        the placeholder of v
    y : dict
        dict with keys(shape and dtype) of output
    indices : a vector.
        indices into the left-most dimension of x
    kernel_name : str
        kernel name, default value is "inplace_add_d"

    Returns
    -------
    output tensor
    """

    res = te.lang.cce.inplace_add(x, indices, v)

    return res


@util.check_input_type(dict, dict, dict, (tuple, list), str)
def inplace_add_d(x, v, y, indices, kernel_name="inplace__add_d"):
    """
    algorithm: inplacea_add_d

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    v : TVM tensor.
        the placeholder of v
    y : dict
        dict with keys(shape and dtype) of output
    indices : a vector.
        indices into the left-most dimension of x
    kernel_name : str
        kernel name, default value is "inplace_add_d"

    Returns
    -------
    None
    """
    check_tuple = ("float16", "float32", "int32")
    shape_x = x.get("shape")
    shape_v = v.get("shape")

    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_v)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_v)
    util.check_dtype_rule(x.get("dtype").lower(), check_tuple)
    util.check_dtype_rule(v.get("dtype").lower(), check_tuple)
    util.check_kernel_name(kernel_name)
    indices = list(indices)

    if len(shape_x) != len(shape_v):
        raise RuntimeError("The number of dimension x must"
                           " be same as dimension v")

    if shape_v[0] != len(indices):
        raise RuntimeError("The length of rank 0 of tensor v must be"
                           " the same as length of indices")

    for i in range(1, len(shape_v)):
        if shape_x[i] != shape_v[i]:
            raise RuntimeError("The length of each rank of tensor x must "
                               "be the same as length of each rank of "
                               "tensor v except the first dimension")

    for i, _ in enumerate(indices):
        indices[i] = (indices[i] % shape_x[0] + shape_x[0]) % shape_x[0]

    data_x = tvm.placeholder(shape_x, name="data_x",
                             dtype=x.get("dtype").lower())
    data_v = tvm.placeholder(shape_v, name="data_v",
                             dtype=v.get("dtype").lower())
    res = inplace_add_d_compute(data_x, data_v, y, indices,
                                kernel_name="inplace_add_d")

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_v, res]}
    te.lang.cce.cce_build_code(sch, config)
