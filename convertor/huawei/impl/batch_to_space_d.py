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

batch_to_space
"""
from impl.batch_to_space_nd_d import batch_to_space_nd_d_compute
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

DIM_CNT = 5
CROPS_LEN = 2


# pylint: disable=locally-disabled,invalid-name
@util.check_input_type(dict, dict, int, (list, tuple), str)
def batch_to_space_d(x, y, block_size, crops, kernel_name="batch_to_space_d"):
    """
    batch_to_space for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_size: int
        the size of block.
    crops: list or tuple
        2-D with shape [2, 2], crops[i] = [crop_start, crop_end].
    kernel_name: str
        cce kernel name, default value is "batch_to_space".

    Returns
    -------
    None.
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    if len(crops) == 4:
        crops = [[crops[0], crops[1]], [crops[2], crops[3]]]
    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)
    util.check_kernel_name(kernel_name)
    check_list = {"float16", "float32"}
    util.check_dtype_rule(input_dtype, check_list)

    if len([x for x in input_shape if isinstance(x, int) and x > 0])\
            != len(input_shape):
        raise RuntimeError("input_shape should be positive integer")

    if len(input_shape) != DIM_CNT:
        raise RuntimeError("the length of input_shape must be 5,\
        while it is: %d" % len(input_shape))

    if not (len(crops) == CROPS_LEN and len(crops[0]) == CROPS_LEN
            and len(crops[1]) == CROPS_LEN):
        raise RuntimeError("shape of crops should be 2*2")

    if not (isinstance(crops[0][0], int) and crops[0][0] >= 0
            and isinstance(crops[0][1], int) and crops[0][1] >= 0
            and isinstance(crops[1][0], int) and crops[1][0] >= 0
            and isinstance(crops[1][1], int) and crops[1][1] >= 0):
        raise RuntimeError("crops  must be >= 0")

    batch_size = input_shape[0]
    if batch_size % (block_size * block_size) != 0:
        raise RuntimeError("batch_size  should be divisible by\
        the square of block_size")
    output_shape = (input_shape[0] // block_size // block_size, input_shape[1],
                    input_shape[2] * block_size - crops[0][0] - crops[0][1],
                    input_shape[3] * block_size - crops[1][0] - crops[1][1],
                    input_shape[4])
    util.check_shape_rule(output_shape)
    util.check_tensor_shape_size(output_shape)

    block_shape = [block_size, block_size]
    data = tvm.placeholder(input_shape, name="data", dtype=input_dtype)

    res = batch_to_space_nd_d_compute(data, y, block_shape, crops, kernel_name)

    sch = tvm.create_schedule(res.op)

    with build_config:
        tvm.build(sch, [data, res], "cce", name=kernel_name)
