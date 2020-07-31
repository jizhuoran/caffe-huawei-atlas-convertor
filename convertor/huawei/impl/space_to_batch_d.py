#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

space_to_batch
"""
from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util
from impl.space_to_batch_nd_d import space_to_batch_nd_d_compute


# pylint: disable=invalid-name,unused-argument
def _check_param(x, y, paddings, block_size, kernel_name):
    """check the parameters including shape, dtype, block_shape, paddings
    and kernel_name.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    dtype_list = ("float16", "float32")
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_dtype_rule(dtype, dtype_list)
    util.check_kernel_name(kernel_name)

    if len(paddings) == 4:
        paddings = [[paddings[0], paddings[1]], [paddings[2], paddings[3]]]
    if len(shape) != 5:
        raise RuntimeError(
            "the shape of image_input should be 5, but got: %d" % len(shape))
    if block_size < 2:
        raise RuntimeError("the attr block_size must be greater than one")

    _check_padding(paddings)

    padding_shape = (shape[0], shape[1],
                     shape[2] + paddings[0][0] + paddings[0][1],
                     shape[3] + paddings[1][0] + paddings[1][1], shape[4])
    util.check_shape_rule(padding_shape)
    util.check_tensor_shape_size(padding_shape)

    padding_height, padding_width = padding_shape[2], padding_shape[3]
    if padding_height % block_size != 0 or padding_width % block_size != 0:
        raise RuntimeError(
            "both height_pad and width_pad must be divisible by block_size")

    output_shape = (padding_shape[0] * block_size * block_size,
                    padding_shape[1], padding_shape[2] // block_size,
                    padding_shape[3] // block_size, padding_shape[4])
    util.check_shape_rule(output_shape)
    util.check_tensor_shape_size(output_shape)


def _check_padding(paddings):
    """
    check the paddings
    """
    if len(paddings) != 2 or len(paddings[0]) != 2 or len(paddings[1]) != 2:
        raise RuntimeError("the shape of paddings should be 2x2")

    def _check_padding_val(val):
        """
        check the padding value
        """
        if not (isinstance(val, int) and val >= 0):
            raise RuntimeError("paddings should be integer and must be >= 0")

    _check_padding_val(paddings[0][0])
    _check_padding_val(paddings[0][1])
    _check_padding_val(paddings[1][0])
    _check_padding_val(paddings[1][1])


@util.check_input_type(dict, dict, int, (tuple, list), str)
def space_to_batch_d(x,
                     y,
                     block_size,
                     paddings,
                     kernel_name="space_to_batch_d"):
    """
    the main function of space_to_batch_d

    Parameters
    ----------
    x: dict,shape and datatype,datatype supports float16,float32
    y: dict,shape and datatype,datatype supports float16,float32
    block_size: must be greater than one. It indicates the block size
    paddings: (tuple, list),the padding of the input with zeros across the
              spatial dimensions as follows:
              paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
    kernel_name: cce kernel name, default value is "space_to_batch_d"
    Returns
    -------
    None
    """
    _check_param(x, y, paddings, block_size, kernel_name)

    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    block_shape = [block_size, block_size]

    data = tvm.placeholder(input_shape, name="data", dtype=input_dtype)
    res = space_to_batch_nd_d_compute(data, y, block_shape, paddings,
                                      kernel_name)
    sch = tvm.create_schedule(res.op)
    with build_config:
        tvm.build(sch, [data, res], "cce", name=kernel_name)
