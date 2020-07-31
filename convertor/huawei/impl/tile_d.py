#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tile
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# shape limit, maximum value of int32
SHAPE_SIZE_LIMIT = 2147483648
# dim size limit
MAX_SHAPE_NUM = 10000000

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("tile_d")
def tile_d_compute(data, output_x, multiples, kernel_name="tile_d"):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    output_x: dict.
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        Cce kernel name, default value is "tile_d".

    Returns
    -------
    None
    """
    shape = te.lang.cce.util.shape_to_list(data.shape)
    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i*multiples_i
        out_shape.append(out_shape_i)
    res = te.lang.cce.broadcast(data, out_shape)

    return res


# pylint: disable=too-many-locals
@util.check_input_type(dict, dict, (list, tuple), str)
def tile_d(input_x, output_x, multiples, kernel_name="tile_d"):
    """algorithm: tile.
    The tile in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The tile op in TBE is different from tf.tile, tile of TBE use broadcast
    api, and only support that at least an axis in shape is 1.The '1' axis
    is to be multipled.
    For example, if shape = [51, 1] and multiples = [1, 77], after computation,
    the output shape will be [51, 77].
    Abnormal condition:
    1. The length of shape and multiples is not the same.
    2. The axis to be multipled in shape is not 1.
    3. The type of kernel_name is not string.
    4. The shape is neither list nor tuple.
    5. The dtype is not float32, float16, or int32.
    6. All of the axises of the multiples is 1.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        kernel name, default value is "tile_d".

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(multiples, max_shape_num=MAX_SHAPE_NUM)
    util.check_dtype_rule(dtype.lower(), ("float16", "float32", "int32"))
    shape = list(shape)
    multiples = list(multiples)

    if len(shape) > len(multiples):
        raise RuntimeError(
            "The len of multiples must be greater or equal"
            "to length of input shape")
    if len(shape) < len(multiples):
        len_error = len(multiples) - len(shape)
        shape = [1]*len_error + shape

    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i*multiples_i
        out_shape.append(out_shape_i)
    util.check_shape_size(out_shape, SHAPE_SIZE_LIMIT)

    shape_adapt = []
    multiples_adapt = []
    for i, shape_i in enumerate(shape):
        multiples_i = multiples[i]
        if multiples_i != 1 and shape_i != 1:
            shape_adapt.append(1)
            multiples_adapt.append(multiples_i)
            multiples_i = 1
        shape_adapt.append(shape_i)
        multiples_adapt.append(multiples_i)

    shape = shape_adapt
    multiples = multiples_adapt

    for shape_i, multiples_i in zip(shape, multiples):
        if not (shape_i == 1 or multiples_i == 1):
            raise RuntimeError(
                "In tile of TBE, any axis of either shape or multiples have "
                "to be 1")

    axis_not_multiple = 0
    for multiples_i in multiples:
        if multiples_i == 1:
            axis_not_multiple += 1
    if axis_not_multiple == len(multiples):
        raise RuntimeError(
            "In tile of TBE, the axis of multiples can't all be 1")

    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    res = tile_d_compute(data, output_x, multiples, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
