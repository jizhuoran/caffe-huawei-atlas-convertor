#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

square
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns

# shape size limit for aicore is 2**31
SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable=unused-argument
@fusion_manager.register("square")
def square_compute(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    res : tvm.tensor
        the result of square
    """
    res = te.lang.cce.vmul(input_x, input_x)
    return res


@util.check_input_type(dict, dict, str)
def square(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "square"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32", "int32"]
    if not dtype in check_list:
        raise RuntimeError("square only support float16, float32, int32")

    shape = util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=dtype.lower())

    with tvm.target.cce():
        res = square_compute(data, output_y, kernel_name)
        sch = generic.auto_schedule(res)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": [data, res]
    }

    te.lang.cce.cce_build_code(sch, config)
