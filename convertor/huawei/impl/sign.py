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

sign
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns

SHAPE_SIZE_LIMIT = 2147483648  # shape limit

# pylint: disable=unused-argument
@fusion_manager.register("sign")
def sign_compute(input_x, output_y, kernel_name="sign"):
    """
    compute for sign
    """
    inp_dtype = input_x.dtype
    fp16_max = tvm.const(32768, dtype=inp_dtype)
    fp16_min = tvm.const(2**(-15), dtype=inp_dtype)
    data_tmp = input_x
    if inp_dtype == "float16":
        data_tmp = te.lang.cce.round_to(input_x, 0.5, -0.5)

    new_data = te.lang.cce.vmuls(data_tmp, fp16_max)
    tmp2 = te.lang.cce.vabs(new_data)
    anuminate = te.lang.cce.vadds(tmp2, fp16_min)
    rec = te.lang.cce.vrec(anuminate)
    fp16_res = te.lang.cce.vmul(new_data, rec)
    int_res = te.lang.cce.round(fp16_res)
    res = te.lang.cce.cast_to(int_res, inp_dtype)
    return res


@util.check_input_type(dict, dict, str)
def sign(input_x, output_y, kernel_name="sign"):
    """
                                 x*32768
    algrithm: sign = round(-------------------------)
                            2 ** (-15) + |x*32768|

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is sign

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32", "int32"]
    inp_dtype = input_x.get("dtype").lower()
    if not inp_dtype in check_list:
        raise RuntimeError("sign only support float16, float32, int32")

    shape = util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=inp_dtype)

    res = sign_compute(data, output_y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}
    te.lang.cce.cce_build_code(sch, config)
