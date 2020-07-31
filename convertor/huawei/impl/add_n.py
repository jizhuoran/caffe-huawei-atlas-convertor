#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

add_n
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te.platform.cce_conf import api_check_support

SHAPE_SIZE_LIMIT = 2147483648  # General limitation of the reduce size for input shape: 2**31


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("add_n")
def add_n_compute_for_fusion(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """
    res = datas[0]
    for i, data_n in enumerate(datas):
        if i == 0:
            continue
        res = te.lang.cce.vadd(res, data_n)

    return res


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def add_n_compute(datas, output, tensor_num, kernel_name="add_n"):
    """
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    datas : list of placeholders
        all input data
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    res : output of the data's add_n
    """

    data_type = datas[0].dtype
    has_covert_float32 = (data_type == "float16" and
                          api_check_support("te.lang.cce.vadd", "float32"))

    first_data = datas[0] if not has_covert_float32 else\
        te.lang.cce.cast_to(datas[0], "float32")

    res = first_data
    for i, data_n in enumerate(datas):
        if i == 0:
            continue
        temp_data = data_n if not has_covert_float32 else \
            te.lang.cce.cast_to(data_n, "float32")
        res = te.lang.cce.vadd(res, temp_data)

    if has_covert_float32:
        res = te.lang.cce.cast_to(res, "float16")
    return res


@util.check_input_type((list, tuple), dict, int, str)
def add_n(inputs, output, tensor_num, kernel_name="add_n"):
    """
    algorithm: add_n
    calculating data's adds, z = a + b + c...

    Parameters
    ----------
    inputs : list or tuple of dict
        A list of Tensor objects, each with same shape and type.
    output : dict
        dict of output
    tensor_num:
        nums of input
    kernel_name : string
        cce kernel name, default value is add_n

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)

    input_num = len(inputs)
    if input_num < 2:
        raise RuntimeError("add_n inputs num should more than 1.")

    if input_num != tensor_num:
        raise RuntimeError("add_n inputs num should equal tensor_num.")

    shape_0 = inputs[0].get("shape")
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_0)

    check_list = ("float16", "float32", "int32")
    data = []
    for i, input_dict in enumerate(inputs):
        shape_input = input_dict.get("shape")
        if list(shape_0) != list(shape_input):
            raise RuntimeError("add_n only support same input shapes")
        util.check_shape_rule(shape_input)
        util.check_shape_size(shape_input, SHAPE_SIZE_LIMIT)
        dtype_input = input_dict.get("dtype").lower()
        util.check_dtype_rule(dtype_input, check_list)
        data.append(tvm.placeholder(fuseshape,
                                    name="data_%d" % i,
                                    dtype=dtype_input))

    res = add_n_compute(data, output, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    data.append(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": data}

    te.lang.cce.cce_build_code(schedule, config)
