#!/usr/bin/env python
# -*- coding:utf-8 -*-
# noinspection PyInterpreter
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

relu_v2

  Op_description :
    Algrithm: relu_v2(x) = x and 1 when x > 0 , else 0, 0

    # relu_v2(
    #   x,
    #   y,
    #   mask,
    #   kernel_name='relu_v2')

  Supportive_dtype_format :
    ['float16', 'float32', 'int8', 'int32', 'uint8']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : the last dim of `x` must be mutiply of 8.
    [2] All : shape size limit is 2147483648.
"""


from te import tvm
import te.lang.cce
from te.platform.cce_conf import api_check_support
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from topi import generic
from topi.cce import util

# const value
CONST_ZERO = 0


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("relu_v2")
def relu_v2_compute(x, y, mask, kernel_name="relu_v2_cce"):
    """
    Algrithm : relu_v2(x) = x and 1 when x > 0 , else 0, 0

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    mask : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of relu_v2_res

    mask: result of relu_v2_mask
    """

    inp_dtype = x.dtype
    shape = x.shape
    compatible_dtype = x.dtype

    if inp_dtype == 'int8' and api_check_support('te.lang.cce.cast_to',
                                                 's82f16'):
        x = te.lang.cce.cast_to(x, 'float16')
        compatible_dtype = 'float16'
    if api_check_support('te.lang.cce.vrelu', compatible_dtype):
        data_res = te.lang.cce.vrelu(x)
    else:
        tensor_zero = te.lang.cce.broadcast(tvm.const(CONST_ZERO,
                                                      compatible_dtype),
                                            shape)
        data_res = te.lang.cce.vmax(x, tensor_zero)

    data_res = te.lang.cce.cast_to(data_res, inp_dtype)
    mask = te.lang.cce.vcmp(x, CONST_ZERO, "gt", "bit")

    return data_res, mask


@util.check_input_type(dict, dict, dict, str)
def relu_v2(x, y, mask, kernel_name="relu_v2"):
    """
    Algrithm: relu_v2(x) = x and 1 when x > 0 , else 0, 0

    Parameters
    ----------
    Algorithm: relu_v2

    Parameters:

    x: the dict of input data, support float16, float32, int8, int32, uint8

    y: the dict of output

    mask: the dict of mask_output

    kernel_name: cce kernel name, default value is "relu_v2".

    Returns
    -------
    None
    """

    shape = x.get("shape")
    dtype = x.get("dtype")

    util.check_kernel_name(kernel_name)
    check_shape(shape)

    if shape[-1] % 8 != 0:
        raise RuntimeError(
            "the last axis if shape must be dive by 8")

    check_list = ("float16", "float32", "int8", "int32", "uint8")
    check_dtype(dtype, check_list)

    dtype = dtype.lower()
    input_data = tvm.placeholder(shape, dtype, "input_data")

    with tvm.target.cce():
        res, res_mask = relu_v2_compute(input_data, y, mask, kernel_name)
        sch = generic.auto_schedule([res, res_mask])

    config = {"name": kernel_name,
              "tensor_list": [input_data, res, res_mask],
              "print_ir": False
              }

    te.lang.cce.cce_build_code(sch, config)
