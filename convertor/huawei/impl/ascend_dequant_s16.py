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

ascend_dequant_s16
"""

import operator

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

NONETYPE = type(None)


# pylint: disable=invalid-name,unused-argument,unnecessary-lambda
# pylint: disable=too-many-arguments,too-many-locals
@fusion_manager.register("ascend_dequant_s16")
def ascend_dequant_s16_compute(x0, deq_scale, x1, y, relu_flag=False,
                               kernel_name='ascend_dequant_s16'):
    """
    int32 -> int16

    Parameters:
     ----------
    x : the placeholder of input

    deq: the placeholder of requant num

    x1:  the placeholder of add input tensor

    y: the dict of output

    relu_flag : the relu mode when true the result to do relu,
                default value is False

    kernel_name : cce kernel name, default value is "ascend_dequant_s16"

    Returns:

    res : the result of ascend_dequant_s16
    -------
    None
    """

    x0_shape = x0.shape
    deq_shape = deq_scale.shape

    x0_shape_list = te.lang.cce.util.shape_to_list(x0_shape)
    deq_shape_list = te.lang.cce.util.shape_to_list(deq_shape)

    align_shape = x0_shape_list
    align_shape[2] = (align_shape[2] + 15) // 16 * 16

    tensor_flag = False
    if operator.eq((deq_shape_list[1] * deq_shape_list[4]),
                   (x0_shape_list[1] * x0_shape_list[3])):
        tensor_flag = True

    if x1 is not None:
        if tensor_flag:
            res_ub = tvm.compute(align_shape, lambda i, j, k, l:
                                 tvm.vdeq_cast(x0(i, j, k, l),
                                               deq_scale(0, j, 0, 0, l),
                                               "int16",
                                               do_relu=relu_flag)
                                 + x1(0, j, 0, 0, l),
                                 name='s32_to_s16', tag="dequant_s16_vector")
        else:
            res_ub = tvm.compute(align_shape, lambda i, j, k, l:
                                 tvm.deq_cast(x0(i, j, k, l),
                                              deq_scale(0, 0, 0, 0, 0),
                                              "int16")
                                 + x1(0, j, 0, 0, l),
                                 name='s32_to_s16', tag="dequant_s16_scale")
    else:
        if tensor_flag:
            res_ub = tvm.compute(align_shape, lambda i, j, k, l:
                                 tvm.vdeq_cast(x0(i, j, k, l),
                                               deq_scale(0, j, 0, 0, l),
                                               "int16",
                                               do_relu=relu_flag),
                                 name='s32_to_s16', tag="dequant_s16_vector")
        else:
            res_ub = tvm.compute(align_shape, lambda i, j, k, l:
                                 tvm.deq_cast(x0(i, j, k, l),
                                              deq_scale(0, 0, 0, 0, 0),
                                              "int16"),
                                 name='s32_to_s16', tag="dequant_s16_scale")
    res_shape = te.lang.cce.util.shape_to_list(res_ub.shape)
    res_shape[2] = x0.shape[2]
    res = tvm.compute(res_shape, lambda *indice: res_ub(*indice),
                      name='dequant_s16_remove_pad',
                      tag="dequant_s16_remove_pad")

    return res


@util.check_input_type((dict), (dict), (dict, NONETYPE), (dict), bool, str)
def ascend_dequant_s16(x0, deq_scale, x1, y, relu_flag=False,
                       kernel_name='ascend_dequant_s16'):
    """
    int32 -> int16

    Parameters:
    ----------
    x0 : the dict of input

    deq_scale: the dict of dequant num

    x1 : the input of add tensor

    y : the dict of output.

    relu_flag : the relu mode when true the result to do relu

    kernel_name : cce kernel name, default value is "ascend_dequant_s16"

    Returns:
    -------
    None
    """

    shape_x0 = x0.get("shape")
    format_x0 = x0.get("format")
    dtype_x0 = x0.get("dtype")

    shape_deq = deq_scale.get("shape")
    format_deq = deq_scale.get("format")
    dtype_deq = deq_scale.get("dtype")

    check_list = [("int32",), ("uint64",), ("int16",)]
    util.check_dtype_rule(dtype_x0, check_list[0])
    util.check_dtype_rule(dtype_deq, check_list[1])

    if len(shape_x0) != 5:
        raise ValueError(
            "the x0 shape must of length 5")

    if len(shape_deq) != 5:
        raise ValueError(
            "the deq_scale shape must of length 5")

    if format_x0 != "NC1HWC0":
        raise ValueError(
            "the x format must be NC1HWC0")

    if format_deq != "NC1HWC0":
        raise ValueError(
            "the deq format must be NC1HWC0")

    if shape_deq[0] != 1 or shape_deq[2] != 1 or shape_deq[3] != 1:
        raise RuntimeError(
            "ascend dequants16 deq shape must be 1 in n,h,w")

    # n, C1, H*W, C0
    shape_x0 = [shape_x0[0], shape_x0[1], shape_x0[2] * shape_x0[3],
                shape_x0[4]]

    input_x0 = tvm.placeholder(shape_x0, dtype_x0, "x0")
    input_deq = tvm.placeholder(shape_deq, dtype_deq, "deq_scale")
    input_x1 = None
    if x1:
        input_x1 = tvm.placeholder(shape_x0, "int16", "x1")

    with tvm.target.cce():
        res = ascend_dequant_s16_compute(input_x0, input_deq, input_x1,
                                         relu_flag, kernel_name)
        generic.auto_schedule(res)
