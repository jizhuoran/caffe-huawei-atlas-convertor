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

ascend_requant_s16
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
@fusion_manager.register("ascend_requant_s16")
def ascend_requant_s16_compute(x, deq_scale, x1, y, y1, dual_output, relu_flag,
                               kernel_name='ascend_requant_s16'):
    """
    int16 -> int8

    Parameters:
     ----------
    x : the placeholder of input

    deq_scale: the placeholder of deq_scale

    x1: the placeholder of x1

    y : the dict of output.

    y1 : the dict of output1.

    dual_output : the sqrt mode when true return 2 result,
                  default value is False

    relu_flag : the relu mode when true the result to do relu,
                default value is False

    kernel_name : cce kernel name, default value is "ascend_requant_s16"

    Returns:

    res : the result of ascend_requant_s16 which is list
    -------
    None
    """
    x_shape = x.shape
    deq_shape = deq_scale.shape

    x_shape_list = te.lang.cce.util.shape_to_list(x_shape)
    deq_shape_list = te.lang.cce.util.shape_to_list(deq_shape)
    align_shape = x_shape_list
    align_shape[1] = (align_shape[1] + 1) // 2 * 2
    align_shape[2] = (align_shape[2] + 15) // 16 * 16

    tensor_flag = False
    if operator.eq((deq_shape_list[1] * deq_shape_list[4]),
                   (x_shape_list[1] * x_shape_list[3])):
        tensor_flag = True

    if x1 is not None:
        if relu_flag:
            res_s16 = tvm.compute(x_shape, lambda *indices: tvm.relu(
                x(*indices) + x1(*indices)), name="res_s16",
                                  tag="requant_s16_vaddrelu")
        else:
            res_s16 = tvm.compute(x_shape,
                                  lambda *indices: x(*indices) + x1(*indices),
                                  name="res_s16", tag="requant_s16_vadd")
    else:
        if relu_flag:
            res_s16 = tvm.compute(x_shape, lambda *indices: tvm.relu(
                x(*indices)), name="res_s16",
                                  tag="requant_s16_relu")
        else:
            res_s16 = tvm.compute(x_shape, lambda *indices: x(*indices),
                                  name="res_s16", tag="requant_s16")

    if tensor_flag:
        res_ub = tvm.compute(align_shape, lambda i, j, k, l:
                             tvm.conv_vdeq(res_s16(i, j, k, l),
                                           deq_scale(0, j, 0, 0, l)).astype(
                                               "int8"),
                             name='s16_to_s8', tag="requant_s16_vector")
    else:
        res_ub = tvm.compute(align_shape, lambda i, j, k, l:
                             tvm.conv_vdeq(res_s16(i, j, k, l),
                                           deq_scale(0, 0, 0, 0, 0)).astype(
                                               "int8"),
                             name='s16_to_s8', tag="requant_s16_scale")

    res = _format_transfer(align_shape, res_ub)

    if dual_output:
        return [res_s16, res]

    return [res]


def _format_transfer(shape, x):
    trans_shape = (shape[0], shape[1] // 2, shape[2], shape[3] * 2)
    res = tvm.compute(trans_shape,
                      lambda i, j, k, l: x[
                          i, (j * 32 + l) // 16, k, (j * 32 + l) % 16],
                      name='data_transfer',
                      tag="requant_s16_data_transfer", )
    return res


@util.check_input_type((dict), (dict), (dict, NONETYPE), (dict),
                       (dict, NONETYPE), bool, bool, str)
def ascend_requant_s16(x, deq_scale, x1, y, y1, dual_output=False,
                       relu_flag=False, kernel_name='ascend_requant_s16'):
    """
    int16 -> int8

    Parameters:
    ----------
    x : the dict of input

    deq_scale: the dict of requant num

    x1: the dict of elewise num

    y : the dict of output.

    y1: the dict of output1

    dual_output : the sqrt mode when true return 2 result,
                  default value is False

    relu_flag : the relu mode when true the result to do relu,
                default value is False

    kernel_name : cce kernel name, default value is "ascend_requant_s16"

    Returns:
    -------
    None
    """

    shape_x = x.get("shape")
    format_x = x.get("format")
    dtype_x = x.get("dtype")

    shape_deq = deq_scale.get("shape")
    format_deq = deq_scale.get("format")
    dtype_deq = deq_scale.get("dtype")

    check_list = [("int16",), ("uint64",)]
    util.check_dtype_rule(dtype_x, check_list[0])
    util.check_dtype_rule(dtype_deq, check_list[1])

    if len(shape_x) != 5:
        raise ValueError(
            "the x shape must of length 5")

    if len(shape_deq) != 5:
        raise ValueError(
            "the deq_scale shape must of length 5")

    if format_x != "NC1HWC0":
        raise ValueError(
            "the x format must be NC1HWC0")

    if format_deq != "NC1HWC0":
        raise ValueError(
            "the deq format must be NC1HWC0")

    if shape_deq[0] != 1 or shape_deq[2] != 1 or shape_deq[3] != 1:
        raise RuntimeError(
            "ascend requant_s16 deq shape must be 1 in n,h,w")

    # n, C1, H*W, C0
    shape_x = [shape_x[0], shape_x[1], shape_x[2] * shape_x[3], shape_x[4]]

    input_x = tvm.placeholder(shape_x, dtype_x, "x")
    input_deq = tvm.placeholder(shape_deq, dtype_deq, "deq_scale")

    if x1:
        input_x1 = tvm.placeholder(shape_x, "int16", "x1")
    else:
        input_x1 = None

    with tvm.target.cce():
        res = ascend_requant_s16_compute(input_x, input_deq, input_x1, y, y1,
                                         dual_output, relu_flag, kernel_name)
        generic.auto_schedule(res)
