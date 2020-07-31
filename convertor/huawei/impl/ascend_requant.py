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

ascend_requant
"""
import operator

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=invalid-name,unused-argument,unnecessary-lambda
# pylint: disable=too-many-arguments,too-many-locals
@fusion_manager.register("ascend_requant")
def ascend_requant_compute(x, req_scale, y, relu_flag=False,
                           kernel_name='ascend_requant'):
    """
    int32 -> int8

    Parameters:
     ----------
    x : the placeholder of input

    req_scale: the placeholder of requant num

    y : the dict of output.

    relu_flag : the relu mode when true the result to do relu

    kernel_name : cce kernel name, default value is "ascend_requant"

    Returns:

    res : the result of ascend_requant
    -------
    None
    """

    x_shape = x.shape
    req_shape = req_scale.shape

    x_shape_list = te.lang.cce.util.shape_to_list(x_shape)
    req_shape_list = te.lang.cce.util.shape_to_list(req_shape)

    tensor_flag = False
    if operator.eq((req_shape_list[1] * req_shape_list[4]),
                   (x_shape_list[1] * x_shape_list[3])):
        tensor_flag = True

    align_shape = x_shape_list
    align_shape[1] = (align_shape[1] + 1) // 2 * 2
    align_shape[2] = (align_shape[2] + 15) // 16 * 16

    if tensor_flag:
        res_ub = tvm.compute(align_shape, lambda i, j, k, l:
                             tvm.vdeq_cast(x(i, j, k, l),
                                           req_scale(0, j, 0, 0, l),
                                           "int8",
                                           do_relu=relu_flag),
                             name='s32_to_s8', tag="requant_vector")
    else:
        res_ub = tvm.compute(align_shape, lambda i, j, k, l:
                             tvm.deq_cast(x(i, j, k, l),
                                          req_scale(0, 0, 0, 0, 0),
                                          "int8"),
                             name='s32_to_s8', tag="requant_scale")

    res_ub_reform = _format_transfer(align_shape, res_ub)
    res_shape = te.lang.cce.util.shape_to_list(res_ub_reform.shape)
    res_shape[2] = x.shape[2]
    res = tvm.compute(res_shape, lambda *indice: res_ub_reform(*indice),
                      name='requant_remove_pad', tag="requant_remove_pad")
    return res


def _format_transfer(shape, x):
    trans_shape = (shape[0], shape[1] // 2, shape[2], shape[3] * 2)
    res = tvm.compute(trans_shape,
                      lambda i, j, k, l: x[
                          i, (j * 32 + l) // 16, k, (j * 32 + l) % 16],
                      name='data_transfer', tag="data_transfer")
    return res


@util.check_input_type((dict), (dict), (dict), bool, str)
def ascend_requant(x, req_scale, y, relu_flag=False,
                   kernel_name='ascend_requant'):
    """
    int32 -> int8

    Parameters:
    ----------
    x : the dict of input

    req_scale: the dict of requant num

    offset: the dict of offset num

    y : the dict of output.

    relu_flag : the relu mode when true the result to do relu

    kernel_name : cce kernel name, default value is "ascend_requant"

    Returns:
    -------
    None
    """

    shape_x = x.get("shape")
    format_x = x.get("format")

    shape_req = req_scale.get("shape")
    format_req = req_scale.get("format")

    dtype_x = x.get("dtype")
    dtype_req = req_scale.get("dtype")

    check_list = [("int32",), ("uint64",)]
    util.check_dtype_rule(dtype_x, check_list[0])
    util.check_dtype_rule(dtype_req, check_list[1])

    if len(shape_x) != 5:
        raise ValueError("the x shape must of length 5")

    if len(shape_req) != 5:
        raise ValueError("the req_scale shape must of length 5")

    if format_x != "NC1HWC0":
        raise ValueError("the x format must be NC1HWC0")

    if format_req != "NC1HWC0":
        raise ValueError("the req format must be NC1HWC0")

    if shape_req[0] != 1 or shape_req[2] != 1 or shape_req[3] != 1:
        raise RuntimeError(
            "ascend requant req shape must be 1 in n,h,w")

    # n, C1, H*W, C0
    shape_x = [shape_x[0], shape_x[1], shape_x[2] * shape_x[3], shape_x[4]]

    input_x = tvm.placeholder(shape_x, dtype_x, "x")
    input_req = tvm.placeholder(shape_req, dtype_req, "req_scale")

    with tvm.target.cce():
        res = ascend_requant_compute(input_x, input_req, relu_flag,
                                     kernel_name)
        generic.auto_schedule(res)
