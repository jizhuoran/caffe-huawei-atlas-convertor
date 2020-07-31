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

read_select
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

READ_SELECT_TAG = "read_select"
PARA_LIST_LEN = 5
EMPTY_LIST_LEN = 0


def _check_para_list_len(total_shape, valid_shape, slice_offset, stride_list):
    if len(total_shape) != PARA_LIST_LEN:
        raise RuntimeError("the len of input shape should be 5")

    if (len(valid_shape) != PARA_LIST_LEN) and (len(valid_shape) != EMPTY_LIST_LEN):
        raise RuntimeError("the len of valid shape should be 5 or 0")

    if (len(slice_offset) != PARA_LIST_LEN) and (len(slice_offset) != EMPTY_LIST_LEN):
        raise RuntimeError("the len of slice offset should be 5 or 0")

    if len(stride_list) != PARA_LIST_LEN:
        raise RuntimeError("the len of stride list should be 5")


# pylint: disable=locally-disabled,too-many-locals,unused-argument,dangerous-default-value
@fusion_manager.register("read_select")
def read_select_compute(input_tensor, output_x, stride_list=[1, 1, 1, 1, 1],
                        kernel_name="read_select"):
    """
    calculating data

    Parameters
    ----------
    input_tensor : TVM tensor
        the placeholder of input_x
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    stride_list : list
        list of stride for 5HD shape
    kernel_name : str
        kernel name, default value is "read_select"

    Returns
    -------
    output tensor
    """
    total_shape = input_tensor.shape
    n_total, c1_total, h_total, w_total, c0_total = total_shape

    # valid_shape and slice_offset are all 5HD shape
    valid_shape = input_tensor.op.attrs['valid_shape']
    slice_offset = input_tensor.op.attrs['slice_offset']
    _check_para_list_len(total_shape, valid_shape, slice_offset, stride_list)

    if len(valid_shape) == EMPTY_LIST_LEN:
        valid_shape = [n_total, c1_total,
                       (h_total + stride_list[2] - 1)//stride_list[2],
                       (w_total + stride_list[3] - 1)//stride_list[3],
                       c0_total]

    if len(slice_offset) == EMPTY_LIST_LEN:
        slice_offset = [0, 0, 0, 0, 0]

    n_valid, c1_valid, h_valid, w_valid, c0_valid = valid_shape

    output_shape = valid_shape
    output_ub_5d = \
        tvm.compute(output_shape,
                    lambda n, c1, h, w, c0:
                    input_tensor(n, c1, slice_offset[2] + h*stride_list[2],
                                 w*stride_list[3], c0),
                    name="output_ub_5d", attrs={"dma_copy":True})

    output_shape_4d = (n_valid, c1_valid, h_valid*w_valid, c0_valid)
    output_ub_4d = \
        tvm.compute(output_shape_4d,
                    lambda n, c1, hw, c0: output_ub_5d(n, c1, hw // w_valid, hw % w_valid, c0),
                    name="output_ub_4d")

    return output_ub_4d


# pylint: disable=locally-disabled,unexpected-keyword-arg,unnecessary-lambda
@util.check_input_type(dict, dict, (tuple, list), str)
def read_select(input_x, output_x, stride_list=[1, 1, 1, 1, 1], kernel_name="read_select"):
    """
    Read data with offset and stride

    Parameters
    ----------
    input_x : dict
        dict of input_x, include keys(shape and dtype)
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    stride_list : list
        list of stride for 5HD shape
    kernel_name : str
        kernel name, default value is "read_select"

    Returns
    -------
    output tensor
    """
    total_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    valid_shape = input_x.get("valid_shape")
    slice_offset = input_x.get("slice_offset")

    util.check_shape_rule(total_shape)
    if len(valid_shape) != EMPTY_LIST_LEN:
        util.check_shape_rule(valid_shape)
        util.check_tensor_shape_size(valid_shape)
    util.check_tensor_shape_size(total_shape)
    util.check_kernel_name(kernel_name)

    _check_para_list_len(total_shape, valid_shape, slice_offset, stride_list)

    check_list = ["float16", "int8"]
    if input_dtype not in check_list:
        raise RuntimeError("read_select only support %s while dtype is %s"
                           % (",".join(check_list), input_dtype))

    src_in_flag = "DDR"
    if "src_in_flag" in input_x:
        src_in_flag = input_x.get("src_in_flag")

    input_tensor = tvm.placeholder(total_shape,
                                   name="input_tensor",
                                   dtype=input_dtype,
                                   attrs={"valid_shape": valid_shape,
                                          "slice_offset": slice_offset,
                                          "src_in_flag": src_in_flag})

    output_4d = read_select_compute(input_tensor, output_x, stride_list,
                                    kernel_name=kernel_name)
    output_5d = output_4d.op.input_tensors
    output_5d_tensor = output_5d[0]

    res = tvm.compute(output_5d_tensor.shape, lambda *indice: output_5d_tensor(*indice),
                      name="res", tag=READ_SELECT_TAG)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_tensor, res]}
    te.lang.cce.cce_build_code(sch, config)
