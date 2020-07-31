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

write_select
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

WRITE_SELECT_TAG = "write_select"
PARA_LIST_LEN = 5
NAME_INDEX = [0]


# pylint: disable=locally-disabled,unnecessary-lambda,unused-argument
@fusion_manager.register("write_select")
def write_select_compute(input_tensor, output_x, kernel_name="write_select"):
    """
    calculating data

    Parameters
    ----------
    input_tensor : TVM tensor
        the input tensor
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "write_select"

    Returns
    -------
    output tensor
    """
    # input_shape = output_x.get("shape")
    input_shape = input_tensor.shape
    valid_shape = output_x.get("valid_shape")

    if len(valid_shape) != PARA_LIST_LEN:
        raise RuntimeError("the len of valid shape should be 5")

    _, _, h_valid, w_valid, c0_valid = valid_shape

    compute_name = "res_write_select" + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1
    res = tvm.compute(input_shape, lambda *indice: input_tensor(*indice),
                      name=compute_name,
                      attrs={"HWC0": h_valid*w_valid*c0_valid},
                      tag=WRITE_SELECT_TAG)

    return res


# pylint: disable=locally-disabled,too-many-locals,unexpected-keyword-arg
@util.check_input_type(dict, dict, str)
def write_select(input_x, output_x, kernel_name="write_select"):
    """
    Write data with offset

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "write_select"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    valid_shape = output_x.get("valid_shape")

    util.check_shape_rule(input_shape)
    util.check_shape_rule(valid_shape)
    util.check_tensor_shape_size(input_shape)
    util.check_tensor_shape_size(valid_shape)
    util.check_kernel_name(kernel_name)

    if util.is_lhisi_version():
        check_list = ["int32", "float16", "int8"]
    else:
        check_list = ["int32", "float16", "float32", "int8"]

    if input_dtype not in check_list:
        raise RuntimeError("write_select only support %s while dtype is %s"
                           % (",".join(check_list), input_dtype))

    if len(valid_shape) != PARA_LIST_LEN:
        raise RuntimeError("the len of valid shape should be 5")

    dst_out_flag = "DDR"
    if "dst_out_flag" in output_x:
        dst_out_flag = output_x.get("dst_out_flag")

    input_tensor_ph = tvm.placeholder(input_shape,
                                      name="input_tensor_ph",
                                      dtype=input_dtype,
                                      attrs={"valid_shape": valid_shape,
                                             "dst_out_flag": dst_out_flag})

    input_tensor = tvm.compute(input_shape,
                               lambda *indice: input_tensor_ph(*indice),
                               name="input_tensor")
    res = write_select_compute(input_tensor, output_x, kernel_name=kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_tensor_ph, res]}
    te.lang.cce.cce_build_code(sch, config)
