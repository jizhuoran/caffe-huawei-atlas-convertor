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

relu_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("relu_grad_v2")
def relu_grad_v2_compute(gradients, mask, backprops,
                         kernel_name="relu_grad_v2"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).

    Parameters
    ----------
    gradients: TVM tensor
        input tensor of grad
    mask: TVM tensor
        input tensor of relu output
    backprops: dict
        output dict of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad_v2"

    Returns
    -------
    res: TVM tensor
        the result of relu_grad_compute
    """
    dtype = gradients.dtype
    trans_type = dtype

    # need cast int8 or uint8 to float16
    if dtype in ("int8", "uint8"):
        gradients = te.lang.cce.cast_to(gradients, "float16")
        trans_type = "float16"

    result = te.lang.cce.vsel(mask, gradients, tvm.const(0, trans_type))

    # cast int8 or uint8 back
    if dtype in ("int8", "uint8"):
        result = te.lang.cce.cast_to(result, dtype, f1628IntegerFlag=True)

    return result


@util.check_input_type(dict, dict, dict, str)
def relu_grad_v2(gradients, mask, backprops, kernel_name="relu_grad_v2"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).
    support dtype:float16,float32,int32,int8,uint8

    Parameters
    ----------
    gradients: dict
        dict of grad
    mask: dict
        dict of relu output mask
    backprops: dict
        output of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad_v2"

    Returns
    -------
    None
    """
    shape_input_gradients = gradients.get("shape")
    shape_input_features = mask.get("shape")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input_gradients)
    util.check_shape_rule(shape_input_features)
    util.check_tensor_shape_size(shape_input_gradients)
    util.check_tensor_shape_size(shape_input_features)

    dtype_input_gradients = gradients.get("dtype").lower()
    dtype_input_features = mask.get("dtype").lower()

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(dtype_input_gradients, check_list)
    util.check_dtype_rule(dtype_input_features, ("uint8"))

    shape_in = list(shape_input_gradients)

    # make sure the last dim of input feature is 8's multipules
    if shape_in[-1] % 8 != 0:
        shape_in[-1] = (shape_in[-1] + 7) // 8 * 8

    data_input_gradients = tvm.placeholder(tuple(shape_in),
                                           name="data_input_gradients",
                                           dtype=dtype_input_gradients)
    data_input_features = tvm.placeholder(shape_input_features,
                                          name="data_input_features",
                                          dtype=dtype_input_features)

    res = relu_grad_v2_compute(data_input_gradients, data_input_features,
                               backprops, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_gradients, data_input_features, res]}
    te.lang.cce.cce_build_code(sch, config)
