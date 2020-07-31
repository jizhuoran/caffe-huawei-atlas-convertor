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
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("relu_grad")
def relu_grad_compute(input_gradients, input_features, output_backprops,
                      kernel_name="relu_grad"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).

    Parameters
    ----------
    input_gradients: TVM tensor
        input tensor of grad
    input_features: TVM tensor
        input tensor of relu output
    output_backprops: dict
        output dict of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad"

    Returns
    -------
    res: TVM tensor
        the result of relu_grad_compute
    """
    dtype = input_gradients.dtype
    trans_type = dtype
    shape_input_gradients = te.lang.cce.util.shape_to_list(input_gradients.shape)
    shape_input_features = te.lang.cce.util.shape_to_list(input_features.shape)
    shape = shape_input_gradients

    # need cast int8 or uint8 to float16
    if dtype in ("int8", "uint8"):
        input_gradients = te.lang.cce.cast_to(input_gradients, "float16")
        input_features = te.lang.cce.cast_to(input_features, "float16")
        trans_type = "float16"

    # broadcast in case the input shapes are not same
    if list(shape_input_gradients) != list(shape_input_features):
        shape_input_gradients, shape_input_features, shape = \
            util.produce_shapes(shape_input_gradients, shape_input_features)
        input_gradients = te.lang.cce.broadcast(input_gradients, shape,
                                                trans_type)
        input_features = te.lang.cce.broadcast(input_features, shape,
                                               trans_type)

    derivative_relu = te.lang.cce.calculate_one_or_zero(input_features,
                                                        shape, trans_type)

    result = te.lang.cce.vmul(input_gradients, derivative_relu)

    # cast int8 or uint8 back
    if dtype in ("int8", "uint8"):
        result = te.lang.cce.cast_to(result, dtype, f1628IntegerFlag=True)

    return result


@util.check_input_type(dict, dict, dict, str)
def relu_grad(input_gradients, input_features, output_backprops,
              kernel_name="relu_grad"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).
    support dtype:float16,float32,int32,int8,uint8

    Parameters
    ----------
    input_gradients: dict
        the backpropagated gradients to the corresponding relu operation
    input_features: dict
        the features passed as output of relu operation
    output_backprops: dict
        the output of relu back propagation
    kernel_name: str
        cce kernel name, default value is "relu_grad"

    Returns
    -------
    None
    """
    shape_input_gradients = input_gradients.get("shape")
    shape_input_features = input_features.get("shape")

    util.compare_tensor_dict_key(input_gradients, input_features, "dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input_gradients)
    util.check_shape_rule(shape_input_features)
    util.check_tensor_shape_size(shape_input_gradients)
    util.check_tensor_shape_size(shape_input_features)

    if list(shape_input_gradients) != list(shape_input_features):
        shape_input_gradients, shape_input_features, shape_max = \
            util.produce_shapes(shape_input_gradients, shape_input_features)
        util.check_tensor_shape_size(shape_max)

    dtype_input_gradients = input_gradients.get("dtype").lower()
    dtype_input_features = input_features.get("dtype").lower()

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(dtype_input_gradients, check_list)
    util.check_dtype_rule(dtype_input_features, check_list)

    shape_input_gradients, shape_input_features = \
        refine_shapes_for_broadcast(shape_input_gradients,
                                    shape_input_features)
    data_input_gradients = tvm.placeholder(shape_input_gradients,
                                           name="data_input_gradients",
                                           dtype=dtype_input_gradients)
    data_input_features = tvm.placeholder(shape_input_features,
                                          name="data_input_features",
                                          dtype=dtype_input_features)

    res = relu_grad_compute(data_input_gradients, data_input_features,
                            output_backprops, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_gradients, data_input_features, res]}
    te.lang.cce.cce_build_code(sch, config)
