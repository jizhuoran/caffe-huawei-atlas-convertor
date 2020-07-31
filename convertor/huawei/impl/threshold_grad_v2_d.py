"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

threshold_grad_v2_d
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


@fusion_manager.register("threshold_grad_v2_d")
def threshold_grad_v2_d_compute(input_gradients, input_features,
                                output_backprops, threshold,
                                kernel_name="threshold_grad_v2_d"):
    """
    calculating data

    Parameters
    ----------
    input_gradients : TVM tensor
        input tensor of gradients
    input_features : TVM tensor
        input tensor of features
    output_backprops : dict
        dict of output_backprops, include keys(shape and dtype)
    threshold:
    kernel_name : str
        kernel name, default value is "threshold_grad_v2_d"

    Returns
    -------
    res: TVM tensor
        the result of threshold_grad_v2_d_compute
    """

    dtype = input_gradients.dtype
    result = te.lang.cce.vcmpsel(input_features, threshold, 'gt',
                                 input_gradients, tvm.const(0, dtype))

    return result


@util.check_input_type(dict, dict, dict, (int, float), str)
def threshold_grad_v2_d(input_gradients, input_features, output_backprops,
                        threshold=0, kernel_name="threshold_grad_v2_d"):
    """
    calculating data

    Parameters
    ----------
    input_gradients : dict
        shape and dtype of input_gradients
    input_features : dict
        shape and dtype of input_features
    output_backprops : dict
        shape and dtype of output_backprops,
        should be same shape and type as inputs
    threshold : dict
        shape and dtype of threshold, 0-dimensional array
    kernel_name : str
        kernel name, default value is "threshold_grad_v2_d"

    Returns
    -------
    None
    """
    shape_input_gradients = input_gradients.get("shape")
    dtype_input_gradients = input_gradients.get("dtype").lower()

    shape_input_features = input_features.get("shape")
    dtype_input_features = input_features.get("dtype").lower()

    util.check_shape_rule(shape_input_gradients)
    util.check_shape_rule(shape_input_features)
    util.check_tensor_shape_size(shape_input_gradients)
    util.check_tensor_shape_size(shape_input_features)
    util.check_kernel_name(kernel_name)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(dtype_input_gradients, check_list)
    util.check_dtype_rule(dtype_input_features, check_list)

    data_input_gradients = tvm.placeholder(shape_input_gradients,
                                           name="data_input_gradients",
                                           dtype=dtype_input_gradients)
    data_input_features = tvm.placeholder(shape_input_features,
                                          name="data_input_features",
                                          dtype=dtype_input_features)
    res = threshold_grad_v2_d_compute(data_input_gradients, data_input_features,
                                      output_backprops, threshold, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_gradients, data_input_features, res]}

    te.lang.cce.cce_build_code(schedule, config)
