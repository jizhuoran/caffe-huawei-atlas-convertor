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

threshold_v2_d
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# shape limit
SHAPE_SIZE_LIMIT = 2147483648

@fusion_manager.register("threshold_v2_d")
def threshold_v2_d_compute(x, y, threshold, value=0, kernel_name="threshold_v2_d_cce"):
    """
    Thresholds each element of the input Tensor
    y = (x > threshold) ? x : value

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    threshold : float
        scale value to threshold at
    value : float
        scale value to replace with, default vaule is 0
    kernel_name : str
        kernel name, default value is "threshold_v2_d_cce"

    Returns
    -------
    output tensor
    """
    dtype_x = x.dtype

    threshold = tvm.const(threshold, dtype_x)
    value = tvm.const(value, dtype_x)

    data_res = te.lang.cce.vcmpsel(x, threshold, operation='gt', slhs=x, srhs=value)
    return data_res


@util.check_input_type(dict, dict, float, float, str)
def threshold_v2_d(x, y, threshold, value=0, kernel_name="threshold_v2_d_cce"):
    """
    Thresholds each element of the input Tensor
    y = (x > threshold) ? x : value

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    threshold : float
        scale value to threshold at
    value : float
        scale value to replace with, default vaule is 0
    kernel_name : str
        kernel name, default value is "threshold_v2_d_cce"

    Returns
    -------
    output tensor
    """
    shape_x = util.scalar2tensor_one(x.get("shape"))

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)

    check_list = ("float16", "float32", "int8", "uint8", "int32")

    dtype_x = x.get("dtype").lower()

    util.check_dtype_rule(dtype_x, check_list)

    data_x = tvm.placeholder(shape=shape_x, name="data_x", dtype=dtype_x)
    res = threshold_v2_d_compute(data_x, y, threshold, value, kernel_name)
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_x, res]}
    te.lang.cce.cce_build_code(schedule, config)
