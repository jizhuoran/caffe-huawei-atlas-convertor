#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cce extended operator builder wrapper
"""

from functools import reduce as reduceIns

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("leaky_relu")
def leaky_relu_compute(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """
    compute for caffe_relu_layer_cce
    """
    inp_dtype = x.dtype.lower()
    shape = x.shape

    # The original relu logic remains unchanged.
    if negative_slope == 0:
        if inp_dtype in ("float32", "int32"):
            tensor_zero = te.lang.cce.broadcast(tvm.const(0, inp_dtype), shape)
            data_res = te.lang.cce.vmax(x, tensor_zero)
        else:
            data_res = te.lang.cce.vrelu(x)

        data_res = te.lang.cce.cast_to(data_res, inp_dtype)

        return data_res

    # negative_slope != 0
    if inp_dtype in ("float16", "float32"):
        slope_tmp = tvm.const(negative_slope, dtype=inp_dtype)
        tmp = te.lang.cce.vmuls(x, slope_tmp)
        if negative_slope <= 1:
            res = te.lang.cce.vmax(x, tmp)
        else:
            res = te.lang.cce.vmin(x, tmp)
    else:
        # inp_dtype in ("int32", "int8")
        slope_tmp = tvm.const(negative_slope, dtype=inp_dtype)
        tmp = te.lang.cce.vmuls(x, slope_tmp)
        tmp_oritype = te.lang.cce.cast_to(tmp, inp_dtype)
        if negative_slope <= 1:
            res = te.lang.cce.vmax(x, tmp_oritype)
        else:
            res = te.lang.cce.vmin(x, tmp_oritype)

        res = te.lang.cce.cast_to(res, inp_dtype)

    return res


@util.check_input_type(dict, dict, (int, float), str)
def leaky_relu(x, y, negative_slope=0, kernel_name="leaky_relu"):
    """leaky_relu op for input tensor

       f(x)= x(x>=0) or negative_slope*x(x<0) equal to
       f(x)=negative_slope*x

    Parameters
    ----------
    x : TVM tensor
        input tensor has shape and dtype attributes
    y : dict
        dict with keys(shape and dtype) of output

    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization

    kernel_name : str
        cce kernel name, default value is "leaky_relu"

    Returns
    ------
    None
    """

    # check input tensor shape
    shape = x.get("shape")
    dtype = x.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)

    # check input tensor data_type
    check_list = ["float16", "float32", "int32", "int8"]
    if dtype.lower() not in check_list:
        raise RuntimeError(
            "leaky relu only support %s while dtype is %s"
            % (",".join(check_list), dtype))
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    inp_dtype = dtype.lower()
    input_data_x = tvm.placeholder(fuseshape, name="input_data_x", dtype=inp_dtype)

    with tvm.target.cce():

        res = leaky_relu_compute(input_data_x, y, negative_slope, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
