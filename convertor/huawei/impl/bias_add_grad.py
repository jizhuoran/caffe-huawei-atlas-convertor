#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

bias_add_grad
"""
import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("bias_add_grad")
def bias_add_grad_compute_nz(x, y, data_format,
                             kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x: TVM tensor
        the placeholder of y input data ,dataformat = "NZ"
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad

    Returns
    -------
    TVM tensor by bias add grad
    """
    dtype = x.dtype
    shape = te.lang.cce.util.shape_to_list(x.shape)
    shape_list = []
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        x = te.lang.cce.cast_to(x, "float32")

    if data_format == "NCHW":
        if len(shape) == 4:
            for i in range(-1 * len(shape), 0):
                if i not in (-1, -4):
                    shape_list += [i + len(shape)]

        elif len(shape) == 5:
            for i in range(-1 * len(shape), 0):
                if i not in (-2, -3):
                    shape_list += [i + len(shape)]

        else:
            shape_list.append(0)
            for i in range(2, len(shape)):
                shape_list = shape_list + [i]
        result = te.lang.cce.sum(x, shape_list)

    else:
        if len(shape) < 4:
            raise RuntimeError(
                "cce_bias_add_grad_nz_2_nhwc only support shape larger than 4D")
        for i in range(-1 * len(shape), 0):
            if i not in (-1, -4):
                shape_list += [i + len(shape)]
        result = te.lang.cce.sum(x, shape_list)

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        result = te.lang.cce.cast_to(result, "float16")

    return result


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("bias_add_grad")
def bias_add_grad_compute(x, y, data_format,
                          kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x: TVM tensor
        the placeholder of y input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad

    Returns
    -------
    TVM tensor by bias add grad
    """
    dtype = x.dtype
    shape = te.lang.cce.util.shape_to_list(x.shape)

    if dtype == "float16":
        x = te.lang.cce.cast_to(x, "float32")

    if data_format == "NCHW":
        shape_list = [0]
        for i in range(2, len(shape)):
            shape_list = shape_list + [i]
        result = te.lang.cce.sum(x, shape_list)
    else:
        if len(shape) < 2:
            raise RuntimeError(
                "cce_bias_add_grad only support shape larger than 2D")
        result = te.lang.cce.sum(x, [x for x in range(len(shape) - 1)])

    if dtype == "float16":
        result = te.lang.cce.cast_to(result, "float16")

    return result


@util.check_input_type(dict, dict, str, str)
def bias_add_grad(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x : dict
        shape and dtype of input, only support float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad
    Returns
    -------
    None
    """
    shape = x.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    dtype = x.get("dtype").lower()
    data_format = data_format.upper()
    check_list = ("float16", "float32")
    util.check_dtype_rule(dtype, check_list)
    data_format_list = ("NCHW", "NHWC")
    input_data_format = x.get("format").upper()

    if data_format not in data_format_list:
        raise RuntimeError(
            "The data_format only support NCHW, NHWC")

    data = tvm.placeholder(shape, dtype, name="data")

    if input_data_format == "FRACTAL_NZ":
        result = bias_add_grad_compute_nz(data, y, data_format, kernel_name)
    else:
        result = bias_add_grad_compute(data, y, data_format, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, result]}
    te.lang.cce.cce_build_code(sch, config)
