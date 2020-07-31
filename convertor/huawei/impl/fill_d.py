#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fill_d

  Op_description :
    This operation creates a tensor of shape `dims` and fills it with `value`.

    # fill_d(
    #   value,
    #   y,
    #   dims,
    #   kernel_name='fill_d')

  Supportive_dtype_format :
    ['int32', 'int8', 'uint8', 'float32', 'float16']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : shape size limit is 2147483648.
"""

from functools import reduce as functools_reduce
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=consider-using-in,invalid-name,redefined-builtin
@fusion_manager.register("fill_d")
def _fill_compute(data_value, data_output, data_dims, kernel_name="fill_d"):
    """
    Process fill operator

    Parameters
    ----------
    data_value: the placeholder of data input

    data_output : the dict of output

    data_dims: the shape of input

    kernel_name : cce kernel name

    Returns
    -------
    res : result of fill
    """

    in_dtype = data_value.dtype
    shape = data_dims
    # te.lang.cce.broadcast supports float16, float32, int32.
    # so convert int8, uint8 to float16
    if in_dtype == "int8" or in_dtype == "uint8":
        data_value = te.lang.cce.cast_to(data_value, "float16")

    if functools_reduce(lambda x, y: x*y, data_dims) == 1:
        if in_dtype == "int32":
            tensor_zero = te.lang.cce.broadcast(tvm.const(0, "int32"), shape)
            res = te.lang.cce.vadd(data_value, tensor_zero)
        elif in_dtype == "float32":
            tensor_zero = te.lang.cce.broadcast(tvm.const(0, "float32"), shape)
            res = te.lang.cce.vadd(data_value, tensor_zero)
        else:
            tensor_zero = te.lang.cce.broadcast(tvm.const(0, "float16"), shape)
            res = te.lang.cce.vadd(data_value, tensor_zero)
    else:
        res = te.lang.cce.broadcast(data_value, shape, in_dtype)
    if in_dtype == "int8" or in_dtype == "uint8":
        res = te.lang.cce.cast_to(res, in_dtype)
    return res


def _check_shape_compatibility(shape_in, shape_out):
    """
    Check if the shape of input tensor is compatible with output tensor.

    Parameters:
    ----------
    shape_in : shape of input tensor.

    shape_out : shape of output tensor.

    Returns:
    -------
    comp_shape_in : new shape_in compatible with shape_out.
    """

    try:
        comp_shape_in, comp_shape_out, shape_max = broadcast_shapes(
            shape_in, shape_out)
        if comp_shape_out != shape_max:
            raise ValueError('shape_in is not compatible with shape_out.')
    except RuntimeError:
        raise ValueError('shape_in is not compatible with shape_out.')
    return comp_shape_in


@util.check_input_type(dict, dict, (tuple, list), str)
def fill_d(value, y, dims, kernel_name="fill_d"):
    """
    do  fill operation

    Parameters:
    ----------
    value:   the dict of input value, include shape and dtype,
             dtype support int8, uint8, int32, float16, float32

    y :  the dict of output

    dims :  the output shape, type support int32

    kernel_name : cce kernel name, default value is "fill_d"

    Returns
    -------
    None
    """
    # get the shape and dtype
    shape_value = value.get("shape")
    dtype_value = value.get("dtype")

    # kernel name check: should be unique
    util.check_kernel_name(kernel_name)

    # check whether the shape is right
    check_shape(dims)
    check_shape(shape_value)

    # check whether dtypes are right
    check_list_value = ("int8", "uint8", "int32", "float16", "float32")
    check_dtype(dtype_value, check_list_value)

    # get 2 input tensors: data_dims, data_value
    compatible_shape_in = _check_shape_compatibility(shape_value, dims)

    dtype_value = dtype_value.lower()
    data_value = tvm.placeholder(compatible_shape_in,
                                 dtype=dtype_value, name="data_value")
    res = _fill_compute(data_value, y, dims, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_value, res),
              "print_ir": False}
    te.lang.cce.cce_build_code(sch, config)
