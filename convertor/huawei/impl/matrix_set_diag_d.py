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

MatrixSetDiagD: Returns a batched matrix tensor with new batched diagonal values
"""
from __future__ import absolute_import

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# define a scalar, value = -1
SCALAR_NEGATIVE_ONE = -1
# define a scalar, value = 2
SCALAR_TWO = 2


def _check_tensor_size(shape_x, shape_y):
    """
    Check whether matrix_set_diag_d is supported or not.

    Parameters
    ----------
    shape_x: list
        shape of the first tensor x
    shape_y: list
        shape of the second tensor y with the same type and shape with x

    Returns
    -------
    None
    """
    len_x = len(shape_x)
    len_y = len(shape_y)

    if (len_x < SCALAR_TWO) or (len_y < SCALAR_TWO):
        raise RuntimeError("Only the rank of input tensors >= 2 are supported!")
    if len_x == len_y:
        for i in range(len_x):
            if shape_x[i] != shape_y[i]:
                raise RuntimeError(
                    "The input_x and input_y are not with the same dimension!")
    else:
        raise RuntimeError("The input_x and input_y are not with the same rank!")


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("matrix_set_diag_d")
def matrix_set_diag_d_compute(input_matrix, input_diagonal, input_help,
                              output_matrix, kernel_name="matrix_set_diag_d"):
    """
    how to make matrix_set_diag_d compute these tensors.
    -----------
    According to the auxiliary matrix and diagonal, res1 matrix is generated,
    then the matrix points with all diagonal zeros are multiplied by the input
    matrix, and finally the sum is added.

    Parameters
    ----------
    input_matrix: TVM tensor
        the placeholder of input_matrix
    input_diagonal: TVM tensor
        the placeholder of input_diagonal
    input_help: TVM tensor
        the placeholder of input_help
    output_matrix: dict
        dict of output
    kernel_name: str
        kernel name, default value is "matrix_set_diag_d"

    Returns
    -------
    res: TVM tensor
        the result of matrix_set_diag_d_compute
    """
    shape_input = te.lang.cce.util.shape_to_list(input_matrix.shape)
    input_dtype = input_matrix.dtype

    if input_dtype in ("int8", "uint8"):
        input_matrix = te.lang.cce.cast_to(input_matrix, "float16")
        input_diagonal = te.lang.cce.cast_to(input_diagonal, "float16")
        input_help = te.lang.cce.cast_to(input_help, "float16")

    diag_tmp = te.lang.cce.broadcast(input_diagonal, shape_input)
    help_tmp = te.lang.cce.vadds(input_help, SCALAR_NEGATIVE_ONE)
    help_y = te.lang.cce.vabs(help_tmp)

    res_vmul_x = te.lang.cce.vmul(input_matrix, help_y)
    res_vmul_y = te.lang.cce.vmul(diag_tmp, input_help)
    res = te.lang.cce.vadd(res_vmul_x, res_vmul_y)

    if input_dtype in ("int8", "uint8"):
        res = te.lang.cce.cast_to(res, input_dtype, f1628IntegerFlag=True)

    return res


# pylint: disable=locally-disabled,too-many-locals
@util.check_input_type(dict, dict, dict, dict, str)
def matrix_set_diag_d(input_matrix, input_diagonal, input_help, output_matrix,
                      kernel_name="matrix_set_diag_d"):
    """
    algorithm: matrix_set_diag_d

    Parameters
    ----------
    input_matrix: dict with keys(shape and dtype)
        dtype only support float16, float32, int32, int8，uint8.
    input_diagonal: dict with keys(shape and dtype)
        dtype only support float16, float32, int32, int8，uint8.
    input_help: dict with keys(shape and dtype)
        dtype only support float16, float32, int32, int8，uint8.
    output_matrix: dict
        dict of output
    kernel_name: str
        kernel name, default value is "matrix_set_diag_d"

    Returns
    -------
    None
    """
    shape_input = input_matrix.get("shape")
    shape_diag = input_diagonal.get("shape")
    dtype_diagonal = input_diagonal.get("dtype")
    dtype_help = input_help.get("dtype")
    help_matrix = input_help.get("shape")
    dtype = input_matrix.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input)
    util.check_shape_rule(shape_diag)
    util.check_shape_rule(help_matrix)

    util.check_tensor_shape_size(shape_input)
    util.check_tensor_shape_size(shape_diag)
    util.check_tensor_shape_size(help_matrix)

    # Check help_matrix can really help.
    _check_tensor_size(shape_input, help_matrix)

    # Adjust diag's shape according to input shape.
    # Extend the shape_diag dimension for broadcast.
    if shape_input[-2] <= shape_input[-1]:
        shape_b_newshape = list(shape_diag) + [1]
    # The penultimate dimension of the shape_diag is extended for broadcast.
    else:
        shape_b_newshape = list(shape_diag)
        shape_b_newshape.insert(-1, 1)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    input_dtype = dtype.lower()
    input_dtype_diagonal = dtype_diagonal.lower()
    input_dtype_help = dtype_help.lower()
    util.check_dtype_rule(input_dtype, check_list)
    util.check_dtype_rule(input_dtype_diagonal, check_list)
    util.check_dtype_rule(input_dtype_help, check_list)

    data_a = tvm.placeholder(shape_input, name="data_a", dtype=input_dtype)
    data_b = tvm.placeholder(shape_b_newshape, name="data_b", dtype=input_dtype)
    help_x = tvm.placeholder(help_matrix, name="help_x", dtype=input_dtype)

    res = matrix_set_diag_d_compute(data_a, data_b, help_x,
                                    output_matrix, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_a, data_b, help_x, res]}
    te.lang.cce.cce_build_code(sch, config)
