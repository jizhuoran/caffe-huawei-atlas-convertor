#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance
with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

batch_matmul
"""
from __future__ import absolute_import

import functools

import te.lang.cce
import te.platform.cce_params as cce
from te import tvm

from topi.cce import util
from topi import generic

from impl.batch_matmul_vector import matmul_vector_cce
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a, trans_b):
    """
    Check the given shape for matrix A, B and bias == legal

    Parameters
    ---------
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    src_dtype: str
            The data type of input, support "float16"
    trans_a: bool
            If True, shape_a is transposed before multiplication
    trans_b: bool
            If True, shape_b is transposed before multiplication

    Returns
    -------
    None
    """
    shape_len = len(shape_a)
    inp_src_dtype = src_dtype.lower()
    k_block_size = cce.BLOCK_REDUCE
    check_list = ("float16")

    if inp_src_dtype not in check_list:
        raise RuntimeError("Dtype of input only support float16")

    if shape_len != len(shape_b):
        raise RuntimeError("length of a and b are not equal")

    if shape_len < 2:
        raise RuntimeError("shape length for batch matmul must large than 2")

    if shape_len == 2:
        raise RuntimeError(
            "batch matmul not support shape length 2, if shape length equal 2, use matmul!")

    if shape_a[:shape_len - 2] != shape_b[:shape_len - 2]:
        raise RuntimeError("batch size of a and b are not equal")

    is_gevm = bool((shape_a[-2] == 1) or (shape_a[-1] == 1))
    is_gemv = bool((shape_b[-2] == 1) or (shape_b[-1] == 1))

    if trans_a:
        m_shape = shape_a[shape_len - 1]
        km_shape = shape_a[shape_len - 2]
    else:
        m_shape = shape_a[shape_len - 2]
        km_shape = shape_a[shape_len - 1]

    if trans_b:
        kn_shape = shape_b[shape_len - 1]
        n_shape = shape_b[shape_len - 2]
    else:
        kn_shape = shape_b[shape_len - 2]
        n_shape = shape_b[shape_len - 1]

    if m_shape == 1:
        if n_shape == 1:
            raise RuntimeError("input shape M and N can't both be 1")

    if km_shape != kn_shape:
        raise RuntimeError("reduce axis not same")

    if m_shape % cce.BLOCK_IN != 0 and m_shape != 1:
        raise RuntimeError(
            "input shape M should be 1 or multiple of %d" % cce.BLOCK_IN)

    if m_shape != 1:
        if km_shape % k_block_size != 0:
            raise RuntimeError(
                "input shape K1 should be multiple of %d" % cce.BLOCK_IN)

    if n_shape % cce.BLOCK_IN != 0 and n_shape != 1:
        raise RuntimeError(
            "input shape N should be 1 or multiple of %d" % cce.BLOCK_IN)

    shape_bias_length = len(shape_bias)

    if shape_bias_length > 0:
        if shape_bias_length == 1:
            if is_gevm or is_gemv:
                if shape_bias[0] != m_shape * n_shape:
                    raise RuntimeError("broadcast case shape bias for gemv must be equal m*n")
            else:
                if shape_bias[0] != n_shape:
                    raise RuntimeError("broadcast bias shape must be equal to shape n")
        elif shape_bias_length == shape_len:
            out_shape = [i for i in shape_a[:-2]] + [m_shape, n_shape]
            if [i for i in shape_bias] != out_shape:
                raise RuntimeError("non broadcast bias shape must be same as output shape")
        else:
            raise RuntimeError("unsupport input shape now for batch bias case")


def _get_bias(shape_bias):
    bias_length = shape_bias[0]
    if bias_length % 16 != 0:
        bias_length = (bias_length // 16) * 16 + 16
        shape_bias = []
        shape_bias.append(bias_length)

    return shape_bias


def _get_input_shape(shape_x):
    shape_length = len(shape_x)
    dim_a = shape_x[shape_length - 2]
    dim_b = shape_x[shape_length - 1]
    shape_length = shape_length - 2
    res = shape_x[:shape_length]
    if dim_a % 16 != 0:
        dim_a = (dim_a // 16) * 16 + 16
        res.append(dim_a)
    else:
        res.append(dim_a)

    if dim_b % 16 != 0:
        dim_b = (dim_b // 16) * 16 + 16
        res.append(dim_b)
    else:
        res.append(dim_b)
    return res


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=dangerous-default-value, no-member
# pylint: disable=too-many-statements, unused-argument
def op_select_format(input_x, input_y, bias=None, output_z={}, trans_a=False,
                     trans_b=False, kernel_name="matmul"):
    """
    provide dynamic format to FE
    """
    src_dtype = input_x.get("dtype")

    if src_dtype == "float16":
        input0 = gen_param(classify="input0", name="x1",
                           datatype="float16",
                           format="FRACTAL_NZ")
        input1 = gen_param(classify="input1", name="x2",
                           datatype="float16",
                           format="FRACTAL_NZ")
        input2 = gen_param(classify="input2", name="bias",
                           datatype="float16",
                           format="ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16",
                            format="FRACTAL_NZ")
    else:
        input0 = gen_param(classify="input0", name="x1",
                           datatype="float16,float,float,int32,int32",
                           format="FRACTAL_NZ,NHWC,ND,NHWC,ND")
        input1 = gen_param(classify="input1", name="x2",
                           datatype="float16,float,float,int32,int32",
                           format="FRACTAL_NZ,NHWC,ND,NHWC,ND")
        input2 = gen_param(classify="input2", name="bias",
                           datatype="float16,float,float,int32,int32",
                           format="ND,NHWC,ND,NHWC,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float,float,int32,int32",
                            format="FRACTAL_NZ,NHWC,ND,NHWC,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-arguments, no-member
# pylint: disable=too-many-statements, unused-argument
def check_supported(input_x, input_y, bias=None, output_z={}, trans_a=False,
                    trans_b=False, kernel_name="matmul"):
    """
    get the op supported situation
    """
    shape_a = input_x.get("shape")
    shape_b = input_y.get("shape")
    src_dtype = input_x.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)
    util.check_shape_size(shape_a, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_b, SHAPE_SIZE_LIMIT)
    src_dtypes = ["float32", "int32"]
    res = True
    if src_dtype in src_dtypes:
        shape_length = len(shape_a)
        shape_length_b = len(shape_b)
        if shape_length != shape_length_b:
            res = False
        elif trans_b:
            if shape_b[shape_length - 2] == 1:
                res = False
        elif bool(1-trans_b):
            if shape_b[shape_length - 1] == 1:
                res = False
        elif trans_a:
            if trans_b:
                if shape_a[shape_length - 2] != shape_b[shape_length - 1]:
                    res = False
            else:
                if shape_a[shape_length - 2] != shape_b[shape_length - 2]:
                    res = False
        else:
            if trans_b:
                if shape_a[shape_length - 1] != shape_b[shape_length - 1]:
                    res = False
            else:
                if shape_a[shape_length - 1] != shape_b[shape_length - 2]:
                    res = False
    elif src_dtype == "float16":
        shape_length = len(shape_a)
        if trans_a:
            k_shape = shape_a[shape_length - 2]
        else:
            k_shape = shape_a[shape_length - 1]

        if trans_b:
            k_b_shape = shape_b[shape_length - 1]
        else:
            k_b_shape = shape_b[shape_length - 2]

        if k_shape != k_b_shape:
            res = False

    return res


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-locals, no-member
# pylint: disable=too-many-statements, dangerous-default-value
@util.check_input_type(dict, dict, (dict, NoneType), dict, bool, bool, str)
def batch_matmul(input_x, input_y, bias=None, output_z={}, trans_a=False,
                 trans_b=False, kernel_name="matmul"):
    """
    algorithm: batch_matmul
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters
    ---------
    input_x1: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    input_x2: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Return
    ------
    None
    """
    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    shape_a_length = len(shape_a)
    shape_b_length = len(shape_b)
    if shape_a is not None:
        if shape_a_length < 2:
            shape_a = input_x.get("shape")

    if shape_b is not None:
        if shape_b_length < 2:
            shape_b = input_y.get("shape")
    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)
        if input_x.get("format") == "FRACTAL_NZ":
            shape_bias = _get_bias(shape_bias)

    src_dtype = input_x.get("dtype").lower()
    dst_dtype = output_z.get("dtype").lower()
    is_fractal = False

    shape_a = list(shape_a)
    shape_b = list(shape_b)
    if input_x.get("format") == "FRACTAL_NZ":
        shape_a = _get_input_shape(shape_a)
        shape_b = _get_input_shape(shape_b)

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)
    util.check_shape_size(shape_a, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_b, SHAPE_SIZE_LIMIT)

    if input_x.get("format") == "FRACTAL_NZ":
        batch_axis = shape_a[:(len(shape_a) - 2)]
        shape_a = batch_axis + [shape_a[len(shape_a) - 1], shape_a[len(shape_a) - 2]]
        trans_a = bool(1 - trans_a)

    if input_y.get("format") == "FRACTAL_NZ":
        batch_axis = shape_a[:(len(shape_a) - 2)]
        shape_b = batch_axis + [shape_b[len(shape_b) - 1], shape_b[len(shape_b) - 2]]
        trans_b = bool(1 - trans_b)

    if src_dtype.lower() == "float32" or src_dtype.lower() == "int32":
        matmul_vector_cce(shape_a, shape_b, src_dtype, trans_a, trans_b, shape_bias, kernel_name)
        return

    _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a, trans_b)
    inp_src_dtype = src_dtype.lower()

    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_a) - 2]
    n_shape = shape_b[len(shape_a) - 1]

    if inp_src_dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE

    block_in = cce.BLOCK_IN
    block_out = cce.BLOCK_OUT

    if trans_a and km_shape == 1:
        block_in = cce.BLOCK_VECTOR

    if not trans_a and m_shape == 1:
        block_in = cce.BLOCK_VECTOR

    if trans_b and kn_shape == 1:
        block_out = cce.BLOCK_VECTOR

    if not trans_b and n_shape == 1:
        block_out = cce.BLOCK_VECTOR

    if trans_a:
        shape_a_dup = (m_shape // block_reduce, km_shape // block_in, block_reduce, block_in)
    else:
        shape_a_dup = (m_shape // block_in, km_shape // block_reduce, block_in, block_reduce)

    if trans_b:
        shape_b_dup = (kn_shape // block_out, n_shape // block_reduce, block_reduce, block_out)
    else:
        shape_b_dup = (kn_shape // block_reduce, n_shape // block_out, block_out, block_reduce)

    if input_x.get("format") == "FORMAT_FRACTAL_Z":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "fractal"
    elif input_x.get("format") == "FRACTAL_NZ":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "FRACTAL_NZ"
    else:
        shape_a_dup = (shape_a[len(shape_a) - 2], shape_a[len(shape_a) - 1])
        format_a = "ND"

    if input_y.get("format") == "FORMAT_FRACTAL_Z":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "fractal"
    elif input_y.get("format") == "FRACTAL_NZ":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "FRACTAL_NZ"
    else:
        shape_b_dup = (shape_b[len(shape_b) - 2], shape_b[len(shape_b) - 1])
        format_b = "ND"

    batch_shape_a = functools.reduce(lambda x, y: x * y, shape_a[:-2])
    batch_shape = batch_shape_a

    if batch_shape >= 1:
        if is_fractal:
            shape_a_dup = (batch_shape,) + shape_a_dup
            shape_b_dup = (batch_shape,) + shape_b_dup
        else:
            shape_a_dup = (batch_shape,) + shape_a_dup
            shape_b_dup = (batch_shape,) + shape_b_dup

    tensor_bias = None
    shape_bias_length = len(shape_bias)
    if shape_bias_length <= 2:
        shape_bias_dup = shape_bias
    else:
        shape_bias_dup = (shape_bias[len(shape_bias) - 2], shape_bias[len(shape_bias) - 1])
        bias_batch_size = functools.reduce(lambda x, y: x * y, shape_bias[:-2])
        shape_bias_dup = (bias_batch_size,) + shape_bias_dup

    tensor_a = tvm.placeholder(shape_a_dup, name='tensor_a',
                               dtype=inp_src_dtype)
    tensor_b = tvm.placeholder(shape_b_dup, name='tensor_b',
                               dtype=inp_src_dtype)

    if shape_bias_length > 0:
        tensor_bias = tvm.placeholder(shape_bias_dup, name='tensor_bias',
                                      dtype=dst_dtype)

    with tvm.target.cce():
        result = te.lang.cce.matmul(tensor_a, tensor_b, trans_a, trans_b,
                                    format_a=format_a, format_b=format_b, dst_dtype=dst_dtype,
                                    tensor_bias=tensor_bias)
        schedule = generic.auto_schedule(result)
    tensor_list = [tensor_a, tensor_b, result]

    if shape_bias_length > 0:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
