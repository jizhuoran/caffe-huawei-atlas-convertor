#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

lamb_update_with_lr
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform

MIN_FP32 = 2 ** (-126)
# min float16 value
MIN_FP16 = 2 ** (-24)
VALUE_ONE = 1
# shape limit, 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=locally-disabled,invalid-name,unused-variable
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals
def real_div_compute(data_1, data_2, output_z, kernel_name="real_div"):
    """
    calculating data's realdiv, c = a / b

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is real_div

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = te.lang.cce.util.shape_to_list(data_1.shape)
    shape_y = te.lang.cce.util.shape_to_list(data_2.shape)
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x = te.lang.cce.broadcast(data_1, shape_max)
    data_y = te.lang.cce.broadcast(data_2, shape_max)
    res = te.lang.cce.vdiv(data_x, data_y)

    return res


def select_compute(condition, x1, x2, y, kernel_name="select"):
    """
    compute for select

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of input x1
    x2: TVM tensor
        the placeholder of input x2
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape = te.lang.cce.util.shape_to_list(x1.shape)
    con_shape = te.lang.cce.util.shape_to_list(condition.shape)
    num_dtype = x1.dtype
    bool_dtype = condition.dtype

    if num_dtype in ("int8", "uint8"):
        x1_dtype = "float32"
        ones = te.lang.cce.broadcast(tvm.const(VALUE_ONE, dtype="float32"),
                                     shape, output_dtype="float32")
        x1 = te.lang.cce.cast_to(x1, "float32")
        x2 = te.lang.cce.cast_to(x2, "float32")
    else:
        x1_dtype = num_dtype
        ones = te.lang.cce.broadcast(tvm.const(VALUE_ONE, dtype=num_dtype),
                                     shape, output_dtype=num_dtype)

    if bool_dtype == "int8":
        if x1_dtype == "int32":
            condition_dtype = te.lang.cce.ceil(condition)
        else:
            condition_dtype = te.lang.cce.cast_to(condition, x1_dtype)
    else:
        if x1_dtype == "int32":
            condition_dtype = condition
        else:
            condition_dtype = te.lang.cce.cast_to(condition, x1_dtype)

    if list(con_shape) != list(shape):
        condition_dtype = te.lang.cce.broadcast(condition_dtype, shape)

    condition_opp = te.lang.cce.vsub(ones, condition_dtype)

    temp_x = te.lang.cce.vmul(x1, condition_dtype)
    temp_y = te.lang.cce.vmul(x2, condition_opp)
    res = te.lang.cce.vadd(temp_x, temp_y)
    if num_dtype in ("int8", "uint8"):
        res = te.lang.cce.cast_to(res, num_dtype)
    return res


def _greater_compare(data, shape, dtype, data_min):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    data : tuple
        two input data
    shape : list or tuple
        shape of input data
    dtype : str
        source data type, support float16,float32,int32,int8,uint8
    data_min : tvm.const
        the minimal data according to dtype

    Returns
    -------
    the compare result
    """
    data_zero = te.lang.cce.broadcast(tvm.const(0, dtype), shape, dtype)
    if dtype == "int32":
        data_one = \
            te.lang.cce.broadcast(tvm.const(1, "float16"), shape, "float16")
    else:
        data_one = te.lang.cce.broadcast(tvm.const(1, dtype), shape, dtype)

    res_sub = te.lang.cce.vsub(data[1], data[0])
    # to amend sub zero result
    res_sub_zero = te.lang.cce.vadd(res_sub, data_min)
    res_min = te.lang.cce.vmin(res_sub_zero, data_min)
    res_max = te.lang.cce.vmax(res_min, data_zero)

    if dtype == "float32":
        # max num of float32 is 2**126
        # but cce can only support 2**62, so use 62/62/2 to adaptor 126
        max_support_fp32 = tvm.const(2 ** 62, dtype=dtype)
        res_mul1 = te.lang.cce.vmuls(res_max, max_support_fp32)
        res_mul2 = te.lang.cce.vmuls(res_mul1, max_support_fp32)
        res_mul = te.lang.cce.vmuls(res_mul2, tvm.const(2 ** 2, dtype=dtype))
    elif dtype == "float16":
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        max_support_fp16 = tvm.const(2 ** 12, dtype=dtype)
        res_mul1 = te.lang.cce.vmuls(res_max, max_support_fp16)
        res_mul = te.lang.cce.vmuls(res_mul1, max_support_fp16)
    else:
        res_mul = te.lang.cce.cast_to(res_max, "float16")
    res = te.lang.cce.vsub(data_one, res_mul)

    return te.lang.cce.cast_to(res, "uint8", True)


def greater_compute(x, y, z, kernel_name="greater"):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    x : Tensor
        input data_x
    y : Tensor
        input data_y
    z : dict
        shape and dtype of output data_z
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    the result
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    shape_y = te.lang.cce.util.shape_to_list(y.shape)
    dtype = x.dtype.lower()
    shape_x, shape_y, shape = util.produce_shapes(shape_x, shape_y)

    if dtype in ("int8", "uint8"):
        x = te.lang.cce.cast_to(x, "float16")
        y = te.lang.cce.cast_to(y, "float16")
        dtype = "float16"

    data_x = te.lang.cce.broadcast(x, shape)
    data_y = te.lang.cce.broadcast(y, shape)

    if dtype == "float32":
        # minimun num of float32 2**(-126)
        data_min = \
            te.lang.cce.broadcast(tvm.const(MIN_FP32, dtype=dtype), shape,
                                  dtype)
    elif dtype == "float16":
        # minimun num of float16 2**(-24)
        data_min = \
            te.lang.cce.broadcast(tvm.const(MIN_FP16, dtype=dtype), shape,
                                  dtype)
    else:
        data_min = te.lang.cce.broadcast(tvm.const(1, dtype=dtype), shape,
                                         dtype)

    return _greater_compare((data_x, data_y), shape, dtype, data_min)


def reduce_sum_d_compute(x,
                         y,
                         axis=None,
                         keepdims=None,
                         kernel_name="reduce_sum_d"):
    """redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    dtype = x.dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        x = te.lang.cce.cast_to(x, "float32")
    res_sum = te.lang.cce.sum(x, axis=axis, keepdims=keepdims)
    res = te.lang.cce.cast_to(res_sum, dtype)

    return res


def maximum(input_x, input_y, output_z, kernel_name="maximum"):
    """
    do element-wise maximum operation between two input tensors

    """
    shape1 = te.lang.cce.util.shape_to_list(input_x.shape)
    shape2 = te.lang.cce.util.shape_to_list(input_y.shape)
    shape1 = util.scalar2tensor_one(shape1)

    shape2 = util.scalar2tensor_one(shape2)

    shape1, shape2, shape_max = util.produce_shapes(shape1, shape2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    data1_tmp1 = te.lang.cce.broadcast(input_x, shape_max)
    data2_tmp1 = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vmax(data1_tmp1, data2_tmp1)
    return res


def minimum(input_x, input_y, output_z, kernel_name="minimum"):
    """
    do element-wise minimum operation between two input tensors
    """
    shape1 = te.lang.cce.util.shape_to_list(input_x.shape)
    shape2 = te.lang.cce.util.shape_to_list(input_y.shape)
    shape1 = util.scalar2tensor_one(shape1)
    shape2 = util.scalar2tensor_one(shape2)

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape1)
    util.check_shape_rule(shape2)
    util.check_shape_size(shape1, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape2, SHAPE_SIZE_LIMIT)

    check_list = ["float16", "float32", "int32"]
    dtype = input_x.dtype
    util.check_dtype_rule(dtype, check_list)

    shape1, shape2, shape_max = util.produce_shapes(shape1, shape2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    data1_tmp1 = te.lang.cce.broadcast(input_x, shape_max)
    data2_tmp1 = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vmin(data1_tmp1, data2_tmp1)
    return res


def sub(input_x, input_y, output_z, kernel_name="sub"):
    """
    do element-wise sub operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32,int32
    input_y : dict
        shape and dtype of input, only support float16, float32,int32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "sub"

    Returns
    -------
    None
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data1_tmp1 = te.lang.cce.broadcast(input_x, shape_max)
    data2_tmp1 = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vsub(data1_tmp1, data2_tmp1)
    return res


def vmul(input_x, input_y, output_z, kernel_name="vmul"):
    """
    do element-wise sub operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32,int32
    input_y : dict
        shape and dtype of input, only support float16, float32,int32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "sub"

    Returns
    -------
    None
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)

    data1_tmp1 = te.lang.cce.broadcast(input_x, shape_max)
    data2_tmp1 = te.lang.cce.broadcast(input_y, shape_max)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    res = te.lang.cce.vmul(data1_tmp1, data2_tmp1)
    return res


@fusion_manager.register("lamb_update_with_lr")
def lamb_update_with_lr_compute(data_input_greater1,
                                data_input_greater_realdiv,
                                data_input_realdiv, data_input_mul0,
                                data_input_mul1, data_input_sub,
                                data_greater_y, data_select_e,
                                data_minimum_y, y,
                                kernel_name="lamb_update_with_lr"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """

    greater0 = greater_compute(data_input_greater1,
                               data_greater_y, {}, kernel_name)
    greater1 = greater_compute(data_input_greater_realdiv,
                               data_greater_y, {}, kernel_name)
    realdiv0 = real_div_compute(data_input_greater_realdiv,
                                data_input_realdiv, {}, kernel_name)

    select0 = select_compute(greater0, realdiv0,
                             data_select_e, {}, kernel_name)
    select1 = select_compute(greater1, select0,
                             data_select_e, {}, kernel_name)

    minimum0 = minimum(select1, data_minimum_y, {}, kernel_name)

    maximum0 = maximum(minimum0, data_greater_y, {}, kernel_name)

    mul0 = vmul(maximum0, data_input_mul0, {}, kernel_name)
    mul1 = vmul(mul0, data_input_mul1, {}, kernel_name)
    res = sub(data_input_sub, mul1, {}, kernel_name)

    return res


def lamb_update_with_lr(input_greater1, input_greater_realdiv, input_realdiv,
                        input_mul0, input_mul1, input_sub,
                        greater_y, select_e, minimum_y, y,
                        kernel_name="lamb_update_with_lr"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    shape_input_greater1 = input_greater1.get("shape")
    shape_input_sub = input_sub.get("shape")
    input_dtype = input_sub.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_input_greater1)
    util.check_shape_rule(shape_input_sub)
    util.check_shape_size(shape_input_greater1, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_input_sub, SHAPE_SIZE_LIMIT)

    shape_input_greater1, shape_input_sub, shape_max = \
        util.produce_shapes(shape_input_greater1, shape_input_sub)

    data_input_greater1 = \
        tvm.placeholder(shape_input_greater1,
                        name="data_input_greater1",
                        dtype=input_dtype)

    data_input_greater_realdiv = \
        tvm.placeholder(shape_input_greater1,
                        name="data_input_greater_realdiv",
                        dtype=input_dtype)
    data_input_realdiv = tvm.placeholder(shape_input_greater1,
                                         name="data_input_realdiv",
                                         dtype=input_dtype)
    data_input_mul0 = tvm.placeholder(shape_input_greater1,
                                      name="data_input_mul0",
                                      dtype=input_dtype)
    data_input_mul1 = tvm.placeholder(shape_input_sub,
                                      name="data_input_mul1",
                                      dtype=input_dtype)
    data_input_sub = tvm.placeholder(shape_input_sub,
                                     name="data_input_sub",
                                     dtype=input_dtype)
    data_greater_y = tvm.placeholder(shape_input_greater1,
                                     name="data_greater_y",
                                     dtype=input_dtype)
    data_select_e = tvm.placeholder(shape_input_greater1,
                                    name="data_select_e",
                                    dtype=input_dtype)
    data_minimum_y = tvm.placeholder(shape_input_greater1,
                                     name="data_minimum_y",
                                     dtype=input_dtype)

    res = lamb_update_with_lr_compute(data_input_greater1,
                                      data_input_greater_realdiv,
                                      data_input_realdiv, data_input_mul0,
                                      data_input_mul1, data_input_sub,
                                      data_greater_y, data_select_e,
                                      data_minimum_y, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_greater1, data_input_greater_realdiv,
                              data_input_realdiv, data_input_mul0,
                              data_input_mul1, data_input_sub, data_greater_y,
                              data_select_e, data_minimum_y, res]}

    te.lang.cce.cce_build_code(sch, config)
