#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

lamb_update_with_lr_v2
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# min float32 value
MIN_FP32 = 2**(-126)
# min float16 value
MIN_FP16 = 2**(-24)
# shape size limit
SHAPE_SIZE_LIMIT = 2**30


# pylint: disable=locally-disabled,unused-argument,invalid-name
# pylint: disable=locally-disabled,too-many-locals,too-many-arguments
def true_div_compute(x1, x2, kernel_name="true_div"):
    """
    calculating data's realdiv, c = a / b

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    kernel_name: str
        cce kernel name, default value is "true_div"

    Returns
    -------
    res : output of the data's divide
    """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = util.produce_shapes(shape_x1, shape_x2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vdiv(data_x1, data_x2)

    return res


def mul_compute(x1, x2, kernel_name="mul"):
    """
   calculating data's element-wise mul, c = a .* b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "mul"

   Returns
   -------
   res : output of the data's element-wise mul
   """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = util.produce_shapes(shape_x1, shape_x2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vmul(data_x1, data_x2)

    return res


def sub_compute(x1, x2, kernel_name="sub"):
    """
   calculating data's sub, c = a - b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "sub"

   Returns
   -------
   res : output of the data's sub
   """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = util.produce_shapes(shape_x1, shape_x2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vsub(data_x1, data_x2)

    return res


def _greater_compare(data_x, data_y, shape, dtype, data_min):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    data_x : TVM tensor
        first input after correction
    data_y: TVM tensor
        second input after correction
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
        data_one = te.lang.cce.broadcast(tvm.const(1, "float16"), shape,
                                         "float16")
    else:
        data_one = te.lang.cce.broadcast(tvm.const(1, dtype), shape, dtype)

    res_sub = te.lang.cce.vsub(data_y, data_x)
    # to amend sub zero result
    res_sub_zero = te.lang.cce.vadd(res_sub, data_min)
    res_min = te.lang.cce.vmin(res_sub_zero, data_min)
    res_max = te.lang.cce.vmax(res_min, data_zero)

    if dtype == "float32":
        # max num of float32 is 2**126
        # but cce can only support 2**62, so use 62/62/2 to adaptor 126
        max_support_fp32 = tvm.const(2**62, dtype=dtype)
        res_mul1 = te.lang.cce.vmuls(res_max, max_support_fp32)
        res_mul2 = te.lang.cce.vmuls(res_mul1, max_support_fp32)
        res_mul = te.lang.cce.vmuls(res_mul2, tvm.const(2**2, dtype=dtype))
    elif dtype == "float16":
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        max_support_fp16 = tvm.const(2**12, dtype=dtype)
        res_mul1 = te.lang.cce.vmuls(res_max, max_support_fp16)
        res_mul = te.lang.cce.vmuls(res_mul1, max_support_fp16)
    else:
        res_mul = te.lang.cce.cast_to(res_max, "float16")
    res = te.lang.cce.vsub(data_one, res_mul)

    return te.lang.cce.cast_to(res, "uint8", True)


def greater_compute(x, y, kernel_name="greater"):
    """
    if x is greater than y, then return 1, else return 0.

    Parameters:
    ----------
    x : TVM tensor
        the placeholder of first input data
    y : TVM tensor
        the placeholder of second input data
    kernel_name : str
        cce kernel name, default value is "greater"

    Returns
    -------
    res : output of the result of comparison
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    shape_y = te.lang.cce.util.shape_to_list(y.shape)
    dtype = x.dtype.lower()
    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)

    if dtype in ("int8", "uint8"):
        x = te.lang.cce.cast_to(x, "float16")
        y = te.lang.cce.cast_to(y, "float16")
        dtype = "float16"

    data_x = te.lang.cce.broadcast(x, shape_max)
    data_y = te.lang.cce.broadcast(y, shape_max)

    if dtype == "float32":
        # minimun num of float32 2**(-126)
        data_min = te.lang.cce.broadcast(tvm.const(MIN_FP32, dtype=dtype),
                                         shape_max, dtype)
    elif dtype == "float16":
        # minimun num of float16 2**(-24)
        data_min = te.lang.cce.broadcast(tvm.const(MIN_FP16, dtype=dtype),
                                         shape_max, dtype)
    else:
        data_min = te.lang.cce.broadcast(tvm.const(1, dtype=dtype), shape_max,
                                         dtype)

    return _greater_compare(data_x, data_y, shape_max, dtype, data_min)


def select_compute(condition, x1, x2, kernel_name="select"):
    """
    compute for select

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    res : output of the result of select compute
    """
    shape = te.lang.cce.util.shape_to_list(x1.shape)
    x1_dtype = x1.dtype
    con_shape = te.lang.cce.util.shape_to_list(condition.shape)
    bool_dtype = condition.dtype

    if x1_dtype in ("int8", "uint8"):
        x1_dtype = "float32"
        ones = te.lang.cce.broadcast(tvm.const(1, dtype=x1_dtype),
                                     shape,
                                     output_dtype=x1_dtype)
        x1 = te.lang.cce.cast_to(x1, "float32")
        x2 = te.lang.cce.cast_to(x2, "float32")
    else:
        ones = te.lang.cce.broadcast(tvm.const(1, dtype=x1_dtype),
                                     shape,
                                     output_dtype=x1_dtype)

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
    if x1_dtype in ("int8", "uint8"):
        res = te.lang.cce.cast_to(res, x1_dtype)

    return res


def _check_broadcast_shape(input0, input1, input2, input3, input4, greater_y,
                           select_e):
    """
    check broadcast shape

    Parameters
    ----------
    all input: dict
        the dict of shape for inputs

    Returns
    -------
    the list of inputs shape after broadcast
    """
    shape0 = input0.get("shape")
    util.check_shape_rule(shape0)
    util.check_shape_size(shape0, SHAPE_SIZE_LIMIT)

    shape1 = input1.get("shape")
    util.check_shape_rule(shape1)
    util.check_shape_size(shape1, SHAPE_SIZE_LIMIT)

    shape2 = input2.get("shape")
    util.check_shape_rule(shape2)
    util.check_shape_size(shape2, SHAPE_SIZE_LIMIT)

    shape3 = input3.get("shape")
    util.check_shape_rule(shape3)
    util.check_shape_size(shape3, SHAPE_SIZE_LIMIT)

    shape4 = input4.get("shape")
    util.check_shape_rule(shape4)
    util.check_shape_size(shape4, SHAPE_SIZE_LIMIT)

    shape_greatery = greater_y.get("shape")
    util.check_shape_rule(shape_greatery)
    util.check_shape_size(shape_greatery, SHAPE_SIZE_LIMIT)

    shape_selecte = select_e.get("shape")
    util.check_shape_rule(shape_selecte)
    util.check_shape_size(shape_selecte, SHAPE_SIZE_LIMIT)

    # broadcast input0,1,2 greater_y, select_e according to input3
    shape2, shape3, shape_max_23 = util.produce_shapes(shape2, shape3)
    shape0, shape3, shape_max_03 = util.produce_shapes(shape0, shape3)
    shape1, shape3, shape_max_13 = util.produce_shapes(shape1, shape3)
    shape_greatery, shape3, shape_max_3y = util.produce_shapes(
        shape_greatery, shape3)
    shape_selecte, shape3, shape_max_select3 = util.produce_shapes(
        shape_selecte, shape3)

    util.check_shape_size(shape_max_23, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_max_03, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_max_13, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_max_3y, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_max_select3, SHAPE_SIZE_LIMIT)

    # broadcast input0 greater_y
    shape0, shape_greatery, shape_max_0y = util.produce_shapes(
        shape0, shape_greatery)
    util.check_shape_size(shape_max_0y, SHAPE_SIZE_LIMIT)

    shape0, shape1, shape_max_01 = util.produce_shapes(shape0, shape1)
    util.check_shape_size(shape_max_01, SHAPE_SIZE_LIMIT)

    shape1, shape_greatery, shape_max_1y = util.produce_shapes(
        shape1, shape_greatery)
    util.check_shape_size(shape_max_1y, SHAPE_SIZE_LIMIT)

    shape_selecte, shape_max_1y, shape_max_select0 = util.produce_shapes(
        shape_selecte, shape_max_1y)
    util.check_shape_size(shape_max_select0, SHAPE_SIZE_LIMIT)

    shape_selecte, shape_max_0y, shape_max_select1 = util.produce_shapes(
        shape_selecte, shape_max_0y)
    util.check_shape_size(shape_max_select1, SHAPE_SIZE_LIMIT)

    shape2, shape_max_select1, shape_max_mul0 = util.produce_shapes(
        shape2, shape_max_select1)
    util.check_shape_size(shape_max_mul0, SHAPE_SIZE_LIMIT)

    shape3, shape_max_mul0, shape_max_mul1 = util.produce_shapes(
        shape3, shape_max_mul0)
    util.check_shape_size(shape_max_mul1, SHAPE_SIZE_LIMIT)

    shape4, shape_max_mul1, shape_max_sub0 = util.produce_shapes(
        shape4, shape_max_mul1)
    util.check_shape_size(shape_max_sub0, SHAPE_SIZE_LIMIT)

    return shape0, shape1, shape2, shape3, shape4,\
           shape_greatery, shape_selecte


@fusion_manager.register("lamb_update_with_lr_v2")
def lamb_update_with_lr_v2_compute(input0,
                                   input1,
                                   input2,
                                   input3,
                                   input4,
                                   greater_y,
                                   select_e,
                                   output,
                                   kernel_name="lamb_update_with_lr_v2"):
    """
    calculating data

    Parameters
    ----------
    input0 : TVM tensor
        the placeholder of input0
    input1 : TVM tensor
        the placeholder of input1
    input2 : TVM tensor
        the placeholder of input2
    input3 : TVM tensor
        the placeholder of input3
    input4 : TVM tensor
        the placeholder of input4
    greater_y : TVM tensor
        the placeholder of greater_y
    select_e : TVM tensor
        the placeholder of select_e
    output : dict
        dict of output, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "apply_one_lamb"

    Returns
    -------
    output tensor
    """
    greater_0 = greater_compute(greater_y, input0, kernel_name="Geater")
    greater_1 = greater_compute(greater_y, input1, kernel_name="Greater_1")
    truediv_0 = true_div_compute(input0, input1, kernel_name="truediv_3")
    select_0 = select_compute(greater_1,
                              truediv_0,
                              select_e,
                              kernel_name="Select")
    select_1 = select_compute(greater_0,
                              select_0,
                              select_e,
                              kernel_name="Select_1")
    mul_5 = mul_compute(input2, select_1, kernel_name="mul_5")
    mul_6 = mul_compute(mul_5, input3, kernel_name="mul_6")
    sub_0 = sub_compute(input4, mul_6, kernel_name="sub")

    return sub_0


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, str)
def lamb_update_with_lr_v2(input0,
                           input1,
                           input2,
                           input3,
                           input4,
                           greater_y,
                           select_e,
                           output,
                           kernel_name="lamb_update_with_lr_v2"):
    """
    calculating data

    Parameters
    ----------
    input0 : dict
        shape and dtype of input0
    input1 : dict
        shape and dtype of input1
    input2 : dict
        shape and dtype of input2
    input3 : dict
        shape and dtype of input3
    input4 : dict
        shape and dtype of input4
    greater_y : dict
        shape and dtype of greater_y
    select_e : dict
        shape and dtype of select_e
    output : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "apply_one_lamb"

    Returns
    -------
    None
    """

    dtype0 = input0.get("dtype")
    dtype1 = input1.get("dtype")
    dtype2 = input2.get("dtype")
    dtype3 = input3.get("dtype")
    dtype4 = input4.get("dtype")
    dtype_greatery = greater_y.get("dtype")
    dtype_selecte = select_e.get("dtype")

    shape0, shape1, shape2, shape3, shape4, shape_greatery, shape_selecte = \
        _check_broadcast_shape(input0, input1, input2, input3, input4,
                               greater_y, select_e)

    util.check_kernel_name(kernel_name)

    input_place0 = tvm.placeholder(shape0, name="input0", dtype=dtype0)
    input_place1 = tvm.placeholder(shape1, name="input1", dtype=dtype1)
    input_place2 = tvm.placeholder(shape2, name="input2", dtype=dtype2)
    input_place3 = tvm.placeholder(shape3, name="input3", dtype=dtype3)
    input_place4 = tvm.placeholder(shape4, name="input4", dtype=dtype4)
    input_greatery = tvm.placeholder(shape_greatery,
                                     name="greater_y",
                                     dtype=dtype_greatery)
    input_selecte = tvm.placeholder(shape_selecte,
                                    name="select_e",
                                    dtype=dtype_selecte)

    res = lamb_update_with_lr_v2_compute(input_place0, input_place1,
                                         input_place2, input_place3,
                                         input_place4, input_greatery,
                                         input_selecte, output, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name":
            kernel_name,
        "tensor_list": (input_place0, input_place1, input_place2, input_place3,
                        input_place4, input_greatery, input_selecte, res)
    }

    te.lang.cce.cce_build_code(sch, config)
