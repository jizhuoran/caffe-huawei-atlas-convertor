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

concat_v2_d: Concatenates tensors along one dimension.
            The number of dimensions of input tensors must match,
            and all dimensions except 'axis' must be equal.
            tf ConcactV2 op

"""
from __future__ import absolute_import

from te import tvm
import te.lang.cce
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.concat_last_dim import ConcatWithVnchw
from impl.concat_tik import ConcatSchedule


# pylint: disable=locally-disabled,unused-argument,too-many-branches
# pylint: disable=too-many-locals,too-many-statements,unused-variable
# pylint: disable=too-many-boolean-expressions
def op_select_format(input_values,
                     output_data,
                     axis,
                     kernel_name="concat_v2_d"):
    """
    select format dynamically
    """
    data_list = []
    datatype_5d_xhs = "float16,int32,int8,int16,int64,uint8,uint16,uint32," \
                      "uint64,bool,float16,int32,int8,int16,int64," \
                      "uint8,uint16,uint32,uint64,bool"
    format_5d_xhs = "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0," \
                    "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,ND,ND,ND,ND,ND," \
                    "ND,ND,ND,ND,ND"
    datatype_4d_xhs = "float16,int32,int8,int16,int64,uint8,uint16,uint32," \
                      "uint64,bool"
    format_4d_xhs = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
    datatype_5d = "float16,float,int32,int8,int16,int64,uint8,uint16,uint32," \
                  "uint64,bool,float16,float,int32,int8,int16,int64,uint8," \
                  "uint16,uint32,uint64,bool"
    format_5d = "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0," \
                "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,ND,ND,ND,ND,ND,ND,ND,ND," \
                "ND,ND,ND"
    datatype_4d = "float16,float,int32,int8,int16,int64,uint8,uint16," \
                  "uint32,uint64,bool"
    format_4d = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"
    ori_format = input_values[0].get("ori_format").upper()
    for i, input_dict in enumerate(input_values):
        shape_input = input_dict.get("ori_shape")
        data_list.append(shape_input)
    divisible = 16
    len_axis = 0

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        if ori_format == "NHWC" and len(data_list[0]) == 4:
            for i, list_element in enumerate(data_list):
                if data_list[i][3] % divisible == 0:
                    len_axis += 1
            if len_axis == len(data_list):
                # NC1HWC0+ND
                input0 = gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_5d_xhs,
                    format=format_5d_xhs)
                output0 = gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_5d_xhs,
                    format=format_5d_xhs)
            else:
                # ND+
                input0 = gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_4d_xhs,
                    format=format_4d_xhs)
                output0 = gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_4d_xhs,
                    format=format_4d_xhs)
        elif ori_format == "NCHW" and len(data_list[0]) == 4:
            for i, list_element in enumerate(data_list):
                if data_list[i][1] % divisible == 0:
                    len_axis += 1
            if len_axis == len(data_list):
                # NC1HWC0+ND
                input0 = gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_5d_xhs,
                    format=format_5d_xhs)
                output0 = gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_5d_xhs,
                    format=format_5d_xhs)
            else:
                # ND+
                input0 = gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_4d_xhs,
                    format=format_4d_xhs)
                output0 = gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_4d_xhs,
                    format=format_4d_xhs)
        else:
            # ND
            input0 = gen_param(
                classify="input0",
                name="input_values",
                datatype=datatype_4d_xhs,
                format=format_4d_xhs)
            output0 = gen_param(
                classify="output0",
                name="output_data",
                datatype=datatype_4d_xhs,
                format=format_4d_xhs)
    else:
        if ori_format == "NHWC" and len(data_list[0]) == 4:
            for i, list_element in enumerate(data_list):
                if data_list[i][3] % divisible == 0:
                    len_axis += 1
            if len_axis == len(data_list):
                # NC1HWC0+ND
                input0 = gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_5d,
                    format=format_5d)
                output0 = gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_5d,
                    format=format_5d)
            else:
                # ND+
                input0 = gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_4d,
                    format=format_4d)
                output0 = gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_4d,
                    format=format_4d)
        elif ori_format == "NCHW" and len(data_list[0]) == 4:
            for i, list_element in enumerate(data_list):
                if data_list[i][1] % divisible == 0:
                    len_axis += 1
            if len_axis == len(data_list):
                # NC1HWC0+ND
                input0 = gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_5d,
                    format=format_5d)
                output0 = gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_5d,
                    format=format_5d)
            else:
                # ND+
                input0 = gen_param(
                    classify="input0",
                    name="input_values",
                    datatype=datatype_4d,
                    format=format_4d)
                output0 = gen_param(
                    classify="output0",
                    name="output_data",
                    datatype=datatype_4d,
                    format=format_4d)
        else:
            # ND
            input0 = gen_param(
                classify="input0",
                name="input_values",
                datatype=datatype_4d,
                format=format_4d)
            output0 = gen_param(
                classify="output0",
                name="output_data",
                datatype=datatype_4d,
                format=format_4d)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def concat_v2_d_compute(input_values,
                        output_data,
                        axis,
                        kernel_name="concat_v2_d"):
    """how to make concat_v2_d compute these tensors.
    -----------
    Parameters
    ----------
    input_values : A list of tensor objects .
    axis : scalar,in the range [-rank(values), rank(values))
    output_data : A dict resulting from concatenation of the input tensors
    kernel_name : cce kernel name, default value is "concat_v2_d"

    Returns
    -------
    res : the result of concat_v2_d_compute
    """
    res = te.lang.cce.concat(input_values, axis)

    return res


@util.check_input_type((list, tuple), dict, int, str)
def concat_v2_d(input_values, output_data, axis, kernel_name="concat_v2_d"):
    """
    algorithm: concat_v2_d

    Parameters
    ----------
    input_values : A list of dict objects.
                 dict with keys(shape and dtype) of tensor
                 dtype only support float32, int8, int16, int32, int64, uint8,
                 uint16, uint32, uint64, float16
    output_data : A dict resulting from concatenation of the input tensors
    axis : scalar,in the range [-rank(values), rank(values))
    kernel_name : cce kernel name, default value is "concat_v2_d"

    Returns
    -------
    None
    """
    shape_value = []
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get("ori_shape")
        shape_value.append(shape_input)
    first_input_shape = input_values[0].get("ori_shape")
    if axis < 0:
        axis_new = len(first_input_shape) + axis
    else:
        axis_new = axis
    for _, element_shape in enumerate(shape_value):
        for j, _ in enumerate(first_input_shape):
            if element_shape[j] != first_input_shape[j] and j != axis_new:
                raise RuntimeError("Axes must equal except merge axis")

    # when format is 5HD check whether concat by C and redefine the axis
    input_format = input_values[0].get("format")
    ori_format = input_values[0].get("ori_format")
    if input_format == "NC1HWC0":
        axis = util.axis_transfrom_5d(axis, ori_format)

    # do check for input
    util.check_kernel_name(kernel_name)
    input_num = len(input_values)
    dim_num = len(input_values[0].get("shape"))

    # check the length of input shape must be equal
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get("shape")
        if len(shape_input) != dim_num:
            raise RuntimeError("The length of each shape must be equal")
    if axis < -dim_num or axis >= dim_num:
        raise RuntimeError("Axis value out of range")

    # begin to check where user branch concat_last_dim with command nchwvconv
    concat_l = ConcatWithVnchw(input_values, output_data, kernel_name)
    if_vnchw_support = concat_l.check_vnchw_supported()
    if if_vnchw_support:
        concat_l.do_concat_vnchw()
        return
    # end to check where user branch concat_last_dim with command nchwvconv

    # begin to check where user branch concat tik
    concat_s = ConcatSchedule(input_values, output_data, axis, kernel_name)
    if_tik_support = concat_s.check_tik_supported()

    if if_tik_support:
        concat_s.concat_compute()
        return
    # end to check where user branch concat tik

    # get 2D(or 1D) new shape and axis
    input_shape_list = concat_s.input_shapes
    axis = concat_s.concat_axis
    check_list = ("float32", "int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16")
    data = []
    for i, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get("shape")
        util.check_shape_rule(shape_input)
        util.check_tensor_shape_size(shape_input)
        inp_dtype = tensor_dict.get("dtype").lower()
        util.check_dtype_rule(inp_dtype, check_list)
        data.append(
            tvm.placeholder(
                input_shape_list[i], name="data_%d" % i, dtype=inp_dtype))

    res = concat_v2_d_compute(data, output_data, axis, kernel_name)
    data.append(res)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": data}

    te.lang.cce.cce_build_code(sch, config)
