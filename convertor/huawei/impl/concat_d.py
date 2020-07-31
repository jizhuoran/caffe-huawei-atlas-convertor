#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

concat
"""
from __future__ import absolute_import
from te import platform as tbe_platform
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.concat_v2_d import concat_v2_d


# pylint: disable=locally-disabled,unused-argument,too-many-branches
# pylint: disable=too-many-locals,too-many-statements,unused-variable
def op_select_format(input_values, output_data, concat_dim,
                     kernel_name="concat"):
    """
    select format dynamically
    """
    data_list = []
    datatype_5d_xhs = "float16,int32,int8,int16,int64,uint8,uint16,uint32," \
                      "uint64,bool,float16,int32,int8,int16,int64," \
                      "uint8,uint16,uint32,uint64,bool"
    format_5d_xhs = "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0," \
                    "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,ND,ND,ND,ND,ND,ND," \
                    "ND,ND,ND,ND"
    datatype_4d_xhs = \
        "float16,int32,int8,int16,int64,uint8,uint16,uint32,uint64,bool"
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
                input0 = gen_param(classify="input0", name="input_values",
                                   datatype=datatype_5d_xhs,
                                   format=format_5d_xhs)
                output0 = gen_param(classify="output0", name="output_data",
                                    datatype=datatype_5d_xhs,
                                    format=format_5d_xhs)
            else:
                # ND+
                input0 = gen_param(classify="input0", name="input_values",
                                   datatype=datatype_4d_xhs,
                                   format=format_4d_xhs)
                output0 = gen_param(classify="output0", name="output_data",
                                    datatype=datatype_4d_xhs,
                                    format=format_4d_xhs)
        elif ori_format == "NCHW" and len(data_list[0]) == 4:
            for i, list_element in enumerate(data_list):
                if data_list[i][1] % divisible == 0:
                    len_axis += 1
            if len_axis == len(data_list):
                # NC1HWC0+ND
                input0 = gen_param(classify="input0", name="input_values",
                                   datatype=datatype_5d_xhs,
                                   format=format_5d_xhs)
                output0 = gen_param(classify="output0", name="output_data",
                                    datatype=datatype_5d_xhs,
                                    format=format_5d_xhs)
            else:
                # ND+
                input0 = gen_param(classify="input0", name="input_values",
                                   datatype=datatype_4d_xhs,
                                   format=format_4d_xhs)
                output0 = gen_param(classify="output0", name="output_data",
                                    datatype=datatype_4d_xhs,
                                    format=format_4d_xhs)
        else:
            # ND
            input0 = gen_param(classify="input0", name="input_values",
                               datatype=datatype_4d_xhs,
                               format=format_4d_xhs)
            output0 = gen_param(classify="output0", name="output_data",
                                datatype=datatype_4d_xhs,
                                format=format_4d_xhs)
    else:
        if ori_format == "NHWC" and len(data_list[0]) == 4:
            for i, list_element in enumerate(data_list):
                if data_list[i][3] % divisible == 0:
                    len_axis += 1
            if len_axis == len(data_list):
                # NC1HWC0+ND
                input0 = gen_param(classify="input0", name="input_values",
                                   datatype=datatype_5d, format=format_5d)
                output0 = gen_param(classify="output0", name="output_data",
                                    datatype=datatype_5d, format=format_5d)
            else:
                # ND+
                input0 = gen_param(classify="input0", name="input_values",
                                   datatype=datatype_4d, format=format_4d)
                output0 = gen_param(classify="output0", name="output_data",
                                    datatype=datatype_4d, format=format_4d)
        elif ori_format == "NCHW" and len(data_list[0]) == 4:
            for i, list_element in enumerate(data_list):
                if data_list[i][1] % divisible == 0:
                    len_axis += 1
            if len_axis == len(data_list):
                # NC1HWC0+ND
                input0 = gen_param(classify="input0", name="input_values",
                                   datatype=datatype_5d, format=format_5d)
                output0 = gen_param(classify="output0", name="output_data",
                                    datatype=datatype_5d, format=format_5d)
            else:
                # ND+
                input0 = gen_param(classify="input0", name="input_values",
                                   datatype=datatype_4d, format=format_4d)
                output0 = gen_param(classify="output0", name="output_data",
                                    datatype=datatype_4d, format=format_4d)
        else:
            # ND
            input0 = gen_param(classify="input0", name="input_values",
                               datatype=datatype_4d, format=format_4d)
            output0 = gen_param(classify="output0", name="output_data",
                                datatype=datatype_4d, format=format_4d)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@util.check_input_type((list, tuple), dict, int, str)
def concat_d(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    algorithm: concat
    Concatenates tensors along one dimension.
    Parameters
    ----------
    input_values : A list of `dict`.dict include keys shape and dtype
    output_data: dict of output_data, dict include keys shape and dtype
    concat_dim : scalar, in the range [-rank(values), rank(values))]
    kernel_name : cce kernel name, default value is "concat"
    Returns
    -------
    None
    """
    # concat_d is the same as concat_v2_d
    # use concat_v2_d to replace
    concat_v2_d(input_values, output_data, concat_dim, kernel_name)
