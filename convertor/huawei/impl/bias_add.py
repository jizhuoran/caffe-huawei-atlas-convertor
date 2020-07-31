#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

bias_add
"""
import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=too-many-locals,redefined-builtin,unused-argument
# pylint: disable=too-many-statements,too-many-branches,invalid-name
def op_select_format(x, bias, y, data_format="NHWC",
                     kernel_name="bias_add"):
    """
    select format dynamically
    """
    shape_bias = bias.get("shape")
    ori_shape_x = x.get("ori_shape")
    c0 = 16
    if len(ori_shape_x) <= 4:
        if shape_bias[0] % c0 == 0 and len(ori_shape_x) == 4:
            # NC1HWC0+ND NCHW+NCHW NHWC+NHWC
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,int32,float16,float,"
                                        "int32,float16,float",
                               format="NC1HWC0,NC1HWC0,NCHW,NCHW,NCHW,NHWC,"
                                      "NHWC,NHWC")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="float16,float,int32,float16,float,"
                                        "int32,float16,float",
                               format="ND,ND,NCHW,NCHW,NCHW,NHWC,NHWC,NHWC")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,int32,float16,float,"
                                         "int32,float16,float",
                                format="NC1HWC0,NC1HWC0,NCHW,NCHW,NCHW,NHWC,"
                                       "NHWC,NHWC")
        elif shape_bias[0] % c0 != 0 and len(ori_shape_x) == 4:
            # NC1HWC0+NC1HWC0 NCHW+NCHW NHWC+NHWC
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,int32,float16,float,"
                                        "int32,float16,float",
                               format="NC1HWC0,NC1HWC0,NCHW,NCHW,NCHW,NHWC,"
                                      "NHWC,NHWC")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="float16,float,int32,float16,float,"
                                        "int32,float16,float",
                               format="NC1HWC0,NC1HWC0,NCHW,NCHW,NCHW,NHWC,"
                                      "NHWC,NHWC")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,int32,float16,float,"
                                         "int32,float16,float",
                                format="NC1HWC0,NC1HWC0,NCHW,NCHW,NCHW,NHWC,"
                                       "NHWC,NHWC")
        else:
            # NCHW+NCHW NHWC+NHWC
            input0 = gen_param(classify="input0", name="x",
                               datatype="int32,float16,float,int32,float16,"
                                        "float",
                               format="NCHW,NCHW,NCHW,NHWC,NHWC,NHWC")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="int32,float16,float,int32,float16,"
                                        "float",
                               format="NCHW,NCHW,NCHW,NHWC,NHWC,NHWC")
            output0 = gen_param(classify="output0", name="y",
                                datatype="int32,float16,float,int32,float16,"
                                         "float",
                                format="NCHW,NCHW,NCHW,NHWC,NHWC,NHWC")
    else:
        if shape_bias[0] % c0 == 0:
            # NDHWC+NDHWC NCDHW+NCDHW NDC1HWC0+NDC1HWC0
            input0 = gen_param(classify="input0", name="x",
                               datatype="int32,float16,float,int32,float16,"
                                        "float,int32,float16,float",
                               format="NDHWC,NDHWC,NDHWC,NCDHW,NCDHW,"
                                      "NCDHW,NDC1HWC0,NDC1HWC0,NDC1HWC0")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="int32,float16,float,int32,float16,"
                                        "float,int32,float16,float",
                               format="ND,ND,ND,ND,ND,ND,ND,ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="int32,float16,float,int32,float16,"
                                        "float,int32,float16,float",
                                format="NDHWC,NDHWC,NDHWC,NCDHW,NCDHW,"
                                       "NCDHW,NDC1HWC0,NDC1HWC0,NDC1HWC0")

        else:
            # NDHWC+NDHWC NCDHW+NCDHW
            input0 = gen_param(classify="input0", name="x",
                               datatype="int32,float16,float,int32,float16,"
                                        "float",
                               format="NDHWC,NDHWC,NDHWC,NCDHW,NCDHW,NCDHW")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="int32,float16,float,int32,float16,"
                                        "float",
                               format="ND,ND,ND,ND,ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="int32,float16,float,int32,float16,"
                                         "float",
                                format="NDHWC,NDHWC,NDHWC,NCDHW,NCDHW,NCDHW")

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def bias_add_compute(x, bias, y, data_format,
                     kernel_name="bias_add"):
    """
    calculating data's bias add

    Parameters
    ----------
    x : tvm tensor
              x data x
    bias : tvm tensor
              x data y
    y : tvm tensor
              y data
    data_format: A string.
                'N...C' and 'NC...' are supported.
    kernel_name : string
                  cce kernel name, default value is "bias_add"

    Returns
    -------
    res : y of the data's bias add
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    bias_broad = te.lang.cce.broadcast(bias, shape_x)

    res = te.lang.cce.vadd(x, bias_broad)

    return res


@util.check_input_type(dict, dict, dict, str, str)
def bias_add(x, bias, y, data_format="NHWC",
             kernel_name="bias_add"):
    """
    algorithm: bias_and
    Reduce a tensor on a certain axis based on min

    Parameters
    ----------
    x : dict
              the shape and dtype of the tensor x
    bias : dict
              the shape and dtype of the tensor y
    y :  dict
              the shape and dtype of the tensor z
    data_format: A string.
                'N...C' and 'NC...' are supported.
    kernel_name : string
                  cce kernel name, default value is "bias_add"
    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_bias = bias.get("shape")
    dtype_x = x.get("dtype").lower()
    dtype_bias = bias.get("dtype").lower()
    dtype_y = y.get("dtype").lower()
    data_format = data_format.upper()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_bias)
    util.check_tensor_shape_size(shape_x, )
    util.check_tensor_shape_size(shape_bias, )

    check_tuple = ("float16", "float32", "int32")
    data_format_list = ("NCHW", "NHWC", "NDHWC", "NCDHW")
    util.check_dtype_rule(dtype_x, check_tuple)
    util.check_dtype_rule(dtype_bias, check_tuple)
    util.check_dtype_rule(dtype_y, check_tuple)
    if dtype_x != dtype_bias:
        raise RuntimeError(
            "The dtype of x and bias must be the same")
    if data_format not in data_format_list:
        raise RuntimeError(
            "The data_format only support NCHW, NHWC, NDHWC, NCDHW")
    if x.get("format") is not None and x.get("format").upper() == "NC1HWC0":
        ori_format_x = x.get("ori_format").upper()
        ori_shape_x = x.get("ori_shape")
        if len(shape_x) != 5:
            raise RuntimeError("bias_add only support shape 5D"
                               "when input format is NC1HWC0")
        if ori_format_x != data_format:
            raise RuntimeError("the input ori_format and"
                               "data_format must be the same")
        if bias.get("format") is not None and \
                bias.get("format").upper() == "NC1HWC0":
            ori_shape_bias = bias.get("ori_shape")
            if ori_format_x == "NCHW" and ori_shape_x[1] != ori_shape_bias[0]:
                raise RuntimeError("data_format is NCHW, shape_bias must"
                                   "be equal to the second axis of shape_x")
            elif ori_format_x == "NHWC" and \
                    ori_shape_x[-1] != ori_shape_bias[0]:
                raise RuntimeError("data_format is NHWC, shape_bias must"
                                   "be equal to the second axis of shape_x")
        else:
            if ori_format_x == "NCHW" and ori_shape_x[1] != shape_bias[0]:
                raise RuntimeError("data_format is NCHW, shape_bias must"
                                   "be equal to the second axis of shape_x")
            elif ori_format_x == "NHWC" and ori_shape_x[-1] != shape_bias[0]:
                raise RuntimeError("data_format is NHWC, shape_bias must"
                                   "be equal to the second axis of shape_x")
        shape_bias = (1, shape_x[1], 1, 1, shape_x[4])

    elif x.get("format") is not None and x.get("format").upper() == "NDHWC":
        if len(shape_x) != 5:
            raise RuntimeError("bias_add only support shape 5D"
                               "when input format is NDHWC")
        if shape_x[4] != shape_bias[0]:
            raise RuntimeError("data_format is NDHWC, shape_bias must"
                               "be equal to the fifth axis of shape_x")
        shape_bias = (1, 1, 1, 1, shape_x[4])

    elif x.get("format") is not None and x.get("format").upper() == "NCDHW":
        if len(shape_x) != 5:
            raise RuntimeError("bias_add only support shape 5D"
                               "when input format is NCDHW")
        if shape_x[1] != shape_bias[0]:
            raise RuntimeError("data_format is NDHWC, shape_bias must"
                               "be equal to the second axis of shape_x")
        shape_bias = (1, shape_x[1], 1, 1, 1)

    elif x.get("format") is not None and x.get("format").upper() == "NDC1HWC0":
        if len(shape_x) != 6:
            raise RuntimeError("bias_add only support shape 6D"
                               "when input format is NDC1HWC0")
        ori_shape_x = x.get("ori_shape")
        if x.get("ori_format").upper() == "NDHWC":
            if ori_shape_x[4] != shape_bias[0]:
                raise RuntimeError("data_format is NDHWC, shape_bias must"
                                   "be equal to the fifth axis of shape_x")
        elif x.get("ori_format").upper() == "NCDHW":
            if ori_shape_x[1] != shape_bias[0]:
                raise RuntimeError("data_format is NDHWC, shape_bias must"
                                   "be equal to the second axis of shape_x")
        shape_bias = (1, 1, shape_x[2], 1, 1, shape_x[5])

    else:
        if data_format == "NCHW":
            if len(shape_x) < 2 or len(shape_x) > 4:
                raise RuntimeError("bias_add only support shape 2D to 4D")
            if shape_x[1] != shape_bias[0]:
                raise RuntimeError("data_format is NCHW, shape_bias must"
                                   "be equal to the second axis of shape_x")
            shape_bias = (1, shape_x[1],)
            for i in range(2, len(shape_x)):
                shape_bias = shape_bias + (1,)
        else:
            if len(shape_x) < 2:
                raise RuntimeError("bias_add only support"
                                   "shape larger than 2D")
            if shape_x[-1] != shape_bias[0]:
                raise RuntimeError("data_format is NHWC, shape_bias must be"
                                   "equal to the last axis of shape_x")
            shape_bias = ()
            for i in range(0, len(shape_x)):
                if i != len(shape_x) - 1:
                    shape_bias = shape_bias + (1,)
                else:
                    shape_bias = shape_bias + (shape_x[-1],)

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)

    bias = tvm.placeholder(shape_bias, name="bias", dtype=dtype_bias)

    res = bias_add_compute(data_x, bias, y, data_format, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_x, bias, res]}
    te.lang.cce.cce_build_code(sch, config)
