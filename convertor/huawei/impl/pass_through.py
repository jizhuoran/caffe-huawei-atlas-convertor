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

pass_through
"""

from topi.cce import util
from impl import pass_through_forward
from impl import pass_through_backward

MINI_STRIDE = 1


def check_param(in_dic, out_dic, stride, reverse, kernel_name):
    """
    check validation of input param

    Parameters
    ----------
    in_dic : dict
        shape/dtype/fmt of input
    out_dic : dict
        shape/dtype/fmt of output, should be same shape and type as input
    stride : int32
        c/h/w stride
    reverse : bool
        forward or reverse flag
    kernel_name : str
        kernel name, default value is "pass_through"

    Returns
    -------
    None
    """
    shape_in = in_dic.get("shape")
    dtype_in = in_dic.get("dtype")
    fmt_in = in_dic.get("format")
    shape_out = out_dic.get("shape")
    dtype_out = out_dic.get("dtype")
    fmt_out = out_dic.get("format")

    util.check_shape_rule(shape_in)
    util.check_tensor_shape_size(shape_in)
    util.check_shape_rule(shape_out)
    util.check_tensor_shape_size(shape_out)
    util.check_dtype_rule(dtype_in.lower(), ["float16", "float32",
                                             "int8", "uint8",
                                             "int16", "uint16",
                                             "int32", "uint32",
                                             "int64", "uint64"])
    util.check_dtype_rule(dtype_out.lower(), ["float16", "float32",
                                              "int8", "uint8",
                                              "int16", "uint16",
                                              "int32", "uint32",
                                              "int64", "uint64"])

    if fmt_in.lower() != "nhwc" or fmt_out.lower() != "nhwc":
        raise ValueError("In Out format must be NHWC")

    if stride < MINI_STRIDE:
        raise ValueError("stride must be greater than 0")

    if reverse is True:
        if (shape_in[3] % (stride * stride)) != 0:
            raise ValueError("Reverse direction: c must be times of stride**2")
    else:
        if (shape_in[1] % stride != 0) or (shape_in[2] % stride != 0):
            raise ValueError("Forward direction: w/h must be times of stride")

    util.check_kernel_name(kernel_name)


@util.check_input_type(dict, dict, int, bool, str)
def pass_through(in_dic, out_dic, stride, reverse, kernel_name="pass_through"):
    """
    pass_through ops interface

    Parameters
    ----------
    in_dic : dict
        shape/dtype/fmt of input
    out_dic : dict
        shape/dtype/fmt of output, should be same shape and type as input
    stride : int32
        c/h/w stride
    reverse : bool
        forward or reverse flag
    kernel_name : str
        kernel name, default value is "pass_through"

    Returns
    -------
    tik instance
    """

    check_param(in_dic, out_dic, stride, reverse, kernel_name)

    if reverse is False:
        tik_instance, input_gm, output_gm = \
            pass_through_backward.pass_through_backward_func(in_dic, stride)
    else:
        tik_instance, input_gm, output_gm = \
            pass_through_forward.pass_through_forward_func(in_dic, stride)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=input_gm,
                          outputs=output_gm)
    return tik_instance
