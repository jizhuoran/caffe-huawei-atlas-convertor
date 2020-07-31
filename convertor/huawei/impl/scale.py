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

scale
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

NONETYPE = type(None)


# pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
# pylint: disable=too-many-boolean-expressions,too-many-locals,unused-variable
def op_select_format(x, scale, bias, y, axis=1, num_axes=1, scale_from_blob=True,
                     kernel_name="scale"):
    """
    select format dynamically
    """
    shape_x_ori = x.get("ori_shape")
    shape_x = x.get("shape")
    shape_scale_ori = scale.get("ori_shape")
    shape_scale = scale.get("shape")

    length_x_ori = len(shape_x_ori)
    length_x = len(shape_x)
    length_scale_ori = len(shape_scale_ori)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        format_scale = "ND,ND,ND,ND"
        format_bias = "ND,ND,ND,ND"
        format_scale_hisi = "ND,ND"
        format_bias_hisi = "ND,ND"
    else:
        format_scale = "NC1HWC0,NC1HWC0,ND,ND"
        format_bias = "NC1HWC0,NC1HWC0,ND,ND"
        format_scale_hisi = "NC1HWC0,ND"
        format_bias_hisi = "NC1HWC0,ND"

    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if length_x_ori == 4:
        # NC1HWC0+ND
        if product_version in ("Hi3796CV300ES"):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16",
                               format="NC1HWC0,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float16",
                               format=format_scale_hisi)
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float16",
                               format=format_bias_hisi)
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16",
                                format="NC1HWC0,ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float",
                               format="NC1HWC0,NC1HWC0,ND,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float,float16,float",
                               format=format_scale)
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float,float16,float",
                               format=format_bias)
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,float16,float",
                                format="NC1HWC0,NC1HWC0,ND,ND")
    else:
        # ND+ND
        if product_version in ("Hi3796CV300ES"):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16",
                               format="ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16",
                               format="ND")
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16",
                               format="ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16",
                                format="ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float",
                               format="ND,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float",
                               format="ND,ND")
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float",
                               format="ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float",
                                format="ND,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def param_scale_check(shape_x, shape_scale):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_scale : list or tuple.
        shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if not(length_scale == 1 and shape_scale[0] == 1):
        if length_x != length_scale:
            raise RuntimeError(
                "length_x and length_scale must be equal")
        for i in range(length_scale):
            if shape_scale[i] != shape_x[i] and shape_scale[i] != 1:
                raise RuntimeError(
                    "shape_scale is not match to broadcast")


def get_param_scale_shape(shape_x, shape_scale):
    """
    Function to calculate the shape of scale.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_scale : list or tuple.
        shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        shape = [1] * length_x
    else:
        shape = list(shape_scale)

    return shape


def _check_dtype(input_dtype):
    """
    Function to check dtype of input data.

    Parameters
    ----------

    input_dtype: str
        dtype of input data
    Returns
    -------
    None
    """

    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if product_version in ("Hi3796CV300ES"):
        if input_dtype == "float32":
            raise RuntimeError("float32 is not support in ES")
        util.check_dtype_rule(input_dtype, ("float16",))
    else:
        util.check_dtype_rule(input_dtype, ("float16", "float32",))


# pylint: disable=too-many-branches
def _check_scale_shape_axis(shape_x, shape_scale, axis, num_axes, scale_from_blob):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes:
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if (axis >= length_x) or (axis < (-length_x)):
        raise RuntimeError("axis out of range index")

    if num_axes < -1:
        raise RuntimeError("num_axes must be non-negative or -1")

    if axis < 0:
        axis_ = length_x + axis
    else:
        axis_ = axis

    # from blob
    if scale_from_blob:
        if num_axes == -1:
            scale_num = length_x - axis_
            if length_scale != scale_num:
                raise RuntimeError(
                    "length_scale and scale_num must be equal")
            for i in range(scale_num):
                if shape_x[axis_ + i] != shape_scale[i]:
                    raise RuntimeError(
                        "Dimensions shape_x and shape_scale must be equal")
        if num_axes == 0:
            if length_scale != 1 or shape_scale[0] != 1:
                raise RuntimeError("scale must be a scalar ")
        if num_axes > 0:
            num_axis = axis_ + num_axes
            if num_axis > length_x:
                raise RuntimeError(
                    "scale shape extends x shape when applied")
            if length_scale != num_axes:
                raise RuntimeError(
                    "length_scale and num_axes must be equal")
            for i in range(num_axes):
                if shape_x[axis_ + i] != shape_scale[i]:
                    raise RuntimeError(
                        "dimensions shape_x and shape_scale must be equal")

    # from bottom
    if not scale_from_blob:
        if not(length_scale == 1 and shape_scale[0] == 1):
            scale_num = axis_ + length_scale
            if scale_num > length_x:
                raise RuntimeError(
                    "scale shape extends x shape when applied")
            for i in range(length_scale):
                if shape_x[axis_ + i] != shape_scale[i]:
                    raise RuntimeError(
                        "Dimensions shape_x and shape_scale must be equal")


def get_scale_shape(shape_x, shape_scale, axis_, num_axes, scale_from_blob):
    """
    Function to calculate shape of scale.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape
    axis_ : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes:
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    shape: list or tuple
        the shape of scale
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)
    if scale_from_blob:
        if num_axes == -1:
            shape_left = [1] * axis_
            shape = shape_left + list(shape_scale)
        elif num_axes == 0:
            shape = [1] * length_x
        else:
            left_length = length_x - num_axes - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_scale) + shape_right
    else:
        if length_scale == 1 and shape_scale[0] == 1:
            shape = [1] * length_x
        else:
            left_length = length_x - length_scale - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_scale) + shape_right

    return shape


# pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_compute(x, scale):
    """
    algorithm: Scale
    y = scale*x

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    dtype_x = x.dtype
    dtype_scale = scale.dtype

    is_cast = False
    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES"):
        if dtype_x == "float16":
            is_cast = True
            x = te.lang.cce.cast_to(x, 'float32')
        if dtype_scale == "float16":
            scale = te.lang.cce.cast_to(scale, 'float32')

    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    scale_broad = te.lang.cce.broadcast(scale, shape_x)

    res = te.lang.cce.vmul(x, scale_broad)

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype_x)

    return res


# pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_bias_compute(x, scale, bias):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data
    bias : TVM tensor
        contains bias data
    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    dtype_x = x.dtype
    dtype_scale = scale.dtype
    dtype_bias = bias.dtype

    is_cast = False
    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES"):
        if dtype_x == "float16":
            is_cast = True
            x = te.lang.cce.cast_to(x, 'float32')
        if dtype_scale == "float16":
            scale = te.lang.cce.cast_to(scale, 'float32')
        if dtype_bias == "float16":
            bias = te.lang.cce.cast_to(bias, 'float32')

    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    scale_broad = te.lang.cce.broadcast(scale, shape_x)
    bias_broad = te.lang.cce.broadcast(bias, shape_x)

    res_tmp = te.lang.cce.vmul(x, scale_broad)
    res = te.lang.cce.vadd(res_tmp, bias_broad)

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype_x)

    return res


# pylint: disable=too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("scale")
def scale_compute(x, scale, bias, y, axis, num_axes, scale_from_blob,
                  kernel_name="scale"):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data
    bias : TVM tensor
        contains bias data
    y : dict
        dict of output,
        A Tensor for output, should be same shape and type as x.
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes: int
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.
    kernel_name : str
        kernel name, default value is "scale"

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    if bias is not None:
        res = _fused_scale_bias_compute(x, scale, bias)
    else:
        res = _fused_scale_compute(x, scale)

    return res


# pylint: disable=too-many-locals,no-member,invalid-name,too-many-statements,line-too-long
@util.check_input_type(dict, dict, (dict, NONETYPE), dict, int, int, bool, str)
def scale(x, scale, bias, y, axis=1, num_axes=1, scale_from_blob=True,
          kernel_name="scale"):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : dict
        dict of input, A Tensor for input data.
    scale : dict
        dict of scale,
        A Tensor for scaling factor, to scale the input data.
    bias : dict
        dict of bias,
        A Tensor for bias, to shift to the input data.
    y : dict
        dict of output,
        A Tensor for y, should be same shape and type as x.
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes: int
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.
    kernel_name : str
        kernel name, default value is "scale"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    _check_dtype(dtype_x.lower())

    shape_scale = scale.get("shape")
    dtype_scale = scale.get("dtype")
    util.check_shape_rule(shape_scale)
    util.check_tensor_shape_size(shape_scale)
    _check_dtype(dtype_scale.lower())
    util.check_kernel_name(kernel_name)

    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        dtype_bias = bias.get("dtype")
        util.check_shape_rule(shape_bias)
        util.check_tensor_shape_size(shape_bias)
        _check_dtype(dtype_bias.lower())

    shape_x_ori = x.get("ori_shape")
    length_x_ori = len(shape_x_ori)

    shape_scale_new = []
    shape_bias_new = []

    if length_x_ori == 4:
        param_scale_check(shape_x, shape_scale)
        shape_scale_new = get_param_scale_shape(shape_x, shape_scale)
        if len(shape_bias) > 0:
            shape_bias_new = shape_scale_new
    else:
        _check_scale_shape_axis(shape_x, shape_scale, axis, num_axes, scale_from_blob)

        length_x = len(shape_x)
        if axis < 0:
            axis_ = length_x + axis
        else:
            axis_ = axis

        shape_scale_new = get_scale_shape(shape_x, shape_scale, axis_, num_axes, scale_from_blob)
        if len(shape_bias) > 0:
            shape_bias_new = shape_scale_new

    bias_input = None
    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    scale_input = tvm.placeholder(shape_scale_new, name="scale_input", dtype=dtype_scale.lower())
    if len(shape_bias) > 0:
        bias_input = tvm.placeholder(shape_bias_new, name="bias_input", dtype=dtype_bias.lower())

    res = scale_compute(x_input, scale_input, bias_input, y,
                        axis, num_axes, scale_from_blob, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = (x_input, scale_input, res)
    if len(shape_bias) > 0:
        tensor_list = (x_input, scale_input, bias_input, res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
