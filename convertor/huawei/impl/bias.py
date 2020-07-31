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

bias
"""

import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
#from te import platform as cceconf
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=too-few-public-methods
# class Version():
#     """
#     version class:
#     product and the core relation
#     """
#     MINI = ['1.1', '1.3']
#     CLOUD = ['1.60']
#     HISI = ['5.10']
#     V200 = ['2.3']

# pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
# pylint: disable=too-many-boolean-expressions,too-many-locals,unused-variable
def op_select_format(x, bias, y, axis=1, num_axes=1, bias_from_blob=True,
                     kernel_name="bias"):
    """
    select format dynamically
    """
    shape_x_ori = x.get("ori_shape")
    shape_x = x.get("shape")
    shape_bias_ori = bias.get("ori_shape")
    shape_bias = bias.get("shape")

    length_x_ori = len(shape_x_ori)
    length_x = len(shape_x)
    length_bias_ori = len(shape_bias_ori)
    length_bias = len(shape_bias)

    if length_bias == 1 and shape_bias[0] == 1:
        format_bias = "ND,ND,ND,ND"
        format_bias_hisi = "ND,ND"
    else:
        format_bias = "NC1HWC0,NC1HWC0,ND,ND"
        format_bias_hisi = "NC1HWC0,ND"

    #product_version = cceconf.CceProductParams().cce_product
    if length_x_ori == 4:
        # NC1HWC0+ND
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES"):
        #if product_version in Version.HISI:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16",
                               format="NC1HWC0,ND")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="float16,float16",
                               format=format_bias_hisi)
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16",
                                format="NC1HWC0,ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float",
                               format="NC1HWC0,NC1HWC0,ND,ND")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="float16,float,float16,float",
                               format=format_bias)
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,float16,float",
                                format="NC1HWC0,NC1HWC0,ND,ND")
    else:
        # ND+ND
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES"):
        #if product_version in Version.HISI:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16",
                               format="ND")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="float16",
                               format="ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16",
                                format="ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float",
                               format="ND,ND")
            input1 = gen_param(classify="input1", name="bias",
                               datatype="float16,float",
                               format="ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float",
                                format="ND,ND")

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def param_bias_check(shape_x, shape_bias):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_bias : list or tuple.
        shape of bias.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_bias = len(shape_bias)

    if not(length_bias == 1 and shape_bias[0] == 1):
        if length_x != length_bias:
            raise RuntimeError(
                "length_x and length_bias must be equal")
        for i in range(length_bias):
            if shape_bias[i] != shape_x[i] and shape_bias[i] != 1:
                raise RuntimeError(
                    "shape_bias is not match to broadcast")


def get_param_bias_shape(shape_x, shape_bias):
    """
    Function to calculate the shape of bias.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_bias : list or tuple.
        shape of bias.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_bias = len(shape_bias)

    if length_bias == 1 and shape_bias[0] == 1:
        shape = [1] * length_x
    else:
        shape = list(shape_bias)

    return shape


# pylint: disable=too-many-branches
def _check_shape_axis(shape_x, shape_bias, axis, num_axes, bias_from_blob):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_bias: list or tuple
        bias's data shape
    axis : int
        A int num indicates shape of bias when bias is from bottom.
    num_axes: int
        A int num indicates shape of bias when bias is from blob.
    bias_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_bias = len(shape_bias)

    if (axis >= length_x) or (axis < (-length_x)):
        raise RuntimeError(
            "axis out of range index")

    if num_axes < -1:
        raise RuntimeError(
            "num_axes must be non-negative or -1")

    if axis < 0:
        axis_ = length_x + axis
    else:
        axis_ = axis

    # from blob
    if bias_from_blob:
        if num_axes == -1:
            bias_num = length_x - axis_
            if length_bias != bias_num:
                raise RuntimeError(
                    "length_bias and bias_num must be equal")
            for i in range(bias_num):
                if shape_x[axis_ + i] != shape_bias[i]:
                    raise RuntimeError(
                        "Dimensions shape_x and shape_bias must be equal")
        if num_axes == 0:
            if length_bias != 1 or shape_bias[0] != 1:
                raise RuntimeError("bias must be a scalar ")
        if num_axes > 0:
            num_axis = axis_ + num_axes
            if num_axis > length_x:
                raise RuntimeError(
                    "bias shape extends x shape when applied")
            if length_bias != num_axes:
                raise RuntimeError(
                    "length_bias and num_axes must be equal")
            for i in range(num_axes):
                if shape_x[axis_ + i] != shape_bias[i]:
                    raise RuntimeError(
                        "Dimensions shape_x and shape_bias must be equal")

    # from bottom
    if not bias_from_blob:
        if not(length_bias == 1 and shape_bias[0] == 1):
            bias_num = axis_ + length_bias
            if bias_num > length_x:
                raise RuntimeError(
                    "bias shape extends x shape when applied")
            for i in range(length_bias):
                if shape_x[axis_ + i] != shape_bias[i]:
                    raise RuntimeError(
                        "Dimensions shape_x and shape_bias must be equal")


def get_shape(shape_x, shape_bias, axis_, num_axes, bias_from_blob):
    """
    Function to calculate shape of bias.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_bias: list or tuple
        bias's data shape
    axis_ : int
        A int num indicates shape of bias when bias is from bottom.
    num_axes:
        A int num indicates shape of bias when bias is from blob.
    bias_from_blob:
        A bool value indicates bias is from blob or bottom.

    Returns
    -------
    shape: list or tuple
        the shape of bias
    """

    length_x = len(shape_x)
    length_bias = len(shape_bias)
    if bias_from_blob:
        if num_axes == -1:
            shape_left = [1] * axis_
            shape = shape_left + list(shape_bias)
        elif num_axes == 0:
            shape = [1] * length_x
        else:
            left_length = length_x - num_axes - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_bias) + shape_right
    else:
        if length_bias == 1 and shape_bias[0] == 1:
            shape = [1] * length_x
        else:
            left_length = length_x - length_bias - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_bias) + shape_right

    return shape


def _check_dtype(dtype_x, dtype_bias):
    """
    Function to check dtype of input data.

    Parameters
    ----------

    dtype_x: str
        dtype of x data
    dtype_bias: str
        dtype of bias data
    Returns
    -------
    None
    """

    #product_version = cceconf.CceProductParams().cce_product
    #if product_version in Version.HISI:
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES"):
        if dtype_x == "float32" or dtype_bias == "float32":
            raise RuntimeError("float32 is not support in ES")
        check_tuple = ("float16",)
        util.check_dtype_rule(dtype_x, check_tuple)
        util.check_dtype_rule(dtype_bias, check_tuple)
    else:
        check_tuple = ("float32", "float16",)
        util.check_dtype_rule(dtype_x, check_tuple)
        util.check_dtype_rule(dtype_bias, check_tuple)


# pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
@fusion_manager.register("bias")
def bias_compute(x, bias, y, axis, num_axes, bias_from_blob,
                 kernel_name="bias"):
    """
    calculating data
    y = x + bias

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    bias : TVM tensor
        the placeholder of bias
    y : dict
        dict of y, include keys(shape and dtype)
    axis : int
        A int num indicates shape of bias when bias is from bottom.
    num_axes: int
        A int num indicates shape of bias when bias is from blob.
    bias_from_blob:
        A bool value indicates bias is from blob or bottom.
    kernel_name : str
        kernel name, default value is "bias"

    Returns
    -------
    output tensor
    """

    dtype_x = x.dtype
    dtype_bias = bias.dtype

    is_cast = False
    #product_version = cceconf.CceProductParams().cce_product

    #if product_version not in Version.HISI:
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vadd","float32"):
    #if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ("Hi3796CV300ES"):
        if dtype_x == "float16":
            is_cast = True
            x = te.lang.cce.cast_to(x, 'float32')
        if dtype_bias == "float16":
            bias = te.lang.cce.cast_to(bias, 'float32')

    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    bias_broad = te.lang.cce.broadcast(bias, shape_x)
    res = te.lang.cce.vadd(x, bias_broad)

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype_x)

    return res


# pylint: disable=too-many-locals
@util.check_input_type(dict, dict, dict, int, int, bool, str)
def bias(x, bias, y, axis=1, num_axes=1, bias_from_blob=True,
         kernel_name="bias"):
    """
    algorithm: Bias
    y = x + bias

    Parameters
    ----------
    x : dict
        dict of input, A Tensor for input data.
    bias : dict
        dict of bias,
        A Tensor for bias, to shift to the input data.
    y : dict
        dict of output,
        A Tensor for y, should be same shape and type as x.
    axis : int
        A int num indicates shape of bias when bias is from bottom.
    num_axes: int
        A int num indicates shape of bias when bias is from blob.
    bias_from_blob:
        A bool value indicates bias is from blob or bottom.
    kernel_name : str
        kernel name, default value is "bias"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_bias = bias.get("shape")
    dtype_x = x.get("dtype")
    dtype_bias = bias.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    util.check_shape_rule(shape_bias)
    util.check_tensor_shape_size(shape_bias)
    _check_dtype(dtype_x.lower(), dtype_bias.lower())

    shape_x_ori = x.get("ori_shape")
    length_x_ori = len(shape_x_ori)

    shape_bias_new = []

    if length_x_ori == 4:
        param_bias_check(shape_x, shape_bias)
        shape_bias_new = get_param_bias_shape(shape_x, shape_bias)
    else:
        _check_shape_axis(shape_x, shape_bias, axis, num_axes, bias_from_blob)

        length_x = len(shape_x)
        if axis < 0:
            axis_ = length_x + axis
        else:
            axis_ = axis

        shape_bias_new = get_shape(shape_x, shape_bias, axis_, num_axes, bias_from_blob)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    bias_input = tvm.placeholder(shape_bias_new, name="bias_input", dtype=dtype_bias.lower())

    res = bias_compute(x_input, bias_input, y, axis, num_axes, bias_from_blob, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = (x_input, bias_input, res)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
