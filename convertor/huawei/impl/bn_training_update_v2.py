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

bn_training_update_v2
"""

from __future__ import absolute_import
from __future__ import division
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
# pylint: disable=redefined-builtin,too-many-arguments
def op_select_format(x, sum, square_sum, scale, offset,
                     y, batch_mean, batch_variance, epsilon,
                     kernel_name="bn_training_update_v2"):
    """
    select format dynamically
    """
    origin_format = x.get("ori_format").upper()
    origin_shape = x.get("ori_shape")

    # can support Nz + ND
    if origin_format == "NCHW" and len(origin_shape) == 4 \
            and origin_shape[0] == 1 and origin_shape[2] == 1:
        input0 = gen_param(classify="input0", name="x",
                           datatype="float16,float,float16,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input1 = gen_param(classify="input1", name="sum",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input2 = gen_param(classify="input2", name="square_sum",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input3 = gen_param(classify="input3", name="scale",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input4 = gen_param(classify="input4", name="offset",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float,float16,float",
                            format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        output1 = gen_param(classify="output1", name="batch_mean",
                            datatype="float,float,float,float",
                            format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        output2 = gen_param(classify="output2", name="batch_variance",
                            datatype="float,float,float,float",
                            format="NCHW,NCHW,NC1HWC0,NC1HWC0")
    # support 5HD + 5HD
    else:
        input0 = gen_param(classify="input0", name="x",
                           datatype="float16,float",
                           format="NC1HWC0,NC1HWC0")
        input1 = gen_param(classify="input1", name="sum",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        input2 = gen_param(classify="input2", name="square_sum",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        input3 = gen_param(classify="input3", name="scale",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        input4 = gen_param(classify="input4", name="offset",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float",
                            format="NC1HWC0,NC1HWC0")
        output1 = gen_param(classify="output1", name="batch_mean",
                            datatype="float,float",
                            format="NC1HWC0,NC1HWC0")
        output2 = gen_param(classify="output2", name="batch_variance",
                            datatype="float,float",
                            format="NC1HWC0,NC1HWC0")

    param_list = [input0, input1, input2, input3,
                  input4, output0, output1, output2]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_format(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NCHW"):
        raise RuntimeError("The data format only supports NC1HWC0 and NCHW.")
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            raise RuntimeError("The origin format only supports "
                               "NCHW when format is NCHW")


# pylint: disable=locally-disabled,too-many-arguments
def _check_shape(shape_x, shape_sum, shape_square_sum,
                 shape_scale, shape_offset):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_sum: list or tuple
        sum's data shape
    shape_square_sum: list or tuple
        square_sum's data shape
    shape_scale: list or tuple
        scale's data shape
    shape_offset: list or tuple
        offset's data shape

    Returns
    -------
    None
    """
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)

    util.check_shape_rule(shape_sum)
    util.check_tensor_shape_size(shape_sum)

    util.check_shape_rule(shape_square_sum)
    util.check_tensor_shape_size(shape_square_sum)

    util.check_shape_rule(shape_scale)
    util.check_tensor_shape_size(shape_scale)

    util.check_shape_rule(shape_offset)
    util.check_tensor_shape_size(shape_offset)

    if len(shape_x) != 5 or len(shape_sum) != 5 \
            or len(shape_square_sum) != 5 or len(shape_scale) != 5 \
            or len(shape_offset) != 5:
        raise RuntimeError(
            "The data format is 5HD, "
            "but some input's shape length is not 5")

    dim_c1 = shape_x[1]
    dim_c0 = shape_x[4]

    if shape_sum[1] != dim_c1 or shape_sum[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and sum must be equal")
    if shape_square_sum[1] != dim_c1 or shape_square_sum[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and square_sum must be equal")
    if shape_scale[1] != dim_c1 or shape_scale[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and scale must be equal")
    if shape_offset[1] != dim_c1 or shape_offset[4] != dim_c0:
        raise RuntimeError(
            "Dimension C of x and offset must be equal")


# pylint: disable=locally-disabled,too-many-arguments
def _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_sum: str
        sum's data type
    dtype_square_sum: str
        square_sum's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type

    Returns
    -------
    None
    """
    util.check_dtype_rule(dtype_x.lower(), ("float16", "float32"))
    util.check_dtype_rule(dtype_sum.lower(), ("float32",))
    util.check_dtype_rule(dtype_square_sum.lower(), ("float32",))
    util.check_dtype_rule(dtype_scale.lower(), ("float32",))
    util.check_dtype_rule(dtype_offset.lower(), ("float32",))


# pylint: disable=locally-disabled,too-many-locals,unused-argument,invalid-name
# pylint: disable=redefined-builtin
@fusion_manager.register("bn_training_update_v2")
def bn_training_update_v2_compute(x, sum, square_sum, scale, offset,
                                  y, batch_mean, batch_variance, epsilon,
                                  kernel_name="bn_training_update_v2"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    factor: float
        A ratio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v2 compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    data_format = y.get("format")
    origin_format = y.get("ori_format")
    axis = list(range(len(shape_x)))

    if data_format == "NC1HWC0":
        axis = [0, 2, 3]
    if data_format == "NCHW":
        if origin_format == "NCHW":
            axis.pop(1)

    num = 1
    for cnt in axis:
        num *= shape_x[cnt]
    num_rec = 1.0/num

    # compute the saved mean of x
    save_mean_reduce = te.lang.cce.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = te.lang.cce.vmuls(square_sum, num_rec)
    variance_square = te.lang.cce.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = te.lang.cce.vsub(variance_div, variance_square)

    # compute the oefficient of y
    multiplier_add = te.lang.cce.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = te.lang.cce.vsqrt(multiplier_add)
    multiplier_div = te.lang.cce.vdiv(scale, multiplier_sqrt)
    multiplier = te.lang.cce.broadcast(multiplier_div, shape_x)

    addend_mul = te.lang.cce.vmul(multiplier_div, save_mean_reduce)
    addend_sub = te.lang.cce.vsub(offset, addend_mul)
    addend = te.lang.cce.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16":
        is_cast = True
        x = te.lang.cce.cast_to(x, "float32")

    res_y = te.lang.cce.vadd(te.lang.cce.vmul(multiplier, x), addend)
    if is_cast:
        res_y = te.lang.cce.cast_to(res_y, "float16")

    res = [res_y, save_mean_reduce, save_variance_reduce]

    return res


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@util.check_input_type(dict, dict, dict, dict, dict,
                       dict, dict, dict, float, str)
def bn_training_update_v2(x, sum, square_sum, scale, offset,
                          y, batch_mean, batch_variance, epsilon,
                          kernel_name="bn_training_update_v2"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A 5HD Tensor for mean.
    offset: dict
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    factor: float
        A retio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v2"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")

    dtype_x = x.get("dtype")
    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")

    data_format = x.get("format")
    origin_format = x.get("ori_format")

    _check_format(data_format, origin_format)

    if data_format == "NC1HWC0":
        _check_shape(shape_x, shape_sum, shape_square_sum,
                     shape_scale, shape_offset)
    else:
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_x[1]
        shape_sum = shape_list
        shape_square_sum = shape_list

    _check_dtype(dtype_x, dtype_sum, dtype_square_sum,
                 dtype_scale, dtype_offset)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input",
                                dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum,
                                       name="square_sum_input",
                                       dtype=dtype_square_sum.lower())
    scale_input = tvm.placeholder(shape_sum, name="scale_input",
                                  dtype=dtype_scale.lower())
    offset_input = tvm.placeholder(shape_sum, name="offset_input",
                                   dtype=dtype_offset.lower())

    res = bn_training_update_v2_compute(x_input, sum_input, square_sum_input,
                                        scale_input, offset_input, y,
                                        batch_mean, batch_variance,
                                        epsilon, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [x_input, sum_input, square_sum_input,
                   scale_input, offset_input, ] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
