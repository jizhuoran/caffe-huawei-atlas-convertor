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

bn_training_update_grad
"""
from __future__ import division

from te import tvm
from te import platform as tbe_platform
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

MAX_SHAPE_NUM = 10000000
SCALAR_ONE = 1


# pylint: disable=locally-disabled,unused-argument,invalid-name
# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
def op_select_format(grads, x, batch_mean, batch_variance,
                     diff_scale, diff_offset, epsilon,
                     kernel_name="bn_training_update_grad"):
    """
    To set supporting formats for reference according to format.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
    x: dict
        dict of x, A 5D Tensor for input x.
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_training_update_grad"

    Returns
    -------
    None
    """
    format_x = x.get("ori_format").upper()
    origin_shape = x.get("ori_shape")

    if format_x == "NCHW" and len(origin_shape) == 4 \
            and origin_shape[0] == 1 and origin_shape[2] == 1:

        input0 = gen_param(classify="input0", name="grads",
                           datatype="float16,float,float16,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input1 = gen_param(classify="input1", name="x",
                           datatype="float16,float,float16,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input2 = gen_param(classify="input2", name="batch_mean",
                           datatype="float,float,float,float",
                           format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        input3 = gen_param(classify="input3", name="batch_variance",
                           datatype="float,float,float,float",
                           format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        output0 = gen_param(classify="output0", name="diff_scale",
                            datatype="float,float,float,float",
                            format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        output1 = gen_param(classify="output1", name="diff_offset",
                            datatype="float,float,float,float",
                            format="NCHW, NCHW,NC1HWC0,NC1HWC0")
    else:
        input0 = gen_param(classify="input0", name="grads",
                           datatype="float16,float",
                           format="NC1HWC0,NC1HWC0")
        input1 = gen_param(classify="input1", name="x",
                           datatype="float16,float",
                           format="NC1HWC0,NC1HWC0")
        input2 = gen_param(classify="input2", name="batch_mean",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        input3 = gen_param(classify="input3", name="batch_variance",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        output0 = gen_param(classify="output0", name="diff_scale",
                            datatype="float,float",
                            format="NC1HWC0,NC1HWC0")
        output1 = gen_param(classify="output1", name="diff_offset",
                            datatype="float,float",
                            format="NC1HWC0,NC1HWC0")

    param_list = [input0, input1, input2, input3, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _check_format_nd(data_format, origin_foramt):
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


@fusion_manager.register("bn_training_update_grad")
def bn_training_update_grad_compute(grads, x, batch_mean, batch_variance,
                                    diff_scale, diff_offset, epsilon,
                                    kernel_name="bn_training_update_grad"):
    """
    Compute for bn_training_update_grad_compute
    x_norm:(x-input_reserve_space_1)*
            np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_scale:np.sum(y*(x-input_reserve_space_1)*
                         np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_offset: np.sum(y)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads. Must be one of the following
        type: `float16`, `float32`.
    x: TVM tensor 5D
        the placeholder of x. Must be one of the following
        type: `float16`, `float32`.
    batch_mean: TVM tensor 5D
        the placeholder of batch_mean. Must be one of the following
        type: `float32`.
    batch_variance: TVM tensor 5D
        the placeholder of batch_variance. Must be one of the following
        type: `float32`.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_training_update_grad"

    Returns
    -------
    res_list: list
       [diff_scale, diff_offset].
   """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    axis = [0, 2, 3]

    if grads.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        grads = te.lang.cce.cast_to(grads, "float32")
    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        x = te.lang.cce.cast_to(x, "float32")
    batch_mean_inverse = \
        te.lang.cce.vmuls(batch_mean,
                          tvm.const(-1, dtype=batch_mean.dtype))
    input_mean = te.lang.cce.broadcast(batch_mean_inverse, shape_x)
    x_sub = te.lang.cce.vadd(x, input_mean)

    data_adds = te.lang.cce.vadds(batch_variance, epsilon)
    data_rsqrt = te.lang.cce.vsqrt(data_adds)
    shape_var = te.lang.cce.util.shape_to_list(batch_variance.shape)
    data_cast = te.lang.cce.broadcast(tvm.const(SCALAR_ONE, "float32"),
                                      shape_var)
    data_rsqrts = te.lang.cce.vdiv(data_cast, data_rsqrt)
    rsqrts_broadcast = te.lang.cce.broadcast(data_rsqrts, shape_x)
    x_norm = te.lang.cce.vmul(x_sub, rsqrts_broadcast)

    scale_mul = te.lang.cce.vmul(grads, x_norm)

    diff_scale, diff_offset = te.lang.cce.tuple_sum([scale_mul, grads],
                                                    axis, True)
    res_list = [diff_scale, diff_offset]
    return res_list


def _check_shape(shape_grads, shape_x, shape_batch_mean, shape_batch_variance):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_grads: list or tuple
        input grads's data shape
    shape_x: list or tuple
        input x's data shape
    shape_batch_mean: list or tuple
        input batch_mean's data shape
    shape_batch_variance: list or tuple
        input batch_variance's data shape
    Returns
    -------
    None
    """
    util.check_shape_rule(shape_grads, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_grads)

    util.check_shape_rule(shape_x, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_x)

    util.check_shape_rule(shape_batch_mean, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_batch_mean)

    util.check_shape_rule(shape_batch_variance, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_batch_variance)

    dim_c1 = shape_grads[1]
    dim_c0 = shape_grads[4]

    if len(shape_grads) != 5 or len(shape_x) != 5:
        raise RuntimeError(
            "This operator can only support 5D")
    if dim_c0 != 16:
        raise RuntimeError("shape_grads last dim must be 16")
    if len(shape_batch_mean) != 5 or len(shape_batch_variance) != 5:
        raise RuntimeError(
            "This operator can only support 5D")

    if shape_batch_mean[0] != 1 or shape_batch_mean[2] != 1 \
            or shape_batch_mean[3] != 1:
        raise RuntimeError(
            "Dimensions except Dimension C must be one for shape_batch_mean")
    if shape_batch_mean[1] != dim_c1 or shape_batch_mean[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal")


@util.check_input_type(dict, dict, dict, dict, dict, dict, float, str)
def bn_training_update_grad(grads, x, batch_mean, batch_variance,
                            diff_scale, diff_offset, epsilon=0.0001,
                            kernel_name="bn_training_update_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_training_update_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
    x: dict
        dict of x, A 5D Tensor for input x.
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_training_update_grad"

    Returns
    -------
    None
    """

    shape_grads = grads.get("shape")
    shape_x = x.get("shape")
    shape_batch_mean = batch_mean.get("shape")
    shape_batch_variance = batch_variance.get("shape")

    dtype_grads = grads.get("dtype")
    dtype_x = x.get("dtype")
    dtype_batch_mean = batch_mean.get("dtype")
    dtype_batch_variance = batch_variance.get("dtype")

    input_grads_dtype = dtype_grads.lower()
    input_x_dtype = dtype_x.lower()
    batch_mean_dtype = dtype_batch_mean.lower()
    batch_variance_dtype = dtype_batch_variance.lower()

    util.check_dtype_rule(input_grads_dtype, ("float32", "float16"))
    util.check_dtype_rule(input_x_dtype, ("float32", "float16"))
    util.check_dtype_rule(batch_mean_dtype, ("float32",))
    util.check_dtype_rule(batch_variance_dtype, ("float32",))
    util.compare_tensor_dict_key(grads, x, "dtype")

    data_format = grads.get("format")
    ori_format = grads.get("ori_format")
    _check_format_nd(data_format, ori_format)

    if data_format == "NC1HWC0":
        _check_shape(shape_grads, shape_x,
                     shape_batch_mean, shape_batch_variance)
    else:
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_x[1]
        shape_batch_mean = shape_list
        shape_batch_variance = shape_list

    util.compare_tensor_dict_key(grads, x, "shape")
    util.compare_tensor_dict_key(batch_mean, batch_variance, "shape")

    grads_input = tvm.placeholder(shape_grads, name="grads_input",
                                  dtype=input_grads_dtype)
    x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_x_dtype)
    batch_mean_input = tvm.placeholder(shape_batch_mean,
                                       name="batch_mean_input",
                                       dtype=batch_mean_dtype)
    batch_variance_input = tvm.placeholder(shape_batch_variance,
                                           name="batch_variance_input",
                                           dtype=batch_variance_dtype)

    res_list = bn_training_update_grad_compute(grads_input, x_input,
                                               batch_mean_input,
                                               batch_variance_input, diff_scale,
                                               diff_offset,
                                               epsilon, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res_list)
    tensor_list = [grads_input, x_input, batch_mean_input,
                   batch_variance_input] + list(res_list)
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
