#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except
in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

bn_training_reduce_grad
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


# pylint: disable=locally-disabled,invalid-name,too-many-arguments
# pylint: disable=locally-disabled,too-many-statements,unused-argument
# pylint: disable=locally-disabled,too-many-locals
def op_select_format(grads, x, diff_scale, diff_offset, scale,
                     batch_mean, batch_variance, y, epsilon,
                     kernel_name="bn_training_reduce_grad"):
    """
    select format dynamically
    """
    format_grads = grads.get("ori_format").upper()
    origin_shape = grads.get("ori_shape")

    # can support ND + ND
    if format_grads == "NCHW" and len(origin_shape) == 4 \
            and origin_shape[0] == 1 and origin_shape[2] == 1:
        input0 = gen_param(classify="input0", name="grads",
                           datatype="float16,float,float16,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input1 = gen_param(classify="input1", name="x",
                           datatype="float16,float,float16,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input2 = gen_param(classify="input2", name="diff_scale",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input3 = gen_param(classify="input3", name="diff_offset",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input4 = gen_param(classify="input4", name="scale",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input5 = gen_param(classify="input5", name="batch_mean",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input6 = gen_param(classify="input6", name="batch_variance",
                           datatype="float,float,float,float",
                           format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float,float16,float",
                            format="NCHW,NCHW,NC1HWC0,NC1HWC0")
    # support 5HD + 5HD
    else:
        input0 = gen_param(classify="input0", name="grads",
                           datatype="float16,float",
                           format="NC1HWC0,NC1HWC0")
        input1 = gen_param(classify="input1", name="x",
                           datatype="float16,float",
                           format="NC1HWC0,NC1HWC0")
        input2 = gen_param(classify="input2", name="diff_scale",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        input3 = gen_param(classify="input3", name="diff_offset",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        input4 = gen_param(classify="input4", name="scale",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        input5 = gen_param(classify="input5", name="batch_mean",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        input6 = gen_param(classify="input6", name="batch_variance",
                           datatype="float,float",
                           format="NC1HWC0,NC1HWC0")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float",
                            format="NC1HWC0,NC1HWC0")

    param_list = [input0, input1, input2, input3,
                  input4, input5, input6, output0]
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


@fusion_manager.register("bn_training_reduce_grad")
def bn_training_reduce_grad_compute(grads, x, diff_scale, diff_offset, scale,
                                    batch_mean, batch_variance, y, epsilon,
                                    kernel_name="bn_training_reduce_grad"):
    """
    Compute for batch_norm_train_reduce_grad
    y:(grads*scale*np.power((batch_variance + epsilon), (-0.5)))+
      np.sum(grads*scale*(-0.5)*x_norm*np.power((batch_variance+epsilon),(-1))))
      *(2/m)+np.sum(grads*scale*(-1)*
      np.power((batch_variance+epsilon),(-0.5)))*(1/m)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads.
        Must be one of the following type: `float16`, `float32`.
    x: TVM tensor 5D
        the placeholder of x.
        Must be one of the following type: `float32`, 'float16.
    diff_scale: TVM tensor 5D
        the placeholder of diff_scale.
        Must be one of the following type: `float32`.
    diff_offset: TVM tensor 5D
         the placeholder of diff_offset.
         Must be one of the following types: `float32`.
    scale: TVM tensor 5D
        the placeholder of scale.
        Must be one of the following types: `float32`.
    batch_mean: dict 5D
        the placeholder of batch_mean.
        Must be one of the following types: `float32`.
    batch_variance: dict 5D
        the placeholder of batch_variance.
        Must be one of the following types: `float32`.
    y: dict
        dict of y, include keys(shape and dtype).
    epsilon: float
        A small float number added to the variance of x.

    kernel_name: str
        kernel name, default value is "bn_training_reduce_grad"

    Returns
    -------
    res: TVM tensor
    """
    shape_grads = te.lang.cce.util.shape_to_list(grads.shape)
    num = shape_grads[0] * shape_grads[2] * shape_grads[3]
    num_rec = 1.0 / num
    is_cast = False
    if grads.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        is_cast = True
        grads = te.lang.cce.cast_to(grads, "float32")

    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        x = te.lang.cce.cast_to(x, "float32")

    data_sqrt = te.lang.cce.vsqrt(te.lang.cce.vadds(batch_variance, epsilon))
    scale_inv = te.lang.cce.vmuls(diff_scale, num_rec)
    scale_inv_reverse = te.lang.cce.vmuls(diff_scale, (-1.0)*num_rec)
    offset_inv_reverse = te.lang.cce.vmuls(diff_offset, (-1.0)*num_rec)

    multiplier = te.lang.cce.vdiv(scale_inv_reverse, data_sqrt)
    addend_div = te.lang.cce.vdiv(batch_mean, data_sqrt)
    addend_mul = te.lang.cce.vmul(addend_div, scale_inv)
    addend = te.lang.cce.vadd(addend_mul, offset_inv_reverse)

    multiplier_broadcast = te.lang.cce.broadcast(multiplier, shape_grads)
    addend_broadcast = te.lang.cce.broadcast(addend, shape_grads)

    coef_mul = te.lang.cce.vmul(multiplier_broadcast, x)
    coef_add = te.lang.cce.vadd(grads, coef_mul)
    coef = te.lang.cce.vadd(coef_add, addend_broadcast)

    mul_scale = te.lang.cce.vdiv(scale, data_sqrt)
    mul_scale_broadcast = te.lang.cce.broadcast(mul_scale, shape_grads)

    res = te.lang.cce.vmul(coef, mul_scale_broadcast)

    if is_cast:
        res = te.lang.cce.cast_to(res, "float16")
    return res


def _check_shape(shape_grads, shape_diff_scale):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_grads: list or tuple
        input grads's data shape
    shape_diff_scale: list or tuple
        input diff_scale's data shape
    Returns
    -------
    None
    """
    util.check_shape_rule(shape_grads, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_grads)

    util.check_shape_rule(shape_diff_scale, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_diff_scale)

    dim_c1 = shape_grads[1]
    dim_c0 = shape_grads[4]

    if len(shape_grads) != 5 or len(shape_diff_scale) != 5:
        raise RuntimeError(
            "This operator can only support 5D, "
            "but some input's shape length is not 5")
    if dim_c0 != 16:
        raise RuntimeError("shape_grads last dim must be 16")

    if shape_diff_scale[0] != 1 or shape_diff_scale[2] != 1 \
            or shape_diff_scale[3] != 1:
        raise RuntimeError(
            "Dimensions except Dimension C must be one for shape_diff_scale")
    if shape_diff_scale[1] != dim_c1 or shape_diff_scale[4] != dim_c0:
        raise RuntimeError(
            "Dimension C must be equal")


@util.check_input_type(dict, dict, dict, dict, dict,
                       dict, dict, dict, float, str)
def bn_training_reduce_grad(grads, x, diff_scale, diff_offset, scale,
                            batch_mean, batch_variance, y, epsilon=0.0001,
                            kernel_name="bn_training_reduce_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_training_reduce_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
        source data type, support "float32", "float16".
    x: dict
        dict of s, A 5D Tensor for input x.
        source data type, support "float32", "float16".
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for input diff_scale.
        The output of bn_training_update_grad.
        source data type, support "float32".
    diff_offset: dict
        dict of diff_offset, A 5HD Tensor for input diff_offset.
        The output of bn_training_update_grad.
        source data type, support "float32".
    scale: dict
        dict of scale, A 5HD Tensor for input scale.
        source data type, support "float32".
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
        source data type, support "float32".
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
        source data type, support "float32".
    y: dict
        dict of output, A `Tensor`. Has the same type as `grads`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce_grad"

    Returns
    -------
    None
    """

    shape_grads = grads.get("shape")
    shape_x = x.get("shape")
    shape_diff_scale = diff_scale.get("shape")
    shape_diff_offset = diff_offset.get("shape")
    shape_scale = scale.get("shape")
    shape_batch_mean = batch_mean.get("shape")
    shape_batch_variance = batch_variance.get("shape")
    util.compare_tensor_dict_key(grads, x, "shape")

    dtype_grads = grads.get("dtype")
    dtype_x = x.get("dtype")
    dtype_diff_scale = diff_scale.get("dtype")
    dtype_diff_offset = diff_offset.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_batch_mean = batch_mean.get("dtype")
    dtype_batch_variance = batch_variance.get("dtype")

    input_grads_dtype = dtype_grads.lower()
    x_dtype = dtype_x.lower()
    diff_scale_dtype = dtype_diff_scale.lower()
    diff_offset_dtype = dtype_diff_offset.lower()
    scale_dtype = dtype_scale.lower()
    batch_mean_dtype = dtype_batch_mean.lower()
    batch_variance_dtype = dtype_batch_variance.lower()

    util.check_dtype_rule(input_grads_dtype, ("float32", "float16"))
    util.check_dtype_rule(x_dtype, ("float32", "float16"))
    util.check_dtype_rule(diff_scale_dtype, ("float32",))
    util.check_dtype_rule(diff_offset_dtype, ("float32",))
    util.check_dtype_rule(scale_dtype, ("float32",))
    util.check_dtype_rule(batch_mean_dtype, ("float32",))
    util.check_dtype_rule(batch_variance_dtype, ("float32",))

    util.compare_tensor_dict_key(diff_scale, diff_offset, "shape")
    util.compare_tensor_dict_key(diff_scale, scale, "shape")
    util.compare_tensor_dict_key(diff_scale, batch_mean, "shape")
    util.compare_tensor_dict_key(diff_scale, batch_variance, "shape")
    util.compare_tensor_dict_key(grads, x, "shape")

    data_format = grads.get("format").upper()
    ori_format = grads.get("ori_format").upper()
    _check_format_nd(data_format, ori_format)

    if data_format == "NC1HWC0":
        _check_shape(shape_grads, shape_diff_scale)
    else:
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_x[1]
        shape_diff_scale = shape_list
        shape_diff_offset = shape_list
        shape_scale = shape_list
        shape_batch_mean = shape_list
        shape_batch_variance = shape_list

    grads_input = tvm.placeholder(shape_grads, name="grads_input",
                                  dtype=input_grads_dtype)
    x_input = tvm.placeholder(shape_x, name="x_input", dtype=x_dtype)
    diff_scale_input = tvm.placeholder(shape_diff_scale,
                                       name="diff_scale_input",
                                       dtype=diff_scale_dtype)
    diff_offset_input = tvm.placeholder(shape_diff_offset,
                                        name="diff_offset_input",
                                        dtype=diff_offset_dtype)
    scale_input = tvm.placeholder(shape_scale, name="scale_input",
                                  dtype=scale_dtype)
    batch_mean_input = tvm.placeholder(shape_batch_mean,
                                       name="batch_mean_input",
                                       dtype=batch_mean_dtype)
    batch_variance_input = tvm.placeholder(shape_batch_variance,
                                           name="batch_variance_input",
                                           dtype=batch_variance_dtype)

    res = bn_training_reduce_grad_compute(grads_input, x_input,
                                          diff_scale_input, diff_offset_input,
                                          scale_input, batch_mean_input,
                                          batch_variance_input, y, epsilon,
                                          kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    tensor_list = [grads_input, x_input, diff_scale_input, diff_offset_input,
                   scale_input, batch_mean_input, batch_variance_input, res]
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
