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

layernorm
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648

SIZE_SIXTEEN = 16


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
def _division_sixteen(shape):

    if len(shape) < 2:
        if shape[-1] == 0:
            raise RuntimeError("value of shape is illegal")
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        raise RuntimeError("value of shape is illegal")

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True
    return False


def op_select_format(input_x, input_gamma, input_beta,
                     output_y, output_mean, output_variance,
                     begin_norm_axis, begin_params_axis,
                     kernel_name="layer_norm"):
    """
    select format dynamically
    """
    shape_x = input_x.get("ori_shape")
    shape_x = util.scalar2tensor_one(shape_x)
    shape_gamma = input_gamma.get("ori_shape")
    shape_gamma = util.scalar2tensor_one(shape_gamma)

    # can not support Nz + ND
    # while len(shape_gamma) >= 2 and  _division_sixteen(shape_x) = False
    if begin_params_axis == 0:
        if len(shape_gamma) >= 2 or (not _division_sixteen(shape_x)):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16,float16,float16,"
                                        "float,float,float,float",
                               format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            input1 = gen_param(classify="input1", name="gamma",
                               datatype="float16,float16,float16,float16,float,"
                                        "float,float,float",
                               format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            input2 = gen_param(classify="input2", name="beta",
                               datatype="float16,float16,float16,float16,float,"
                                        "float,float,float",
                               format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16,float16,float16,float,"
                                         "float,float,float",
                                format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            output1 = gen_param(classify="output1", name="mean",
                                datatype="float16,float16,float16,float16,float,"
                                         "float,float,float",
                                format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            output2 = gen_param(classify="output2", name="variance",
                                datatype="float16,float16,float16,float16,float,"
                                         "float,float,float",
                                format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float16,float16,"
                                        "float16,float,float,float,float",
                               format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NC1HWC0,NHWC,"
                                      "ND,NCHW,NC1HWC0,NHWC,ND")

            input1 = gen_param(classify="input1", name="gamma",
                               datatype="float16,float,float16,float16,float16,"
                                        "float16,float,float,float,float",
                               format="ND,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,"
                                      "NHWC,ND")

            input2 = gen_param(classify="input2", name="beta",
                               datatype="float16,float,float16,float16,float16,"
                                        "float16,float,float,float,float",
                               format="ND,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,"
                                      "NHWC,ND")

            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,float16,float16,float16,"
                                         "float16,float,float,float,float",
                                format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NC1HWC0,NHWC,ND,"
                                       "NCHW,NC1HWC0,NHWC,ND")

            output1 = gen_param(classify="output1", name="mean",
                                datatype="float16,float,float16,float16,float16,"
                                         "float16,float,float,float,float",
                                format="ND,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,"
                                       "NHWC,ND")

            output2 = gen_param(classify="output2", name="variance",
                                datatype="float16,float,float16,float16,float16,"
                                         "float16,float,float,float,float",
                                format="ND,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,"
                                       "NHWC,ND")
    else:
        if len(shape_gamma) >= 2 or (not _division_sixteen(shape_x)):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16,float16,"
                                        "float,float,float",
                               format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            input1 = gen_param(classify="input1", name="gamma",
                               datatype="float16,float16,float16,"
                                        "float,float,float",
                               format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            input2 = gen_param(classify="input2", name="beta",
                               datatype="float16,float16,float16,"
                                        "float,float,float",
                               format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16,float16,"
                                         "float,float,float",
                                format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            output1 = gen_param(classify="output1", name="mean",
                                datatype="float16,float16,float16,"
                                         "float,float,float",
                                format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            output2 = gen_param(classify="output2", name="variance",
                                datatype="float16,float16,float16,"
                                         "float,float,float",
                                format="NCHW,NHWC,ND,NCHW,NHWC,ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float16,"
                                        "float16,float,float,float",
                               format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,"
                                      "ND,NCHW,NHWC,ND")

            input1 = gen_param(classify="input1", name="gamma",
                               datatype="float16,float,float16,float16,"
                                        "float16,float,float,float",
                               format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                      "NHWC,ND")

            input2 = gen_param(classify="input2", name="beta",
                               datatype="float16,float,float16,float16,"
                                        "float16,float,float,float",
                               format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                      "NHWC,ND")

            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,float16,float16,"
                                         "float16,float,float,float",
                                format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,ND,"
                                       "NCHW,NHWC,ND")

            output1 = gen_param(classify="output1", name="mean",
                                datatype="float16,float,float16,float16,"
                                         "float16,float,float,float",
                                format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                       "NHWC,ND")

            output2 = gen_param(classify="output2", name="variance",
                                datatype="float16,float,float16,float16,"
                                         "float16,float,float,float",
                                format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                       "NHWC,ND")

    param_list = [input0, input1, input2, output0, output1, output2]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def to_frac_z_axis(ori_shape, ori_axis):
    """
    judge the format is fractal NZ

    Parameters
    ----------
    ori_shape: list or tuple
        original shape of input
    ori_axis: list or tuple
        original axis of original shape to operate

    Returns
    -------
    output: list
        axis of the fractal Nz shape
    """

    frac_z_axis = list(ori_axis)
    shape_len = len(ori_shape)
    axis_count = len(frac_z_axis)
    axis_negative_1 = shape_len - 1
    axis_negative_2 = shape_len - 2
    for i in range(axis_count):
        axis_index = (frac_z_axis[i] + shape_len) % shape_len
        if axis_index == axis_negative_1:
            if frac_z_axis[i] > shape_len - 2:
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 1)
            else:
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 2)
        elif axis_index == axis_negative_2:
            frac_z_axis[i] = axis_index + 1
            frac_z_axis.append(axis_index + 2)
        else:
            frac_z_axis[i] = axis_index
    return frac_z_axis


def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = te.lang.cce.util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = te.lang.cce.broadcast(tensor, temp_shape)
    tensor = te.lang.cce.broadcast(tensor, shape)
    return tensor


def layer_norm_compute_nz(input_x, input_gamma, input_beta,
                          output_y, output_mean, output_variance,
                          begin_norm_axis, begin_params_axis,
                          ori_shape, epsilon, kernel_name="layer_norm"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_data: dict
        shape and dtype of output
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "cce_layernorm"

    Returns
    -------
    res_tuple: tuple
        (mean, variance, result)
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    cast_dtype = "float16"
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vexp", "float32"):
        cast_dtype = "float32"
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_gamma = te.lang.cce.cast_to(input_gamma, "float32")
        input_beta = te.lang.cce.cast_to(input_beta, "float32")

    # Calculate the scaling ratio of the average
    reduce_elts = 1.0
    index_list = tuple(index for index, _ in enumerate(ori_shape))
    reduce_axis = index_list[begin_norm_axis:]
    for i in reduce_axis:
        reduce_elts *= ori_shape[i]
    reduce_axis = to_frac_z_axis(ori_shape, reduce_axis)
    mean_cof = reduce_elts ** (-1)
    # DSL description of the mean calculation process
    mean_muls = te.lang.cce.vmuls(input_x, mean_cof)
    mean = te.lang.cce.sum(mean_muls, axis=reduce_axis, keepdims=True)

    # DSL description of the variance calculation process
    mean_variance_broadcast = _broadcast_nz(mean, shape_x)
    variance_sub = te.lang.cce.vsub(input_x, mean_variance_broadcast)
    variance_mul = te.lang.cce.vmul(variance_sub, variance_sub)
    variance_muls = te.lang.cce.vmuls(variance_mul, mean_cof)
    variance = te.lang.cce.sum(variance_muls, axis=reduce_axis, keepdims=True)

    # DSL description of the normalize calculation process
    mean_normalize_broadcast = _broadcast_nz(mean, shape_x)
    normalize_sub = te.lang.cce.vsub(input_x, mean_normalize_broadcast)
    epsilon = tvm.const(epsilon, dtype=cast_dtype)
    variance_normalize_broadcast = _broadcast_nz(variance, shape_x)
    normalize_add = te.lang.cce.vadds(variance_normalize_broadcast, epsilon)
    normalize_log = te.lang.cce.vlog(normalize_add)
    normalize_log_mul = \
        te.lang.cce.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
    normalize_exp = te.lang.cce.vexp(normalize_log_mul)
    normalize_mul = te.lang.cce.vmul(normalize_sub, normalize_exp)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = te.lang.cce.vmul(input_gamma, normalize_mul)
        res = te.lang.cce.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = _broadcast_nz(input_gamma, shape_x)
        beta_broadcast = _broadcast_nz(input_beta, shape_x)
        scale_mul = te.lang.cce.vmul(gamma_broadcast, normalize_mul)
        res = te.lang.cce.vadd(scale_mul, beta_broadcast)

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vexp", "float32"):
        mean = te.lang.cce.cast_to(mean, "float16")
        variance = te.lang.cce.cast_to(variance, "float16")
        res = te.lang.cce.cast_to(res, "float16")

    return mean, variance, res


@fusion_manager.register("layer_norm")
def layer_norm_compute(input_x, input_gamma, input_beta,
                       output_y, output_mean, output_variance,
                       begin_norm_axis, begin_params_axis,
                       epsilon, kernel_name="layer_norm"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_data: dict
        shape and dtype of output
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "cce_layernorm"

    Returns
    -------
    res_tuple: tuple
        (mean, variance, result)
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    cast_dtype = "float16"
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vexp", "float32"):
        cast_dtype = "float32"
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_gamma = te.lang.cce.cast_to(input_gamma, "float32")
        input_beta = te.lang.cce.cast_to(input_beta, "float32")

    # Calculate the scaling ratio of the average
    index_list = tuple(index for index, _ in enumerate(shape_x))
    reduce_axis = index_list[begin_norm_axis:]

    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    mean_cof = reduce_elts ** (-1)

    # DSL description of the mean calculation process
    mean_muls = te.lang.cce.vmuls(input_x, mean_cof)
    mean = te.lang.cce.sum(mean_muls, axis=reduce_axis, keepdims=True)

    # DSL description of the variance calculation process
    mean_variance_broadcast = te.lang.cce.broadcast(mean, shape_x)
    variance_sub = te.lang.cce.vsub(input_x, mean_variance_broadcast)
    variance_mul = te.lang.cce.vmul(variance_sub, variance_sub)
    variance_muls = te.lang.cce.vmuls(variance_mul, mean_cof)
    variance = te.lang.cce.sum(variance_muls, axis=reduce_axis, keepdims=True)

    # DSL description of the normalize calculation process
    mean_normalize_broadcast = te.lang.cce.broadcast(mean, shape_x)
    normalize_sub = te.lang.cce.vsub(input_x, mean_normalize_broadcast)
    epsilon = tvm.const(epsilon, dtype=cast_dtype)
    variance_normalize_broadcast = te.lang.cce.broadcast(variance, shape_x)
    normalize_add = te.lang.cce.vadds(variance_normalize_broadcast, epsilon)
    normalize_log = te.lang.cce.vlog(normalize_add)
    normalize_log_mul = \
        te.lang.cce.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
    normalize_exp = te.lang.cce.vexp(normalize_log_mul)
    normalize_mul = te.lang.cce.vmul(normalize_sub, normalize_exp)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = te.lang.cce.vmul(input_gamma, normalize_mul)
        res = te.lang.cce.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = te.lang.cce.broadcast(input_gamma, shape_x)
        beta_broadcast = te.lang.cce.broadcast(input_beta, shape_x)
        scale_mul = te.lang.cce.vmul(gamma_broadcast, normalize_mul)
        res = te.lang.cce.vadd(scale_mul, beta_broadcast)

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vexp", "float32"):
        mean = te.lang.cce.cast_to(mean, "float16")
        variance = te.lang.cce.cast_to(variance, "float16")
        res = te.lang.cce.cast_to(res, "float16")

    return mean, variance, res


@util.check_input_type(dict, dict, dict, dict, dict, dict, int, int, float, str)
def layer_norm(input_x, input_gamma, input_beta,
               output_y, output_mean, output_variance,
               begin_norm_axis, begin_params_axis,
               epsilon = 1e-12, kernel_name="layer_norm"):
    """
    layernorm operator interface implementation
    calculating: x, gamma, beta
        mean  = np.mean(x, reduce_axis, keepdims=True)
        variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True)
        result = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

    Parameters
    ----------
    input_x : dict
        shape and dtype of input x, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    input_beta: dict
        shape and dtype of input beta, only support float16, float32
    output_y: dict
        shape and dtype of output, only support float16, float32
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layernorm"

    Returns
    -------
    None
    """


    shape_x = list(input_x.get("shape"))
    input_gamma_shape = input_gamma.get("shape")
    input_beta_shape = input_beta.get("shape")
    ori_shape_x = list(input_x.get("ori_shape"))
    input_format = input_x.get("format").upper()
    input_gamma_format = input_gamma.get("format").upper()
    input_beta_format = input_beta.get("format").upper()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_gamma_shape)
    util.check_tensor_shape_size(input_gamma_shape)
    util.check_shape_rule(input_beta_shape)
    util.check_tensor_shape_size(input_beta_shape)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(shape_x)

    check_list = ("float16", "float32")
    dtype = input_x.get("dtype").lower()
    dtype_gamma = input_gamma.get("dtype").lower()
    dtype_beta = input_gamma.get("dtype").lower()
    util.check_dtype_rule(dtype, check_list)
    util.check_dtype_rule(dtype_gamma, check_list)
    util.check_dtype_rule(dtype_beta, check_list)

    shape_gamma = list(input_gamma.get("shape"))
    shape_beta = list(input_beta.get("shape"))

    if input_format == "FRACTAL_NZ":
        begin_norm_axis = util.axis_check(len(ori_shape_x), begin_norm_axis)
        begin_params_axis = util.axis_check(len(ori_shape_x), begin_params_axis)

        if input_gamma_format == "FRACTAL_NZ" or \
                input_beta_format == "FRACTAL_NZ":
            raise RuntimeError("gamma and beta not support Nz in bert")
        if shape_gamma != shape_beta:
            raise RuntimeError("gamma and beta's must be same.")
        if ori_shape_x[begin_params_axis:] != shape_gamma:
            raise RuntimeError("x or gamma or begin_params_axis is wrong.")
        if len(shape_gamma) > 1:
            raise RuntimeError("shape of gamma or beta only support 1D in bert")

        # make shape_x,shape_gamma,shape_beta dim same
        if begin_params_axis != 0:
            for i in range(begin_params_axis):
                shape_gamma.insert(i, 1)
        shape_gamma[-2] = shape_x[-4]
        shape_gamma[-1] = 1
        shape_gamma.append(1)
        shape_gamma.append(shape_x[-1])
        if begin_params_axis > len(ori_shape_x) - 2:
            shape_x[-3:] = [shape_x[-3]*shape_x[-2], shape_x[-1]]
            shape_gamma[-3:] = [shape_gamma[-3]*shape_gamma[-2], shape_gamma[-1]]
        shape_beta = shape_gamma
    else:
        begin_norm_axis = util.axis_check(len(shape_x), begin_norm_axis)
        begin_params_axis = util.axis_check(len(shape_x), begin_params_axis)

        if shape_gamma != shape_beta:
            raise RuntimeError("gamma and beta's must be same.")
        if shape_x[begin_params_axis:] != shape_gamma:
            raise RuntimeError("x or gamma or begin_params_axis is wrong.")

        # make shape_x,shape_gamma,shape_beta dim same
        if begin_params_axis != 0:
            for i in range(begin_params_axis):
                shape_gamma.insert(i, 1)
                shape_beta.insert(i, 1)

    data_x = tvm.placeholder(shape_x, name="x", dtype=dtype)
    data_gamma = tvm.placeholder(shape_gamma, name="gamma", dtype=dtype)
    data_beta = tvm.placeholder(shape_beta, name="beta", dtype=dtype)

    if input_format == "FRACTAL_NZ":

        mean, variance, res = \
            layer_norm_compute_nz(data_x, data_gamma, data_beta,
                                  output_y, output_mean, output_variance,
                                  begin_norm_axis, begin_params_axis,
                                  ori_shape_x, epsilon, kernel_name)
    else:

        mean, variance, res = \
            layer_norm_compute(data_x, data_gamma, data_beta,
                               output_y, output_mean,
                               output_variance,
                               begin_norm_axis, begin_params_axis,
                               epsilon, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule([res, mean, variance])

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, data_gamma,
                              data_beta, res, mean, variance]}

    te.lang.cce.cce_build_code(sch, config)
