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

layernorm_grad
"""
import operator

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi.cce import util
from topi import generic

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648

# Minimum positive number greater than 0
EPSLON = 1e-12


# pylint: disable=too-many-locals,too-many-arguments
# pylint: disable=locally-disabled,unused-argument
def _check_params(params_map):
    """
    check parameters including shape_dy, shape_x, shape_var,
    shape_mean, shape_gamma, dtype and kernel_name

    Parameters
    ----------
    params_map: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
        "shape_mean": shape_mean, "shape_gamma": shape_gamma,
        "dtype": dtype, "kernel_name": kernel_name}

    Returns
    -------
    None
    """
    util.check_kernel_name(params_map.get("kernel_name"))

    check_list = ("float16", "float32")
    util.check_dtype_rule(params_map.get("dtype"), check_list)

    _check_shape(params_map)


def _check_shape(params_map):
    """
    check parameters including shape_dy, shape_x, shape_var,
    shape_mean and shape_gamma

    Parameters
    ----------
    params_map: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
         "shape_mean": shape_mean, "shape_gamma": shape_gamma,
         "dtype": dtype, "kernel_name": kernel_name}

    Returns
    -------
    None
    """
    if operator.ne(tuple(params_map.get("shape_dy")),
                   tuple(params_map.get("shape_x"))):
        raise RuntimeError("dy and x must have the same shape")

    if operator.ne(tuple(params_map.get("shape_var")),
                   tuple(params_map.get("shape_mean"))):
        raise RuntimeError("variance and mean must have the same shape")

    shape_x = params_map.get("shape_x")
    shape_mean = params_map.get("shape_mean")
    shape_gamma = params_map.get("shape_gamma")

    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_mean)
    util.check_shape_rule(shape_gamma)

    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_mean, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_gamma, SHAPE_SIZE_LIMIT)

    _check_shape_mean(shape_x, shape_mean)
    _check_shape_gamma(shape_x, shape_gamma)


def _check_shape_mean(shape_x, shape_mean):
    """
    check if parameter shape_mean meets the requirements of function

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_mean: list or tuple
        shape of mean

    Returns
    -------
    None
    """
    if len(shape_x) != len(shape_mean):
        raise RuntimeError("length of shape_x and shape_mean must be the same")

    if shape_mean[-1] != 1:
        raise RuntimeError("value of shape_mean's last dim must be 1")

    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != mean:
            flag = i
            break

    if flag != -1:
        for i, mean in enumerate(shape_mean):
            if i < flag:
                continue
            if mean != 1:
                raise RuntimeError("value of shape_mean is wrong")


def _check_shape_gamma(shape_x, shape_gamma):
    """
    check if parameter shape_gamma meets the requirements of function

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    None
    """
    if len(shape_gamma) > len(shape_x):
        raise RuntimeError("length of shape_gamma can not be "
                           "longer than shape_x")

    for xtem, gamma in zip(reversed(shape_x), reversed(shape_gamma)):
        if xtem != gamma:
            raise RuntimeError("value of shape_gamma is wrong")


def _update_gamma_shape(shape_x, shape_gamma):
    """
    update shape_gamma for subsequent calculation

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    shape_gamma_new: tuple
        new shape_gamma after update
    params_axis: tuple
        the list of axis for gamma reduce_sum
    """
    params_axis_tmp = []
    if len(shape_x) != len(shape_gamma):
        sub = len(shape_x) - len(shape_gamma)
        shape_gamma = list(shape_gamma)
        for i in range(sub):
            shape_gamma.insert(0, 1)
            params_axis_tmp.append(i)

    shape_gamma_new = tuple(shape_gamma)
    params_axis = tuple(params_axis_tmp)

    return shape_gamma_new, params_axis


def _get_data_gm(shapes, dtype):
    """
    get placeholders of data_dy, data_x, data_variance, data_mean and data_gamma

    Parameters
    ----------
    shapes: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
         "shape_mean": shape_mean, "shape_gamma": shape_gamma}
    dtype: str
        the data type

    Returns
    -------
    data_gm: tuple
        (data_dy, data_x, data_variance, data_mean, data_gamma)
    """
    data_dy = tvm.placeholder(shapes.get("shape_dy"),
                              name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(shapes.get("shape_x"),
                             name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(shapes.get("shape_var"),
                                    name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(shapes.get("shape_mean"),
                                name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(shapes.get("shape_gamma"),
                                 name="data_gamma", dtype=dtype)

    data_gm = (data_dy, data_x, data_variance, data_mean, data_gamma)

    return data_gm


def _get_params(shape_x, shape_mean, shape_gamma):
    """
    compute parameters including param_axis, reduce_axis and mean_num

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_mean: list or tuple
        shape of mean
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    """
    param_axis = _update_gamma_shape(shape_x, shape_gamma)[1]

    reduce_axis_tmp = []
    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != mean:
            flag = i
            break
    if flag != -1:
        for i in range(flag, len(shape_x)):
            reduce_axis_tmp.append(i)
    else:
        reduce_axis_tmp.append(len(shape_x) - 1)
    reduce_axis = tuple(reduce_axis_tmp)

    mean_num = 1.0
    for i in reduce_axis:
        mean_num *= shape_x[i]

    params = {"param_axis": param_axis, "reduce_axis": reduce_axis,
              "mean_num": mean_num}

    return params


def _get_pd_xl(data, shape_x):
    """
    compute pd_xl according to data_dy, data_gamma and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    shape_x: list or tuple
        shape of x

    Returns
    -------
    pd_xl: tvm.tensor
        data_dy*data_gamma
    """
    data_gamma_cast = te.lang.cce.broadcast(data.get("data_gamma"), shape_x)
    pd_xl = te.lang.cce.vmul(data_gamma_cast, data.get("data_dy"))

    return pd_xl


def _get_pd_var_front(data, cast_dtype):
    """
    compute front part of pd_var according to data_variance

    Parameters
    ----------
    data: dict
        placeholders after cast

    Returns
    -------
    pd_var_1: tvm.tensor
        np.power((data_variance + EPSLON), (-1.5))
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    """
    var_elta = te.lang.cce.vadds(data.get("data_variance"),
                                 tvm.const(EPSLON, dtype=cast_dtype))
    var_elta_log = te.lang.cce.vlog(var_elta)
    var_elta_mul = te.lang.cce.vmuls(var_elta_log,
                                     tvm.const(-0.5, dtype=cast_dtype))
    var_elta_2 = te.lang.cce.vexp(var_elta_mul)
    pdvar1_mul = te.lang.cce.vmul(var_elta_2, var_elta_2)
    pd_var_1 = te.lang.cce.vmul(pdvar1_mul, var_elta_2)

    return pd_var_1, var_elta_2


def _get_pd_var(data, params, shape_x, pd_xl, cast_dtype):
    """
    compute pd_var according to data_x, data_mean, reduce_axis and pd_xl

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    pd_xl: tvm.tensor
        data_dy*data_gamma

    Returns
    -------
    pd_var: tvm.tensor
        np.sum(((-0.5)*pd_xl*(data_x - data_mean)
        *np.power((data_variance + EPSLON), (-1.5))), reduce_axis,
        keepdims=True)
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    pd_var_1, var_elta_2 = _get_pd_var_front(data, cast_dtype)

    data_mean_cast = te.lang.cce.broadcast(data.get("data_mean"), shape_x)
    sub_x_mean = te.lang.cce.vsub(data.get("data_x"), data_mean_cast)

    pdvar_mul1 = te.lang.cce.vmul(pd_xl, sub_x_mean)
    pdvar_sum = te.lang.cce.sum(pdvar_mul1, params.get("reduce_axis"),
                                keepdims=True)
    pdvar_mul3 = te.lang.cce.vmul(pdvar_sum, pd_var_1)
    pd_var = te.lang.cce.vmuls(pdvar_mul3, tvm.const(-0.5, dtype=cast_dtype))

    return pd_var, var_elta_2, sub_x_mean


def _get_pd_mean(params, pd_xl, pd_var, var_elta_2, sub_x_mean, cast_dtype):
    """
    compute pd_mean according to reduce_axis, pd_xl, pd_var,
    var_elta_2 and sub_x_mean

    Parameters
    ----------
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    pd_xl: tvm.tensor
        data_dy*data_gamma
    pd_var: tvm.tensor
        np.sum(((-0.5)*pd_xl*(data_x - data_mean)
        *np.power((data_variance + EPSLON), (-1.5))),
        reduce_axis, keepdims=True)
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean

    Returns
    -------
    pd_mean: tvm.tensor
        np.sum(((-1.0)*pd_xl
        *np.power((data_variance + EPSLON), (-0.5))), reduce_axis,
        keepdims=True)
        + pd_var*(1.0/m)*np.sum(((-2.0)*(data_x - data_mean)),
        reduce_axis, keepdims=True)
    """
    pdmean1_sum = te.lang.cce.sum(pd_xl, params.get("reduce_axis"),
                                  keepdims=True)
    pdmean1_mul = te.lang.cce.vmul(pdmean1_sum, var_elta_2)
    pd_mean_1 = te.lang.cce.vmuls(pdmean1_mul,
                                  tvm.const(-1.0, dtype=cast_dtype))

    pdmean2_mul1 = te.lang.cce.vmuls(sub_x_mean,
                                     tvm.const(-2.0, dtype=cast_dtype))
    pdmean2_sum = te.lang.cce.sum(pdmean2_mul1, params.get("reduce_axis"),
                                  keepdims=True)
    pdmean2_mul3 = te.lang.cce.vmuls(pdmean2_sum,
                                     tvm.const((params.get("mean_num")**(-1)),
                                               dtype=cast_dtype))
    pd_mean_2 = te.lang.cce.vmul(pd_var, pdmean2_mul3)

    pd_mean = te.lang.cce.vadd(pd_mean_2, pd_mean_1)

    return pd_mean


def _get_pd_x_front(data, params, shape_x, cast_dtype):
    """
    compute front part of pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x

    Returns
    -------
    pd_x_1: tvm.tensor
        pd_xl*np.power((data_variance + EPSLON), (-0.5))
    pd_x_2: tvm.tensor
        pd_var*(2.0/m)*(data_x - data_mean)
    pd_x_3: tvm.tensor
        pd_mean*(1.0/m)
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    pd_xl = _get_pd_xl(data, shape_x)

    pd_var, var_elta_2, sub_x_mean = _get_pd_var(data, params, shape_x, pd_xl,
                                                 cast_dtype)

    pd_mean = _get_pd_mean(params, pd_xl, pd_var, var_elta_2,
                           sub_x_mean, cast_dtype)

    var_elta_2_cast = te.lang.cce.broadcast(var_elta_2, shape_x)
    pd_x_1 = te.lang.cce.vmul(var_elta_2_cast, pd_xl)
    pdx2_broad = te.lang.cce.broadcast(pd_var, shape_x)
    pdx2_mul = te.lang.cce.vmul(pdx2_broad, sub_x_mean)
    pd_x_2 = te.lang.cce.vmuls(pdx2_mul,
                               tvm.const((2*(params.get("mean_num")**(-1))),
                                         dtype=cast_dtype))
    pd_x_3 = te.lang.cce.vmuls(pd_mean,
                               tvm.const((params.get("mean_num")**(-1)),
                                         dtype=cast_dtype))

    return pd_x_1, pd_x_2, pd_x_3, var_elta_2_cast, sub_x_mean


def _get_pd_x(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    pd_x_1, pd_x_2, pd_x_3, var_elta_2_cast, sub_x_mean = \
        _get_pd_x_front(data, params, shape_x, cast_dtype)

    pdx_broad = te.lang.cce.broadcast(pd_x_3, shape_x)
    pdx_add = te.lang.cce.vadd(pd_x_1, pd_x_2)
    pd_x_ub = te.lang.cce.vadd(pdx_add, pdx_broad)
    if dtype == "float16" and cast_dtype == "float32":
        pd_x = te.lang.cce.cast_to(pd_x_ub, dtype)
    else:
        pd_x = pd_x_ub

    return pd_x, var_elta_2_cast, sub_x_mean


def _get_pd_gamma(data, params, var_elta_2_cast, sub_x_mean, dtype,
                  cast_dtype):
    """
    compute pd_gamma according to data, params, var_elta_2_cast and sub_x_mean

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    dtype: str
        the data type

    Returns
    -------
    pd_gamma: tvm.tensor
        partial derivation of gamma
    """
    xl_mul = te.lang.cce.vmul(var_elta_2_cast, sub_x_mean)
    pdga_mul = te.lang.cce.vmul(data.get("data_dy"), xl_mul)

    if params.get("param_axis"):
        pd_gamma_ub = te.lang.cce.sum(pdga_mul, params.get("param_axis"),
                                      keepdims=True)
        if dtype == "float16" and cast_dtype == "float32":
            pd_gamma = te.lang.cce.cast_to(pd_gamma_ub, dtype)
        else:
            return pd_gamma_ub
    else:
        if dtype == "float16" and cast_dtype == "float32":
            pd_gamma = te.lang.cce.cast_to(pdga_mul, dtype)
        else:
            return pdga_mul

    return pd_gamma


def _get_pd_beta(data, params, dtype, cast_dtype):
    """
    compute pd_beta according to data and params

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    dtype: str
        the data type

    Returns
    -------
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    if params.get("param_axis"):
        pd_beta_ub = te.lang.cce.sum(data.get("data_dy"),
                                     params.get("param_axis"), keepdims=True)
        if dtype == "float16" and cast_dtype == "float32":
            pd_beta = te.lang.cce.cast_to(pd_beta_ub, dtype)
        else:
            return pd_beta_ub
    elif (not params.get("param_axis")) and dtype == "float16":
        pd_beta = te.lang.cce.cast_to(data.get("data_dy"), dtype)
    else:
        pd_beta = te.lang.cce.vadds(data.get("data_dy"),
                                    tvm.const(0, dtype="float32"))

    return pd_beta


def _get_res(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x, pd_gamma, pd_beta according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    pd_gamma: tvm.tensor
        partial derivation of gamma
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    pd_x, var_elta_2_cast, sub_x_mean = _get_pd_x(data, params, shape_x,
                                                  dtype, cast_dtype)

    pd_gamma = _get_pd_gamma(data, params, var_elta_2_cast, sub_x_mean,
                             dtype, cast_dtype)

    pd_beta = _get_pd_beta(data, params, dtype, cast_dtype)

    return pd_x, pd_gamma, pd_beta


def _get_pds(data_dy, data_x, data_variance, data_mean,
             data_gamma, shape_gamma_ori):
    """
    get params and data, compute pd_x, pd_gamma, pd_beta.

    Parameters
    ----------
    data_dy: TVM tensor
        the placeholder of dy input data
    data_x: TVM tensor
        the placeholder of x input data
    data_variance: TVM tensor
        the placeholder of variance input data
    data_mean: TVM tensor
        the placeholder of mean input data
    data_gamma: TVM tensor
        the placeholder of gamma input data
    shape_gamma_ori: list or tuple
        original shape of gamma

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    pd_gamma: tvm.tensor
        partial derivation of gamma
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    dtype = data_dy.dtype.lower()
    shape_x = te.lang.cce.util.shape_to_list(data_x.shape)
    shape_mean = te.lang.cce.util.shape_to_list(data_mean.shape)

    params = _get_params(shape_x, shape_mean, shape_gamma_ori)

    has_improve_precision = False
    cast_dtype = dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        cast_dtype = "float32"

    if has_improve_precision:
        data_dy = te.lang.cce.cast_to(data_dy, "float32")
        data_x = te.lang.cce.cast_to(data_x, "float32")
        data_variance = te.lang.cce.cast_to(data_variance, "float32")
        data_mean = te.lang.cce.cast_to(data_mean, "float32")
        data_gamma = te.lang.cce.cast_to(data_gamma, "float32")

    data = {"data_dy": data_dy, "data_x": data_x,
            "data_variance": data_variance,
            "data_mean": data_mean, "data_gamma": data_gamma}

    pd_x, pd_gamma, pd_beta = _get_res(data, params, shape_x,
                                       dtype, cast_dtype)

    return pd_x, pd_gamma, pd_beta


@fusion_manager.register("layer_norm_grad")
def layer_norm_grad_compute(input_dy, input_x, input_variance, input_mean,
                            input_gamma, output_pd_x, output_pd_gamma,
                            output_pd_beta, kernel_name="layer_norm_grad"):
    """
    DSL description of the layernorm_grad operator's mathematical
    calculation process

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_pd_x: dict
        shape and dtype of output_pd_x, only support float16, float32
    output_pd_gamma: dict
        shape and dtype of output_pd_gamma, only support float16, float32
    output_pd_beta: dict
        shape and dtype of output_pd_beta, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_grad"

    Returns
    -------
    res_tuple: tuple
        (pd_x, pd_gamma, pd_beta)
    """
    pd_x, pd_gamma, pd_beta = _get_pds(input_dy, input_x, input_variance,
                                       input_mean, input_gamma,
                                       input_gamma.shape)
    res_list = [pd_x, pd_gamma, pd_beta]

    return res_list


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, str)
def layer_norm_grad(input_dy, input_x, input_variance, input_mean,
                    input_gamma, output_pd_x, output_pd_gamma,
                    output_pd_beta, kernel_name="layer_norm_grad"):
    """
    algorithm: layernorm_grad
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
        pd_xl    = data_dy*data_gamma
        pd_var   = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-1.5))),
                   reduce_axis, keepdims=True)
        pd_mean  = np.sum(((-1.0)*pd_xl
                   *np.power((data_variance + EPSLON), (-0.5))),
                   reduce_axis, keepdims=True)
                   + pd_var*(1.0/m)
                   *np.sum(((-2.0)*(data_x - data_mean)), reduce_axis,
                   keepdims=True)
        pd_x     = pd_xl*np.power((data_variance + EPSLON), (-0.5))
                   + pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
        pd_gamma = np.sum((data_dy*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-0.5))), param_axis,
                   keepdims=True)
        pd_beta  = np.sum(data_dy, param_axis, keepdims=True)

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_y: dict
        shape and dtype of output, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_grad"

    Returns
    -------
    None
    """
    dtype = input_dy.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_variance = input_variance.get("shape")
    shape_mean = input_mean.get("shape")
    shape_gamma = input_gamma.get("shape")

    _check_params({"shape_dy": shape_dy, "shape_x": shape_x,
                   "shape_var": shape_variance,
                   "shape_mean": shape_mean, "shape_gamma": shape_gamma,
                   "dtype": dtype, "kernel_name": kernel_name})

    shape_gamma = _update_gamma_shape(shape_x, shape_gamma)[0]

    data_gm = _get_data_gm({"shape_dy": shape_dy, "shape_x": shape_x,
                            "shape_var": shape_variance,
                            "shape_mean": shape_mean,
                            "shape_gamma": shape_gamma}, dtype)

    res_list = layer_norm_grad_compute(data_gm[0], data_gm[1], data_gm[2],
                                       data_gm[3], data_gm[4], output_pd_x,
                                       output_pd_gamma, output_pd_beta)

    with tvm.target.cce():
        sch = generic.auto_schedule(res_list)

    tensor_list = list(data_gm) + list(res_list)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
