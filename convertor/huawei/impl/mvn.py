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

mvn
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

# const value
CONST_HALF = 0.5
CONST_SQRT_ITER = 3


# pylint: disable=too-few-public-methods


def _check_format_shape(data_format, shape):
    """
    Function to check format and shape of input data.

    Parameters
    ----------

    data_format: str
        format of input data
    shape: list or tuple
        data shape of input data
    Returns
    -------
    None
    """

    if data_format != "NCHW":
        raise RuntimeError("format only support NCHW")

    if len(shape) != 4:
        raise RuntimeError("shape only support 4D Tensor")


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

    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES",):
        if input_dtype == "float32":
            raise RuntimeError("float32 is not support in ES")
        util.check_dtype_rule(input_dtype, ("float16",))
    else:
        util.check_dtype_rule(input_dtype, ("float16", "float32",))


# pylint: disable=too-many-arguments,too-many-locals,protected-access
# pylint: disable=too-many-branches,unused-argument,invalid-name
@fusion_manager.register("mvn")
def mvn_compute(x, y, normalize_variance, across_channels,
                eps, kernel_name="mvn"):
    """
    algorithm: MVN
    y = (x-mean(x))/(std(x) + eps)

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    normalize_variance: bool
        A bool value indicates the operation for normalize_variance.
    across_channels: bool
        A bool value indicates the operation for across_channels.
    eps: float
        A small float number added to the variance of x. Defaults to `1e-9`.
    kernel_name : str
        kernel name, default value is "mvn"

    Returns
    -------
    output tensor
    """

    dtype_x = x.dtype
    is_cast = False

    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vmuls","float32"):
        if dtype_x == "float16":
            is_cast = True
            x = te.lang.cce.cast_to(x, 'float32')

    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    if across_channels:
        axis = [1, 2, 3]
        num = shape_x[1] * shape_x[2] * shape_x[3]
    else:
        axis = [2, 3]
        num = shape_x[2] * shape_x[3]
    if num != 0:
        num_rec = 1.0/num

    # compute subtract mean
    mean_sum = te.lang.cce.sum(x, axis, True)    # sum
    mean_muls = te.lang.cce.vmuls(mean_sum, num_rec)
    mean_broad = te.lang.cce.broadcast(mean_muls, shape_x)  # mean

    if normalize_variance:
        mean_sub = te.lang.cce.vsub(x, mean_broad)   # x - mean

        var_mul = te.lang.cce.vmul(mean_sub, mean_sub)
        var_sum = te.lang.cce.sum(var_mul, axis, True)
        var_muls = te.lang.cce.vmuls(var_sum, num_rec)
        var = te.lang.cce.broadcast(var_muls, shape_x)   # var

        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend910"):
            y_sqrt = te.lang.cce.vsqrt(var)
            y_add = te.lang.cce.vadds(y_sqrt, eps)
            res = te.lang.cce.vdiv(mean_sub, y_add)
        elif tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES") and \
                hasattr(te.lang.cce, "te_compute") and \
                hasattr(te.lang.cce.te_compute, "elewise_compute") and \
                hasattr(te.lang.cce.te_compute.elewise_compute, "__binary_elewise_op"):
            y_sqrt = te.lang.cce.vsqrt(var, priority_flag=1)
            y_add = te.lang.cce.vadds(y_sqrt, eps)
            res = te.lang.cce.te_compute.elewise_compute. \
                __binary_elewise_op(mean_sub, y_add, "elewise_binary_div")
        elif tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310","Ascend610","Ascend620"):
            y_sqrt = te.lang.cce.vsqrt(var)

            for _ in range(CONST_SQRT_ITER):
                data_sqrt = te.lang.cce.vdiv(var, y_sqrt)
                data_sqrt = te.lang.cce.vadd(data_sqrt, y_sqrt)
                data_sqrt = te.lang.cce.vmuls(data_sqrt, tvm.const(CONST_HALF, var.dtype))
                y_sqrt = data_sqrt

            y_add = te.lang.cce.vadds(y_sqrt, eps)
            res = te.lang.cce.vdiv(mean_sub, y_add)
        else:
            y_sqrt = te.lang.cce.vsqrt(var, priority_flag=1)  # calculate_by_newton
            y_add = te.lang.cce.vadds(y_sqrt, eps)
            res = te.lang.cce.vdiv(mean_sub, y_add)
    else:
        res = te.lang.cce.vsub(x, mean_broad)   # x - mean

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype_x)

    return res


@util.check_input_type(dict, dict, bool, bool, float, str)
def mvn(x, y, normalize_variance=True, across_channels=False,
        eps=1e-9, kernel_name="mvn"):
    """
    algorithm: MVN
    y = (x-mean(x))/(std(x) + eps)

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    normalize_variance: bool
        A bool value indicates the operation for normalize_variance.
    across_channels: bool
        A bool value indicates the operation for across_channels.
    eps: float
        A small float number added to the variance of x. Defaults to `1e-9`.
    kernel_name : str
        kernel name, default value is "mvn"
    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    input_dtype = dtype_x.lower()

    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)

    _check_dtype(input_dtype)

    data_format = x.get("format")
    _check_format_shape(data_format, shape_x)

    util.check_kernel_name(kernel_name)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_dtype)
    res = mvn_compute(x_input, y, normalize_variance, across_channels,
                      eps, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [x_input, res]}

    te.lang.cce.cce_build_code(sch, config)
