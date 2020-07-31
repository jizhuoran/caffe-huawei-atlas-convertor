"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

group_norm
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


def _shape_check(shape_x, shape_scale, shape_offset, data_format, num_groups):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        shape_scale's data shape
    shape_offset: list or tuple
        shape_offset's data shape
    data_format: str
        data format of input x
    num_groups: int
        groups of channel

    Returns
    -------
    None
    """
    if data_format == "NCHW":
        c_index_ = 1
    elif data_format == "NHWC":
        c_index_ = 3
    else:
        raise RuntimeError("not support this %s format" % data_format)

    if len(shape_x) != 4:
        raise RuntimeError("The input shape only support 4D Tensor")

    if (shape_scale != shape_offset) or (len(shape_scale) != 1):
        raise RuntimeError("scale and offset's shape must be same and should "
                           "be 1D Tensor")

    if (shape_x[c_index_] != shape_scale[0]) or \
            (shape_x[c_index_] != shape_offset[0]):
        raise RuntimeError("scale, offset, x dimensions must be equal")

    if shape_x[c_index_] % num_groups != 0:
        raise RuntimeError("num_groups must divide C channel")

    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_scale)
    util.check_shape_rule(shape_offset)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_scale)
    util.check_tensor_shape_size(shape_offset)


# pylint: disable=locally-disabled,unused-argument,too-many-statements
# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=locally-disabled,invalid-name
@fusion_manager.register("group_norm")
def group_norm_compute(x, scale, offset, epsilon, data_format,
                       kernel_name="group_norm"):
    """
    DSL description of the group_norm operator's calculation process

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x input data
    scale: TVM tensor
        the placeholder of gamma input data
    offset: TVM tensor
        the placeholder of beta input data
    epsilon: float,
        Minimum positive number greater than 0
    data_format: str
        format string of input x
    kernel_name: str
        cce kernel name, default value is "group_norm"

    Returns
    -------
    res: TVM tensor
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    dtype_x = x.dtype.lower()

    if dtype_x == "float16":
        x = te.lang.cce.cast_to(x, "float32")

    if data_format == "NCHW":
        reduce_axis = [2, 3, 4]
    else:
        reduce_axis = [1, 2, 4]

    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    mean_cof = reduce_elts ** (-1)

    # DSL description of the mean calculation process
    # mu = sum(x / n)
    mean_muls = te.lang.cce.vmuls(x, mean_cof)
    mean = te.lang.cce.sum(mean_muls, axis=reduce_axis, keepdims=True)
    mean_broadcast = te.lang.cce.broadcast(mean, shape_x)

    # DSL description of the variance calculation process
    # var = sum((x-mean)^2 / n)
    x_mean_sub = te.lang.cce.vsub(x, mean_broadcast)
    variance_mul = te.lang.cce.vmul(x_mean_sub, x_mean_sub)
    variance_muls = te.lang.cce.vmuls(variance_mul, mean_cof)
    variance = te.lang.cce.sum(variance_muls, axis=reduce_axis, keepdims=True)
    variance_broadcast = te.lang.cce.broadcast(variance, shape_x)

    # DSL description of the normalize calculation process
    # rsqrt(x) = exp(-0.5ln(var + eps))
    # norm = (x - mean) * rsqrt * gamma + beta
    epsilon = tvm.const(epsilon, dtype="float32")
    normalize_add = te.lang.cce.vadds(variance_broadcast, epsilon)
    normalize_log = te.lang.cce.vlog(normalize_add)
    normalize_log_mul = \
        te.lang.cce.vmuls(normalize_log, tvm.const(-0.5, dtype="float32"))
    normalize_exp = te.lang.cce.vexp(normalize_log_mul)
    normalize_mul = te.lang.cce.vmul(x_mean_sub, normalize_exp)

    # norm * gamma + beta
    gamma_broadcast = te.lang.cce.broadcast(scale, shape_x)
    beta_broadcast = te.lang.cce.broadcast(offset, shape_x)
    scale_mul = te.lang.cce.vmul(gamma_broadcast, normalize_mul)
    res = te.lang.cce.vadd(scale_mul, beta_broadcast)

    if dtype_x == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


# pylint: disable=locally-disabled,unused-argument,too-many-arguments
@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, float, str, bool, int, str)
def group_norm(x,
               scale,
               offset,
               mean,
               variance,
               y,
               batch_mean,
               batch_variance,
               reserve_space_1,
               reserve_space_2,
               epsilon=1e-4,
               data_format="NHWC",
               is_training=True,
               num_groups=2,
               kernel_name="group_norm"):
    """
    algorithm: group_norm
    Group normalization.
    The input tensor's dimension C should be equal.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    scale: dict
        dict of scale,
        A Tensor for scaling factor, to scale the normalized x.
    offset: dict
        dict of offset, A Tensor for offset, to shift to the normalized x.
    mean: dict
        dict of mean, A Tensor for population mean.
        Used for inference only, must be empty for training.
    variance: dict
        dict of variance, A Tensor for population variance.
        Used for inference only, must be empty for training.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`. Has the same type as `mean`.
    batch_variance: dict
        dict of batch_var, A `Tensor`. Has the same type as `variance`.
    reserve_space_1: dict
        dict of reserve_space_1, A `Tensor`.
    reserve_space_2: dict
        dict of reserve_space_2, A `Tensor`.
    reserve_space_3: dict
        dict of reserve_space_3, A `Tensor`.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.001`.
    data_format: str
        The data format for x and y. Support "NCHW" and "NHWC" only.
    is_training: bool
        A bool value indicates the operation for train (default) or inference.
    num_groups: int
        A integer value indicates the group in channel.
    kernel_name: str
        kernel name, default value is "group_norm"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_scale = scale.get("shape")
    shape_offset = offset.get("shape")

    dtype_x = x.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")

    input_dtype = dtype_x.lower()

    _shape_check(shape_x, shape_scale, shape_offset, data_format, num_groups)

    util.check_kernel_name(kernel_name)
    util.check_dtype_rule(dtype_x.lower(), ("float16", "float32"))
    util.check_dtype_rule(dtype_scale.lower(), ("float32",))
    util.check_dtype_rule(dtype_offset.lower(), ("float32",))

    # Reshape NCHW -> N[GD]HW
    if data_format == "NCHW":
        shape_x = [shape_x[0], num_groups, shape_x[1] // num_groups,
                   shape_x[2], shape_x[3]]
        shape_scale = [1] + [num_groups, shape_scale[0] // num_groups] + [1, 1]
        shape_offset = [1] + [num_groups, shape_offset[0] // num_groups] + \
                       [1, 1]
    # Reshape NHWC -> NHW[GD]
    elif data_format == "NHWC":
        shape_x = [shape_x[0], shape_x[1], shape_x[2], num_groups,
                   shape_x[3] // num_groups]
        shape_scale = [1, 1, 1] + [num_groups, shape_scale[0] // num_groups]
        shape_offset = [1, 1, 1] + [num_groups, shape_offset[0] // num_groups]

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_dtype)
    scale_input = tvm.placeholder(shape_scale, name="scale_input",
                                  dtype=dtype_scale)
    offset_input = tvm.placeholder(shape_offset, name="offset_input",
                                   dtype=dtype_offset)

    res = group_norm_compute(x_input, scale_input, offset_input, epsilon,
                             data_format, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [x_input, scale_input, offset_input, res]}

    te.lang.cce.cce_build_code(schedule, config)
