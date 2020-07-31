#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

dequantize op, Dequantize the 'input' tensor into a float tensor.
"""
import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

SHAPE_SIZE_LIMIT = 2**30  # shape limit


def _min_combined_mode_compute(input_tensor, broadcast_min_range,
                               broadcast_max_range):
    """
    Computation of MIN_COMBINED mode.

    Parameters:
    ----------
    input_tensor: the tensor of input data, dtype must be one of the following:
      only support `int8`, `uint8`, `int32`,

    broadcast_min_range: the tensor of input_min_range, dtype must be `float32`.
      The minimum scalar value possibly produced for the input.
      the shape is the same as the input_tensor

    broadcast_max_range: the tensor of input_max_range, dtype must be `float32`.
      The maximum scalar value possibly produced for the input.
      the shape is the same as the input_tensor

    Returns
    -------
    res : output of the dequantization's computation under MIN_COMBINED mode.
    """
    dtype = input_tensor.dtype

    if dtype in ("int8", "uint8"):
        size = 1
    elif dtype == "int32":
        size = 4
    num_bits = size*8

    if dtype in ("int8", "int32"):
        # cast input_tensor from int to float32
        input_tensor_float32_tmp = te.lang.cce.cast_to(input_tensor,
                                                       "float32")
        val_max_int = (1 << num_bits) / 2
        input_tensor_float32 = te.lang.cce.vadds(input_tensor_float32_tmp,
                                                 val_max_int)
    if dtype == "uint8":
        # cast input_tensor from uint to float32
        input_tensor_float32 = te.lang.cce.cast_to(input_tensor, "float32")
    sub_tensor = te.lang.cce.vsub(broadcast_max_range, broadcast_min_range)
    val_scalar = 1.0 / ((1 << num_bits) - 1)
    sub_mul_tensor = te.lang.cce.vmul(input_tensor_float32, sub_tensor)
    res_tmp = te.lang.cce.vmuls(sub_mul_tensor, val_scalar)
    res = te.lang.cce.vadd(res_tmp, broadcast_min_range)
    return res


def _min_first_8bits_offset(res, broadcast_min_range, sub_tensor):
    """
    Compute the offset for 8bits data type under the MIN_FIRST mode.

    Parameters:
    ----------
    res: the tensor of the output without offset,
      dtype must be one of the following: only support `int8`, `uint8`

    broadcast_min_range: the tensor of input_min_range, dtype must be `float32`.
      The minimum scalar value possibly produced for the input.
      the shape is the same as the input_tensor

    sub_tensor: the tensor of broadcast_max_range subtracts broadcast_min_range,
      dtype must be `float32`.
      The maximum scalar value possibly produced for the input.
      the shape is the same as the input_tensor

    Returns
    -------
    res : output of the dequantization's computation under MIN_FIRST mode.
    """
    sub_rec_tensor = te.lang.cce.vrec(sub_tensor)
    # muls the range of the 8bits data type
    muls_tensor = te.lang.cce.vmuls(broadcast_min_range,
                                    (1 << 8) - 1)
    muls_mul_sub_rec_tensor = te.lang.cce.vmul(muls_tensor,
                                               sub_rec_tensor)
    least_quantize = te.lang.cce.round(muls_mul_sub_rec_tensor)
    neg_least_quantize = te.lang.cce.vmuls(least_quantize, -1)
    least_quantize_fp32 = te.lang.cce.cast_to(neg_least_quantize,
                                              "float32")
    least_quantize_fp32_mul = te.lang.cce.vmul(least_quantize_fp32,
                                               sub_tensor)
    # the reciprocal of the range of the 8bits data type
    val_scalar = 1.0 / ((1 << 8) - 1)
    least_quantize_fp32_mul_muls = te.lang.cce.vmuls(
        least_quantize_fp32_mul, val_scalar)
    offset = te.lang.cce.vadd(broadcast_min_range,
                              least_quantize_fp32_mul_muls)
    res = te.lang.cce.vsub(res, offset)

    return res


def _min_first_mode_compute(input_tensor, broadcast_min_range,
                            broadcast_max_range):
    """
    Computation of MIN_FIRST mode.

    Parameters:
    ----------
    input_tensor: the tensor of x, dtype must be one of the following:
      only support `int8`, `uint8`, `int32`,

    broadcast_min_range: the tensor of input_min_range, dtype must be `float32`.
      The minimum scalar value possibly produced for the input.
      the shape is the same as the input_tensor

    broadcast_max_range: the tensor of input_max_range, dtype must be `float32`.
      The maximum scalar value possibly produced for the input.
      the shape is the same as the input_tensor

    Returns
    -------
    res : output of the dequantization's computation under MIN_FIRST mode.
    """
    dtype = input_tensor.dtype

    if dtype in ("int8", "uint8"):
        size = 1
    elif dtype == "int32":
        size = 4
    num_bits = size*8

    if dtype in ("int8", "int32"):
        # cast input_tensor from int to float32
        input_tensor_float32_tmp = te.lang.cce.cast_to(input_tensor,
                                                       "float32")
        val_max_int = (1 << num_bits) / 2
        input_tensor_float32 = te.lang.cce.vadds(input_tensor_float32_tmp,
                                                 val_max_int)
    if dtype == "uint8":
        # cast input_tensor from uint to float32
        input_tensor_float32 = te.lang.cce.cast_to(input_tensor, "float32")
    sub_tensor = te.lang.cce.vsub(broadcast_max_range, broadcast_min_range)
    val_scalar = 1.0 / ((1 << num_bits) - 1)
    sub_muls_tensor = te.lang.cce.vmuls(sub_tensor, val_scalar)
    res_tmp_1 = te.lang.cce.vmul(input_tensor_float32, sub_muls_tensor)
    res = te.lang.cce.vadd(res_tmp_1, broadcast_min_range)

    # for offset int8 and uint8
    if dtype in ("int8", "uint8"):
        res = _min_first_8bits_offset(res, broadcast_max_range, sub_tensor)
    return res


def _scaled_mode_compute(input_tensor, broadcast_max_range):
    """
    Computation of SCALED mode.

    Parameters:
    ----------
    input_tensor: the tensor of x, dtype must be one of the following:
      only support `int8`, `uint8`, `int32`,

    broadcast_max_range: the tensor of input_max_range, dtype must be `float32`.
      The maximum scalar value possibly produced for the input.
      the shape is the same as the input_tensor

    Returns
    -------
    res : output of the dequantization's computation under SCALED mode.
    """
    dtype = input_tensor.dtype

    if dtype in ("int8", "uint8"):
        size = 1
    elif dtype == "int32":
        size = 4
    num_bits = size*8

    val_scalar = 0
    if dtype in ("int8", "int32"):
        # min&max fixed is float32
        [min_fixed, max_fixed] = [-((1 << (num_bits - 1)) - 1),
                                  (1 << (num_bits - 1)) - 1]
        val_scalar = 2.0 / (max_fixed - min_fixed)
    if dtype == "uint8":
        [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
        val_scalar = 1.0 / (max_fixed - min_fixed)
    # cast input from int to float32
    muls_tensor = te.lang.cce.vmuls(broadcast_max_range, val_scalar)
    input_tensor_float32 = te.lang.cce.cast_to(input_tensor, "float32")
    res = te.lang.cce.vmul(input_tensor_float32, muls_tensor)

    return res


# pylint: disable=locally-disabled, too-many-arguments, unused-argument, unnecessary-lambda
# pylint: disable=locally-disabled,invalid-name,too-many-locals
@fusion_manager.register("dequantize")
def dequantize_compute(x, min_range,
                       max_range, y,
                       mode="MIN_COMBINED", kernel_name="dequantize"):
    """
    Computation for dequantize the 'input' tensor into a float tensor.

    Parameters:
    ----------
    x: input data, dtype must be one of the following:
      only support `int8`, `uint8`, `int32`,

    min_range: input min_range, dtype must be `float32`.
      The minimum scalar value possibly produced for the input.

    max_range: input max_range, dtype must be `float32`.
      The maximum scalar value possibly produced for the input.

    y: the dict of output_data, dtype must be `float32`.

    mode: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST", "SCALED"`.
      Defaults to `"MIN_COMBINED"`.

    kernel_name : cce kernel name, default value is "dequantize".

    Returns
    -------
    res : output of the dequantization's computation.
    """

    input_tensor = x

    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    shape_range = te.lang.cce.util.shape_to_list(max_range.shape)

    shape_x, shape_range, shape_max = util.produce_shapes(
        shape_x, shape_range)

    broadcast_min_range = te.lang.cce.broadcast(min_range, shape_max)
    broadcast_max_range = te.lang.cce.broadcast(max_range, shape_max)

    if mode == "MIN_COMBINED":
        res = _min_combined_mode_compute(input_tensor, broadcast_min_range,
                                         broadcast_max_range)

    elif mode == "MIN_FIRST":
        res = _min_first_mode_compute(input_tensor, broadcast_min_range,
                                      broadcast_max_range)

    elif mode == "SCALED":
        res = _scaled_mode_compute(input_tensor, broadcast_max_range)

    return res


@util.check_input_type(dict, dict, dict, dict, str, str)
def dequantize(x, min_range, max_range, y,
               mode="MIN_COMBINED", kernel_name="dequantize"):
    """
    Dequantize the 'input' tensor into a float tensor.

    [min_range, max_range] are scalar floats that specify the range for
    the 'input' data.

    The 'mode' attribute controls exactly which calculations are used
    to convert the float to their quantized equivalents.


    In 'MIN_COMBINED' mode,
    each value of the tensor will undergo the following:

    ```
    if T == int8 or T == int32: in[i] += (range(T) + 1) / 2.0
    out[i] = min_range + (in[i] * (max_range - min_range) / range(T))
    ```
    here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

    Note that if quantizedtype is int8, the operation will additionally add
    each value by 128 prior to casting.


    If the mode is 'MIN_FIRST', then this approach is used:

    ```
    num_discrete_values = 1 << (# of bits in T)
    range_adjust = num_discrete_values / (num_discrete_values - 1)
    range = (range_max - range_min) * range_adjust
    range_scale = range / num_discrete_values
    if T == int32:
        result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
    else if T == int8 or T == uint8:
        least_quantitize = -round(min_range * ((1 << num_bits) - 1) /
                            (max_range - min_range))
        offset = min_range + least_quantitize * 1.0 * (max_range - min_range) /
                            ((1 << num_bits) - 1)
        res_tmp = range_min + ((input - numeric_limits<T>::min()) * range_scale)
        result = res_tmp - offset
    ```


    In `SCALED` mode,

    ```
    m = input_max
    num_bits = sizeof(T) * 8
    if T == int8 or T == int32:
        [min_fixed, max_fixed] =
            [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
        s = (2.0 * m) / (max_fixed - min_fixed)
    if T == uint8:
        [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
        s = 1.0 * m / (max_fixed - min_fixed)
    result = input * s
    ```

    Parameters:
    ----------
    x: the dict of x, dtype must be one of the following:
      cloud version only supports `int8`, `uint8`, `int32`,
      mini version only supports `int8`, `uint8`.

    min_range: the dict of input_min_range, dtype must be `float32`.
      The minimum scalar value possibly produced for the input.

    max_range: the dict of input_max_range, dtype must be `float32`.
      The maximum scalar value possibly produced for the input.

    y: the dict of output_data, dtype must be `float32`.

    mode: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST", "SCALED"`.
      Defaults to `"MIN_COMBINED"`.

    kernel_name : cce kernel name, default value is "dequantize"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    shape_input_min_range = min_range.get("shape")
    shape_input_max_range = max_range.get("shape")
    shape_output_data = y.get("shape")
    if len(shape_input_min_range) != len(shape_input_max_range):
        raise RuntimeError("shape_input_min_range and shape_input_max_range"
                           " must be equal")
    if shape_output_data != shape_x:
        raise RuntimeError("shape_output_data and shape_x must be equal.")
    shape_range = shape_input_min_range
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_range)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_range, 1)

    dtype_x = x.get("dtype")
    dtype_input_min_range = min_range.get("dtype")
    dtype_input_max_range = max_range.get("dtype")
    dtype_output_data = y.get("dtype")
    dtype_x = dtype_x.lower()
    dtype_input_min_range = dtype_input_min_range.lower()
    dtype_input_max_range = dtype_input_max_range.lower()
    dtype_output_data = dtype_output_data.lower()
    check_list = ("int8", "uint8", "int32")
    s322f32_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.cast_to", "s322f32")
    if dtype_x == "int32" and not s322f32_support:
        raise RuntimeError("not support on the platform")
    vmul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    if not vmul_support:
        raise RuntimeError("not support on the platform")
    util.check_dtype_rule(dtype_x, check_list)
    util.check_dtype_rule(dtype_input_min_range, ("float32",))
    util.check_dtype_rule(dtype_input_max_range, ("float32",))
    util.check_dtype_rule(dtype_output_data, ("float32",))

    if mode not in ("MIN_COMBINED", "MIN_FIRST", "SCALED"):
        raise RuntimeError("mode only support MIN_COMBINED, MIN_FIRST, SCALED.")

    util.check_kernel_name(kernel_name)

    shape_x, shape_range, _ = util.produce_shapes(
        shape_x,
        shape_range)

    shape_x, shape_range = refine_shapes_for_broadcast(shape_x, shape_range)
    input_tensor = tvm.placeholder(shape_x,
                                   dtype=dtype_x,
                                   name="x")
    min_range = tvm.placeholder(shape_range,
                                dtype="float32",
                                name="input_min_range")
    max_range = tvm.placeholder(shape_range,
                                dtype="float32",
                                name="input_max_range")

    res = dequantize_compute(input_tensor, min_range, max_range,
                             y, mode, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_tensor, min_range, max_range, res]}
    te.lang.cce.cce_build_code(sch, config)

