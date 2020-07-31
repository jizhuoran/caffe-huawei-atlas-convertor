#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not
use this file except in compliance with the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
fake_quant_with_min_max_args
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals
@fusion_manager.register("fake_quant_with_min_max_args")
def fake_quant_with_min_max_args_compute(x, y, min=-6, max=6, num_bits=8,
                                         narrow_range=False,
                                         kernel_name="fake_quant_with_min_"
                                                     "max_args"):
    """
    Computes Fake-quantize the 'x' tensor,
    type float32 to 'y' tensor of same type
    calculating data's :
    y = (floor(clamped_shifted * inv_nudged_scale + 0.5f)))*scale+nudged_min
    scale=(max-min)/(quant_max-quant_min)

    Parameters
    ----------
    x: TVM tenor
        the placeholder of input data,type is float32
    y: dict
        the dict of output data
    min: scalar float int
        Defaults to -6
    max: scalar float int
        Defaults to 6
        [min; max] define the clamping range for the x data
    num_bits: float int
        Defaults to 8.num_bits is the bitwidth of the quantization,
        between 2 and 16
    narrow_range: bool
        True or False.if None,narrow_range=False
        if True x values are quantized into the quantization range
        [1; 2^num_bits - 1]
        if False x values are quantized into the quantization range
        [0; 2^num_bits - 1]
    kernel_name: str
        cce kernel name, default value is "fake_quant_with_min_max_args"

    Returns
    -------
    res: TVM tensor
        the result of fake_quant_with_min_max_args_compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    output_dtype = x.dtype

    nudged_min, nudged_max, scale = _nudge_min_max(min, max, num_bits,
                                                   narrow_range)

    zero_tensor = te.lang.cce.broadcast(0, shape_x, output_dtype=output_dtype)
    nudged_max_tensor = te.lang.cce.vadds(zero_tensor, nudged_max)
    nudged_min_tensor = te.lang.cce.vadds(zero_tensor, nudged_min)
    inv_nudged_scale = 1.00 / scale
    inv_nudged_scale_const = tvm.const(inv_nudged_scale, dtype=output_dtype)

    # Transform the input between nudged_max and nudged_min
    clamped_vmin = te.lang.cce.vmin(x, nudged_max_tensor)
    clamped = te.lang.cce.vmax(clamped_vmin, nudged_min_tensor)

    # Calculate the quantized and dequantized results
    clamped_shifted = te.lang.cce.vsub(clamped, nudged_min_tensor)
    vmul_shifted = te.lang.cce.vmuls(clamped_shifted, inv_nudged_scale_const)
    vadds_shifted = te.lang.cce.vadds(vmul_shifted, tvm.const(0.5,
                                                              dtype="float32"))
    floor_vadds_shifted = te.lang.cce.floor(vadds_shifted)
    floor_cast = te.lang.cce.cast_to(floor_vadds_shifted, output_dtype)
    res_scale = te.lang.cce.vmuls(floor_cast, scale)
    res = te.lang.cce.vadd(res_scale, nudged_min_tensor)

    return res


def _nudge_min_max(min, max, num_bits, narrow_range):
    """
    Calculate the maximum and minimum values of the quantization

    Parameters
    ----------
    min: scalar
        input min
    max: TVM tenor
        input max
    num_bits: scalar
        Defaults to 8.num_bits is the bitwidth of the quantization,
        between 2 and 16
    narrow_range: bool

    Returns
    -------
    res: nudged_min, nudged_max, scale
    """
    quant_max = (2**num_bits) - 1

    if narrow_range is False:
        quant_min = 0.00
    else:
        quant_min = 1.00

    scale = (max - min) / (float(quant_max) - quant_min)

    zeor_point_from_min = quant_min - min / scale

    # Calculate the maximum and minimum values of the quantization
    if zeor_point_from_min < quant_min:
        nudged_zero_point = quant_min
    elif zeor_point_from_min > quant_max:
        nudged_zero_point = quant_max
    else:
        nudged_zero_point = (zeor_point_from_min + 0.5) // 1

    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale

    return nudged_min, nudged_max, scale


@util.check_input_type(dict, dict, (float, int), (float, int), int, bool, str)
def fake_quant_with_min_max_args(x, y, min=-6, max=6, num_bits=8,
                                 narrow_range=False, kernel_name="fake_quant_"
                                                                 "with_min_max_args"):
    """
    Computes Fake-quantize the 'x' tensor,
    type float32 to 'y' tensor of same type
    calculating data's :
    y = (floor(clamped_shifted * inv_nudged_scale + 0.5f)))*scale+nudged_min
    scale=(max-min)/(quant_max-quant_min)

    Parameters
    ----------
    x: dict
        shape and dtype of input,only support float32
    y: dict
        the dict of output data
    min: scalar float int
        Defaults to -6
    max: scalar float int
        Defaults to 6
        [min; max] define the clamping range for the x data
    num_bits: float int
        Defaults to 8.num_bits is the bitwidth of the quantization,
        between 2 and 16
    narrow_range: bool or None
        True or False.if None,narrow_range=False
        if True x values are quantized into the quantization range
        [1; 2^num_bits - 1]
        if False x values are quantized into the quantization range
        [0; 2^num_bits - 1]
    kernel_name: str
        cce kernel name, default value is "fake_quant_with_min_max_args"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    input_dtype = x.get("dtype").lower()
    util.check_dtype_rule(input_dtype, ["float32"])
    if min >= max:
        raise RuntimeError("min must be less than max")
    if num_bits < 2 or num_bits > 16:
        raise RuntimeError("num_bits is between 2 and 16")
    shape_x = (functools_reduce(lambda x, y: x * y, shape_x[:]),)
    x = tvm.placeholder(shape_x, name="x", dtype=input_dtype)
    res = fake_quant_with_min_max_args_compute(x, y, float(min), float(max),
                                               num_bits, narrow_range,
                                               kernel_name)

    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [x, res]}
    te.lang.cce.cce_build_code(auto_sch, config)
