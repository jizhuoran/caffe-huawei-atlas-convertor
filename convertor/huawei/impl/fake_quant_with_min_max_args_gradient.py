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
fake_quant_with_min_max_args_gradient
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals
@fusion_manager.register("fake_quant_with_min_max_args_gradient")
def fake_quant_with_min_max_args_gradient_compute(gradients, x, y,
                                                  min=-6, max=6, num_bits=8,
                                                  narrow_range=False,
                                                  kernel_name="fake_quant_with_min"
                                                              "_max_args_gradient"):
    """
    Compute gradients for a FakeQuantWithMinMaxArgs operation.
    calculating data's :
    y = gradients*(if x>=nudged_min and <=nudged_max 1 else 0)

    Parameters
    ----------
    gradients: TVM rensor
        the placeholder of input data,type is float32,
        Backpropagated gradients above the FakeQuantWithMinMaxArgs operation
    x: TVM tenor
        the placeholder of input data,type is float32
    y: dict
        the dict of output data
    min: scalar int or float
        Defaults to -6
    max: scalar int or float
        Defaults to 6
        [min; max] define the clamping range for the x data
    num_bits: int  or float
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
        the result of fake_quant_with_min_max_args_gradient_compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    output_dtype = x.dtype
    nudged_min, nudged_max = _nudge_min_max_gradient(min, max, num_bits,
                                                     narrow_range)
    zero_tensor = te.lang.cce.broadcast(0, shape_x, output_dtype=output_dtype)
    nudged_max_tensor = te.lang.cce.vadds(zero_tensor,
                                          tvm.const(nudged_max,
                                                    dtype="float32"))
    nudged_min_tensor = te.lang.cce.vadds(zero_tensor, nudged_min)

    # where((x<=nudged_max)&(x>=nudged_min),1,0),Convert the input to 0 and 1 tensor
    between_nudged_min_max = _cmpare_value(x, nudged_min_tensor,
                                           nudged_max_tensor)

    res = te.lang.cce.vmul(gradients, between_nudged_min_max)

    return res


def _nudge_min_max_gradient(min, max, num_bits, narrow_range):
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
   res: nudged_min, nudged_max
   """
    quant_max = (2 ** num_bits) - 1

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

    return nudged_min, nudged_max


def _cmpare_value(x, nudged_min, nudged_max):
    """
    where((x<=nudged_max)&(x>=nudged_min),1,0)

    Parameters
    ----------
    x: TVM rensor
        Input data
    nudged_min: TVM tenor
        Minimum value of comparison
    nudged_max: TVM rensor
        Maximum value of comparison

    Returns
    -------
    res: TVM tensor
        the result of f_cmpare_value
    """
    min_value = tvm.const(2 ** (-126), dtype="float32")
    # (2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1
    # so min_value*max_value*max_value*max_value_one = 1
    max_value = tvm.const(2 ** (62), dtype="float32")
    max_value_one = tvm.const(2 ** (2), dtype="float32")
    data_zero = te.lang.cce.vmuls(x, 0)
    max_value_tensor = te.lang.cce.vadds(data_zero, max_value)
    min_value_tensor = te.lang.cce.vadds(data_zero, min_value)
    max_value_one_tensor = te.lang.cce.vadds(data_zero, max_value_one)

    sub_tmp = te.lang.cce.vsub(x, nudged_min)
    sub_min = te.lang.cce.vadds(sub_tmp, min_value)
    vmax_tmp = te.lang.cce.vmax(sub_min, data_zero)

    sub_tmp_max = te.lang.cce.vsub(nudged_max, x)
    sub_max = te.lang.cce.vadds(sub_tmp_max, min_value)
    vmin_tmp = te.lang.cce.vmax(sub_max, data_zero)

    one_tmp = te.lang.cce.vmul(vmax_tmp, vmin_tmp)
    one_min = te.lang.cce.vmin(one_tmp, min_value_tensor)

    vmul_max_value = te.lang.cce.vmul(one_min, max_value_tensor)
    vmul_max_value_one = te.lang.cce.vmul(vmul_max_value, max_value_tensor)
    between_nudged_min_max = te.lang.cce.vmul(vmul_max_value_one,
                                              max_value_one_tensor)

    return between_nudged_min_max


@util.check_input_type(dict, dict, dict, (float, int), (float, int), int, bool,
                       str)
def fake_quant_with_min_max_args_gradient(gradients, x, y, min=-6,
                                          max=6, num_bits=8, narrow_range=False,
                                          kernel_name="fake_quant_"
                                                      "with_min_max_args"):
    """
    Compute gradients for a FakeQuantWithMinMaxArgs operation.
    calculating data's :
    y = gradients*(if x>=nudged_min and <=nudged_max 1 else 0)

    Parameters
    ----------
    gradients:dict
              shape and dtype of input gradients,only support float32
    x: dict
        shape and dtype of input x,only support float32
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
        True or False
        if True x values are quantized into the quantization range
        [1; 2^num_bits - 1]
        if False x values are quantized into the quantization range
        [0; 2^num_bits - 1]
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_args_gradient"

    Returns
    -------
    None
    """
    shape_gradients = gradients.get("shape")
    shape_x = x.get("shape")
    if shape_gradients != shape_x:
        raise RuntimeError("shape of two input must be same")
    util.compare_tensor_dict_key(gradients, x, "dtype")

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
    gradients = tvm.placeholder(shape_x, name="gradients", dtype=input_dtype)
    x = tvm.placeholder(shape_x, name="x", dtype=input_dtype)
    res = fake_quant_with_min_max_args_gradient_compute(gradients, x,
                                                        y, float(min),
                                                        float(max),
                                                        num_bits, narrow_range,
                                                        kernel_name)
    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [gradients, x, res]}
    te.lang.cce.cce_build_code(auto_sch, config)
