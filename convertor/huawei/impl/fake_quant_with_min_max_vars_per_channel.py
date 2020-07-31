#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fake_quant_with_min_max_vars_per_channel:
Fake-quantize the 'inputs' tensor of type float and one of the shapes: [d],[b, d] [b, h, w, d]
via per-channel floats min and max of shape [d] to 'outputs' tensor of same shape as inputs.
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# define a scalar for add
HALF_ONE = 0.5
# define zero and one for broadcast
ZERO_VALUE = 0
ONE_VALUE = 1


def _less_compare_float32(data_x, data_y):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    data_x : TVM tensor
        tensor x
    data_y : TVM tensor
        tensor y

    Returns
    -------
    the compare result
    """
    shape_inputs = te.lang.cce.util.shape_to_list(data_x.shape)
    # minimun num of float32 2**(-126)
    data_min = te.lang.cce.broadcast(tvm.const(2 ** (-126), dtype="float32"),
                                     shape_inputs, "float32")
    data_zero = te.lang.cce.broadcast(tvm.const(0, dtype="float32"),
                                      shape_inputs, "float32")
    res_sub = te.lang.cce.vsub(data_y, data_x)
    res_min = te.lang.cce.vmin(res_sub, data_min)
    res_max = te.lang.cce.vmax(res_min, data_zero)
    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 62/62/2 to adaptor 126
    res_mul_fierst = te.lang.cce.vmuls(res_max,
                                       tvm.const(2 ** 62, dtype="float32"))
    res_mul_second = te.lang.cce.vmuls(res_mul_fierst,
                                       tvm.const(2 ** 62, dtype="float32"))
    res = te.lang.cce.vmuls(res_mul_second, tvm.const(2 ** 2, dtype="float32"))

    return res


# pylint: disable=locally-disabled,too-many-locals
def _nudged_min_max_compute(min_broadcast, max_broadcast, num_bits,
                            narrow_range):
    """
    compute nudged_min and nudged_max by input parameters

    Parameters:
    ----------
    min : TVM tensor
        tensor min has broadcast to x shape
    max : TVM tensor
        tensor max has broadcast to x shape
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False

    Returns
    -------
    res: tensor list
        [nudged_min, nudged_max, scale]
    """
    dtype = min_broadcast.dtype
    if narrow_range is False:
        quant_min = 0
    else:
        quant_min = 1
    quant_max = 2 ** num_bits - 1
    tensor_zero = te.lang.cce.vmuls(min_broadcast, tvm.const(ZERO_VALUE, dtype))
    quant_min_float = te.lang.cce.vadds(tensor_zero,
                                        tvm.const(quant_min, dtype))
    quant_max_float = te.lang.cce.vadds(tensor_zero,
                                        tvm.const(quant_max, dtype))
    max_sub_min = te.lang.cce.vsub(max_broadcast, min_broadcast)
    quant_max_sub_quant_min = te.lang.cce.vsub(quant_max_float, quant_min_float)
    scale = te.lang.cce.vdiv(max_sub_min, quant_max_sub_quant_min)
    min_div_scale = te.lang.cce.vdiv(min_broadcast, scale)
    zero_point_from_min = te.lang.cce.vsub(quant_min_float, min_div_scale)

    bool_less_quant_min_float = _less_compare_float32(zero_point_from_min,
                                                      quant_min_float)
    bool_more_quant_max_float = _less_compare_float32(quant_max_float,
                                                      zero_point_from_min)
    less_quant_min_float = te.lang.cce.vmul(quant_min_float,
                                            bool_less_quant_min_float)
    more_quant_max_float = te.lang.cce.vmul(quant_max_float,
                                            bool_more_quant_max_float)
    tensor_one = te.lang.cce.vadds(tensor_zero, tvm.const(ONE_VALUE, dtype))
    bool_not_less_quant_min_float = te.lang.cce.vsub(tensor_one,
                                                     bool_less_quant_min_float)
    bool_not_more_quant_max_float = te.lang.cce.vsub(tensor_one,
                                                     bool_more_quant_max_float)
    bool_between_min_max = te.lang.cce.vmul(bool_not_less_quant_min_float,
                                            bool_not_more_quant_max_float)
    between_min_max_float = te.lang.cce.vmul(zero_point_from_min,
                                             bool_between_min_max)
    between_min_max_add_half_one = te.lang.cce.vadds(between_min_max_float,
                                                     tvm.const(HALF_ONE, dtype))
    between_min_max_round = te.lang.cce.floor(between_min_max_add_half_one)
    nudged_zero_point_tmp = te.lang.cce.vadd(less_quant_min_float,
                                             more_quant_max_float)
    nudged_zero_point = te.lang.cce.vadd(nudged_zero_point_tmp,
                                         between_min_max_round)

    nudged_min_tmp = te.lang.cce.vsub(quant_min_float, nudged_zero_point)
    nudged_max_tmp = te.lang.cce.vsub(quant_max_float, nudged_zero_point)
    nudged_min = te.lang.cce.vmul(nudged_min_tmp, scale)
    nudged_max = te.lang.cce.vmul(nudged_max_tmp, scale)
    res = [nudged_min, nudged_max, scale]

    return res


def _bool_both_zero_compute(juduged_min, juduged_max):
    """
    if input min and max are both zero then output_date will be all zero
    so need a juduge compute tensor

    Parameters:
    ----------
    min : TVM tensor
        tensor min
    max : TVM tensor
        tensor max

    Returns
    -------
    res : TVM tensor
        a tensor for juduge compute
    """
    dtype = juduged_min.dtype
    tensor_zero = te.lang.cce.vmuls(juduged_min, tvm.const(ZERO_VALUE, dtype))
    min_abs = te.lang.cce.vabs(juduged_min)
    max_abs = te.lang.cce.vabs(juduged_max)
    min_max_replace = te.lang.cce.vadd(min_abs, max_abs)
    bool_min_max_product_less_zero = _less_compare_float32(min_max_replace,
                                                           tensor_zero)
    bool_min_max_product_more_zero = _less_compare_float32(tensor_zero,
                                                           min_max_replace)
    bool_both_zero = te.lang.cce.vadd(bool_min_max_product_less_zero,
                                      bool_min_max_product_more_zero)
    res = bool_both_zero

    return res


# pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-arguments
# pylint: disable=locally-disabled,invalid-name,redefined-builtin
@fusion_manager.register("fake_quant_with_min_max_vars_per_channel")
def fake_quant_with_min_max_vars_per_channel_compute(x, min, max,
                                                     y,
                                                     num_bits=8,
                                                     narrow_range=False,
                                                     kernel_name="fake_quant_with"
                                                     "_min_max_vars_"
                                                     "per_channel"):
    """
    Fake-quantize the 'inputs' tensor of type float and one of the shapes:
                  [d],[b, d] [b, h, w, d]
    via per-channel floats min and max of shape [d] to 'outputs' tensor
                  of same shape as inputs.

    Parameters
    ----------
    x: TVM tensor
        input tensor has shape and dtype attributes
        shape, x_shape equals y_shape,
        dtype, x_dtype equals y_dtype, only support fp32
    min: TVM tensor
        input tensor has shape and dtype attributes
        shape of min,min shape equals to max shape
        The shapes of min,max and shape_inputs last one dimension shoud be same
        the min data type,only support fp32
    max: TVM tensor
        input tensor has shape and dtype attributes
        shape of max,min shape equals to max shape
        The shapes of min,max and shape_inputs last one dimension shoud be same
        the max data type,only support fp32
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_vars_per_channel"

    Returns
    ------
    res: TVM tensor
        output tensor has shape and dtype attributes
        shape, y_shape equals x_shape
        dtype, y_dtype equals x_dtype, only support fp32
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    dtype = x.dtype
    min_broadcast = te.lang.cce.broadcast(min, shape, dtype)
    max_broadcast = te.lang.cce.broadcast(max, shape, dtype)
    # get nudged_min and nudged_max by _nudged_min_max_compute function
    nudged_min_nudged_max = _nudged_min_max_compute(min_broadcast,
                                                    max_broadcast, num_bits,
                                                    narrow_range)
    clamped_tmp = te.lang.cce.vmin(x, nudged_min_nudged_max[1])
    clamped = te.lang.cce.vmax(clamped_tmp, nudged_min_nudged_max[0])
    clamped_shifted = te.lang.cce.vsub(clamped, nudged_min_nudged_max[0])
    clamped_shifted_div_scale = te.lang.cce.vdiv(clamped_shifted,
                                                 nudged_min_nudged_max[2])
    result_tmp = te.lang.cce.vadds(clamped_shifted_div_scale,
                                   tvm.const(0.5, dtype))
    floor_result_tmp = te.lang.cce.floor(result_tmp)
    scale_product = te.lang.cce.vmul(floor_result_tmp, nudged_min_nudged_max[2])
    tmp_res = te.lang.cce.vadd(scale_product, nudged_min_nudged_max[0])
    # get bool_both_zero_value by _bool_both_zero_compute function
    bool_both_zero_value = _bool_both_zero_compute(min_broadcast, max_broadcast)
    res = te.lang.cce.vmul(tmp_res, bool_both_zero_value)

    return res


# pylint: disable=locally-disabled,redefined-builtin,invalid-name
@util.check_input_type(dict, dict, dict, dict, int, bool, str)
def fake_quant_with_min_max_vars_per_channel(x, min, max, y,
                                             num_bits=8,
                                             narrow_range=False,
                                             kernel_name="fake_quant_with_min_"
                                             "max_vars_per_channel"):
    """
    Generate fake_quant_with_min_max_vars_per_channel cce operator use
    fake_quant_with_min_max_vars_per_channel_compute

    Parameters
    ----------
    x: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data, assume x_shape equals y_shape,
        the data type, src_dtype equals dst_dtype, support fp32
    min: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of min,min shape equals to max shape and only 1 rank
        The shapes of min,max and shape_inputs last one dimension shoud be same
        the min data type,only support fp32
    max: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of max,min shape equals to max shape and only 1 rank
        The shapes of min,max and shape_inputs last one dimension shoud be same
        the max data type,only support fp32
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_vars_per_channel"

    Returns
    ------
    None
    """
    # get dtype and shape attributes
    shape_inputs = x.get("shape")
    dtype_inputs = x.get("dtype")
    shape_min = min.get("shape")
    dtype_min = min.get("dtype")
    shape_max = max.get("shape")
    dtype_max = max.get("dtype")
    # check_kernel_name & shape
    dtype_inputs = dtype_inputs.lower()
    dtype_min = dtype_min.lower()
    dtype_max = dtype_max.lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_inputs)
    util.check_shape_rule(shape_min, min_dim=1, max_dim=1)
    util.check_shape_rule(shape_max, min_dim=1, max_dim=1)
    util.check_tensor_shape_size(shape_inputs)
    util.check_tensor_shape_size(shape_min)
    util.check_tensor_shape_size(shape_max)
    # check input tensor data_type
    util.check_dtype_rule(dtype_inputs, "float32")
    util.check_dtype_rule(dtype_min, "float32")
    util.check_dtype_rule(dtype_max, "float32")
    # check shape_min & shape_max
    if list(shape_min) != list(shape_max):
        raise RuntimeError("The shapes of min and max shoud be same")
    if shape_min[0] != shape_inputs[-1]:
        raise RuntimeError(
            "The shapes of min,max and shape_inputs last"
            "one dimension shoud be same")
    # check num_bits range
    if num_bits > 16 or num_bits < 2:
        raise RuntimeError("numbits should be range[2,16]")

    # produce shape_min and shape_max for palceholder
    shape_min_broadcast, _, _ = util.produce_shapes(shape_min, shape_inputs)

    # definition of three input placeholders
    min_inputs = tvm.placeholder(shape_min_broadcast, name="min_inputs",
                                 dtype=dtype_min)
    max_inputs = tvm.placeholder(shape_min_broadcast, name="max_inputs",
                                 dtype=dtype_max)
    data_inputs = tvm.placeholder(shape_inputs, name="data_inputs",
                                  dtype=dtype_inputs)

    # get output by fake_quant_with_min_max_vars_per_channel_compute function
    res = fake_quant_with_min_max_vars_per_channel_compute(data_inputs,
                                                           min_inputs,
                                                           max_inputs, y,
                                                           num_bits,
                                                           narrow_range,
                                                           kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": (data_inputs, min_inputs, max_inputs, res)}
    te.lang.cce.cce_build_code(sch, config)
