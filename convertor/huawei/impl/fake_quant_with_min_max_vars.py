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

fake_quant_with_min_max_vars
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce

# define a scalar for add
HALF_ONE = 0.5
# define zero for broadcast
ZERO_VALUE = 0
# define one for broadcast
ONE_VALUE = 1


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals
def _less_compare_float32(data_x, data_y):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    data_x : tensor
        tensor x
    data_y : tensor
        tensor y

    Returns
    -------
    the compare result
    """
    shape_inputs = te.lang.cce.util.shape_to_list(data_x.shape)
    # minimum num of float32 2**(-126)
    data_min = te.lang.cce.broadcast(tvm.const(2 ** (-126), dtype="float32"),
                                     shape_inputs, "float32")
    data_zero = te.lang.cce.broadcast(tvm.const(0, dtype="float32"),
                                      shape_inputs, "float32")
    res_sub = te.lang.cce.vsub(data_y, data_x)
    res_min = te.lang.cce.vmin(res_sub, data_min)
    res_max = te.lang.cce.vmax(res_min, data_zero)
    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 62/62/2 to adaptor 126
    res_muled = te.lang.cce.vmuls(res_max, tvm.const(2 ** 62, dtype="float32"))
    res_mul = te.lang.cce.vmuls(res_muled, tvm.const(2 ** 62, dtype="float32"))
    res = te.lang.cce.vmuls(res_mul, tvm.const(2 ** 2, dtype="float32"))

    return res


def _bool_both_zero_compute(juduged_min, juduged_max):
    """
    if input min and max are both zero then output_date will be all zero
    so need a juduge compute tensor

    Parameters:
    ----------
    juduged_min : tensor
        tensor min
    juduged_max : tensor
        tensor max

    Returns
    -------
    res : tensor
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


def _nudged_min_max_compute(zero_point_from_min, quant_min, quant_max, scale,
                            min):
    """
        Compute nudged_min, nudged_max operation.

        Parameters
        ----------
        zero_point_from_min: TVM tensor
               the placeholder of zerp_point_from_min
        quant_min: TVM tensor
               the placeholder of quant_min
        quant_max: TVM tensor
               the placeholder of quant_max
        scale: TVM tensor
                the placeholder of scale
        min: TVM tensor
                the placeholder of min

        Returns
        ------
        res: list
            the calculation results
        """
    tensor_zero = te.lang.cce.vmuls(min, tvm.const(ZERO_VALUE, "float32"))
    bool_less_quant_min_float = _less_compare_float32(zero_point_from_min,
                                                      quant_min)
    bool_more_quant_max_float = _less_compare_float32(quant_max,
                                                      zero_point_from_min)
    less_quant_min_float = te.lang.cce.vmul(quant_min,
                                            bool_less_quant_min_float)
    more_quant_max_float = te.lang.cce.vmul(quant_max,
                                            bool_more_quant_max_float)
    tensor_one = te.lang.cce.vadds(tensor_zero, tvm.const(ONE_VALUE, "float32"))
    bool_not_less_quant_min_float = te.lang.cce.vsub(tensor_one,
                                                     bool_less_quant_min_float)
    bool_not_more_quant_max_float = te.lang.cce.vsub(tensor_one,
                                                     bool_more_quant_max_float)
    bool_between_min_max = te.lang.cce.vmul(bool_not_less_quant_min_float,
                                            bool_not_more_quant_max_float)
    between_min_max_float = te.lang.cce.vmul(zero_point_from_min,
                                             bool_between_min_max)
    between_min_max_add_half_one = te.lang.cce.vadds(between_min_max_float,
                                                     tvm.const(HALF_ONE,
                                                               "float32"))
    between_min_max_round = te.lang.cce.floor(between_min_max_add_half_one)
    nudged_zero_point_tmp = te.lang.cce.vadd(less_quant_min_float,
                                             more_quant_max_float)
    nudged_zero_point = te.lang.cce.vadd(nudged_zero_point_tmp,
                                         between_min_max_round)
    nudged_min_tmp = te.lang.cce.vsub(quant_min, nudged_zero_point)
    nudged_max_tmp = te.lang.cce.vsub(quant_max, nudged_zero_point)
    nudged_min = te.lang.cce.vmul(nudged_min_tmp, scale)
    nudged_max = te.lang.cce.vmul(nudged_max_tmp, scale)

    return nudged_min, nudged_max


@fusion_manager.register("fake_quant_with_min_max_vars")
def fake_quant_with_min_max_vars_compute(x, min, max, y, num_bits,
                                         narrow_range,
                                         kernel_name="fake_quant_"
                                                     "with_min_max_vars"):
    """
    Compute FakeQuantWithMinMaxVars operation.

    Parameters
    ----------
    x: TVM tensor
           the placeholder of x
    min: TVM tensor
           the placeholder of min
    max: TVM tensor
           the placeholder of max
    y: dict
            shape and dtype of fake quant output
    num_bits: int
            define the range of quant max
    narrow_range: bool
            define the range of quant min
    kernel_name : string
            cce kernel name, default value is "bitwise_or"

    Returns
    ------
    res: tensor
        the calculation results
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    quant_max = 2 ** num_bits - 1

    if not narrow_range:
        quant_min = 0
    else:
        quant_min = 1

    quant_max = te.lang.cce.broadcast(quant_max, shape)
    quant_min = te.lang.cce.broadcast(quant_min, shape)
    max = te.lang.cce.broadcast(max, shape)
    min = te.lang.cce.broadcast(min, shape)

    scale = te.lang.cce.vdiv(te.lang.cce.vsub(max, min),
                             te.lang.cce.vsub(quant_max, quant_min))
    zero_point_from_min = te.lang.cce.vsub(quant_min,
                                           te.lang.cce.vdiv(min, scale))
    nudged_min, nudged_max = _nudged_min_max_compute(zero_point_from_min,
                                                     quant_min,
                                                     quant_max, scale, min)

    clamped_tmp = te.lang.cce.vmin(x, nudged_max)
    clamped = te.lang.cce.vmax(clamped_tmp, nudged_min)
    clamped_shifted = te.lang.cce.vsub(clamped, nudged_min)
    result_tmp = te.lang.cce.floor(
        te.lang.cce.vadds(te.lang.cce.vdiv(clamped_shifted, scale),
                          tvm.const(0.5, "float32")))
    result = te.lang.cce.vadd(te.lang.cce.vmul(result_tmp, scale), nudged_min)

    bool_both_zero_value = _bool_both_zero_compute(min, max)
    res = te.lang.cce.vmul(result, bool_both_zero_value)

    return res


# pylint: disable=locally-disabled,redefined-builtin,invalid-name
@util.check_input_type(dict, dict, dict, dict, int, bool, str)
def fake_quant_with_min_max_vars(x, min, max, y, num_bits,
                                 narrow_range,
                                 kernel_name="fake_quant_with_min_max_vars"):
    """
    algorithm: calculate the fake quant value of input tensor
    calculating data's fake quant

    Parameters
    ----------
    x: dict
           shape and dtype of input data
    min: dict
         shape and dtype of min
    max: dict
         shape and dtype of max
    y: dict
            shape and dtype of fake quant output
    num_bits: int
                  define the range of quant max
    narrow_range: bool
                  define the range of quant min
    kernel_name : string
                  cce kernel name, default value is
                  "fake_quant_with_min_max_vars"

    Returns
    -------
    None
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    min_shape = min.get("shape")
    min_dtype = min.get("dtype")
    max_shape = max.get("shape")
    max_dtype = max.get("dtype")

    min_shape = util.scalar2tensor_one(min_shape)
    max_shape = util.scalar2tensor_one(max_shape)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_shape_rule(min_shape, 1, 1, 1)
    util.check_shape_rule(max_shape, 1, 1, 1)
    util.check_tensor_shape_size(input_shape)
    util.check_tensor_shape_size(min_shape)
    util.check_tensor_shape_size(max_shape)

    if num_bits > 16 or num_bits < 2:
        raise RuntimeError(
            "The value of num_bits must be between"
            "2 and 16")

    check_tuple = ("float32",)
    x_type = input_dtype.lower()
    min_dtype = min_dtype.lower()
    max_dtype = max_dtype.lower()
    util.check_dtype_rule(x_type, check_tuple)
    util.check_dtype_rule(min_dtype, check_tuple)
    util.check_dtype_rule(max_dtype, check_tuple)
    input_shape = (functools_reduce(lambda x, y: x * y, input_shape[:]),)
    shape_min, shape_max, shape_broadcast = util.produce_shapes(min_shape,
                                                                input_shape)
    data = tvm.placeholder(input_shape, dtype=x_type, name="data_input")
    data_min = tvm.placeholder(shape_min, dtype=min_dtype, name="data_min")
    data_max = tvm.placeholder(shape_min, dtype=max_dtype, name="data_max")

    res = fake_quant_with_min_max_vars_compute(data, data_min, data_max,
                                               y, num_bits, narrow_range,
                                               kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data, data_min, data_max, res)}

    te.lang.cce.cce_build_code(schedule, config)
