#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

softmax_cross_entropy_with_logits
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util

# compute needed,scalar -1
SCALAR_MINUS_ONE = -1

# limit of input dimvalue
MAX_SHAPE_NUM = 10000000


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@fusion_manager.register("softmax_cross_entropy_with_logits")
def softmax_cross_entropy_with_logits_nchw_compute(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        kernel_name="softmax_cross_entropy_with_logits"):
    """
    Computes softmax cross entropy cost.
    softmax = e^(x-max) / ∑(e^(x-max))
    log(softmax) = (x-max) - log(∑e^(x-max))
    cross_entropy = -∑(y * log⁡(softmax))

    Parameters
    # ----------
    input_features: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    input_labels: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_features".
    """
    shape_features = te.lang.cce.util.shape_to_list(input_features.shape)
    shape_labels = te.lang.cce.util.shape_to_list(input_labels.shape)
    dtype = input_features.dtype.lower()

    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            util.produce_shapes(shape_features, shape_labels)
        input_features = te.lang.cce.broadcast(input_features, shape_broadcast,
                                               dtype)
        input_labels = te.lang.cce.broadcast(input_labels, shape_broadcast,
                                             dtype)
    else:
        shape_broadcast = shape_features

    data_max = te.lang.cce.reduce_max(input_features, axis=1, keepdims=True)
    data_max_broadcast = te.lang.cce.broadcast(data_max, shape_broadcast)
    data_sub = te.lang.cce.vsub(input_features, data_max_broadcast)
    data_exp = te.lang.cce.vexp(data_sub)
    data_sum = te.lang.cce.sum(data_exp, axis=1, keepdims=True)
    data_sum_broadcast = te.lang.cce.broadcast(data_sum, shape_broadcast)
    data_div = te.lang.cce.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = te.lang.cce.vlog(data_sum_broadcast)
    data_log = te.lang.cce.vsub(data_sub, data_log_tmp)
    data_mul = te.lang.cce.vmul(input_labels, data_log)
    data_muls = te.lang.cce.vmuls(data_mul, SCALAR_MINUS_ONE)
    loss = te.lang.cce.sum(data_muls, axis=1, keepdims=True)
    backprop = te.lang.cce.vsub(data_div, input_labels)

    res = [loss, backprop]
    return res


@fusion_manager.register("softmax_cross_entropy_with_logits")
def softmax_cross_entropy_with_logits_compute(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        kernel_name="softmax_cross_entropy_with_logits"):
    """
    Computes softmax cross entropy cost.
    softmax = e^(x-max) / ∑(e^(x-max))
    log(softmax) = (x-max) - log(∑e^(x-max))
    cross_entropy = -∑(y * log⁡(softmax))

    Parameters
    # ----------
    input_features: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    input_labels: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_features".
    """
    shape_features = te.lang.cce.util.shape_to_list(input_features.shape)
    shape_labels = te.lang.cce.util.shape_to_list(input_labels.shape)
    dtype = input_features.dtype.lower()

    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            util.produce_shapes(shape_features, shape_labels)
        input_features = te.lang.cce.broadcast(input_features, shape_broadcast,
                                               dtype)
        input_labels = te.lang.cce.broadcast(input_labels, shape_broadcast,
                                             dtype)
    else:
        shape_broadcast = shape_features

    # Last axis is too large, use L1 workspace compute
    # and special designed schedule
    current_csize_maximum_fp32 = 15360
    high_perf_csize_maximum_fp32 = 20000
    if current_csize_maximum_fp32 < shape_broadcast[1] < \
            high_perf_csize_maximum_fp32:
        return softmax_cross_entropy_with_logits_compute_ex(input_features,
                                                            input_labels)
    has_improve_precision = False
    if dtype == "float16" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                "float32"):
        input_features = te.lang.cce.cast_to(input_features, "float32")
        input_labels = te.lang.cce.cast_to(input_labels, "float32")
        has_improve_precision = True

    data_max = te.lang.cce.reduce_max(input_features, axis=-1, keepdims=True)
    data_max_broadcast = te.lang.cce.broadcast(data_max, shape_broadcast)
    data_sub = te.lang.cce.vsub(input_features, data_max_broadcast)
    data_exp = te.lang.cce.vexp(data_sub)
    data_sum = te.lang.cce.sum(data_exp, axis=-1, keepdims=True)
    data_sum_broadcast = te.lang.cce.broadcast(data_sum, shape_broadcast)
    data_div = te.lang.cce.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = te.lang.cce.vlog(data_sum_broadcast)
    data_log = te.lang.cce.vsub(data_sub, data_log_tmp)
    data_mul = te.lang.cce.vmul(input_labels, data_log)
    data_muls = te.lang.cce.vmuls(data_mul, SCALAR_MINUS_ONE)
    loss = te.lang.cce.sum(data_muls, axis=-1)
    backprop = te.lang.cce.vsub(data_div, input_labels)

    if has_improve_precision:
        loss = te.lang.cce.cast_to(loss, "float16")
        backprop = te.lang.cce.cast_to(backprop, "float16")

    res = [loss, backprop]

    return res


def softmax_cross_entropy_with_logits_compute_ex(input_features,
                                                 input_labels):
    """
    Computes softmax cross entropy cost.
    softmax = e^(x-max) / ∑(e^(x-max))
    log(softmax) = (x-max) - log(∑e^(x-max))
    cross_entropy = -∑(y * log⁡(softmax))

    Parameters
    # ----------
    input_features: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    input_labels: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_features".
    """
    shape_features = te.lang.cce.util.shape_to_list(input_features.shape)
    shape_labels = te.lang.cce.util.shape_to_list(input_labels.shape)
    dtype = input_features.dtype.lower()

    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            util.produce_shapes(shape_features, shape_labels)
        input_features = te.lang.cce.broadcast(input_features, shape_broadcast,
                                               dtype)
        input_labels = te.lang.cce.broadcast(input_labels, shape_broadcast,
                                             dtype)
    else:
        shape_broadcast = shape_features

    if dtype == "float16":
        input_features = te.lang.cce.cast_to(input_features, "float32")
        input_labels = te.lang.cce.cast_to(input_labels, "float32")

    with tvm.tag_scope("last_axis_reduce_max"):
        reduce_axis = tvm.reduce_axis((0, shape_broadcast[1]), name="rax0")
        data_max = tvm.compute((shape_broadcast[0], 1),
                               lambda upper, lower:
                               tvm.max(input_features[upper, reduce_axis],
                                       axis=reduce_axis),
                               name="last_axis_reduce_max")
    with tvm.tag_scope("elewise_binary_sub_scalar_L1"):
        data_sub = tvm.compute(input_features.shape,
                               lambda higher, lower:
                               input_features[higher][lower] -
                               data_max[higher][0],
                               name="manual_sub_0")
    data_exp = te.lang.cce.vexp(data_sub)
    data_sum = te.lang.cce.sum(data_exp, axis=-1, keepdims=True)
    with tvm.tag_scope("elewise_binary_div"):
        data_div = tvm.compute(data_exp.shape,
                               lambda higher, lower:
                               data_exp[higher][lower] / data_sum[higher][0],
                               name="manual_div_0")
    data_log_tmp = te.lang.cce.vlog(data_sum)
    with tvm.tag_scope("elewise_get_L1_workspace"):
        fake_buffer = tvm.compute(data_sub.shape,
                                  lambda higher, lower: tvm.const(0, "float32"),
                                  name="get_L1_workspace")
    with tvm.tag_scope("elewise_binary_sub"):
        data_log = tvm.compute(data_sub.shape,
                               lambda higher, lower:
                               fake_buffer[higher][lower] -
                               data_log_tmp[higher][0],
                               name="manual_sub_1")
    data_mul = te.lang.cce.vmul(input_labels, data_log)
    with tvm.tag_scope("last_axis_reduce_sum_reuse"):
        reduce_axis = tvm.reduce_axis((0, shape_broadcast[1]), name="rax1")
        loss = tvm.compute((shape_broadcast[0], 1),
                           lambda upper, lower:
                           tvm.sum(data_mul[upper, reduce_axis],
                                   axis=reduce_axis),
                           name="last_axis_reduce_sum_reuse")
    loss = te.lang.cce.vmuls(loss, SCALAR_MINUS_ONE)
    backprop = te.lang.cce.vsub(data_div, input_labels)

    if dtype == "float16":
        loss = te.lang.cce.cast_to(loss, "float16")
        backprop = te.lang.cce.cast_to(backprop, "float16")

    res = [loss, backprop]

    return res


@util.check_input_type(dict, dict, dict, dict, str)
def softmax_cross_entropy_with_logits(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        kernel_name="softmax_cross_entropy_with_logits"):
    """
    Computes softmax cross entropy cost.

    Parameters
    ----------
    input_features: dict
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    input_labels: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"

    Returns:
    None
    """
    shape_features = input_features.get("shape")
    shape_labels = input_labels.get("shape")

    util.compare_tensor_dict_key(input_features, input_labels, "dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_features, max_shape_num=MAX_SHAPE_NUM)
    util.check_shape_rule(shape_labels, max_shape_num=MAX_SHAPE_NUM)
    util.check_tensor_shape_size(shape_features)
    util.check_tensor_shape_size(shape_labels)

    check_list = ("float16", "float32")
    input_dtype = input_features.get("dtype").lower()
    util.check_dtype_rule(input_dtype, check_list)

    if len(shape_features) == 4:
        if len(shape_features) != len(shape_labels):
            raise RuntimeError("The length of two inputs must be same")
        if input_dtype != "float32":
            raise RuntimeError("Not supported dtype!")
        data_features = tvm.placeholder(shape_features, dtype=input_dtype,
                                        name="data_features")
        data_labels = tvm.placeholder(shape_labels, dtype=input_dtype,
                                      name="data_labels")
        res = softmax_cross_entropy_with_logits_nchw_compute(data_features,
                                                             data_labels,
                                                             output_loss,
                                                             output_backprop)
    else:
        if len(shape_features) == 1 and len(shape_labels) == 1:
            raise RuntimeError("The rank of two inputs can not be 1 at the same"
                               "time")
        if len(shape_features) > 2 or len(shape_labels) > 2:
            raise RuntimeError("logits and labels must be either 2-dimensional,"
                               "or broadcasted to 2-dimensional")
        if len(shape_features) == 1 or len(shape_labels) == 1:
            shape_features, shape_labels, shape_broadcast = \
                util.produce_shapes(shape_features, shape_labels)
            util.check_tensor_shape_size(shape_broadcast)

        data_features = tvm.placeholder(shape_features, dtype=input_dtype,
                                        name="data_features")
        data_labels = tvm.placeholder(shape_labels, dtype=input_dtype,
                                      name="data_labels")
        res = softmax_cross_entropy_with_logits_compute(data_features,
                                                        data_labels,
                                                        output_loss,
                                                        output_backprop)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [data_features, data_labels] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
