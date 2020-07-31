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

binary_cross_entropy
"""
import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# eps value
SCALAR_EPS = 1e-12
# the type of None
NoneType = type(None)


# pylint: disable=invalid-name,too-many-arguments
# pylint: disable=unused-argument,too-many-locals
def op_select_format(x, y, weight, output,
                     reduction="mean",
                     kernel_name="binary_cross_entropy"):
    """op_select_format.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: list or tuple
        the list of output tensor.
    weight :
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size nbatch
    output :
        loss result after compute
    reduction :
        reduce configuration parameter: mean/sum/none. Default: mean
    kernel_name : str
        kernel name, default value is "binary_cross_entropy"

    Returns
    -------
    None.
    """
    is_support_5hd = True
    support_ori_format = ["NCHW", "NHWC"]
    input_ori_shape = x.get("ori_shape")
    input_ori_format = x.get("ori_format")
    shape_5hd_c0 = 16

    if input_ori_format not in support_ori_format \
            or len(input_ori_shape) != 4:
        is_support_5hd = False

    if input_ori_format == "NCHW":
        shape_c = input_ori_shape[1]
    else:
        shape_c = input_ori_shape[3]

    if shape_c % shape_5hd_c0 != 0:
        is_support_5hd = False

    if reduction in ("none",):
        is_support_5hd = True

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
        dtype_base = ["float16"]
    else:
        dtype_base = ["float16", "float"]

    dtype_base_out = dtype_base.copy()
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_5hd:
        dtype_base_out = dtype_base_out + dtype_base
        format_base_out = format_base_out + ["NC1HWC0"] * len(dtype_base)

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = gen_param(
        classify="input0", name="x", datatype=dtype_str,
        format=format_str)
    input1 = gen_param(
        classify="input1", name="y", datatype=dtype_str,
        format=format_str)
    input2 = gen_param(
        classify="input2", name="weight", datatype=dtype_str,
        format=format_str)
    output0 = gen_param(
        classify="output0", name="output", datatype=dtype_str,
        format=format_str)
    param_list = [input0, input1, input2, output0]

    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@fusion_manager.register("binary_cross_entropy")
def binary_cross_entropy_compute(x, y, weight, output,
                                 reduction, kernel_name):
    """
    calculating binary_cross_entropy

    Parameters
    ----------
    x : TVM tensor
        the output of previous layer
    y : TVM tensor
        label
    weight :
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size nbatch
    output :
        loss result after compute
    reduction :
        reduce configuration parameter: mean/sum/none. Default: mean
    kernel_name : str
        kernel name, default value is "binary_cross_entropy"

    Returns
    -------
    result : TVM tensor
        output tensor
    """
    ori_dtype = x.dtype
    trans_dtype = ori_dtype
    shape = te.lang.cce.util.shape_to_list(x.shape)
    if ori_dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        x = te.lang.cce.cast_to(x, "float32")
        y = te.lang.cce.cast_to(y, "float32")
        if weight is not None:
            weight = te.lang.cce.cast_to(weight, "float32")
        trans_dtype = "float32"

    const_one = tvm.const(1, trans_dtype)
    const_neg_one = tvm.const(-1, trans_dtype)
    # calcu value : y * log(x)
    x = te.lang.cce.vmaxs(x, tvm.const(SCALAR_EPS, trans_dtype))
    x_log_tmp = te.lang.cce.vlog(x, priority_flag=1)
    data_mul1 = te.lang.cce.vmul(x_log_tmp, y)
    # calcu value : (1-y) * log(1-x)
    x_neg_tmp = te.lang.cce.vmuls(x, const_neg_one)
    x1_tmp = te.lang.cce.vadds(x_neg_tmp, const_one)
    y_neg_tmp = te.lang.cce.vmuls(y, const_neg_one)
    y1_tmp = te.lang.cce.vadds(y_neg_tmp, const_one)
    x1_tmp = te.lang.cce.vmaxs(x1_tmp, tvm.const(SCALAR_EPS, trans_dtype))
    x1_log_tmp = te.lang.cce.vlog(x1_tmp, priority_flag=1)
    data_mul2 = te.lang.cce.vmul(x1_log_tmp, y1_tmp)
    # calcu value : y * log(x) + (1-y) * log(1-x)
    data_sum = te.lang.cce.vadd(data_mul1, data_mul2)
    # calcu value : -(y * log(x) + (1-y) * log(1-x))
    result = te.lang.cce.vmuls(data_sum, const_neg_one)

    if weight is not None:
        result = te.lang.cce.vmul(result, weight)

    # get total number of tensor
    reduce_elts = 1.0
    for i in shape:
        reduce_elts *= i
    cof = reduce_elts**(-1)

    # get total axis for reduce
    axis_d = []
    for i, _ in enumerate(shape):
        axis_d.append(i)
    axis_d = util.axis_check(len(shape), axis_d)

    if reduction == "mean":
        result = te.lang.cce.vmuls(result, cof)
        result = te.lang.cce.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "sum":
        result = te.lang.cce.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "none":
        pass

    if ori_dtype == "float16":
        result = te.lang.cce.cast_to(result, "float16")

    return result


@util.check_input_type(dict, dict, (dict, NoneType), dict, str, str)
def binary_cross_entropy(x, y, weight, output,
                         reduction="mean",
                         kernel_name="binary_cross_entropy"):
    """
    calculating data
    res = -w (y ln(x) + (1-y) ln(1-x))
    if reduction == sum:  res = reduce_sum(res)            output a scalar
    if reduction == mean:  res = reduce_sum(res)/data_len  output a scalar
    if reduction == none: res = res   output a tensor

    Parameters
    ----------
    x : dict
        shape and dtype of tensor predict
    y : dict
        shape and dtype of tensor target,
        should be same shape and dtype as predict
    weight : None or TVM tensor
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size nbatch
    output : dict
        shape and dtype of output, loss result after compute
    reduction : str
        Specifies the reduction to apply to the output:'none' | 'mean' | 'sum'
         Default: 'mean'
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number
                of elements in the output
        'sum': the output will be summed. Note: size_average and reduce
               are in the process of being deprecated
               and in the meantime, specifying either of those
               two args will override reduction.
    kernel_name : str
        kernel name, default value is "binary_cross_entropy"

    Returns
    -------
    None
    """
    predict_shape = x.get("shape")
    predict_dtype = x.get("dtype")
    predict_dtype_lower = predict_dtype.lower()

    target_shape = y.get("shape")
    target_dtype = y.get("dtype")
    target_dtype_lower = target_dtype.lower()

    # check dtype
    dtype_list = ("float16", "float32")
    util.check_dtype_rule(predict_dtype, dtype_list)
    util.check_dtype_rule(target_dtype, dtype_list)
    util.compare_tensor_dict_key(x, y, "dtype")

    # check shape
    util.check_shape_rule(predict_shape)
    util.check_shape_rule(target_shape)
    util.compare_tensor_dict_key(x, y, "shape")

    # check kernel_name
    util.check_kernel_name(kernel_name)

    data_weight = None
    if weight is not None:
        weight_shape = weight.get("shape")
        weight_dtype = weight.get("dtype")
        weight_dtype_lower = weight_dtype.lower()
        util.check_dtype_rule(weight_dtype, dtype_list)
        util.compare_tensor_dict_key(x, weight, "dtype")
        util.check_shape_rule(weight_shape)
        shape_size = util.check_tensor_shape_size(weight_shape)
        util.compare_tensor_dict_key(x, weight, "shape")
        data_weight = tvm.placeholder([shape_size], name="data_weight",
                                      dtype=weight_dtype_lower)
    shape_size = util.check_tensor_shape_size(predict_shape)
    data_predict = tvm.placeholder([shape_size], name="data_predict",
                                   dtype=predict_dtype_lower)
    shape_size = util.check_tensor_shape_size(target_shape)
    data_target = tvm.placeholder([shape_size], name="data_target",
                                  dtype=target_dtype_lower)

    if reduction not in ("mean", "sum", "none"):
        raise RuntimeError("reduction type should in mean/sum/none")

    res = binary_cross_entropy_compute(data_predict, data_target,
                                       data_weight, output,
                                       reduction, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    if weight is None:
        config = {"name": kernel_name,
                  "tensor_list": [data_predict, data_target,
                                  res]}
    else:
        config = {"name": kernel_name,
                  "tensor_list": [data_predict, data_target,
                                  data_weight, res]}

    te.lang.cce.cce_build_code(schedule, config)

