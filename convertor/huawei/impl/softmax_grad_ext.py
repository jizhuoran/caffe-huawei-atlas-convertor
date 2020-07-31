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

confusion_softmax_grad_ext
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


SIZE_SIXTEEN = 16


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=locally-disabled,too-many-locals,unused-variable
# pylint: disable=locally-disabled,invalid-name,unidiomatic-typecheck
# pylint: disable=locally-disabled,too-many-branches
def _division_sixteen(shape):

    if len(shape) < 2:
        if shape[-1] == 0:
            raise RuntimeError("value of shape is illegal")
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        raise RuntimeError("value of shape is illegal")

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True

    return False


def op_select_format(grad, x1, x2, y, axis, keepdims,
                     kernel_name="softmax_grad_ext"):
    """
    select format dynamically
    """
    origin_shape0 = util.scalar2tensor_one(grad.get("ori_shape"))
    origin_shape1 = util.scalar2tensor_one(x1.get("ori_shape"))
    origin_shape2 = util.scalar2tensor_one(x2.get("ori_shape"))

    condition_0 = len(origin_shape2) == 1 and origin_shape2[0] == 1
    condition_1 = _division_sixteen(origin_shape0)
    condition_2 = _division_sixteen(origin_shape1)

    if condition_0 and condition_1 and condition_2:
        # NZ + NZ + Scalar
        input0 = gen_param(classify="input0", name="grad",
                           datatype="float16,float",
                           format="FRACTAL_NZ, FRACTAL_NZ")
        input1 = gen_param(classify="input1", name="x1",
                           datatype="float16,float",
                           format="FRACTAL_NZ, FRACTAL_NZ")
        input2 = gen_param(classify="input2", name="x2",
                           datatype="float16,float",
                           format="ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float",
                            format="FRACTAL_NZ,FRACTAL_NZ")
    else:
        # ND+ND+ND
        input0 = gen_param(classify="input0", name="grad",
                           datatype="float16,float",
                           format="ND,ND")
        input1 = gen_param(classify="input1", name="x1",
                           datatype="float16,float",
                           format="ND,ND")
        input2 = gen_param(classify="input2", name="x2",
                           datatype="float16,float",
                           format="ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float",
                            format="ND,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_nz_rule(grad, x1, x2, axis):

    shape_grad = util.scalar2tensor_one(grad.get("shape"))
    shape_x1 = util.scalar2tensor_one(x1.get("shape"))
    shape_x2 = util.scalar2tensor_one(x2.get("shape"))

    ori_shape = util.scalar2tensor_one(grad.get("ori_shape"))

    format_grad = grad.get("format")
    format_x1 = x1.get("format")
    format_x2 = x2.get("format")

    format_target = [["FRACTAL_NZ", "FRACTAL_NZ", "ND"],
                     ["FRACTAL_NZ", "FRACTAL_NZ", "NCHW"],
                     ["FRACTAL_NZ", "FRACTAL_NZ", "NHWC"]]
    format_list = [format_grad, format_x1, format_x2]

    if format_list not in format_target:
        raise RuntimeError("Combination of format is illegal in nz+nd ")

    if not(len(shape_x2) == 1 and shape_x2[0] == 1):
        raise RuntimeError("the last input tensor should be scalar")


    forward = [i for i in range(len(ori_shape))]
    back_forwad = [i-len(ori_shape) for i in range(len(ori_shape))]
    if type(axis) in [list, tuple]:
        axis = list(axis)
        flag_0 = False
        flag_1 = False
        axis_new = []
        for value in axis:
            if value in [forward[-1], back_forwad[-1]]:
                flag_0 = True
            elif value in [forward[-2], back_forwad[-2]]:
                flag_1 = True
            else:
                axis_new.append(value)

        if flag_0:
            axis_new.append(-1)
            axis_new.append(-4)
        if flag_1:
            axis_new.append(-2)
            axis_new.append(-3)
    else:
        if axis >= 0:
            if axis == forward[-1]:
                axis_new = [-1, -4]
            elif axis == forward[-2]:
                axis_new = [-2, -3]
            else:
                axis_new = axis

        else:
            if axis == back_forwad[-1]:
                axis_new = [-1, -4]
            elif axis == back_forwad[-2]:
                axis_new = [-2, -3]
            else:
                axis_new = axis

    return axis_new


def shape_broadcast(data_1, data_2):
    """
    broadcast the two input

    Parameters
    ----------
    data_1: TVM tensor
        the placeholder of first input data
    data_2: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input

    Returns
    -------
    res : output of the data's divide
    """
    shape_x = te.lang.cce.util.shape_to_list(data_1.shape)
    shape_y = te.lang.cce.util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
        util.check_tensor_shape_size(shape_max)
        data_1 = _broadcast_nz(data_1, shape_max)
        data_2 = _broadcast_nz(data_2, shape_max)

    return data_1, data_2


def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = te.lang.cce.util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = te.lang.cce.broadcast(tensor, temp_shape)
    tensor = te.lang.cce.broadcast(tensor, shape)
    return tensor


@fusion_manager.register("softmax_grad_ext")
def softmax_grad_ext_compute(data_grad, data_x1, data_x2,
                             y, axis, keepdims,
                             kernel_name="softmax_grad_ext"):
    """
    apply one adam calculation function

    Parameters
    ----------
    data_grad: TVM tensor
         the input tensor of mul and sub
    data_x1: TVM tensor
         the input tensor of mul and mul_1
    data_x2: TVM tensor
         the input tensor of mul_1
    y: dict
         the output tensor of mul_grad
    axis: int, list, tuple
        the axis for reduce.
    keepdims: bool
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "softmax_grad_ext"

    Returns
    -------
    output tensor
    """

    # mul
    data_grad, data_x1 = shape_broadcast(data_grad, data_x1)
    mul_result = te.lang.cce.vmul(data_grad, data_x1)

    # sum
    dtype = mul_result.dtype
    if dtype == "float16":
        mul_result = te.lang.cce.cast_to(mul_result, "float32")
    sum_result = te.lang.cce.sum(mul_result, axis=axis, keepdims=keepdims)
    if dtype == "float16":
        sum_result = te.lang.cce.cast_to(sum_result, "float16")

    # sub
    data_grad, sum_result = shape_broadcast(data_grad, sum_result)
    sub_result = te.lang.cce.vsub(data_grad, sum_result)

    # mul_1
    data_x1, data_x2 = shape_broadcast(data_x1, data_x2)
    mul_1_result = te.lang.cce.vmul(data_x1, data_x2)

    # mul_grad
    sub_result, mul_1_result = shape_broadcast(sub_result, mul_1_result)
    mul_grad_result = te.lang.cce.vmul(sub_result, mul_1_result)
    res = [mul_grad_result]

    return res


@util.check_input_type(dict, dict, dict, dict, (int, list, tuple), bool, str)
def softmax_grad_ext(grad, x1, x2, y, axis, keepdims,
                     kernel_name="softmax_grad_ext"):
    """
    function: softmax_grad_ext

    Parameters
    ----------
    grad: dict
         the input tensor of mul and sub
    x1: dict
         the input tensor of mul and mul_1
    x2: dict
         the input tensor of mul_1
    y: dict
         the output tensor of mul_grad
    axis: int, list, tuple
        the axis for reduce.
    keepdims: bool
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "softmax_grad_ext"

    Returns
    -------
    None
    """
    shape_grad = util.scalar2tensor_one(grad.get("shape"))
    shape_x1 = util.scalar2tensor_one(x1.get("shape"))
    shape_x2 = util.scalar2tensor_one(x2.get("shape"))

    dtype_grad = grad.get("dtype").lower()
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()

    if grad.get("format") == "FRACTAL_NZ":
        axis = _check_nz_rule(grad, x1, x2, axis)

    util.check_kernel_name(kernel_name)

    shape_grad, shape_x1, shape_max_mul = \
        util.produce_shapes(shape_grad, shape_grad)
    shape_x1, shape_x2, shape_max_mul1 = \
        util.produce_shapes(shape_x1, shape_x2)

    data_grad = tvm.placeholder(shape_grad,
                                name="data_grad",
                                dtype=dtype_grad)
    data_x1 = tvm.placeholder(shape_x1,
                              name="data_x1",
                              dtype=dtype_x1)
    data_x2 = tvm.placeholder(shape_x2,
                              name="data_x2",
                              dtype=dtype_x2)

    res = softmax_grad_ext_compute(data_grad, data_x1, data_x2,
                                   y, axis, keepdims, kernel_name)

    inputlist = [data_grad, data_x1, data_x2]

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}

    te.lang.cce.cce_build_code(sch, config)
