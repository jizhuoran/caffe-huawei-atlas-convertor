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

confusion_mul_grad
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=locally-disabled,too-many-locals,unused-variable
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
        data_1 = te.lang.cce.broadcast(data_1, shape_max)
        data_2 = te.lang.cce.broadcast(data_2, shape_max)

    return data_1, data_2


@fusion_manager.register("confusion_mul_grad")
def confusion_mul_grad_compute(data_input0, data_input1, data_input2,
                               output0, output1,
                               axis, keep_dims,
                               kernel_name="confusion_mul_grad"):
    """
    mul_grad calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul and mul_1
    data_input2: TVM tensor
         the input tensor of mul_1
    output0: TVM tensor
         the output tensor of mul
    output1: TVM tensor
         the output tensor of sum
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keep_dims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "confusion_mul_grad"

    Returns
    -------
    output tensor
    """

    # mul
    data_input0, data_input1 = shape_broadcast(data_input0, data_input1)
    result0 = te.lang.cce.vmul(data_input0, data_input1)

    # mul_1
    data_input1, data_input2 = shape_broadcast(data_input1, data_input2)
    mul_1_result = te.lang.cce.vmul(data_input1, data_input2)

    # temp compute for tvm
    zero_tmp = te.lang.cce.vmuls(result0, 0)
    mul_1_result = te.lang.cce.vadd(mul_1_result, zero_tmp)

    # sum
    dtype = mul_1_result.dtype
    if dtype == "float16":
        mul_1_result = te.lang.cce.cast_to(mul_1_result, "float32")
    result1 = te.lang.cce.sum(mul_1_result, axis=axis, keepdims=keep_dims)
    if dtype == "float16":
        result1 = te.lang.cce.cast_to(result1, "float16")

    res = [result0, result1]

    return res


@util.check_input_type(dict, dict, dict, dict, dict,
                       (int, list, tuple), bool, str)
def confusion_mul_grad(input0, input1, input2,
                       output0, output1,
                       axis, keep_dims,
                       kernel_name="confusion_mul_grad"):
    """
    function: mul_grad

    Parameters
    ----------
    input0: dict
         the dict of input of mul, and dtype supports 'float16', 'float32'
    input1: dict
         the dict of input of mul and mul_1,
         and dtype supports 'float16', 'float32'
    input2: dict
         the dict of input of mul_1, and dtype supports 'float16', 'float32'
    output0: dict
         the dict of output of mul, and dtype supports 'float16', 'float32'
    output1: dict
         the dict of output of sum, and dtype supports 'float16', 'float32'
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keep_dims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is confusion_mul_grad

    Returns
    -------
    None
    """
    shape_input0 = util.scalar2tensor_one(input0.get("shape"))
    shape_input1 = util.scalar2tensor_one(input1.get("shape"))
    shape_input2 = util.scalar2tensor_one(input2.get("shape"))

    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()
    data_format1 = input1.get("format").upper()

    util.check_kernel_name(kernel_name)

    shape_input0, shape_input1, shape_max_mul = \
        util.produce_shapes(shape_input0, shape_input1)
    shape_input1, shape_input2, shape_max_mul1 = \
        util.produce_shapes(shape_input1, shape_input2)

    data_input0 = tvm.placeholder(shape_input0,
                                  name="data_input0",
                                  dtype=dtype_input0)
    data_input1 = tvm.placeholder(shape_input1,
                                  name="data_input1",
                                  dtype=dtype_input1)
    data_input2 = tvm.placeholder(shape_input2,
                                  name="data_input2",
                                  dtype=dtype_input2)

    res = confusion_mul_grad_compute(data_input0, data_input1, data_input2,
                                     output0, output1,
                                     axis, keep_dims, kernel_name)

    inputlist = [data_input0, data_input1, data_input2]

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}

    te.lang.cce.cce_build_code(sch, config)
