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

mul_no_nan
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


@fusion_manager.register("mul_no_nan")
def mul_no_nan_compute(input_x1, input_x2, output_y, kernel_name="mul_no_nan"):
    """
    calculating data

    Parameters
    ----------
    input_x1 : TVM tensor
        the placeholder of input_x1
    input_x2 : TVM tensor
        the placeholder of input_x2
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "mul_no_nan"

    Returns
    -------
    output tensor
    """
    """
    np.where(np.equal(y, 0.), np.zeros((), dtype=dtype), np.multiply(x, y))
    """
    src_dtype = input_x1.dtype.lower()
    shape_x1 = te.lang.cce.util.shape_to_list(input_x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(input_x2.shape)

    shape_x1, shape_x2, shape_max = util.produce_shapes(shape_x1, shape_x2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    input_x1 = te.lang.cce.broadcast(input_x1, shape_max)
    input_x2 = te.lang.cce.broadcast(input_x2, shape_max)

    mul_res = te.lang.cce.vmul(input_x1, input_x2)
    zero = tvm.const(0, dtype=src_dtype)
    zeros = te.lang.cce.vmuls(input_x1, zero)
    res = te.lang.cce.vcmpsel(input_x2,
                              zeros,
                              operation='eq',
                              slhs=zeros,
                              srhs=mul_res)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, OPTION_OUTPUT, KERNEL_NAME)
def mul_no_nan(x1, x2, y, kernel_name="mul_no_nan"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input1
    x2: dict
        shape and dtype of input2
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "mul_no_nan"

    Returns
    -------
    None
    """
    input_x = x1
    input_y = x2
    output_z = y
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")

    check_tuple = ("float16", "float32", "int32")
    input_data_type = input_x.get("dtype").lower()
    check_dtype(input_data_type, check_tuple)

    shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y)
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    check_shape(shape_max)
    data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_data_type)
    data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_data_type)
    res = mul_no_nan_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": (data_x, data_y, res)
    }
    te.lang.cce.cce_build_code(schedule, config)
