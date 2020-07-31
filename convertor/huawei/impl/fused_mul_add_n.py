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

add_n
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def mul_add_n_compute(data_x, data_y, data_z):
    """
    fused mul+add_n, output = x * z + y
    res : output of the data's mul+add_n
    """
    data_z = te.lang.cce.broadcast(data_z, data_x.shape)
    res = te.lang.cce.vmul(data_x, data_z)
    res = te.lang.cce.vadd(data_y, res)
    return res


@util.check_input_type(dict, dict, dict, dict, str)
def fused_mul_add_n(input_x, input_y, input_z, output,
                    kernel_name="fused_mul_add_n"):
    """
    algorithm: fused mul+add_n
    calculating output = input_x * input_z + input_y

    Parameters
    ----------
    input_x : dict of input_x, tensor
    input_y: dict of input_y, tensor
    input_z: dict of input_z, scalar
    output : dict of output

    kernel_name : string
        cce kernel name, default value is fused_mul_add_n

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)

    check_list = ("float16", "float32", "int32")

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    data_x = tvm.placeholder(shape_x, name="input_x", dtype=dtype_x)
    util.check_shape_rule(shape_x)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_dtype_rule(dtype_x, check_list)

    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")

    data_y = tvm.placeholder(shape_y, name="input_y", dtype=dtype_y)
    util.check_shape_rule(shape_y)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)
    util.check_dtype_rule(dtype_y, check_list)

    dtype_z = input_z.get("dtype")
    shape_z = [1 for i in range(len(shape_x))]
    data_z = tvm.placeholder(shape_z, name="input_z", dtype=dtype_z)
    util.check_shape_rule(shape_z)
    util.check_shape_size(shape_z, SHAPE_SIZE_LIMIT)
    util.check_dtype_rule(dtype_z, check_list)

    res = mul_add_n_compute(data_x, data_y, data_z)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    tensor_list = [data_x, data_y, data_z, res]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
