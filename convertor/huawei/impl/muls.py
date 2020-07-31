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

muls
"""

import te.lang.cce
from te import tvm
from te.utils.op_utils import check_op_params, check_dtype, REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT, REQUIRED_ATTR_FLOAT, KERNEL_NAME
from te.platform.fusion_manager import fusion_manager
from topi import generic


@fusion_manager.register("muls")
def muls_compute(x, scalar, kernel_name="muls"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    scalar : a number of float or int
    kernel_name : str
        kernel name, default value is "muls"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    res = te.lang.cce.vmuls(x, scalar)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_FLOAT,
                 KERNEL_NAME)
def muls(x, y, value, kernel_name="muls"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as x
    value : float
        scale
    kernel_name : str
        kernel name, default value is "muls"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    input_dtype = dtype.lower()

    check_list = ["float16", "float32", "int32", "int16"]
    check_dtype(input_dtype, check_list)

    scalar = tvm.const(value, dtype=input_dtype)
    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = muls_compute(data_input, scalar)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)
