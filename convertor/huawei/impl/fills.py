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

fill

  Op_description :
    This operation creates a tensor of shape `dims` and fills it with `value`.

    # fill(
    #   x,
    #   y,
    #   value,
    #   kernel_name='fill'
    # )

  Supportive_dtype_format :
    ['int32', 'float32', 'float16']
    all format

  Constraint :
    [1] All : shape size limit is 2147483648.
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce
from te.utils.op_utils import *


@fusion_manager.register("fills")
def fills_compute(x, value, dtype, kernel_name="fills"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    value : a number of float or int
    dtype : string
        the type of input
    kernel_name : str
        kernel name, default value is "fills"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    data_mul = te.lang.cce.vmuls(x, tvm.const(0, dtype=dtype))
    res = te.lang.cce.vadds(data_mul, tvm.const(value, dtype=dtype))
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_FLOAT, KERNEL_NAME)
def fills(x, y, value, kernel_name="fills"):
    """
    do  fill operation

    Parameters:
    ----------
    x : the dict of output
    y :  the dict of output
    value:  scalar  value,
    kernel_name : cce kernel name, default value is "fill"

    Returns
    -------
    None
    """
    # get the shape and dtype
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    # check whether dtypes are right
    check_list = ("int32", "float16", "float32")
    check_dtype(dtype, check_list)

    data_x = tvm.placeholder(shape, dtype=dtype, name="data_x")
    res = fills_compute(data_x, value, dtype)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": (data_x, res),
        "print_ir": False
    }
    te.lang.cce.cce_build_code(sch, config)
