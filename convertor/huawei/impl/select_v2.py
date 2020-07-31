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

select
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce

# define a VALUE, value = 1
VALUE_ONE = 1

# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# pylint: disable=too-many-locals, invalid-name, unused-argument
@fusion_manager.register("select")
def select_v2_compute(condition, x1, x2, y, kernel_name="select_v2"):
    """
    compute for select_v2

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of input x1
    x2: TVM tensor
        the placeholder of input x2
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "select_v2"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    num_dtype = x1.dtype
    condition_dtype = condition.dtype
    x1 = te.lang.cce.cast_to(x1, "float32")
    x2 = te.lang.cce.cast_to(x2, "float32")
    condition = te.lang.cce.cast_to(condition, "float32")
    shape_x1list = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2list = te.lang.cce.util.shape_to_list(x2.shape)
    con_shapelist = te.lang.cce.util.shape_to_list(condition.shape)
    shape_x1list, con_shapelist, shape_max_x1 = util.produce_shapes(shape_x1list, con_shapelist)
    shape_x2list, shape_max_x1, shape_max = util.produce_shapes(shape_x2list, shape_max_x1)
    x1 = te.lang.cce.broadcast(x1, shape_max)
    x2 = te.lang.cce.broadcast(x2, shape_max)
    condition = te.lang.cce.broadcast(condition, shape_max)

    ones = te.lang.cce.broadcast(tvm.const(VALUE_ONE, dtype="float32"),
                                 shape_max, output_dtype="float32")

    res = te.lang.cce.vcmpsel(condition, rhs=ones,
                              operation='eq', slhs=x1, srhs=x2)
    res = te.lang.cce.cast_to(res, num_dtype)
    return res


@util.check_input_type(dict, dict, dict, dict, str)
def select_v2(condition, x1, x2, y, kernel_name="select_v2"):
    """
      Selects elements from `x1` or `x2`, depending on `condition`.

      Parameters
      ----------
      condition: dict
          dict of condition, include keys(shape and dtype),
          only support bool
      x1: dict
          dict of x1, only support float16, float32, int32, int8, uint8
      x2: dict
          dict of x2, only support float16, float32, int32, int8, uint8
      y: dict
          dict of output
      kernel_name: str
          cce kernel name, default value is "select"

      Returns
      -------
      None
      """
    shape_x1 = x1.get("shape")
    dtype_x1 = x1.get("dtype")
    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype")
    bool_dtype = condition.get("dtype")
    con_shape = condition.get("shape")

    shape_x1, con_shape, shape_max_x1 = util.produce_shapes(shape_x1, con_shape)
    shape_x2, con_shape, shape_max_x2 = util.produce_shapes(shape_x2, con_shape)

    if shape_x1[-1] == 1 and shape_x2[-1] == 1 and con_shape[-1] == 1 \
            and shape_max_x1[-1] == 1:
        shape_x1 = shape_x1 if len(shape_x1) == 1 else shape_x1[:-1]
        shape_x2 = shape_x2 if len(shape_x2) == 1 else shape_x2[:-1]
        con_shape = con_shape if len(con_shape) == 1 else con_shape[:-1]

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x1)
    util.check_tensor_shape_size(shape_x1)

    if shape_x1 == shape_x2 == con_shape:
        shape_x1 = (functools_reduce(lambda x, y: x * y, shape_x1[:]),)
        shape_x2 = (functools_reduce(lambda x, y: x * y, shape_x2[:]),)
        con_shape = (functools_reduce(lambda x, y: x * y, con_shape[:]),)

    dtype_x1 = dtype_x1.lower()
    dtype_x2 = dtype_x2.lower()
    check_list = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(dtype_x1, check_list)
    if dtype_x1 != dtype_x2:
        raise RuntimeError("Dtype of tensor x1 and x2 must be equal!")

    bool_dtype = bool_dtype.lower()
    bool_check_list = ("bool", "int8", "uint8")
    util.check_dtype_rule(bool_dtype, bool_check_list)

    condition = tvm.placeholder(con_shape, name="condition", dtype=bool_dtype)
    input_then = tvm.placeholder(shape_x1, name="input_then", dtype=dtype_x1)
    input_else = tvm.placeholder(shape_x2, name="input_else", dtype=dtype_x2)

    with tvm.target.cce():
        res = select_v2_compute(condition, input_then, input_else, y, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [condition, input_then, input_else, res],
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(sch, config)

