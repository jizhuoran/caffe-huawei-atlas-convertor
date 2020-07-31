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
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
# define a VALUE, value = 1
VALUE_ONE = 1


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# pylint: disable=locally-disabled,too-many-statements,too-many-branches
def op_select_format(condition, x1, x2, y, kernel_name="select"):
    """
    select format dynamically
    """
    shape_condition = condition.get("ori_shape")
    shape_x1 = x1.get("ori_shape")
    shape_x2 = x2.get("ori_shape")

    format_4d_list = ["NCHW", "NHWC", "HWCN"]

    format_condition = condition.get("ori_format")
    format_x1 = x1.get("ori_format")
    format_x2 = x2.get("ori_format")

    format_list = []
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul",
                                               "float32"):
        dtype_list = ["float16", "float", "int32", "int8", "uint8"]
    else:
        dtype_list = ["float16", "int32", "int8", "uint8"]
    dtype_total = []
    dtype_total0 = []
    dtype_total0.append("bool")
    format_list1 = []
    #NZ+NZ ND+ND 5HD+5HD FZ+FZ
    if (len(shape_condition) != 1) or \
            (len(shape_condition) == 1 and len(shape_x1) == 1
             and len(shape_x2) == 1):
        format_list.append("ND")
        if format_condition == format_x1 == format_x2 and \
                format_x1 in format_4d_list and \
                list(shape_condition) == list(shape_x1) == list(shape_x2):
            format_list.append("FRACTAL_Z")
            format_list.append("FRACTAL_NZ")
            format_list.append("NC1HWC0")

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        dtype_total0 = dtype_total0*len(dtype_total)
        format_list = format_list * len(dtype_list)
        input0 = gen_param(classify="input0", name="condition",
                           datatype=",".join(dtype_total0),
                           format=",".join(format_list))
        input1 = gen_param(classify="input1", name="x1",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list))
        input2 = gen_param(classify="input2", name="x2",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list))
        output0 = gen_param(classify="output0", name="y",
                            datatype=",".join(dtype_total),
                            format=",".join(format_list))
    else:
        format_list.append("ND")
        if format_x1 == format_x2:
            if len(shape_x1) == 4 and len(shape_x2) == 4 and \
                    format_x1 in format_4d_list and format_x2 in format_4d_list:
                format_list1.append("FRACTAL_NZ")
                format_list1.append("ND")
                if format_x1 in ("NHWC", "NCHW"):
                    format_list1.append("NC1HWC0")
            elif len(shape_x1) > 2 and len(shape_x2) > 2 and \
                    format_x1 in format_4d_list and format_x2 in format_4d_list:
                format_list1.append("FRACTAL_NZ")
                format_list1.append("ND")
            else:
                format_list1.append("ND")
        else:
            format_list1.append("ND")

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list1)
        dtype_total0 = dtype_total0*len(dtype_total)
        format_list1 = format_list1*len(dtype_list)
        format_list = format_list*len(dtype_total)
        input0 = gen_param(classify="input0", name="condition",
                           datatype=",".join(dtype_total0),
                           format=",".join(format_list))
        input1 = gen_param(classify="input1", name="x1",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list1))
        input2 = gen_param(classify="input2", name="x2",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list1))
        output0 = gen_param(classify="output0", name="y",
                            datatype=",".join(dtype_total),
                            format=",".join(format_list1))

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json

# pylint: disable=too-many-locals, invalid-name, unused-argument
@fusion_manager.register("select")
def select_compute(condition, x1, x2, y, kernel_name="select"):
    """
    compute for select

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
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape = te.lang.cce.util.shape_to_list(x1.shape)
    con_shape = te.lang.cce.util.shape_to_list(condition.shape)
    num_dtype = x1.dtype

    if (num_dtype in ("float32", "int32")) and \
            (not (tbe_platform.cce_conf.api_check_support("te.lang.cce.vsel",
                                                          "float32"))):
        if num_dtype == "int32":
            condition = te.lang.cce.ceil(condition)
        else:
            condition = te.lang.cce.cast_to(condition, num_dtype)
        condition = te.lang.cce.broadcast(condition, shape)
        ones = te.lang.cce.broadcast(tvm.const(VALUE_ONE, dtype=num_dtype),
                                     shape, output_dtype=num_dtype)
        condition_opp = te.lang.cce.vsub(ones, condition)
        temp_x = te.lang.cce.vmul(x1, condition)
        temp_y = te.lang.cce.vmul(x2, condition_opp)
        res = te.lang.cce.vadd(temp_x, temp_y)
        return res

    if num_dtype in ("int8", "uint8", "int32"):
        if tbe_platform.cce_conf.api_check_support("te.lang.cce.vsel",
                                                   "float32"):
            x1_dtype = "float32"
            ones = te.lang.cce.broadcast(tvm.const(VALUE_ONE, dtype="float32"),
                                         shape, output_dtype="float32")
            x1 = te.lang.cce.cast_to(x1, "float32")
            x2 = te.lang.cce.cast_to(x2, "float32")
        else:
            x1_dtype = "float16"
            ones = te.lang.cce.broadcast(tvm.const(VALUE_ONE, dtype="float16"),
                                         shape, output_dtype="float16")
            x1 = te.lang.cce.cast_to(x1, "float16")
            x2 = te.lang.cce.cast_to(x2, "float16")
    else:
        x1_dtype = num_dtype
        ones = te.lang.cce.broadcast(tvm.const(VALUE_ONE, dtype=num_dtype),
                                     shape, output_dtype=num_dtype)
    if list(con_shape) == list(shape):
        res = te.lang.cce.vsel(condition, x1, x2)
    else:
        condition = te.lang.cce.cast_to(condition, x1_dtype)
        condition = te.lang.cce.broadcast(condition, shape)
        res = te.lang.cce.vcmpsel(condition, rhs=ones, operation='eq',
                                  slhs=x1, srhs=x2)
    if num_dtype in ("int8", "uint8", "int32"):
        res = te.lang.cce.cast_to(res, num_dtype)
    return res

@util.check_input_type(dict, dict, dict, dict, str)
def select(condition, x1, x2, y, kernel_name="select"):
    """
      Selects elements from `x1` or `x2`, depending on `condition`.

      Parameters
      ----------
      condition: dict
          dict of condition, include keys(shape and dtype),
          only support int8,int32
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
    dtype_x1 = x1.get("dtype").lower()
    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype").lower()
    con_shape = condition.get("shape")
    bool_dtype = condition.get("dtype").lower()
    if bool_dtype == "bool":
        bool_dtype = "int8"
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x1)
    util.check_tensor_shape_size(shape_x1)
    check_list = ("float16", "float32", "int32", "int8", "uint8")
    util.check_dtype_rule(dtype_x1, check_list)

    if shape_x1 != shape_x2:
        raise RuntimeError("Shape of tensor x1 and x2 must be equal!")

    if dtype_x1 != dtype_x2:
        raise RuntimeError("Dtype of tensor x1 and x2 must be equal!")

    x_len = len(shape_x1)
    con_shape = list(con_shape)
    if len(con_shape) == 1 and x_len != 1:
        if con_shape[0] != shape_x1[0]:
            raise RuntimeError("Shape of tensor condition and x1 dim[0] "
                               "must be equal!")
        while x_len > len(con_shape):
            con_shape += [1]
    else:
        if list(con_shape) != list(shape_x1):
            raise RuntimeError("Shape of tensor condition and x1 must be "
                               "equal!")

    con_shape, shape_x1 = refine_shapes_for_broadcast(con_shape, shape_x1)

    flag_cloud = tbe_platform.cce_conf.api_check_support("te.lang.cce.vsel",
                                                         "float32")
    flag_dtype = dtype_x1 in ("float32", "int32")
    if (list(con_shape) != list(shape_x1)) or \
            ((not flag_cloud) and flag_dtype):
        condition = tvm.placeholder(con_shape, name="condition",
                                    dtype=bool_dtype)
    else:
        condition = tvm.placeholder(con_shape, name="condition", dtype="bool")
    input_x1 = tvm.placeholder(shape_x1, name="input_x1", dtype=dtype_x1)
    input_x2 = tvm.placeholder(shape_x1, name="input_x2", dtype=dtype_x2)

    with tvm.target.cce():
        res = select_compute(condition, input_x1, input_x2, y, kernel_name)
        sch = generic.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [condition, input_x1, input_x2, res],
              "bool_storage_as_1bit": False}
    te.lang.cce.cce_build_code(sch, config)
