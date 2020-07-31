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

floordiv
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform

# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("floor_div")
def floor_div_compute(input_x, input_y, output_z, kernel_name='floor_div'):
    """
       floordiv compute
       calculating data's floordiv, res =floor(x / y)

       Parameters
       ----------
       input_x: TVM tensor
           the placeholder of input_x
       input_y: TVM tensor
           the placeholder of input_y
       output_z: dict
           dict with keys(shape and dtype) of output
       kernel_name: str
           kernel name, default value is "floordiv"

       Returns
       -------
       res: TVM tensor
           the result of floordiv compute
    """
    input_x_shape = te.lang.cce.util.shape_to_list(input_x.shape)
    input_y_shape = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_list = util.produce_shapes(input_x_shape, input_y_shape)

    if input_x.dtype != 'float16' and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        cast_x = te.lang.cce.cast_to(input_x, 'float32')
        cast_y = te.lang.cce.cast_to(input_y, 'float32')
        broadcast_x = te.lang.cce.broadcast(cast_x, shape_list[2])
        broadcast_y = te.lang.cce.broadcast(cast_y, shape_list[2])
    else:
        broadcast_x = te.lang.cce.broadcast(input_x, shape_list[2])
        broadcast_y = te.lang.cce.broadcast(input_y, shape_list[2])

    div_res = te.lang.cce.vdiv(broadcast_x, broadcast_y)
    floor_res = te.lang.cce.floor(div_res)
    res = te.lang.cce.cast_to(floor_res, input_x.dtype)

    return res


@util.check_input_type(dict, dict, dict, str)
def floor_div(input_x, input_y, output_z, kernel_name="floor_div"):
    """
      algorithm: floordiv
      calculating data's floordiv, res =floor(x / y)

      Parameters
      ----------
      input_x: dict
          dict with keys(shape and dtype) of input_x
      input_y: dict
          dict with keys(shape and dtype) of input_y
      output_z: dict
          dict with keys(shape and dtype) of output
      kernel_name: str
          kernel name, default value is "floordiv"

      Returns
      -------
      None
    """
    # check dtype of input_x/input_y
    input_dtype_x = input_x.get("dtype").lower()
    input_dtype_y = input_y.get("dtype").lower()
    check_list = ('int8', 'uint8', 'int32', 'float16', 'float32')
    util.check_dtype_rule(input_dtype_x, check_list)
    if input_dtype_x != input_dtype_y:
        raise RuntimeError("The dtype of input_x and input_y must be the same")

    # check shape of input_x/input_y
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)
    shape_list = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_list[2])

    # check kernel_name of the input kernel_name
    util.check_kernel_name(kernel_name)

    # compute result for floordiv() with floordiv_compute()
    shape_x, shape_y = refine_shapes_for_broadcast(shape_list[0],
                                                   shape_list[1])
    data_x = tvm.placeholder(shape_x, dtype=input_dtype_x, name='data_x')
    data_y = tvm.placeholder(shape_y, dtype=input_dtype_y, name='data_y')
    res = floor_div_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
