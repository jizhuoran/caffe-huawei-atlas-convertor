#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

sub
"""
import te.lang.cce
from te import tvm
from topi import generic
from te.platform.fusion_manager import fusion_manager
from topi.cce import util


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("sub")
def sub_compute(input_x, input_y, output_z, kernel_name="sub"):
    """
    calculating data's sub, c = a - b

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is sub

    Returns
    -------
    res : output of the data's sub
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_max)
    input_x = te.lang.cce.broadcast(input_x, shape_max)
    input_y = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vsub(input_x, input_y)

    return res


@util.check_input_type(dict, dict, dict, str)
def sub(input_x, input_y, output_z, kernel_name="sub"):
    """
    do element-wise sub operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32,int32
    input_y : dict
        shape and dtype of input, only support float16, float32,int32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "sub"

    Returns
    -------
    None
    """
    shape_x = util.scalar2tensor_one(input_x.get("shape"))
    shape_y = util.scalar2tensor_one(input_y.get("shape"))
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_x)
    util.check_tensor_shape_size(shape_y)

    check_list = ["float16", "float32", "int32"]
    dtype = input_x.get("dtype").lower()
    if not dtype in check_list:
        raise RuntimeError(
            "sub only support float16, float32, int32")

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_tensor_shape_size(shape_max)

    data1 = tvm.placeholder(shape_x, dtype=dtype, name="data1")
    data2 = tvm.placeholder(shape_y, dtype=dtype, name="data2")

    res = sub_compute(data1, data2, output_z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data1, data2, res]}
    te.lang.cce.cce_build_code(sch, config)
