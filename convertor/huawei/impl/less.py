#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright 2019 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

less
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager

from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast

# shape size limit for aicore is 2**31
SHAPE_SIZE_LIMIT = 2147483648


def _less_compare(data, shape, dtype, data_min):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    data : tuple
        two input data
    shape : list or tuple
        shape of input data
    dtype : str
        source data type, support float16,float32,int32,int8,uint8
    data_min : tvm.const
        the minimal data according to dtype

    Returns
    -------
    the compare result
    """
    data_zero = te.lang.cce.broadcast(tvm.const(0, dtype), shape, dtype)

    res_sub = te.lang.cce.vsub(data[1], data[0])
    res_min = te.lang.cce.vmin(res_sub, data_min)
    res_max = te.lang.cce.vmax(res_min, data_zero)

    if dtype == "float32":
        # max num of float32 is 2**126
        # but cce can only support 2**62, so use 62/62/2 to adaptor 126
        res_mul1 = te.lang.cce.vmuls(res_max, tvm.const(2**62, dtype=dtype))
        res_mul2 = te.lang.cce.vmuls(res_mul1, tvm.const(2**62, dtype=dtype))
        res = te.lang.cce.vmuls(res_mul2, tvm.const(2**2, dtype=dtype))
    elif dtype == "float16":
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        res_mul1 = te.lang.cce.vmuls(res_max, tvm.const(2**12, dtype=dtype))
        res = te.lang.cce.vmuls(res_mul1, tvm.const(2**12, dtype=dtype))
    else:
        res = te.lang.cce.cast_to(res_max, "float16")

    return te.lang.cce.cast_to(res, "uint8", True)

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("less")
def less_compute(input_x, input_y, output_z, kernel_name="less"):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_x: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is less

    Returns
    -------
    the result
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape = util.produce_shapes(shape_x, shape_y)

    dtype = input_x.dtype
    if dtype in ("uint8", "int8"):
        input_x = te.lang.cce.cast_to(input_x, "float16")
        input_y = te.lang.cce.cast_to(input_y, "float16")
        dtype = "float16"

    input_x = te.lang.cce.broadcast(input_x, shape)
    input_y = te.lang.cce.broadcast(input_y, shape)

    if dtype == "float32":
        # minimun num of float32 2**(-126)
        data_min = te.lang.cce.broadcast(tvm.const(2**(-126),
                                                   dtype=dtype), shape, dtype)
    elif dtype == "float16":
        # minimun num of float16 2**(-24)
        data_min = te.lang.cce.broadcast(tvm.const(2**(-24), dtype=dtype),
                                         shape, dtype)
    else:
        data_min = te.lang.cce.broadcast(tvm.const(1, dtype=dtype),
                                         shape, dtype)

    return _less_compare((input_x, input_y), shape, dtype, data_min)


@util.check_input_type(dict, dict, dict, str)
def less(input_x, input_y, output_z, kernel_name="less"):
    """
    do element-wise less operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of first input, support float16,float32,int32,
        int8,uint8
    input_y : dict
        shape and dtype of second input, support float16,float32,int32,
        int8,uint8
    output_x: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is less

    Returns
    -------
    None
    """
    shape_x = util.scalar2tensor_one(input_x.get("shape"))
    shape_y = util.scalar2tensor_one(input_y.get("shape"))
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    input_dtype = input_x.get("dtype").lower()
    util.check_dtype_rule(input_dtype, check_list)

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x,
                                                      shape_y)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)

    shape_x, shape_y = refine_shapes_for_broadcast(shape_x,
                                                   shape_y)
    data_x = tvm.placeholder(shape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=input_dtype, name="data_y")

    res = less_compute(data_x, data_y, output_z, kernel_name="less")
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
