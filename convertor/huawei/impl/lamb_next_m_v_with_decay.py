#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

lamb_next_m_v_with_decay
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=locally-disabled,invalid-name,unused-variable
# pylint: disable=locally-disabled,too-many-statements
# pylint: disable=locally-disabled,redefined-builtin,too-many-locals
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


@fusion_manager.register("lamb_next_m_v_with_decay")
def lamb_next_m_v_with_decay_compute(data_input_mul3, data_input_mul2,
                                     data_input_realdiv1, data_input_mul1,
                                     data_input_mul0, data_input_realdiv0,
                                     data_input_mul4, data_mul0_x,
                                     data_mul1_sub, data_mul2_x,
                                     data_mul3_sub1, data_mul4_x,
                                     data_add2_y, y1, y2, y3, y4,
                                     kernel_name="lamb_next_m_v_with_decay"):
    """
    apply one lamb calculation function

    Parameters
    ----------
    data_input_mul3: TVM tensor
         the input tensor of mul_3
    data_input_mul2: TVM tensor
         the input tensor of  mul_2
    data_input_realdiv1: TVM tensor
         the input tensor of  truediv_1
    data_input_mul1: TVM tensor
        the input tensor of mul_1
    data_input_mul0: TVM tensor
         the input tensor of mul
    data_input_realdiv0: TVM tensor
         the input tensor of truediv
    data_input_mul4: TVM tensor
         the input tensor of mul_4
    data_mul0_x: TVM tensor
         the input tensor of mul
    data_mul1_sub: TVM tensor
         the input tensor of mul_1
    data_mul2_x: TVM tensor
         the input tensor of mul_2
    data_mul3_sub1: TVM tensor
         the input tensor of mul_3
    data_mul4_x: TVM tensor
         the input tensor of mul_4
    data_add2_y: TVM tensor
         the input tensor of add_2 and add_4
    y1: dict
         the dict of output of add_3
    y2: dict
         the dict of output of add
    y3: dict
         the dict of output of add_1
    y4: dict
         the dict of output of add_5
    kernel_name: str
        cce kernel name, default value is lamb_next_m_v_with_decay

    Returns
    -------
    output tensor
    """
    # mul_3
    data_input_mul3, data_mul3_sub1 = \
        shape_broadcast(data_input_mul3, data_mul3_sub1)
    mul_3_result = te.lang.cce.vmul(data_input_mul3, data_mul3_sub1)

    # mul_2
    data_input_mul2, data_mul2_x = \
        shape_broadcast(data_input_mul2, data_mul2_x)
    mul_2_result = te.lang.cce.vmul(data_input_mul2, data_mul2_x)

    # add_1
    mul_2_result, mul_3_result = shape_broadcast(mul_2_result, mul_3_result)
    add_1_result = te.lang.cce.vadd(mul_2_result, mul_3_result)

    # truediv_1
    add_1_result, data_input_realdiv1 = \
        shape_broadcast(add_1_result, data_input_realdiv1)
    truediv_1_result = te.lang.cce.vdiv(add_1_result, data_input_realdiv1)

    # add_2
    truediv_1_result, data_add2_y = \
        shape_broadcast(truediv_1_result, data_add2_y)
    add_2_result = te.lang.cce.vadd(truediv_1_result, data_add2_y)

    # sqrt, actually op is rsqrt
    sqrt_rsqrt_result = te.lang.cce.vrsqrt(add_2_result)

    # sqrt_1
    sqrt_1_result = te.lang.cce.vsqrt(truediv_1_result)

    # add_4
    sqrt_1_result, data_add2_y = \
        shape_broadcast(sqrt_1_result, data_add2_y)
    add_4_result = te.lang.cce.vadd(sqrt_1_result, data_add2_y)

    # mul_1
    data_input_mul1, data_mul1_sub = \
        shape_broadcast(data_input_mul1, data_mul1_sub)
    mul_1_result = te.lang.cce.vmul(data_input_mul1, data_mul1_sub)

    # mul_0
    data_input_mul0, data_mul0_x = \
        shape_broadcast(data_input_mul0, data_mul0_x)
    mul_0_result = te.lang.cce.vmul(data_input_mul0, data_mul0_x)

    # add
    mul_0_result, mul_1_result = shape_broadcast(mul_0_result, mul_1_result)
    add_0_result = te.lang.cce.vadd(mul_0_result, mul_1_result)

    # truediv
    add_0_result, data_input_realdiv0 = \
        shape_broadcast(add_0_result, data_input_realdiv0)
    truediv_0_result = te.lang.cce.vdiv(add_0_result, data_input_realdiv0)

    # truediv_2, actually op is mul
    sqrt_rsqrt_result, truediv_0_result = \
        shape_broadcast(sqrt_rsqrt_result, truediv_0_result)
    truediv_2_mul_result = te.lang.cce.vmul(sqrt_rsqrt_result, truediv_0_result)

    # truediv_4
    truediv_0_result, add_4_result = \
        shape_broadcast(truediv_0_result, add_4_result)
    truediv_4_result = te.lang.cce.vdiv(truediv_0_result, add_4_result)

    # mul_4
    data_input_mul4, data_mul4_x = \
        shape_broadcast(data_input_mul4, data_mul4_x)
    mul_4_result = te.lang.cce.vmul(data_input_mul4, data_mul4_x)

    # add_3
    mul_4_result, truediv_2_mul_result = \
        shape_broadcast(mul_4_result, truediv_2_mul_result)
    add_3_result = te.lang.cce.vadd(mul_4_result, truediv_2_mul_result)

    # add_5
    mul_4_result, truediv_4_result = \
        shape_broadcast(mul_4_result, truediv_4_result)
    add_5_result = te.lang.cce.vadd(mul_4_result, truediv_4_result)

    res = [add_3_result, add_0_result, add_1_result, add_5_result]

    return res


def lamb_next_m_v_with_decay(input_mul3, input_mul2, input_realdiv1,
                             input_mul1, input_mul0, input_realdiv0,
                             input_mul4, mul0_x, mul1_sub, mul2_x,
                             mul3_sub1, mul4_x, add2_y,
                             y1, y2, y3, y4,
                             kernel_name="lamb_next_m_v_with_decay"):
    """
    function: For bert lamb fuse

    Parameters
    ----------
    input_mul3: dict
        the dict of input of mul_3, and dtype supports 'float16', 'float32'
    input_mul2: dict
        the dict of input of mul_2, and dtype supports 'float16', 'float32'
    input_realdiv1
        the dict of input of truediv_1,
        and dtype supports 'float16', 'float32'
    input_mul1: dict
        the dict of input of mul_1, and dtype supports 'float16', 'float32'
    input_mul0: dict
        the dict of input of mul, and dtype supports 'float16', 'float32'
    input_realdiv0
        the dict of input of truediv, and dtype supports 'float16', 'float32'
    input_mul4: dict
        the dict of input of mul_4, and dtype supports 'float16', 'float32'
    mul0_x: dict
        the dict of input of mul, and dtype supports 'float16', 'float32'
    mul1_sub: dict
        the dict of input of mul_1, and dtype supports 'float16', 'float32'
    mul2_x: dict
        the dict of input of mul_2, and dtype supports 'float16', 'float32'
    mul3_sub1: dict
        the dict of input of mul_3, and dtype supports 'float16', 'float32'
    mul4_x: dict
        the dict of input of mul_4, and dtype supports 'float16', 'float32'
    add2_y: dict
        the dict of input of add_2 and add_4,
        and dtype supports 'float16', 'float32'
    y1: dict
        the dict of output of add_3, and dtype supports 'float16', 'float32'
    y2: dict
        the dict of output of add, and dtype supports 'float16', 'float32'
    y3: dict
        the dict of output of add_1, and dtype supports 'float16', 'float32'
    y4: dict
        the dict of output of add_5, and dtype supports 'float16', 'float32'
    kernel_name: str
       cce kernel name, default value is lamb_next_m_v_with_decay

    Returns
    -------
    None
    """
    shape_input_mul3 = util.scalar2tensor_one(input_mul3.get("shape"))
    shape_input_mul2 = util.scalar2tensor_one(input_mul2.get("shape"))
    shape_input_realdiv1 = util.scalar2tensor_one(input_realdiv1.get("shape"))
    shape_input_mul1 = util.scalar2tensor_one(input_mul1.get("shape"))
    shape_input_mul0 = util.scalar2tensor_one(input_mul0.get("shape"))
    shape_input_realdiv0 = util.scalar2tensor_one(input_realdiv0.get("shape"))
    shape_input_mul4 = util.scalar2tensor_one(input_mul4.get("shape"))
    shape_mul0_x = util.scalar2tensor_one(mul0_x.get("shape"))
    shape_mul1_sub = util.scalar2tensor_one(mul1_sub.get("shape"))
    shape_mul2_x = util.scalar2tensor_one(mul2_x.get("shape"))
    shape_mul3_sub1 = util.scalar2tensor_one(mul3_sub1.get("shape"))
    shape_mul4_x = util.scalar2tensor_one(mul4_x.get("shape"))
    shape_add2_y = util.scalar2tensor_one(add2_y.get("shape"))

    input_dtype = input_mul3.get("dtype").lower()
    util.check_kernel_name(kernel_name)

    shape_input_mul3, shape_mul3_sub1, shape_max_mul3 = \
        util.produce_shapes(shape_input_mul3, shape_mul3_sub1)
    shape_input_mul2, shape_mul2_x, shape_max_mul2 = \
        util.produce_shapes(shape_input_mul2, shape_mul2_x)
    shape_max_mul2, shape_max_mul3, shape_max_add1 = \
        util.produce_shapes(shape_max_mul2, shape_max_mul3)
    shape_input_realdiv1, shape_max_add1, shape_max_truediv1 = \
        util.produce_shapes(shape_input_realdiv1, shape_max_add1)
    shape_max_truediv1, shape_add2_y, shape_max_add2 = \
        util.produce_shapes(shape_max_truediv1, shape_add2_y)
    shape_input_mul1, shape_mul1_sub, shape_max_mul1 = \
        util.produce_shapes(shape_input_mul1, shape_mul1_sub)
    shape_input_mul0, shape_mul0_x, shape_max_mul0 = \
        util.produce_shapes(shape_input_mul0, shape_mul0_x)
    shape_max_mul0, shape_max_mul1, shape_max_add0 = \
        util.produce_shapes(shape_max_mul0, shape_max_mul1)
    shape_max_add0, shape_input_realdiv0, shape_max_truediv0 = \
        util.produce_shapes(shape_max_add0, shape_input_realdiv0)
    shape_input_mul4, shape_mul4_x, shape_max_mul4 = \
        util.produce_shapes(shape_input_mul4, shape_mul4_x)

    data_input_mul3 = tvm.placeholder(shape_input_mul3,
                                      name="data_input_mul3",
                                      dtype=input_dtype)
    data_input_mul2 = tvm.placeholder(shape_input_mul2,
                                      name="data_input_mul2",
                                      dtype=input_dtype)
    data_input_realdiv1 = tvm.placeholder(shape_input_realdiv1,
                                          name="data_input_realdiv1",
                                          dtype=input_dtype)
    data_input_mul1 = tvm.placeholder(shape_input_mul1,
                                      name="data_input_mul1",
                                      dtype=input_dtype)
    data_input_mul0 = tvm.placeholder(shape_input_mul0,
                                      name="data_input_mul0",
                                      dtype=input_dtype)
    data_input_realdiv0 = tvm.placeholder(shape_input_realdiv0,
                                          name="data_input_realdiv0",
                                          dtype=input_dtype)
    data_input_mul4 = tvm.placeholder(shape_input_mul4,
                                      name="data_input_mul4",
                                      dtype=input_dtype)
    data_mul0_x = tvm.placeholder(shape_mul0_x,
                                  name="data_mul0_x",
                                  dtype=input_dtype)
    data_mul1_sub = tvm.placeholder(shape_mul1_sub,
                                    name="data_mul1_sub",
                                    dtype=input_dtype)
    data_mul2_x = tvm.placeholder(shape_mul2_x,
                                  name="data_mul2_x",
                                  dtype=input_dtype)
    data_mul3_sub1 = tvm.placeholder(shape_mul3_sub1,
                                     name="data_mul3_sub1",
                                     dtype=input_dtype)
    data_mul4_x = tvm.placeholder(shape_mul4_x,
                                  name="data_mul4_x",
                                  dtype=input_dtype)
    data_add2_y = tvm.placeholder(shape_add2_y,
                                  name="data_add2_y",
                                  dtype=input_dtype)

    res = lamb_next_m_v_with_decay_compute(data_input_mul3, data_input_mul2,
                                           data_input_realdiv1, data_input_mul1,
                                           data_input_mul0, data_input_realdiv0,
                                           data_input_mul4, data_mul0_x,
                                           data_mul1_sub, data_mul2_x,
                                           data_mul3_sub1, data_mul4_x,
                                           data_add2_y, y1, y2, y3, y4,
                                           kernel_name)

    inputlist = [data_input_mul3, data_input_mul2, data_input_realdiv1,
                 data_input_mul1, data_input_mul0, data_input_realdiv0,
                 data_input_mul4, data_mul0_x, data_mul1_sub, data_mul2_x,
                 data_mul3_sub1, data_mul4_x, data_add2_y]

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}

    te.lang.cce.cce_build_code(sch, config)
