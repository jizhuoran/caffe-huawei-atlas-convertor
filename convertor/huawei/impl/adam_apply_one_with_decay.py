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

adam_apply_one_with_decay
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform

# shape size limit
SHAPE_SIZE_LIMIT = 2**30


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=locally-disabled,too-many-locals,too-many-statements
# pylint: disable=locally-disabled,invalid-name,too-many-locals
def square_compute(x, kernel_name="square"):
    """
    calculating data's square,y= x*x

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    kernel_name: str
        cce kernel name, default value is "square"

    Returns
    -------
    res: the result of square
    """
    res = te.lang.cce.vmul(x, x)
    return res


def mul_compute(x1, x2, kernel_name="mul"):
    """
   calculating data's element-wise mul, c = a .* b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "mul"

   Returns
   -------
   res: output of the data's element-wise mul
   """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = util.produce_shapes(shape_x1, shape_x2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vmul(data_x1, data_x2)

    return res


def add_compute(x1, x2, kernel_name="add"):
    """
   calculating data's element-wise add, c = a + b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "add"

   Returns
   -------
   res: output of the data's add
   """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = util.produce_shapes(shape_x1, shape_x2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vadd(data_x1, data_x2)

    return res


def sqrt_compute(x, kernel_name="sqrt"):
    """
    calculating data sqrt,y= x**0.5, mini not support vsqrt, use exp(0.5*log(x))

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input data
    kernel_name: str
        cce kernel name, default value is sqrt

    Returns
    -------
    res:  the result of sqrt
    """
    input_dtype = x.dtype
    has_improve_precision = False
    if input_dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt",
                                                    "float32"):
        x = te.lang.cce.cast_to(x, "float32")
        has_improve_precision = True

    res = te.lang.cce.vsqrt(x)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


def true_div_compute(x1, x2, kernel_name="true_div"):
    """
    calculating data's realdiv, y = x1 / x2

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of first input data
    x2: TVM tensor
        the placeholder of second input data
    kernel_name: str
        cce kernel name, default value is "true_div"

    Returns
    -------
    res: output of the data's divide
    """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = util.produce_shapes(shape_x1, shape_x2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)

    res = te.lang.cce.vdiv(data_x1, data_x2)

    return res


def sub_compute(x1, x2, kernel_name="sub"):
    """
   calculating data's sub, c = a - b

   Parameters
   ----------
   x1: TVM tensor
       the placeholder of first input data
   x2: TVM tensor
       the placeholder of second input data
   kernel_name: str
       cce kernel name, default value is "sub"

   Returns
   -------
   res : output of the data's sub
   """
    shape_x1 = te.lang.cce.util.shape_to_list(x1.shape)
    shape_x2 = te.lang.cce.util.shape_to_list(x2.shape)

    shape_x1, shape_x2, shape_max = util.produce_shapes(shape_x1, shape_x2)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x1 = te.lang.cce.broadcast(x1, shape_max)
    data_x2 = te.lang.cce.broadcast(x2, shape_max)
    res = te.lang.cce.vsub(data_x1, data_x2)

    return res


def _check_broadcast_shape(input0, input1, input2, input3, input4,
                           const_mul_x, const_mul1_x, const_mul2_x,
                           const_mul3_x, const_mul4_x, add2_y):
    """
    check broadcast shape

    Parameters
    ----------
    all inputs: dict
        the dict of inputs

    Returns
    -------
    the list of inputs shape after broadcast
    """
    shape0 = input0.get("shape")
    util.check_shape_rule(shape0)
    util.check_shape_size(shape0, SHAPE_SIZE_LIMIT)

    shape1 = input1.get("shape")
    util.check_shape_rule(shape1)
    util.check_shape_size(shape1, SHAPE_SIZE_LIMIT)

    shape2 = input2.get("shape")
    util.check_shape_rule(shape2)
    util.check_shape_size(shape2, SHAPE_SIZE_LIMIT)

    shape3 = input3.get("shape")
    util.check_shape_rule(shape3)
    util.check_shape_size(shape3, SHAPE_SIZE_LIMIT)

    shape4 = input4.get("shape")
    util.check_shape_rule(shape4)
    util.check_shape_size(shape4, SHAPE_SIZE_LIMIT)

    shapecm0 = const_mul_x.get("shape")
    util.check_shape_rule(shapecm0)
    util.check_shape_size(shapecm0, SHAPE_SIZE_LIMIT)

    shapecm1 = const_mul1_x.get("shape")
    util.check_shape_rule(shapecm1)
    util.check_shape_size(shapecm1, SHAPE_SIZE_LIMIT)

    shapecm2 = const_mul2_x.get("shape")
    util.check_shape_rule(shapecm2)
    util.check_shape_size(shapecm2, SHAPE_SIZE_LIMIT)

    shapecm3 = const_mul3_x.get("shape")
    util.check_shape_rule(shapecm3)
    util.check_shape_size(shapecm3, SHAPE_SIZE_LIMIT)

    shapecm4 = const_mul4_x.get("shape")
    util.check_shape_rule(shapecm4)
    util.check_shape_size(shapecm4, SHAPE_SIZE_LIMIT)

    shapey = add2_y.get("shape")
    util.check_shape_rule(shapey)
    util.check_shape_size(shapey, SHAPE_SIZE_LIMIT)

    # broadcast mul_3 shape
    shape0, shapecm3, shape_max_03 = util.produce_shapes(shape0, shapecm3)
    util.check_shape_size(shape_max_03, SHAPE_SIZE_LIMIT)
    # broadcast mul_2 shape
    shape1, shapecm2, shape_max_02 = util.produce_shapes(shape1, shapecm2)
    util.check_shape_size(shape_max_02, SHAPE_SIZE_LIMIT)
    # broadcast add_1 shape
    shape_max_02, shape_max_03, shape_max_add1 = util.produce_shapes(
        shape_max_02, shape_max_03)
    util.check_shape_size(shape_max_add1, SHAPE_SIZE_LIMIT)
    # broadcast add_2 shape
    shapey, shape_max_add1, shape_max_add2 = util.produce_shapes(
        shapey, shape_max_add1)
    util.check_shape_size(shape_max_add2, SHAPE_SIZE_LIMIT)

    # broadcast mul_0 shape
    shape2, shapecm0, shape_max_20 = util.produce_shapes(shape2, shapecm0)
    util.check_shape_size(shape_max_20, SHAPE_SIZE_LIMIT)
    # broadcast mul_1 shape
    shape0, shapecm1, shape_max_01 = util.produce_shapes(shape0, shapecm1)
    util.check_shape_size(shape_max_01, SHAPE_SIZE_LIMIT)
    # broadcast add_0 shape
    shape_max_20, shape_max_01, shape_max_add0 = util.produce_shapes(
        shape_max_20, shape_max_01)
    util.check_shape_size(shape_max_add0, SHAPE_SIZE_LIMIT)

    # broadcast truediv_0 shape
    shape_max_add0, shape_max_add2, shape_max_truediv = util.produce_shapes(
        shape_max_add0, shape_max_add2)
    util.check_shape_size(shape_max_truediv, SHAPE_SIZE_LIMIT)

    # broadcast mul_4 shape
    shape3, shapecm4, shape_max_34 = util.produce_shapes(shape3, shapecm4)
    util.check_shape_size(shape_max_34, SHAPE_SIZE_LIMIT)
    # broadcast add_3 shape
    shape_max_34, shape_max_truediv, shape_max_add3 = util.produce_shapes(
        shape_max_34, shape_max_truediv)
    util.check_shape_size(shape_max_add3, SHAPE_SIZE_LIMIT)

    # broadcast mul_5 shape
    shape4, shape_max_add3, shape_max_4add3 = util.produce_shapes(
        shape4, shape_max_add3)
    util.check_shape_size(shape_max_4add3, SHAPE_SIZE_LIMIT)
    # broadcast sub_0 shape
    shape3, shape_max_4add3, shape_max_sub = util.produce_shapes(
        shape3, shape_max_4add3)
    util.check_shape_size(shape_max_sub, SHAPE_SIZE_LIMIT)

    return shape0, shape1, shape2, shape3, shape4,\
           shapecm0, shapecm1, shapecm2, shapecm3, shapecm4, shapey


@fusion_manager.register("adam_apply_one_with_decay")
def adam_apply_one_with_decay_compute(input0, input1, input2, input3, input4,
                                      const_mul_x, const_mul1_x, const_mul2_x,
                                      const_mul3_x, const_mul4_x, add2_y):
    """
    calculating data

    Parameters
    ----------
    input0: TVM tensor
        the placeholder of input0
    input1: TVM tensor
        the placeholder of input1
    input2: TVM tensor
        the placeholder of input2
    input3: TVM tensor
        the placeholder of input3
    input4: TVM tensor
        the placeholder of input4
    const_mul_x: TVM tensor
        the placeholder of const_mul_x
    const_mul1_x: TVM tensor
        the placeholder of const_mul1_x
    const_mul2_x: TVM tensor
        the placeholder of const_mul2_x
    const_mul3_x: TVM tensor
        the placeholder of const_mul3_x
    const_mul4_x: TVM tensor
        the placeholder of const_mul4_x
    add2_y: TVM tensor
        the placeholder of add2_y

    Returns
    -------
    y0: TVM tensor
        the tensor of y0
    y1: TVM tensor
        the tensor of y1
    y2: TVM tensor
        the tensor of y2
    """
    square_0 = square_compute(input0, kernel_name="square")
    mul_3 = mul_compute(square_0, const_mul3_x, kernel_name="mul_3")
    mul_2 = mul_compute(input1, const_mul2_x, kernel_name="mul_2")

    y0 = add_compute(mul_2, mul_3, kernel_name="add_1")

    sqrt_0 = sqrt_compute(y0, kernel_name="sqrt")
    add_2 = add_compute(sqrt_0, add2_y, kernel_name="add_2")
    mul_0 = mul_compute(input2, const_mul_x, kernel_name="mul_0")
    mul_1 = mul_compute(input0, const_mul1_x, kernel_name="mul_1")

    y1 = add_compute(mul_0, mul_1, kernel_name="add_0")

    truediv_0 = true_div_compute(y1, add_2, kernel_name="truediv")
    mul_4 = mul_compute(input3, const_mul4_x, kernel_name="mul_4")
    add_3 = add_compute(truediv_0, mul_4, kernel_name="add_3")
    mul_5 = mul_compute(add_3, input4, kernel_name="mul_5")

    y2 = sub_compute(input3, mul_5, kernel_name="sub")

    return y0, y1, y2


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, dict, str)
def adam_apply_one_with_decay(input0,
                              input1,
                              input2,
                              input3,
                              input4,
                              const_mul_x,
                              const_mul1_x,
                              const_mul2_x,
                              const_mul3_x,
                              const_mul4_x,
                              add2_y,
                              output0,
                              output1,
                              output2,
                              kernel_name="adam_apply_one_with_decay"):
    """
    calculating data

    Parameters
    ----------
    input0: dict
        shape and dtype of input0
    input1: dict
        shape and dtype of input1
    input2: dict
        shape and dtype of input2
    input3: dict
        shape and dtype of input3
    input4: dict
        shape and dtype of input4
    const_mul_x: dict
        shape and dtype of const_mul_x
    const_mul1_x: dict
        shape and dtype of const_mul1_x
    const_mul2_x: dict
        shape and dtype of const_mul2_x
    const_mul3_x: dict
        shape and dtype of const_mul3_x
    const_mul4_x: dict
        shape and dtype of const_mul4_x
    add2_y: dict
        shape and dtype of add2_y
    output0: dict
        shape and dtype of output0
    output1: dict
        shape and dtype of output1
    output2: dict
        shape and dtype of output2
    kernel_name: str
        kernel name, default value is "adam_apply_one_with_decay"

    Returns
    -------
    None
    """
    dtype0 = input0.get("dtype")
    dtype1 = input1.get("dtype")
    dtype2 = input2.get("dtype")
    dtype3 = input3.get("dtype")
    dtype4 = input4.get("dtype")
    dtypecm0 = const_mul_x.get("dtype")
    dtypecm1 = const_mul1_x.get("dtype")
    dtypecm2 = const_mul2_x.get("dtype")
    dtypecm3 = const_mul3_x.get("dtype")
    dtypecm4 = const_mul4_x.get("dtype")
    dtypey = add2_y.get("dtype")

    shape0, shape1, shape2, shape3, shape4,\
    shapecm0, shapecm1, shapecm2, shapecm3, shapecm4, \
    shapey = _check_broadcast_shape(input0, input1, input2, input3, input4,
                                    const_mul_x, const_mul1_x, const_mul2_x,
                                    const_mul3_x, const_mul4_x, add2_y)

    util.check_kernel_name(kernel_name)

    input_place0 = tvm.placeholder(shape0, name="input0", dtype=dtype0)
    input_place1 = tvm.placeholder(shape1, name="input1", dtype=dtype1)
    input_place2 = tvm.placeholder(shape2, name="input2", dtype=dtype2)
    input_place3 = tvm.placeholder(shape3, name="input3", dtype=dtype3)
    input_place4 = tvm.placeholder(shape4, name="input4", dtype=dtype4)

    input_cm0 = tvm.placeholder(shapecm0, name="const_mul_x", dtype=dtypecm0)
    input_cm1 = tvm.placeholder(shapecm1, name="const_mul1_x", dtype=dtypecm1)
    input_cm2 = tvm.placeholder(shapecm2, name="const_mul2_x", dtype=dtypecm2)
    input_cm3 = tvm.placeholder(shapecm3, name="const_mul3_x", dtype=dtypecm3)
    input_cm4 = tvm.placeholder(shapecm4, name="const_mul4_x", dtype=dtypecm4)

    input_y = tvm.placeholder(shapey, name="add2_y", dtype=dtypey)

    y1, y2, y3 = adam_apply_one_with_decay_compute(
        input_place0, input_place1, input_place2, input_place3, input_place4,
        input_cm0, input_cm1, input_cm2, input_cm3, input_cm4, input_y)

    with tvm.target.cce():
        sch = generic.auto_schedule([y1, y2, y3])

    config = {
        "name":
            kernel_name,
        "tensor_list": (input_place0, input_place1, input_place2, input_place3,
                        input_place4, input_cm0, input_cm1, input_cm2,
                        input_cm3, input_cm4, input_y, y1, y2, y3)
    }

    te.lang.cce.cce_build_code(sch, config)
