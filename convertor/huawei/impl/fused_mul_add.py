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

fuesd_mul_add
"""

import math
import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

SHAPE_SIZE_LIMIT = 2 ** 30  # shape limit
SIZE_SIXTEEN = 16


# pylint: disable=locally-disabled,unused-variable,unused-argument
# pylint: disable=locally-disabled,too-many-locals,too-many-statements
# pylint: disable=locally-disabled,too-many-branches,unused-variable
def _division_sixteen(shape):

    if len(shape) < 2:
        if shape[-1] == 0:
            raise RuntimeError("value of shape is illegal")
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        raise RuntimeError("value of shape is illegal")

    return shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0


def op_select_format(input0, input1, input2, output,
                     kernel_name="fused_mul_add"):
    """
    _division_sixteen : judge whether the last two dimensions are divided by 16
    scalar2tensor_one : convert scalar to tensor
    """
    shape_0 = input0.get("ori_shape")
    shape_1 = input1.get("ori_shape")
    shape_2 = input2.get("ori_shape")

    shape_0 = util.scalar2tensor_one(shape_0)
    shape_1 = util.scalar2tensor_one(shape_1)
    shape_2 = util.scalar2tensor_one(shape_2)

    if _division_sixteen(shape_0) and not _division_sixteen(shape_1)\
        and not _division_sixteen(shape_2):
        # Nz+ND+ND
        input0 = gen_param(classify="input0", name="x1",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        input1 = gen_param(classify="input1", name="x2",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input2 = gen_param(classify="input2", name="x3",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                            format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")

    elif _division_sixteen(shape_0) and not _division_sixteen(shape_1) \
            and _division_sixteen(shape_2):
        # Nz+ND+Nz
        input0 = gen_param(classify="input0", name="x1",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        input1 = gen_param(classify="input1", name="x2",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input2 = gen_param(classify="input2", name="x3",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                            format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")

    elif not _division_sixteen(shape_0) and _division_sixteen(shape_1)\
        and not _division_sixteen(shape_2):
        # ND+NZ+ND
        input0 = gen_param(classify="input0", name="x1",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input1 = gen_param(classify="input1", name="x2",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        input2 = gen_param(classify="input2", name="x3",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                            format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")

    elif not _division_sixteen(shape_0) and not _division_sixteen(shape_1)\
        and _division_sixteen(shape_2):
        # ND+ND+NZ
        input0 = gen_param(classify="input0", name="x1",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input1 = gen_param(classify="input1", name="x2",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,ND")
        input2 = gen_param(classify="input2", name="x3",
                           datatype="float16,float16,float16,float16,float16,\
                                     float,float,float,float,float,\
                                     int32,int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                   NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float16,float16,float16,float16,\
                                      float,float,float,float,float,\
                                      int32,int32,int32,int32,int32",
                            format="NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ,\
                                    NCHW,NC1HWC0,NHWC,ND,FRACTAL_NZ")
    else:
        # ND+ND
        input0 = gen_param(classify="input0", name="x1",
                           datatype="float16,float16,float16,float16,\
                                     float,float,float,float,\
                                     int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND")
        input1 = gen_param(classify="input1", name="x2",
                           datatype="float16,float16,float16,float16,\
                                     float,float,float,float,\
                                     int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND")
        input2 = gen_param(classify="input2", name="x3",
                           datatype="float16,float16,float16,float16,\
                                     float,float,float,float,\
                                     int32,int32,int32,int32",
                           format="NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,float16,float16,float16,\
                                      float,float,float,float,\
                                      int32,int32,int32,int32",
                            format="NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND,\
                                   NCHW,NC1HWC0,NHWC,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def check_format(format_input0, format_input1, format_input2):
    """
    check the format_list
    """
    list_format = [format_input0, format_input1, format_input2]
    list_pattern = [["FRACTAL_NZ", "ND", "ND"],
                    ["FRACTAL_NZ", "NHWC", "NHWC"],
                    ["FRACTAL_NZ", "NCHW", "NCHW"],
                    ["ND", "FRACTAL_NZ", "ND"],
                    ["NHWC", "FRACTAL_NZ", "NHWC"],
                    ["NCHW", "FRACTAL_NZ", "NCHW"],
                    ["ND", "ND", "FRACTAL_NZ"],
                    ["NHWC", "NHWC", "FRACTAL_NZ"],
                    ["NCHW", "NCHW", "FRACTAL_NZ"]
                   ]
    if list_format in list_pattern:
        format_pattern = list_pattern.index(list_format)
        format_pattern = math.ceil((format_pattern+1)/3)
    else:
        format_pattern = 0

    return format_pattern


def check_ori_shape(input0, input1, input2, format_pattern):
    """
    check the ND shapes whether they can be broadcasted
    """
    shape_0 = list(util.scalar2tensor_one(input0.get("ori_shape")))
    shape_1 = list(util.scalar2tensor_one(input1.get("ori_shape")))
    shape_2 = list(util.scalar2tensor_one(input2.get("ori_shape")))
    shape_input0, shape_input1, shape_max_mul =\
        util.produce_shapes(shape_0, shape_1)
    shape_input2, shape_max_mul, shape_max_add0 =\
        util.produce_shapes(shape_0, shape_2)


def _infer_shape(shape_input0, shape_input1, shape_input2, format_pattern):
    """
    shape_input0 : FRACTAL_NZ, [N,...,A,B,16,16]
    last_two_dims : [B*16, A*16]
    """
    if format_pattern == 2:
        shape_input0, shape_input1 = shape_input1, shape_input0
    if format_pattern == 3:
        shape_input0, shape_input2 = shape_input2, shape_input0

    last_two_dims = [shape_input0[-2]*shape_input0[-3],
                     shape_input0[-4]*shape_input0[-1]]
    condition2 = (len(shape_input1) == 1 and shape_input1[0] == 1)
    if not condition2:
        if len(shape_input1) == 1:
            shape_input1.insert(0, 1)
        condition0 = (shape_input1[-1] == last_two_dims[-1])
        condition1 = (shape_input1[-2] == last_two_dims[-2])

    condition5 = (len(shape_input2) == 1 and shape_input2[0] == 1)
    if not condition5:
        if len(shape_input2) == 1:
            shape_input2.insert(0, 1)
        condition3 = (shape_input2[-1] == last_two_dims[-1])
        condition4 = (shape_input2[-2] == last_two_dims[-2])

    if condition2:
        shape_input0, shape_input1, shape_max_mul =\
            util.produce_shapes(shape_input0, shape_input1)
    elif condition0 and not condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-4] = shape_input0[-4]
        shape_input1[-1] = shape_input0[-1]
        shape_input1[-2] = 1
        shape_input1[-3] = 1
        shape_input0, shape_input1, shape_max_mul =\
            util.produce_shapes(shape_input0, shape_input1)
    elif not condition0 and condition1:
        shape_input1.append(1)
        shape_input1.append(1)
        shape_input1[-2] = shape_input0[-2]
        shape_input1[-3] = shape_input0[-3]
        shape_input1[-4] = 1
        shape_input1[-1] = 1
        shape_input0, shape_input1, shape_max_mul =\
            util.produce_shapes(shape_input0, shape_input1)
    else:
        raise RuntimeError("shape of input1 or input0 is illegal")

    if condition5:
        shape_input2, shape_max_mul, shape_max_add0 =\
            util.produce_shapes(shape_input2, shape_max_mul)
    elif condition3 and not condition4:
        shape_input2.append(1)
        shape_input2.append(1)
        shape_input2[-4] = shape_input0[-4]
        shape_input2[-1] = shape_input0[-1]
        shape_input2[-2] = 1
        shape_input2[-3] = 1
        shape_input2, shape_max_mul, shape_max_add0 =\
            util.produce_shapes(shape_input2, shape_max_mul)
    elif not condition3 and condition4:
        shape_input2.append(1)
        shape_input2.append(1)
        shape_input2[-2] = shape_input0[-2]
        shape_input2[-3] = shape_input0[-3]
        shape_input2[-4] = 1
        shape_input2[-1] = 1
        shape_input2, shape_max_mul, shape_max_add0 =\
            util.produce_shapes(shape_input2, shape_max_mul)
    else:
        raise RuntimeError("shape of input2 or input0 is illegal")

    if format_pattern == 2:
        shape_input0, shape_input1 = shape_input1, shape_input0
    if format_pattern == 3:
        shape_input0, shape_input2 = shape_input2, shape_input0

    return shape_input0, shape_input1, shape_input2


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


def fused_mul_add_compute(data_input0, data_input1, data_input2,
                          output, kernel_name="fused_mul_add"):
    """
    mul+add calculation function

    Parameters
    ----------
    data_input0: TVM tensor
         the input tensor of mul
    data_input1: TVM tensor
         the input tensor of mul
    data_input2: TVM tensor
         the input tensor of add
    output: TVM tensor
         the output tensor of add
    kernel_name : str
        kernel name, default value is "fuesd_mul_add"

    Returns
    -------
    output tensor
    """

    # mul
    data_input0, data_input1 = shape_broadcast(data_input0, data_input1)
    mul_result = te.lang.cce.vmul(data_input0, data_input1)

    # add
    mul_result, data_input2 = shape_broadcast(mul_result, data_input2)
    res = te.lang.cce.vadd(mul_result, data_input2)

    return res


#@util.check_input_type(dict, dict, dict, dict, str)
def fused_mul_add(input0, input1, input2,
                  output, kernel_name="fused_mul_add"):
    """
    function: fused for mul+add

    Parameters
    ----------
    input0: dict
         the dict of input of mul, support float16,float32,int32
    input1: dict
         the dict of input of mul, support float16,float32,int32
    input2: dict
         the dict of input of add, support float16,float32,int32
    output: dict
         the dict of output of add, support float16,float32,int32
    kernel_name: str
        cce kernel name, default value is fused_mul_add

    Returns
    -------
    None
    """
    shape_input0 = list(util.scalar2tensor_one(input0.get("shape")))
    shape_input1 = list(util.scalar2tensor_one(input1.get("shape")))
    shape_input2 = list(util.scalar2tensor_one(input2.get("shape")))

    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()

    format_input0 = input0.get("format").upper()
    format_input1 = input1.get("format").upper()
    format_input2 = input2.get("format").upper()

    util.check_kernel_name(kernel_name)

    format_pattern = check_format(format_input0, format_input1, format_input2)
    if format_pattern != 0:
        check_ori_shape(input0, input1, input2, format_pattern)
        shape_input0, shape_input1, shape_input2 =\
            _infer_shape(shape_input0, shape_input1,
                         shape_input2, format_pattern)
    else:
        shape_input0, shape_input1, shape_max_mul =\
            util.produce_shapes(shape_input0, shape_input1)
        shape_input2, shape_max_mul, shape_max_add0 =\
            util.produce_shapes(shape_input2, shape_max_mul)

    data_input0 = tvm.placeholder(shape_input0,
                                  name="data_input0",
                                  dtype=dtype_input0)
    data_input1 = tvm.placeholder(shape_input1,
                                  name="data_input1",
                                  dtype=dtype_input1)
    data_input2 = tvm.placeholder(shape_input2,
                                  name="data_input2",
                                  dtype=dtype_input2)

    res = fused_mul_add_compute(data_input0, data_input1, data_input2,
                                output, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_input0, data_input1, data_input2, res)}

    te.lang.cce.cce_build_code(sch, config)
