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

add
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from te import platform as tbe_platform

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
SIZE_SIXTEEN = 16


def _can_division_sixteen(shape):
    if shape[-1] == 0 or shape[-2] == 0:
        raise RuntimeError("value of shape is illegal")

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True

    return False


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=invalid-name,too-many-locals,too-many-branches,unused-variable
# pylint: disable=too-many-statements,too-many-boolean-expressions
def op_select_format(input_x, input_y, output_z, kernel_name="add"):
    """
    select format dynamically
    """
    shape_x = input_x.get("ori_shape")
    shape_y = input_y.get("ori_shape")

    shape_x = util.scalar2tensor_one(shape_x)
    shape_y = util.scalar2tensor_one(shape_y)

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cce_product in ("Hi3796CV300ES",):
        dtype_list = ["float16", "int32"]
    else:
        dtype_list = ["float16", "float32", "int32"]

    format_x = input_x.get("ori_format")
    format_y = input_y.get("ori_format")

    dtype_total = []
    format_nd = ["ND"]
    format_list = ["ND"]
    format_nz = ["FRACTAL_NZ"]
    len_format_list = len(dtype_list)
    add_nd_nz = False
    add_nz_nd = False
    if len(shape_x) == 1 and len(shape_y) >= 2 and shape_x[-1] == shape_y[-1]:
        for i in range(0, len(shape_y)):
            if shape_x[0] == shape_y[i] and shape_x [0] % 16 == 0:
                add_nd_nz = True
                break
    if len(shape_y) == 1 and len(shape_x) >= 2 and shape_x[-1] == shape_y[-1]:
        for i in range(0, len(shape_x)):
            if shape_y[0] == shape_x[i] and shape_y[0] % 16 == 0:
                add_nz_nd = True
                break

    if len(shape_x) == 4 and len(shape_y) == 4 and \
            format_x in format_4d_list and format_y in format_4d_list:
        x_cdim = shape_x[format_x.index("C")]
        x_wdim = shape_x[format_x.index("W")]
        x_hdim = shape_x[format_x.index("H")]
        x_ndim = shape_x[format_x.index("N")]
        y_cdim = shape_y[format_y.index("C")]
        y_wdim = shape_y[format_y.index("W")]
        y_hdim = shape_y[format_y.index("H")]
        y_ndim = shape_y[format_y.index("N")]
    if (len(shape_y) == 1 and shape_y[0] == 1 and len(shape_x) == 4) and \
            format_x in format_4d_list:
        x_cdim = shape_x[format_x.index("C")]
        x_ndim = shape_x[format_x.index("N")]
    if (len(shape_x) == 1 and shape_x[0] == 1 and len(shape_y) == 4) and \
            format_y in format_4d_list:
        y_cdim = shape_y[format_y.index("C")]
        y_ndim = shape_y[format_y.index("N")]

    #ND+ND NZ+NZ 5HD+5HD FZ+FZ
    if len(shape_x) >= 2 and len(shape_y) >= 2 and shape_x[-2:] == shape_y[-2:]:
        format_list.append("FRACTAL_NZ")
        if len(shape_x) == 4 and len(shape_y) == 4 and \
                format_x in format_4d_list and format_y in format_4d_list:
            if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                if format_x == format_y == "NCHW" and \
                        (x_cdim == y_cdim or x_cdim // 16 == 1 or y_cdim // 16 == 1) and \
                        (x_ndim == y_ndim or x_ndim == 1 or y_ndim == 1):
                    format_list.append("NC1HWC0")
                if format_x == format_y == "HWCN":
                    if x_hdim == y_hdim and (x_wdim == 1 or y_wdim == 1):
                        format_list.append("NC1HWC0")
                    if x_wdim == y_wdim and (x_hdim == 1 or y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if x_wdim == y_wdim and x_hdim == y_hdim:
                        format_list.append("NC1HWC0")
                    if (x_wdim == x_hdim == 1) or (y_hdim == y_wdim == 1):
                        format_list.append("NC1HWC0")
                    if (x_hdim == y_wdim == 1) or (x_wdim == y_hdim == 1):
                        format_list.append("NC1HWC0")
                if format_x == format_y == "NHWC":
                    if x_hdim == y_hdim and (x_ndim == 1 or y_ndim == 1):
                        format_list.append("NC1HWC0")
                    if x_ndim == y_ndim and (x_hdim == 1 or y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if x_ndim == y_ndim and x_hdim == y_hdim:
                        format_list.append("NC1HWC0")
                    if (x_ndim == x_hdim == 1) or (y_ndim == y_hdim == 1):
                        format_list.append("NC1HWC0")
                    if (x_ndim == 1 and y_hdim == 1) or (x_hdim == 1 and y_ndim == 1):
                        format_list.append("NC1HWC0")
            if x_cdim % 16 == 0 and y_cdim % 16 == 0 and \
                    y_ndim % 16 == 0 and x_ndim % 16 == 0:
                if (format_x == format_y == "NHWC" and
                        list(shape_x) == list(shape_y)) or \
                        (format_x == format_y == "NCHW" and
                         list(shape_x) == list(shape_y)):
                    format_list.append("FRACTAL_Z")
                if format_x == format_y == "HWCN" and \
                        x_wdim*x_hdim == y_wdim*y_hdim:
                    format_list.append("FRACTAL_Z")
            if list(shape_x) == list(shape_y):
                format_list.append("NC1HWC0")
                format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        input0 = gen_param(classify="input0", name="x1",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list))
        input1 = gen_param(classify="input1", name="x2",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list))
        output0 = gen_param(classify="output0", name="y",
                            datatype=",".join(dtype_total),
                            format=",".join(format_list))

    #NZ+ND,ND+ND,5HD+5HD,FZ+FZ,ND+NZ
    elif len(shape_x) >= 2 and len(shape_y) >= 2 and \
            ((_can_division_sixteen(shape_x) and
              not _can_division_sixteen(shape_y)) or
             (not _can_division_sixteen(shape_x) and
              _can_division_sixteen(shape_y))):
        if len(shape_x) == 4 and len(shape_y) == 4 and \
                format_x in format_4d_list and format_y in format_4d_list:
            if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                if x_cdim == y_cdim or x_cdim // 16 == 1 or y_cdim // 16 == 1:
                    format_list.append("NC1HWC0")
            if x_cdim % 16 == 0 and x_ndim % 16 == 0 and \
                    y_cdim % 16 == 0 and y_ndim % 16 == 0:
                if format_x == format_y == "NCHW" and \
                        x_hdim*x_wdim == y_hdim*y_wdim and x_cdim == y_cdim:
                    if x_ndim == y_ndim:
                        format_list.append("FRACTAL_Z")
                    if (x_ndim // 16 == 1 and y_ndim % 16 == 0) or \
                            (y_ndim // 16 == 1 and x_ndim % 16 == 0):
                        format_list.append("FRACTAL_Z")
                if format_x == format_y == "NHWC" and \
                        x_hdim*x_wdim == y_hdim*y_wdim and \
                        x_ndim == y_ndim and x_cdim == y_cdim:
                    format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*1
        format_list0 = format_list + format_nz*len_format_list
        format_list1 = format_list + format_nd*len_format_list
        if _can_division_sixteen(shape_x) and not _can_division_sixteen(shape_y):
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list0))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list1))
            output0 = gen_param(classify="output0", name="y",
                                datatype=",".join(dtype_total),
                                format=",".join(format_list0))
        else:
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list1))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list0))
            output0 = gen_param(classify="output0", name="y",
                                datatype=",".join(dtype_total),
                                format=",".join(format_list1))

    elif add_nd_nz or add_nz_nd:
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*1
        format_list0 = format_list + format_nz*len_format_list
        format_list1 = format_list + format_nd*len_format_list
        if add_nz_nd:
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list0))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list1))
            output0 = gen_param(classify="output0", name="y",
                                datatype=",".join(dtype_total),
                                format=",".join(format_list0))
        else:
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list1))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_total),
                               format=",".join(format_list0))
            output0 = gen_param(classify="output0", name="y",
                                datatype=",".join(dtype_total),
                                format=",".join(format_list1))

    #5HD+scalar,ND+ND,FZ+scalar
    elif len(shape_x) >= 2 and len(shape_y) == 1 and shape_y[0] == 1:
        if len(shape_x) == 4 and len(shape_y) == 1 and format_x in format_4d_list:
            if x_cdim % 16 == 0:
                format_list.append("NC1HWC0")
            if x_cdim % 16 == 0 and x_ndim % 16 == 0:
                format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*1
        format_list0 = format_list + format_nd*len_format_list
        format_list1 = format_nd*len(format_list) + format_nd*len_format_list
        input0 = gen_param(classify="input0", name="x1",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list0))
        input1 = gen_param(classify="input1", name="x2",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list1))
        output0 = gen_param(classify="output0", name="y",
                            datatype=",".join(dtype_total),
                            format=",".join(format_list0))

    #ND+ND,scalar+5HD,scalar+FZ
    elif len(shape_y) >= 2 and len(shape_x) == 1 and shape_x[0] == 1:
        if len(shape_x) == 1 and len(shape_y) == 4 and format_y in format_4d_list:
            if y_cdim % 16 == 0:
                format_list.append("NC1HWC0")
            if y_cdim % 16 == 0 and y_ndim % 16 == 0:
                format_list.append("FRACTAL_Z")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        format_list = format_list*len_format_list
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*1
        format_list0 = format_list + format_nd*len_format_list
        format_list1 = format_nd*len(format_list) + format_nd*len_format_list
        input0 = gen_param(classify="input0", name="x1",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list1))
        input1 = gen_param(classify="input1", name="x2",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list0))
        output0 = gen_param(classify="output0", name="y",
                            datatype=",".join(dtype_total),
                            format=",".join(format_list0))
    #ND+ND,5HD+5HD
    else:
        if len(shape_x) == 1 and len(shape_y) == 1 and \
                shape_x[0] % 16 == 0 and shape_y[0] % 16 == 0:
            format_list.append("NC1HWC0")
        if len(shape_x) == 4 and len(shape_y) == 4 and format_x in format_4d_list and format_y in format_4d_list:
            if format_x == format_y == "NCHW" or format_x == format_y == "HWCN" \
                    or format_x == format_y == "NHWC":
                if x_cdim % 16 == 0 and y_cdim % 16 == 0:
                    if (x_cdim // 16 == 1 or y_cdim // 16 == 1) or (x_cdim == y_cdim):
                        if (x_ndim == y_ndim):
                            if x_hdim == y_hdim and (x_wdim == 1 or y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if x_wdim == y_wdim and (x_hdim == 1 or y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if x_hdim == y_hdim and x_wdim == y_wdim:
                                format_list.append("NC1HWC0")
                            if (x_wdim == x_hdim == 1) or (y_wdim == y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_hdim == 1 and y_wdim == 1) or (x_wdim == 1 and y_hdim == 1):
                                format_list.append("NC1HWC0")
                        if (x_hdim == y_hdim):
                            if x_ndim == y_ndim and (x_wdim == 1 or y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if x_wdim == y_wdim and (x_ndim == 1 or y_ndim == 1):
                                format_list.append("NC1HWC0")
                            if x_ndim == y_ndim and x_wdim == y_wdim:
                                format_list.append("NC1HWC0")
                            if (x_ndim == x_wdim == 1) or (y_ndim == y_wdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_ndim == 1 and y_wdim == 1) or (x_wdim == 1 and y_ndim == 1):
                                format_list.append("NC1HWC0")
                        if (x_wdim == y_wdim):
                            if x_ndim == y_ndim and (x_hdim == 1 or y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if x_hdim == y_hdim and (x_ndim == 1 or y_ndim == 1):
                                format_list.append("NC1HWC0")
                            if x_ndim == y_ndim and x_hdim == y_hdim:
                                format_list.append("NC1HWC0")
                            if (x_ndim == x_hdim == 1) or (y_ndim == y_hdim == 1):
                                format_list.append("NC1HWC0")
                            if (x_ndim == 1 and y_hdim == 1) or (x_hdim == 1 and y_ndim == 1):
                                format_list.append("NC1HWC0")
        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype]*len(format_list)
        len_format_list = len(dtype_list)
        format_list = format_list*len_format_list
        input0 = gen_param(classify="input0", name="x1",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list))
        input1 = gen_param(classify="input1", name="x2",
                           datatype=",".join(dtype_total),
                           format=",".join(format_list))
        output0 = gen_param(classify="output0", name="y",
                            datatype=",".join(dtype_total),
                            format=",".join(format_list))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _add_check_format(x, y):
    format_pattern = 0
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = util.scalar2tensor_one(shape1)
    shape2 = util.scalar2tensor_one(shape2)
    check_list = [["FRACTAL_NZ", "ND"],
                  ["ND", "FRACTAL_NZ"],
                  ["FRACTAL_NZ", "NHWC"],
                  ["NHWC", "FRACTAL_NZ"],
                  ["FRACTAL_NZ", "NCHW"],
                  ["NCHW", "FRACTAL_NZ"]]
    if list_format == check_list[0] \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[1] \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[2] \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[3] \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    elif list_format == check_list[4] \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format == check_list[5] \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2

    return format_pattern


def _infer_shape(format_pattern, x, y):
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    ori_shape_x = x.get("ori_shape")
    ori_shape_y = y.get("ori_shape")
    shape_x = util.scalar2tensor_one(shape_x)
    shape_y = util.scalar2tensor_one(shape_y)

    if format_pattern == 1:
        ori_shape_x, shape_y, shape_max = util.produce_shapes(ori_shape_x,
                                                              shape_y)
        if shape_y[-2] == 1 and shape_y[-1] == ori_shape_x[-1]:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-3] = 1
            shape_y[-1] = shape_x[-1]
            shape_y[-4] = shape_x[-4]

        elif shape_y[-2] == ori_shape_x[-2] and shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-4] = 1
            shape_y[-2] = shape_x[-2]
            shape_y[-3] = shape_x[-3]

        elif shape_y[-2] == shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)

    elif format_pattern == 2:
        shape_x, ori_shape_y, shape_max = util.produce_shapes(shape_x,
                                                              ori_shape_y)
        if shape_x[-2] == 1 and shape_x[-1] == ori_shape_y[-1]:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-3] = 1
            shape_x[-1] = shape_y[-1]
            shape_x[-4] = shape_y[-4]

        elif shape_x[-2] == ori_shape_y[-2] and shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-4] = 1
            shape_x[-2] = shape_y[-2]
            shape_x[-3] = shape_y[-3]

        elif shape_x[-2] == shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)

    return shape_x, shape_y


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("add")
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_data: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output of the data's add
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    input_x = te.lang.cce.broadcast(input_x, shape_max)
    input_y = te.lang.cce.broadcast(input_y, shape_max)
    res = te.lang.cce.vadd(input_x, input_y)

    return res


@util.check_input_type(dict, dict, dict, str)
def add(input_x, input_y, output_z, kernel_name="add"):
    """
    algorithm: add
    calculating data's add, c = a + b

    Parameters
    ----------
    input_x : dict
        shape and dtype of first input, only support float16, float32, int32
    input_y : dict
        shape and dtype of second input, only support float16, float32, int32
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name : str
        cce kernel name, default value is add

    Returns
    -------
    None
    """
    # format_pattern = 1  Nz and vector
    # format_pattern = 2  vector and Nz
    # format_pattern = 0  Nz scalar  Nz Nz  ND ND
    format_pattern = _add_check_format(input_x, input_y)
    shape_x, shape_y = _infer_shape(format_pattern, input_x, input_y)
    shape_x = util.scalar2tensor_one(shape_x)
    shape_y = util.scalar2tensor_one(shape_y)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_rule(shape_y)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_y, SHAPE_SIZE_LIMIT)

    check_tuple = ("float16", "float32", "int32")
    input_data_type = input_x.get("dtype").lower()
    util.check_dtype_rule(input_data_type, check_tuple)

    shape_x, shape_y, shape_max = util.produce_shapes(shape_x, shape_y)
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    util.check_shape_size(shape_max, SHAPE_SIZE_LIMIT)
    data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_data_type)
    data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_data_type)

    res = add_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": (data_x, data_y, res)}
    te.lang.cce.cce_build_code(schedule, config)
