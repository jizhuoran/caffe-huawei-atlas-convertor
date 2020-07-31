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

tile_with_axis
"""

# pylint: disable=import-error
import te.lang.cce
from te import tvm, platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from impl.util.util_select_op_base import gen_param, get_dynamic_param_in_json


@fusion_manager.register("tile_with_axis")
def tile_with_axis_compute(data, shape_y):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    shape_y: tuple or list.
        The shape of output.

    Returns
    -------
    res the compute results
    """
    res = te.lang.cce.broadcast(data, shape_y)

    return res


# pylint: disable=unused-argument
def op_select_format(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    select format dynamically
    """
    ori_format = input_x.get("ori_format")
    ori_shape = input_x.get("ori_shape")

    if ori_shape is not None:
        axis = util.axis_check(len(ori_shape), axis)

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    # for 5hd, axis is only valid for n,h,w
    if ((ori_format == "NHWC" and axis != 3) or (ori_format == "NCHW" and axis != 1)) and \
            len(ori_shape) == 4:
        # NC1HWC0+ND
        if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
            # fp16
            input0 = gen_param(
                classify="input0", name="x",
                datatype="float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64,"
                         "float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,"
                       "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
            output0 = gen_param(
                classify="output0", name="y",
                datatype="float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64,"
                         "float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,"
                       "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
        else:
            # fp16/fp32
            input0 = gen_param(
                classify="input0", name="x",
                datatype="float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64,"
                         "float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,NC1HWC0,"
                       "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
            output0 = gen_param(
                classify="output0", name="y",
                datatype="float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64,"
                         "float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,NC1HWC0,"
                       "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
    else:
        # ND
        if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
            # fp16
            input0 = gen_param(
                classify="input0", name="x",
                datatype="float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND")
            output0 = gen_param(
                classify="output0", name="y",
                datatype="float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND")
        else:
            # fp16/fp32
            input0 = gen_param(
                classify="input0", name="x",
                datatype="float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND")
            output0 = gen_param(
                classify="output0", name="y",
                datatype="float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND")

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


@util.check_input_type(dict, dict, int, int, str)
def tile_with_axis(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    algorithm: tile.
    Expanding the input tensor according to a specified dimension,
    and the expansion multiple is specified by the tiles param.
    For example, tiling [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11,
    12]]], which shape is (2, 3, 2), by axis:1 and tiles:2 produces
    [[[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12], [7, 8], [9, 10], [11, 12]]]
    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    axis: int
         The index of the axis to tile
    tiles: int
        The number of copies (tiles) of the blob to output.
    kernel_name : str
        kernel name, default value is "tile_with_axis"

    Returns
    -------
    tik_instance
    """

    axis, shape_x, shape_y, dtype_x = check_param(input_x, output_y, tiles,
                                                  axis, kernel_name)

    input_data = tvm.placeholder(shape_x, name="input_data", dtype=dtype_x)

    if tiles > 1:
        res = tile_with_axis_compute(input_data, shape_y)
    else:
        zero_data = tvm.const(0, dtype=dtype_x)
        res = te.lang.cce.vadds(input_data, zero_data)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input_data, res]}

    te.lang.cce.cce_build_code(sch, config)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def check_param(input_x, output_y, tiles, axis, kernel_name):
    """
    Check the input parameter

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    axis: int
         The index of the axis to tile
    tiles: int
        The number of copies (tiles) of the blob to output.
    kernel_name : str
        kernel name, default value is "tile_with_axis"

    Returns
    ----------
    axis: int
         The index of the axis to tile which is adjusted to positive
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()
    shape_y = output_y.get("shape")
    dtype_y = output_y.get("dtype").lower()

    util.check_shape_rule(shape_x)
    util.check_tensor_shape_size(shape_x)
    util.check_shape_rule(shape_y)
    util.check_tensor_shape_size(shape_y)

    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32")
    util.check_dtype_rule(dtype_x, check_list)
    util.check_dtype_rule(dtype_y, check_list)

    if dtype_x != dtype_y:
        raise RuntimeError(
            "output's data type must be the same as input's data type")

    if tiles <= 0:
        raise RuntimeError(
            "In tile of TBE, the tiles can't be less than 1")

    shape_x_len = len(shape_x)

    # check for 5HD
    input_format = input_x.get("format")
    if input_format == "NC1HWC0":
        shape_x_ori = input_x.get("ori_shape")
        ori_format = input_x.get("ori_format")
        length_x_ori = len(shape_x_ori)

        if ori_format not in ("NCHW", "NHWC"):
            raise RuntimeError("input_x's ori_format is invalid for 5D Tensor")
        if shape_x_len != 5:
            raise RuntimeError("input_x's shape is invalid for 5D Tensor")
        if length_x_ori != 4:
            raise RuntimeError("input_x's ori_shape is invalid for 5D Tensor")
        axis = util.axis_check(length_x_ori, axis)
        axis = util.axis_transfrom_5d(axis, ori_format)
        if axis in (1, 4):
            raise RuntimeError("axis is invalid for 5D Tensor")
    else:
        if axis >= shape_x_len or axis < -shape_x_len:
            raise RuntimeError(
                "In tile of TBE, the axis can't all be more than the input "
                "dimensions")

        if axis < 0:
            axis += shape_x_len

    shape_y_expected = [0] * shape_x_len
    shape_y_expected[0:shape_x_len] = shape_x[0:shape_x_len]
    shape_y_expected[axis] *= tiles

    if not check_same_shape(shape_y, shape_y_expected):
        raise RuntimeError("output_y's shape is incorrect")

    shape_x_adapt = []
    shape_y_adapt = []
    for i in range(shape_x_len):
        if i == axis:
            shape_x_adapt.append(1)
            shape_y_adapt.append(tiles)
            if shape_x[i] == 1:
                continue
        shape_x_adapt.append(shape_x[i])
        shape_y_adapt.append(shape_x[i])

    util.check_kernel_name(kernel_name)

    return axis, shape_x_adapt, shape_y_adapt, dtype_x


def check_same_shape(shape_x, shape_y):
    """
    check shape_x is the same shape as shape_y

    Parameters
    ----------
    shape_x: a tuple or list
    shape_y: a tuple or list

    Returns
    -------
    boolean: True, if the same shape otherwise False
    """
    shape_x_len = len(shape_x)
    shape_y_len = len(shape_y)

    if shape_x_len != shape_y_len:
        return False

    for i in range(shape_x_len):
        if shape_x[i] != shape_y[i]:
            return False

    return True
