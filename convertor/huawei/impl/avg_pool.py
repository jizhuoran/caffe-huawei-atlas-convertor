#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

tf avg_pool
"""
from __future__ import absolute_import
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# shape limit
# int32's max value
SHAPE_SIZE_LIMIT = 2 ** 31 - 1
# c0 size
C0SIZE = 16

NoneType = type(None)


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
def check_window_rule(ksize, strides, data_format):
    """
    check ksize and strides of window in pooling
    """
    if data_format in ("NHWC",):
        if len(ksize) != 4:
            raise RuntimeError("Invalid ksize params, ksize dim must be 4.")
        elif ksize[0] != 1 or ksize[3] != 1:
            raise RuntimeError("Only supports pooling across width/height,"
                               "and other ksize dimension should be one")
        if len(strides) != 4:
            raise RuntimeError("Invalid strides params, strides dim must be 4.")
        elif strides[0] != 1 or strides[3] != 1:
            raise RuntimeError("Only supports pooling across width/height,"
                               "and other strides dimension should be one")
    elif data_format in ("NC1HWC0", "NCHW"):
        if len(ksize) != 4:
            raise RuntimeError("Invalid ksize params, ksize dim must be 4.")
        elif ksize[0] != 1 or ksize[1] != 1:
            raise RuntimeError("Only supports pooling across width/height,"
                               "and other ksize dimension should be one")
        if len(strides) != 4:
            raise RuntimeError("Invalid strides params, strides dim must be 4.")
        elif strides[0] != 1 or strides[1] != 1:
            raise RuntimeError("Only supports pooling across width/height,"
                               "and other strides dimension should be one")
    else:
        raise RuntimeError("The data_format is not supported")


def avg_pool_check_rule(input_shape, input_dtype, output_dtype,
                        ksize, strides, data_format, kernel_name):
    """
    :param input_shape: shape of input_data
    :param input_dtype: dtype of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of avgpooling
    :param strides: the stride of avgpooling window
    :param data_format: NHWC default
    :param kernel_name: cce kernel name
    :return: None

    """
    # check input and output
    util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(input_shape)
    util.check_dtype_rule(input_dtype, ["float16"])
    util.check_dtype_rule(output_dtype, ["float16"])
    # check ksize and strides of window
    check_window_rule(ksize, strides, data_format)
    # check kernel name
    util.check_kernel_name(kernel_name)

# pylint: disable=unnecessary-lambda
@fusion_manager.register("avg_pool")
def avg_pool_compute(x, y, ksize, strides,
                     padding="VALID", data_format="NHWC",
                     kernel_name="avg_pool_cce"):
    """
    describe compute
    return: tensor
    """
    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

    window = list(window)
    stride = list(stride)

    # l1 fusion params assign
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = x.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in x.op.attrs else -1
    in_l1_flag = x.op.attrs["addr_type"].value == 1 \
        if "addr_type" in x.op.attrs else False
    in_valid_shape = x.op.attrs["valid_shape"] \
        if "valid_shape" in x.op.attrs else []
    in_slice_offset = x.op.attrs["slice_offset"] \
        if "slice_offset" in x.op.attrs else []
    in_select_read_flag = bool(in_valid_shape)
    in_split_index = x.op.attrs["split_index"].value \
        if "split_index" in x.op.attrs else 0
    out_l1_flag = y.get("addr_type") == 1
    out_valid_shape = y.get("valid_shape", [])
    out_select_write_flag = bool(out_valid_shape)
    out_shape = y.get("shape")
    out_total_shape = y.get("valid_shape") \
        if out_select_write_flag else y.get("shape")
    out_slice_offset = y.get("slice_offset", [0, 0, 0, 0, 0])
    fusion_params = {"l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag,
                     "out_l1_flag": out_l1_flag,
                     "in_select_read_flag": in_select_read_flag,
                     "in_split_index": in_split_index,
                     "out_select_write_flag": out_select_write_flag,
                     "out_total_shape": out_total_shape,
                     "out_shape": out_shape,
                     "out_slice_offset": out_slice_offset}

    if in_select_read_flag:
        select_tensor_in = tvm.compute(in_valid_shape,
                                       lambda n, c1, h, w, c0:
                                       x(n, c1, h + in_slice_offset[2], w, c0),
                                       name="tensor_read_select",
                                       attrs=x.op.attrs)
        res = te.lang.cce.pooling2d(select_tensor_in, window, stride, "AVG",
                                    padding, fusion_params=fusion_params)
    elif l1_fusion_type == 1:
        x.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(x.shape,
                                         lambda n, c1, h, w, c0:
                                         x(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=x.op.attrs)
        res = te.lang.cce.pooling2d(l1_width_fusion_in, window, stride,
                                    "AVG", padding,
                                    fusion_params=fusion_params)
    else:
        res = te.lang.cce.pooling2d(x, window, stride, "AVG", padding,
                                    fusion_params=fusion_params)

    return res


@util.check_input_type(dict, dict, (list, tuple), (list, tuple),
                       str, str,
                       str)
def avg_pool(x, y, ksize, strides,
             padding="VALID", data_format="NHWC",
             kernel_name="avg_pool_cce"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4 dims, format is NCHW

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling in H or W

    strides : list or tuple, the stride of avgpooling window, only support avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    None
    """
    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()

    # check others parameter
    avg_pool_check_rule(input_shape, input_dtype, output_dtype,
                        ksize, strides, data_format, kernel_name)

    # set tensor attrs, during L1 fusion these attrs will assign by te_fusion
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    split_index = x.get("split_index", 0)
    l1_fusion_type = x.get("L1_fusion_type", -1)
    attr = {"addr_type": addr_type,
            "valid_shape": valid_shape,
            "slice_offset": slice_offset,
            "split_index": split_index,
            "L1_fusion_type": l1_fusion_type}
    is_l1fusion = l1_fusion_type in (0, 1)

    # compute
    # create tensor_in
    tensor_in = tvm.placeholder(input_shape, name="tensor_in",
                                dtype=input_dtype, attrs=attr)
    res = avg_pool_compute(tensor_in, y, ksize, strides, padding, data_format,
                           kernel_name)

    # schedule
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    # build
    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": [tensor_in, res],
              "l1_fusion_option": is_l1fusion}

    te.lang.cce.cce_build_code(sch, config)
