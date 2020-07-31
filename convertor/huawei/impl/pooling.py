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

caffe pooling
"""

from __future__ import absolute_import
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
import impl
from te.lang.cce.te_compute.pooling2d_compute import get_caffe_out_size_and_pad
from te.platform.cce_policy import get_L1_info

# shape limit
# int32's max value
SHAPE_SIZE_LIMIT = 2 ** 31 - 1
# c0 size
C0SIZE = 16

NoneType = type(None)


# pylint: disable=locally-disabled,unused-argument,invalid-name
# pylint: disable=too-many-arguments,too-many-locals
def pooling_check_rule(input_shape, output_dtype, window, stride, kernel_name):
    """
    :param input_shape: shape of input_data
    :param output_dtype: dtype of output_data
    :param window: shape of window
    :param stride: shape of stride
    :param kernel_name: cce kernel name
    :return: None
    """
    # check input and output
    util.check_shape_size(input_shape, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(input_shape)
    util.check_dtype_rule(output_dtype, ["float16", "int32"])
    # check window and stride length
    if len(window) != 2:
        raise RuntimeError(
            "Invalid shape params, window shape must be 2 dims, including "
            "window_h and window_w.")
    if len(stride) != 2:
        raise RuntimeError(
            "Invalid shape params, stride shape must be 2 dims, including "
            "stride_h and stride_w.")
    # check kernel name
    util.check_kernel_name(kernel_name)


def get_fusion_params(input_data, output_data):
    """
    :param input_data: tensor of input_data
    :param output_data: dict of output_data
    :return: dict fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in input_data.op.attrs else -1
    in_l1_flag = input_data.op.attrs["addr_type"].value == 1 \
        if "addr_type" in input_data.op.attrs else False
    in_valid_shape = input_data.op.attrs["valid_shape"] \
        if "valid_shape" in input_data.op.attrs else []
    in_slice_offset = input_data.op.attrs["slice_offset"] \
        if "slice_offset" in input_data.op.attrs else []
    in_select_read_flag = bool(in_valid_shape)
    out_l1_flag = output_data.get("addr_type") == 1
    out_valid_shape = output_data.get("valid_shape", [])
    out_select_write_flag = bool(out_valid_shape)
    out_shape = output_data.get("shape")
    out_total_shape = output_data.get("valid_shape") \
        if out_select_write_flag else output_data.get("shape")
    out_slice_offset = output_data.get("slice_offset", [0, 0, 0, 0, 0])
    fusion_params = {"l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag,
                     "out_l1_flag": out_l1_flag,
                     "in_select_read_flag": in_select_read_flag,
                     "out_select_write_flag": out_select_write_flag,
                     "out_total_shape": out_total_shape,
                     "out_shape": out_shape,
                     "out_slice_offset": out_slice_offset,
                     "in_slice_offset": in_slice_offset,
                     "in_valid_shape": in_valid_shape}

    return fusion_params


@fusion_manager.register("pooling")
# Example: pylint: disable=unnecessary-lambda
def pool_fuse_compute(input_data, matrix, bias, output_data, window,
                      stride, offset_x=0, mode=0, pad=(0, 0, 0, 0),
                      global_pooling=False, ceil_mode=0,
                      dilation=(1, 1, 1, 1),
                      kernel_name="pool_fuse"):
    """
    Performs pooling on the input.

    Parameters
    ----------
    input_data: TVM tensor
        A `Tensor`. Must be one of the following types: `float16`
        4-D input to pool over.
    matrix: TVM tensor, shape and dtype of right matrix, only support float16,
        shape is 4 dims, format is NCHW
    bias: TVM tensor, use it to modify bias for fusion op.
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    window: list or tuple
        A list of `ints` that has length 2.
        The size of the window for H, W dimension of the input tensor.
    stride: list or tuple
        A list of `ints` that has length 2.
        The stride of the sliding window for H, W .
    offset_x: avg quantize params.
    pad : list or tuple, the pad of pooling, only support pooling in H or W

    global_pooling : global pooling params.

    mode: str
        A int which stands6 for kinds of  pooling.

    dilation : reserved.

    ceil_mode : caffe round_mode params, 0:CEIL, 1:FLOOR, default value is
    DOMI_POOLING_CEIL

    kernel_name: str
        kernel name, default value is 'pool_fuse'

    Returns:
    -------
    res: TVM tensor
        output tensor. Has the same type as `input_data`.
    """
    # get input_shape
    input_x = input_data.shape
    input_h = input_x[2].value
    input_w = input_x[3].value

    # convert mode&pad_mode to str for pooling2d
    pad = list(pad)

    if mode == 0:
        conv_pooling_flag = False
        temp_tensor = input_data
        while temp_tensor.op.input_tensors:
            if temp_tensor.op.tag == "convolution_C":
                conv_pooling_flag = True
                break
            temp_tensor = temp_tensor.op.input_tensors[0]
        if conv_pooling_flag:
            window_h, window_w = window[0], window[1]
            stride_h, stride_w = stride[0], stride[1]
            res = te.lang.cce.max_pool_v200(input_data, (window_h, window_w),
                                            (stride_h, stride_w), "SAME", pad)
        else:
            # call pooling2d for max(pooling)&gmp
            mode_max = "MAX"
            if (input_h == window[0] and input_w == window[1] and
                    pad == [0, 0, 0, 0]) or \
                    global_pooling:
                mode_max = "GMP"
            window = list(window)

            # l1 fusion and l2 fusion
            l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value \
                if "L1_fusion_type" in input_data.op.attrs else -1

            # l1 fusion params assign
            fusion_params = get_fusion_params(input_data, output_data)
            in_select_read_flag = fusion_params.get("in_select_read_flag")
            in_valid_shape = fusion_params.get("in_valid_shape")
            in_slice_offset = fusion_params.get("in_slice_offset")

            if in_select_read_flag:
                select_tensor_in = \
                    tvm.compute(in_valid_shape,
                                lambda n, c1, h, w, c0:
                                input_data(n, c1, h + in_slice_offset[2],
                                           w, c0),
                                name="tensor_read_select",
                                attrs=input_data.op.attrs)
                res = te.lang.cce.pooling2d(select_tensor_in,
                                            window,
                                            stride,
                                            mode_max,
                                            pad=pad, data_mode=0,
                                            ceil_mode=ceil_mode,
                                            fusion_params=fusion_params)
            elif l1_fusion_type == 1:
                input_data.op.attrs["addr_type"].value = 1
                in_l1_flag = True
                fusion_params["in_l1_flag"] = in_l1_flag

                l1_width_fusion_in = \
                    tvm.compute(input_data.shape,
                                lambda n, c1, h, w, c0:
                                input_data(n, c1, h, w, c0),
                                name="l1_width_fusion_tensor_in",
                                attrs=input_data.op.attrs)
                res = te.lang.cce.pooling2d(l1_width_fusion_in, window,
                                            stride,
                                            mode_max, pad=pad, data_mode=0,
                                            ceil_mode=ceil_mode,
                                            fusion_params=fusion_params)
            else:
                res = te.lang.cce.pooling2d(input_data,
                                            window,
                                            stride,
                                            mode_max,
                                            pad=pad,
                                            data_mode=0,
                                            ceil_mode=ceil_mode,
                                            fusion_params=fusion_params)
    elif mode == 1:
        mode_avg = "AVG"
        if (input_h == window[0] and input_w == window[1] and
                pad == [0, 0, 0, 0]) or \
                global_pooling:
            mode_avg = "GAP"

        # call conv2d_compute to fuse for avg_cube
        if mode_avg == "AVG" and matrix is not None:
            # conv2d interface strides is 4D
            strides = (1, 1, stride[0], stride[1])
            # get real pad
            _, _, pad_top, pad_bottom, pad_left, pad_right \
                = get_caffe_out_size_and_pad(ceil_mode, input_h, input_w,
                                             window[0], window[1],
                                             stride[0], stride[1],
                                             dilation[0], dilation[1],
                                             pad[0], pad[1], pad[2],
                                             pad[3])
            conv2d_pad = (pad_top, pad_bottom, pad_left, pad_right)
            # call conv2d_compute for avg
            res = impl.conv2d_compute(input_data, matrix, bias, None,
                                      output_data,
                                      strides, conv2d_pad,
                                      dilation, groups=1,
                                      data_format='NCHW',
                                      offset_x=offset_x,
                                      kernel_name="conv2d")
        else:
            # call pooling2d for gap&avg_old
            window = list(window)

            # l1 fusion and l2 fusion
            l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value \
                if "L1_fusion_type" in input_data.op.attrs else -1

            # l1 fusion params assign
            fusion_params = get_fusion_params(input_data, output_data)
            in_select_read_flag = fusion_params.get("in_select_read_flag")
            in_valid_shape = fusion_params.get("in_valid_shape")
            in_slice_offset = fusion_params.get("in_slice_offset")

            if in_select_read_flag:
                select_tensor_in = \
                    tvm.compute(in_valid_shape,
                                lambda n, c1, h, w, c0:
                                input_data(n, c1, h + in_slice_offset[2],
                                           w, c0),
                                name="tensor_read_select",
                                attrs=input_data.op.attrs)
                res = te.lang.cce.pooling2d(select_tensor_in,
                                            window,
                                            stride,
                                            mode_avg,
                                            pad=pad, data_mode=0,
                                            ceil_mode=ceil_mode,
                                            fusion_params=fusion_params)
            elif l1_fusion_type == 1:
                input_data.op.attrs["addr_type"].value = 1
                in_l1_flag = True
                fusion_params["in_l1_flag"] = in_l1_flag

                l1_width_fusion_in = \
                    tvm.compute(input_data.shape,
                                lambda n, c1, h, w, c0:
                                input_data(n, c1, h, w, c0),
                                name="l1_width_fusion_tensor_in",
                                attrs=input_data.op.attrs)
                res = te.lang.cce.pooling2d(l1_width_fusion_in, window,
                                            stride,
                                            mode_avg, pad=pad, data_mode=0,
                                            ceil_mode=ceil_mode,
                                            fusion_params=fusion_params)
            else:
                res = te.lang.cce.pooling2d(input_data,
                                            window,
                                            stride,
                                            mode_avg,
                                            pad=pad,
                                            data_mode=0,
                                            ceil_mode=ceil_mode,
                                            fusion_params=fusion_params)
    else:
        raise RuntimeError("Invalid mode parameters, mode must set 0 or 1.")

    return res


# pylint: disable=unnecessary-lambda
def pooling_compute(x, matrix, y, window, stride,
                    mode=0, pad=(0, 0, 0, 0),
                    global_pooling=False, ceil_mode=0,
                    kernel_name="pooling_cce"):
    """
    describe compute
    return: tensor
    """
    input_x = x.shape
    input_h = input_x[2].value
    input_w = input_x[3].value

    # convert mode&pad_mode to str for pooling2d
    pad = list(pad)
    if mode == 0:
        mode = "MAX"
        if (input_h == window[0] and input_w == window[1] and
            pad == [0, 0, 0, 0]) or \
                global_pooling:
            mode = "GMP"
    elif mode == 1:
        mode = "AVG"
        if (input_h == window[0] and input_w == window[1] and
            pad == [0, 0, 0, 0]) or \
                global_pooling:
            mode = "GAP"
    else:
        raise RuntimeError("Invalid mode parameters, mode must set 0 or 1.")

    window = list(window)

    # l1 fusion params assign
    fusion_params = get_fusion_params(x, y)
    in_select_read_flag = fusion_params.get("in_select_read_flag")
    in_valid_shape = fusion_params.get("in_valid_shape")
    in_slice_offset = fusion_params.get("in_slice_offset")
    l1_fusion_type = fusion_params.get("l1_fusion_type")

    if in_select_read_flag:
        select_tensor_in = tvm.compute(in_valid_shape,
                                       lambda n, c1, h, w, c0:
                                       x(n, c1, h + in_slice_offset[2], w, c0),
                                       name="tensor_read_select",
                                       attrs=x.op.attrs)
        res = te.lang.cce.pooling2d(select_tensor_in, window, stride, mode,
                                    pad=pad, data_mode=0, ceil_mode=ceil_mode,
                                    fusion_params=fusion_params)
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
                                    mode, pad=pad, data_mode=0,
                                    ceil_mode=ceil_mode,
                                    fusion_params=fusion_params)
    else:
        res = te.lang.cce.pooling2d(x, window, stride, mode, pad=pad,
                                    data_mode=0,
                                    ceil_mode=ceil_mode,
                                    fusion_params=fusion_params)

    return res


@util.check_input_type(dict, (dict, NoneType), (dict, NoneType), dict,
                       (list, tuple), (list, tuple),
                       int, int, (list, tuple),
                       bool, int, (list, tuple),
                       str)
def pooling(x, matrix, bias, y, window=(1, 1), stride=(1, 1),
            offset_x=0, mode=0, pad=(0, 0, 0, 0),
            global_pooling=False, ceil_mode=0, dilation=(1, 1, 1, 1),
            kernel_name="pooling_cce"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16,
    shape is 4 dims, format is NCHW

    matrix: dict, shape and dtype of right matrix, only support float16,
    shape is 4 dims, format is NCHW

    bias: dict, shape and dtype of bias, only support float16,
    shape is 4 dims, format is NCHW, only use bias in conv2d

    y : output dict, shape and dtype of output_data, only support float16

    window : list or tuple, the window of pooling, only support
    pooling in H or W

    stride : list or tuple, the stride of pooling window, only support pooling
    in H or W

    offset_x : avg quantize parmas

    mode : int, the mode of pooling, support 0:max pooling, 1:avg pooling.

    pad : list or tuple, the pad of pooling, only support pooling in H or W

    global_pooling : global pooling params.

    ceil_mode : caffe round_mode params, 0:CEIL, 1:FLOOR, default value is
    DOMI_POOLING_CEIL

    dilation : reserved.

    kernel_name : cce kernel name, default value is "pooling_cce"

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
    pooling_check_rule(input_shape, output_dtype, window, stride, kernel_name)

    input_h = input_shape[2]
    input_w = input_shape[3]

    # convert mode&pad_mode to str for pooling2d
    pad = list(pad)
    if mode == 0:
        modes = "MAX"
        if (input_h == window[0] and input_w == window[1] and
            pad == [0, 0, 0, 0]) or \
                global_pooling:
            modes = "GMP"
    elif mode == 1:
        modes = "AVG"
        if (input_h == window[0] and input_w == window[1] and
            pad == [0, 0, 0, 0]) or \
                global_pooling:
            modes = "GAP"
    else:
        raise RuntimeError("Invalid mode parameters, mode must set 0 or 1.")

    # avg pooling calls conv2d interface to implement
    if modes == "AVG" and matrix:
        # input origin shape should be set to [N*C1, C0, H, W]
        input_ori_shape = (input_shape[0] * input_shape[1], input_shape[4],
                           input_shape[2], input_shape[3])
        x["ori_shape"] = input_ori_shape

        # conv2d interface strides is 4D
        strides = (1, 1, stride[0], stride[1])
        # get real pad
        _, _, pad_top, pad_bottom, pad_left, pad_right \
            = get_caffe_out_size_and_pad(ceil_mode, input_h, input_w,
                                         window[0], window[1], stride[0],
                                         stride[1], dilation[0], dilation[1],
                                         pad[0], pad[1], pad[2],
                                         pad[3])
        pad = (pad_top, pad_bottom, pad_left, pad_right)

        impl.conv2d(x, matrix, None, None, y, strides, pad,
                    dilations=(1, 1, 1, 1), kernel_name=kernel_name)
    else:
        # set tensor attrs
        addr_type = x.get("addr_type", 0)
        valid_shape = x.get("valid_shape", [])
        slice_offset = x.get("slice_offset", [])
        l1_fusion_type = x.get("L1_fusion_type", -1)
        attr = {"addr_type": addr_type,
                "valid_shape": valid_shape,
                "slice_offset": slice_offset,
                "L1_fusion_type": l1_fusion_type}
        is_l1fusion = l1_fusion_type in (0, 1)

        tensor_in = tvm.placeholder(input_shape, name="tensor_in",
                                    dtype=input_dtype, attrs=attr)
        res = pooling_compute(tensor_in, matrix, y, window, stride, mode, pad,
                              global_pooling, ceil_mode, kernel_name)
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
