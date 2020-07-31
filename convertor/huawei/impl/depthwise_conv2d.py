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

depthwise_conv2d
"""

from te import platform as tbe_platform
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util

# cube min cell 16

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 5

# shape's dim of filter must be 4
FILTER_DIM = 6

# shape's dim of strides must be 2
STRIDES_DIM = 4

#General limitation of the size for input shape
SHAPE_SIZE_LIMIT = 1 << 30

NONETYPE = type(None)


# pylint: disable=locally-disabled, too-many-locals, too-many-arguments,
# pylint: disable=unused-argument
# pylint: disable=locally-disabled, bad-continuation, import-error
# pylint: disable=too-many-statements, redefined-builtin, invalid-name
@fusion_manager.register("depthwise_conv2d")
def depthwise_compute(fmap, filter, bias, offset_w, out,
                      strides, dilations, pads, \
                      data_format='NHWC', offset_x=0, dsl_flag=True,\
                      kernel_name="depthwise_conv2d"):
    """
    algorithm: depthwise conv2d compute
    calculating  depthwise compute
    Parameters
    ----------
    fmap : a tensor of featureMap
    filter : a tensor of filter
    bias : a tensor of bias
    offset_w : a tensor of filter offset
    out : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.
    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]
    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]
    pads : padding added to each dimension of the input
    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]
    offset_x : offset of the input
    Returns
    -------
    None
    """
    out_dtype = out.get("dtype")
    DIM_H, DIM_W = 2, 3
    if data_format == 'NHWC':
        DIM_H, DIM_W = 1, 2

    strides_2d = strides[DIM_H], strides[DIM_W]
    dilations_2d = dilations[DIM_H], dilations[DIM_W]

    out = te.lang.cce.te_compute.depthwise_conv2d_compute(
        fmap, filter, out_dtype.lower(), strides_2d, pads, dilations_2d, {
            "bias_tensor": bias,
            "dsl_flag": dsl_flag,
            "offset_x": offset_x
        })
    return out


@util.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                       (list, tuple), (list, tuple), (list, tuple), str, int,
                       str)
def depthwise_conv2d(
        x,
        filter,
        bias,
        offset_w,
        y,
        strides,
        dilations=(1, 1, 1, 1),
        pads=(0, 0, 0, 0),
        data_format='NHWC',
        offset_x=0,
        kernel_name="depthwise_conv2d",
):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution

    Parameters
    ----------
    x : a dict of featureMap
        {"shape", "dtype", "format"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    filter : a dict of filter
        {"shape", "dtype"}
        shape of filter tensor [C1, H, W, K, Co, C0],
        K is depthwise_multiplier, support int.

    bias : a dict of bias
        {"shape", "dtype"}
        shape of bias tensor [C1*C0,]
        support int8.

    offset_w : a dict of filter offset
        {"shape", "dtype"}
        shape of offset tensor [C1, H, W, K, Co, C0]
        support float16.

    y : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]

    pads : padding added to each dimension of the input

    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    offset_x : offset of the input

    kernel_name : str
       cce kernel name

    Returns
    -------
    None

    """
    shape_w = filter.get("shape")
    shape_in = x.get("shape")
    output_dtype = y.get("dtype")
    in_dtype = x.get("dtype")
    w_dtype = filter.get("dtype")
    fmap_data_format = x.get("format")

    util.check_kernel_name(kernel_name)
    util.check_dtype_rule(in_dtype.lower(), ['float16', 'int8'])
    util.check_dtype_rule(w_dtype.lower(), ['float16', 'int8'])
    util.check_dtype_rule(output_dtype.lower(), ['float16', 'int32'])
    util.check_shape_size(shape_in, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_w, SHAPE_SIZE_LIMIT)
    util.check_shape_rule(shape_in, FEATURE_MAP_DIM, FEATURE_MAP_DIM)
    util.check_shape_rule(shape_w, FILTER_DIM, FILTER_DIM)
    util.check_shape_rule(strides, STRIDES_DIM, STRIDES_DIM)

    if fmap_data_format != "NC1HWC0":
        raise RuntimeError("only supported 5HD")

    def _check_shape(fmap_shape, filter_shape):
        """check input shape"""
        _, in_c1, _, _, _ = fmap_shape
        filter_c1, _, _, filter_k, _, _ = filter_shape

        # check feature map API feature map  shape is 5hd
        # The shape of feature map and filter must be 5HD
        if len(fmap_shape) != FEATURE_MAP_DIM:
            raise RuntimeError("The shape of feature map and filter must"
                               " be 5HD, tensorflow format!")

        # check feature map shape of c, equal filter of c
        if in_c1 != filter_c1:
            raise RuntimeError(
                "The input shape in feature map or filter is invalid!")

        # check multiplier equal 1
        if filter_k != 1:
            raise RuntimeError("The depthwise_multiplier of filter must be 1!")

    # fmap shape reshape, c ceil 16, 6d shape;
    # c must be 16x, if data not 16x, framework reshape c 16x
    in_n, in_c1, in_h, in_w, in_c0 = shape_in
    fmap_shape_5d = in_n, in_c1, in_h, in_w, in_c0
    shape_w_5d = shape_w[0], shape_w[1], shape_w[2], shape_w[4], shape_w[5]

    #filter shape: C1HWNCoC0
    filter_c1, filter_h, filter_w, _, _, _ = shape_w

    if data_format != 'NCHW' and data_format != 'NHWC':
        raise RuntimeError("The format of input in depthwise_conv2d only "
                           "supported NCHW and NHWC.")

    _check_shape(shape_in, shape_w)

    DIM_N, DIM_C, DIM_H, DIM_W = 0, 1, 2, 3  # NCHW
    if data_format == 'NHWC':
        DIM_N, DIM_H, DIM_W, DIM_C = 0, 1, 2, 3

# check strides is list, strides[0] ==shape_in[1]
# strides list, and h w value equal
    if not isinstance(strides, (list, tuple)) and len(strides) == 4:
        raise RuntimeError("depthwise_conv2d only allow strides of int,"
                           " or a list/tuple of four ints")

    if strides[DIM_N] != 1 or strides[DIM_C] != 1:
        raise RuntimeError("stride only support 1 in N axis and C axis.")
    if strides[DIM_H] != strides[DIM_W]:
        raise RuntimeError("only supports equal strides")
    if dilations[DIM_N] != 1 or dilations[DIM_C] != 1:
        raise RuntimeError("dilation only support 1 in N axis and C axis.")
    if dilations[DIM_H] != dilations[DIM_W]:
        raise RuntimeError("only supports equal dilations")

    # check pad parameter
    if len(pads) != 4:
        raise RuntimeError("pads shape should be 4d.")

    strides_2d = strides[DIM_H], strides[DIM_W]
    dilations_2d = dilations[DIM_H], dilations[DIM_W]
    bias_tensor = None
    if bias is not None and bias != {}:
        bias_tensor = tvm.placeholder((filter_c1 * 16, ),
                                      name='bias_tensor',
                                      dtype=output_dtype.lower())
    fmap_placeholder = tvm.placeholder(fmap_shape_5d,
                                       dtype=in_dtype.lower(),
                                       name='fmap')
    filter_placeholder = tvm.placeholder(shape_w_5d,
                                         dtype=w_dtype.lower(),
                                         name='filter')
    dsl_flag = False
    out = te.lang.cce.te_compute.depthwise_conv2d_compute(
        fmap_placeholder, filter_placeholder, output_dtype.lower(), strides_2d,
        pads, dilations_2d, {
            "bias_tensor": bias_tensor,
            "dsl_flag": dsl_flag,
            "offset_x": offset_x
        })

    tensor_list = [fmap_placeholder, filter_placeholder, out]
    if bias_tensor is not None:
        tensor_list = [fmap_placeholder, filter_placeholder, bias_tensor, out]

    with tvm.target.cce():
        sch = generic.auto_schedule(out)

    with tbe_platform.build_config:
        tvm.build(sch, tensor_list, "cce", name=kernel_name)
