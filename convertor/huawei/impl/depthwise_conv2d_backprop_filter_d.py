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

depthwise conv2d backprop filter
"""
from te import platform as tbe_platform
from te import tvm
import te.lang.cce
from te.platform.cce_build import build_config
from topi.cce import util

BLOCK_SIZE = tbe_platform.cce_params.BLOCK_REDUCE

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 4

# shape's dim of filter must be 4
FILTER_DIM = 4

# shape's dim of strides must be 4
STRIDES_DIM = 4

# shape's dim of dilation must be 4
DILATION_DIM = 4

#General limitation of the size for input shape
SHAPE_SIZE_LIMIT = 1 << 30

# pylint: disable=locally-disabled, too-many-locals, bad-continuation
# pylint: disable=locally-disabled, too-many-arguments, invalid-name
# pylint: disable=too-many-statements, redefined-builtin, too-many-branches
@util.check_input_type((dict), (dict), (dict), (list, tuple), (list, tuple),
                       (list, tuple), (list, tuple), str, str)
def depthwise_conv2d_backprop_filter_d(
        input_fm,
        out_backprop,
        filter_grad,
        filter_size,
        strides,
        dilations=(1, 1, 1, 1),
        pads=(0, 0, 0, 0),
        data_format='NHWC',
        kernel_name="depthwise_conv2d_backprop_filter"):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution backward filter

    Parameters
    ----------
    input_fm : a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    out_backprop: a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    filter_grad : a dict.
        4-D origin shape of filter tensor [H, W, C, K],
        K is depthwise_multiplier, support float32.

    filter_size : a list/tuple of four ints
        1-D origin shape of filter tensor with [H, W, C, K],
        K is depthwise_multiplier, support int.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1].

    dilations : a list/tuple of four ints
        dilations size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1].

    pads : a list/tuple of four ints
        padding added to each dimension of the input.

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C].

    kernel_name : str
        cce kernel name

    Returns
    -------
    None

    """
    def _ceil(x):
        """
        Return the least multiple of 16 integer number
        which is greater than or equal to x.
        """
        return ((x + BLOCK_SIZE - 1) // BLOCK_SIZE)*BLOCK_SIZE

    def _check_shape(fmap_shape, dout_shape, filter_shape):
        """Check input shape."""
        fmap_n, fmap_c, _, _ = fmap_shape
        dout_n, dout_c, _, _ = dout_shape
        _, _, filter_c, filter_n = filter_shape

        if filter_n != 1:
            raise RuntimeError("The depthwise_multiplier of filter must be 1!")
        if fmap_c != dout_c or fmap_n != dout_n:
            raise RuntimeError(
                "The input shape(N or C) in feature map or dout is not equal!")
        if fmap_c != filter_c:
            raise RuntimeError(
                "The input shape(C) in feature map or dfilter is not equal!")

    shape_in = input_fm.get('ori_shape')
    shape_w = filter_size
    shape_dout = out_backprop.get('ori_shape')
    in_dtype = input_fm.get('dtype')
    w_dtype = filter_grad.get('dtype')
    dout_dtype = out_backprop.get('dtype')

    if data_format != 'NCHW' and data_format != 'NHWC':
        raise RuntimeError(
            "The format of input_fm in depthwise_conv2d_backprop_filter only "
            "supported NCHW and NHWC.")
    if shape_w != filter_grad.get('ori_shape'):
        raise RuntimeError(
            "The output shape of depthwise_conv2d_backprop_filter must be"
            " same with filter_size.")
    input_ori_format = input_fm.get('ori_format')
    if input_ori_format not in ('NCHW', 'NHWC'):
        raise RuntimeError(
            "The format of input_fm in depthwise_conv2d_backprop_filter only "
            "supported NCHW and NHWC, which current format is ",
            input_ori_format)
    dout_ori_format = out_backprop.get('ori_format')
    if dout_ori_format not in ('NCHW', 'NHWC'):
        raise RuntimeError(
            "The format of out_backprop in depthwise_conv2d_backprop_filter "
            "only supported NCHW and NHWC, which current format is ",
            dout_ori_format)
    filter_grad_ori_format = filter_grad.get('ori_format')
    if filter_grad_ori_format not in ('HWCK', 'HWCN', 'NCHW'):
        raise RuntimeError(
            "The format of filter_grad in depthwise_conv2d_backprop_filter "
            "only supported HWCK(HWCN)/NCHW, which current format is ",
            filter_grad_ori_format)
    if filter_grad_ori_format in ('NCHW',):
        # NCHW to HWCK(HWCN)
        shape_w = (shape_w[2], shape_w[3], shape_w[1], shape_w[0])

    util.check_dtype_rule(in_dtype.lower(), ['float16'])
    util.check_dtype_rule(dout_dtype.lower(), ['float16'])
    util.check_dtype_rule(w_dtype.lower(), ['float32'])

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_in, FEATURE_MAP_DIM, FEATURE_MAP_DIM)
    util.check_shape_rule(shape_w, FILTER_DIM, FILTER_DIM)
    util.check_shape_rule(shape_dout, FEATURE_MAP_DIM, FEATURE_MAP_DIM)
    util.check_shape_rule(strides, STRIDES_DIM, STRIDES_DIM)
    util.check_shape_rule(dilations, DILATION_DIM, DILATION_DIM)
    util.check_shape_size(shape_in, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_w, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_dout, SHAPE_SIZE_LIMIT)
    # index of the origin dimension
    DIM_N, DIM_C, DIM_H, DIM_W = 0, 1, 2, 3  # NCHW
    if input_ori_format == 'NHWC':
        DIM_N, DIM_H, DIM_W, DIM_C = 0, 1, 2, 3
        shape_in = [
            shape_in[DIM_N], shape_in[DIM_C], shape_in[DIM_H], shape_in[DIM_W]
        ]
    if dout_ori_format == 'NHWC':
        shape_dout = [
            shape_dout[0], shape_dout[3], shape_dout[1], shape_dout[2]
        ]

    _check_shape(shape_in, shape_dout, shape_w)

    if dilations[DIM_N] != 1 or dilations[DIM_C] != 1:
        raise RuntimeError("dilation only support 1 in N axis and C axis.")
    if dilations[DIM_H] != 1 or dilations[DIM_W] != 1:
        raise RuntimeError("dilation only support 1 in H axis and W axis.")
    if strides[DIM_N] != 1 or strides[DIM_C] != 1:
        raise RuntimeError("Stride only support 1 in N axis and C axis.")
    if strides[DIM_H] != strides[DIM_W]:
        raise RuntimeError(
            "current implementation only supports equal length strides "
            "in the row and column dimensions.")

    # check pad parameter
    if len(pads) != 4:
        raise RuntimeError("pads shape should be 4d.")

    n, c, h, w = shape_in
    shape_in = [n, _ceil(c) // BLOCK_SIZE, 1, h, w, BLOCK_SIZE]
    fmap_placeholder = tvm.placeholder(shape_in,
                                       dtype=in_dtype.lower(),
                                       name='fmap')
    n, c, h, w = shape_dout
    shape_dout = [n, _ceil(c) // BLOCK_SIZE, 1, h, w, BLOCK_SIZE]
    dout_placeholder = tvm.placeholder(shape_dout,
                                       dtype=dout_dtype.lower(),
                                       name='dout')

    h, w, _, _ = shape_w
    res = te.lang.cce.depthwise_conv2d_backprop_filter_d_compute(
        fmap_placeholder, dout_placeholder, h, w,
        (strides[DIM_H], strides[DIM_W]), pads,
        (dilations[DIM_H], dilations[DIM_W]), w_dtype.lower())
    s = te.lang.cce.te_schedule.depthwise_conv2d_backprop_filter_d_schedule(res)

    with build_config:
        tvm.build(s, [fmap_placeholder, dout_placeholder, res],
                  "cce",
                  name=kernel_name)
