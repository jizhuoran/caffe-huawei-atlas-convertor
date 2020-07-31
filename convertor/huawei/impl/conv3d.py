#!/usr/bin/env python
# -*- coding:utf-8 -*-
# pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-lines
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

conv3d
"""
from __future__ import absolute_import
from te.platform import CUBE_MKN
import te.lang.cce
from te.lang.cce.te_compute import conv3d_compute
from te import tvm
from topi import generic
from topi.cce import util

Nonetype = type(None)
# [strides_batch, strides_depth, strides_height,
#  strides_width, strides_channel]
STRIDE_LENGTH = 5
#
DILATION_LENGTH = 5
# [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
PADS_LENGTH = 6
# NDHWC or NCDHW
SHAPE_DIMS = 5

def _get_mad_dtype(w_dtype):
    """
    algorithm: get the dtype of mad

    Parameters
    ----------
    w_dtype: the dtype of filter

    Returns
    -------
    mad dtype
    """

    return 'int32' if w_dtype == 'int8' else 'float32'


def _conv3d_compute(shape_fm, shape_filter, bias, stride_dhw, pads, fmp_dtype,
                    w_dtype, res_dtype):
    """
    algorithm: compute conv3d

    Parameters
    ----------
    shape_fm: the shape of feature,
        a list/tuple of 'int' that has length `== 5`

    shape_filter: the shape of filter, a list of 'int' that has length `== 5`

    bias: dict with keys(shape and dtype) or None
        input bias tensor

    stride_dhw: A list of `ints` that has length `== 3`.

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    res_dtype: the dtype of output

    Returns
    -------
    list of tensor
    """
    batch, cin, fmp_d, fmp_h, fmp_w = shape_fm
    fmp_block_k = CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fmp_ndc1hwc0 = (batch, fmp_d, cin // fmp_block_k, fmp_h, fmp_w,
                          fmp_block_k)

    cout, cin, w_d, w_h, w_w = shape_filter
    w_block_k = CUBE_MKN[w_dtype]['mac'][1]
    w_block_n = CUBE_MKN[w_dtype]['mac'][2]
    shape_w_frac_z = (w_d * cin * w_h * w_w // w_block_k, cout // w_block_n,
                      w_block_n, w_block_k)

    mad_dtype = _get_mad_dtype(w_dtype)

    data = tvm.placeholder(shape_fmp_ndc1hwc0, name='Fmap', dtype=fmp_dtype)
    weight = tvm.placeholder(shape_w_frac_z, name='Filter', dtype=w_dtype)
    bias_tensor = None
    if bias is not None:
        bias_tensor = tvm.placeholder((cout, ),
                                      name='bias_tensor',
                                      dtype=res_dtype)
    conv3d_dict = {
        "bias_tensor": bias_tensor,
        "pads": pads,
        "shape_filter_ncdhw": shape_filter,
        "stride_dhw": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype
    }
    conv_res = te.lang.cce.te_compute.conv3d(data, weight, conv3d_dict)
    if bias is not None:
        tensor_list = [data, weight, bias_tensor, conv_res]
    else:
        tensor_list = [data, weight, conv_res]

    return tensor_list

def check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    res_dtype: the dtype of output

    Returns
    -------
    None
    """

    util.check_dtype_rule(fmp_dtype, ('float16', ))
    util.check_dtype_rule(w_dtype, ('float16', ))
    util.check_dtype_rule(res_dtype, ('float16', ))


def format_normalize(fmp_format, w_format, fmp_shape, w_shape, strides,
                     dilations):
    """
    algorithm: unified format

    Parameters
    ----------
    fmp_format: The data format of the input feature.

    w_format: The data format of the input filter.

    fmp_shape: the shape of feature,
        a list/tuple of 'int' that has length `== 5`

    w_shape: the shape of filter, a list of 'int' that has length `== 5`

    strides: A list of `ints` that has length `== 5`.

    dilations: tuple/list of 5 integers.
        dilation on D/H/W, format sensitive,
        Dilations in the batch and depth dimensions must be 1.

    Returns
    -------
    shape_fm, shape_filter, stride_dhw, dilation_hw
    """
    if fmp_format == "NCDHW":
        shape_fm = list(fmp_shape)
        stride_dhw = strides[2:]
        dilation_hw = dilations[3:]
    elif fmp_format == "NDHWC":
        shape_fm = [
            fmp_shape[0], fmp_shape[4], fmp_shape[1], fmp_shape[2],
            fmp_shape[3]
        ]
        stride_dhw = strides[1:4]
        dilation_hw = dilations[2:4]
    else:
        raise RuntimeError("inputs format should be NCDHW or NDHWC.")

    if w_format == "NCDHW":
        shape_filter = list(w_shape)
    elif w_format == "NDHWC":
        shape_filter = [
            w_shape[0], w_shape[4], w_shape[1], w_shape[2], w_shape[3]
        ]
    elif w_format == "DHWCN":
        shape_filter = [
            w_shape[4], w_shape[3], w_shape[0], w_shape[1], w_shape[2]
        ]
    else:
        raise RuntimeError("weights format should be NCDHW or NDHWC or DHWCN.")

    return shape_fm, shape_filter, stride_dhw, dilation_hw


def check_input_param(fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype,
                      fmp_format, w_format, strides, pads, dilations):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------
    fmp_shape: the shape of feature,
        a list/tuple of 'int' that has length `== 5`

    w_shape: the shape of filter, a list of 'int' that has length `== 5`

    bias: dict with keys(shape and dtype) or None
        input bias tensor

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    res_dtype: the dtype of output

    fmp_format: The data format of the input feature.

    w_format: The data format of the input filter.

    strides: A list of `ints` that has length `== 5`.

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers.
        dilation on D/H/W, format sensitive,
        Dilations in the batch and depth dimensions must be 1.

    Returns
    -------
    None
    """
    if len(strides) != STRIDE_LENGTH:
        raise RuntimeError("strides shape should be 5d.")
    if len(dilations) != DILATION_LENGTH:
        raise RuntimeError("dilations shape should be 5d.")
    # check dilations for it1
    if len(set(dilations)) != 1 or dilations[2] != 1:
        raise RuntimeError("dilations only support (1,1,1,1,1) for it1.")

    if len(pads) != PADS_LENGTH:
        raise RuntimeError("pads shape should be 6d.")

    util.check_shape_rule(fmp_shape, min_dim=SHAPE_DIMS, max_dim=SHAPE_DIMS)
    util.check_shape_rule(w_shape, min_dim=SHAPE_DIMS, max_dim=SHAPE_DIMS)

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, dilation_hw = format_normalize(
        fmp_format, w_format, fmp_shape, w_shape, strides, dilations)

    check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype)

    te.lang.cce.te_compute.check_conv3d_shape(shape_fm,
        shape_filter, pads, stride_dhw, fmp_dtype, w_dtype)

    return shape_fm, shape_filter, stride_dhw, dilation_hw


@util.check_input_type(dict, dict, (dict, Nonetype), dict, (tuple, list),
                       (tuple, list), str, (tuple, list), str)
def conv3d(fmap,
           weight,
           bias,
           output,
           strides,
           pads,
           data_format="NDHWC",
           dilations=(1, 1, 1, 1, 1),
           kernel_name="conv3d"):
    """
    algorithm: conv3d

    Parameters
    ----------
    fmap: dict with keys(shape and dtype)
        input 5d feature map tensor

    weight: dict with keys(shape and dtype)
        input 5d weight tensor

    bias: dict with keys(shape and dtype) or None
        input bias tensor

    output: dict with keys(shape and dtype)
        output tensor, dtype must be assigned

    strides: tuple/list of 5 integers, format sensitive
        [strides_batch, strides_depth, strides_height,
         strides_width, strides_channel]

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    data_format: The data format of the input and output data. With the
        default format "NDHWC",

    dilations: tuple/list of 5 integers.
        dilation on D/H/W, format sensitive,
        Dilations in the batch and depth dimensions must be 1.

    kernel_name: str
        kernel name, default value is "conv3d"

    Returns
    -------
    None
    """
    fmp_shape = fmap.get("ori_shape")
    fmp_dtype = fmap.get("dtype")
    fmp_format = data_format
    w_shape = weight.get("ori_shape")
    w_dtype = weight.get("dtype")
    w_format = weight.get("ori_format")
    res_dtype = output.get("dtype")

    fmp_dtype = fmp_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, _ = \
        check_input_param(fmp_shape, w_shape, fmp_dtype, w_dtype,
                          res_dtype, fmp_format, w_format, strides,
                          pads, dilations)

    pads = list(pads)
    stride_dhw = list(stride_dhw)

    # C and Cout align 16
    shape_fm = list(shape_fm)
    fmp_block_k = CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fm[1] = (
        (shape_fm[1] + fmp_block_k - 1) // fmp_block_k) * fmp_block_k
    w_block_k = CUBE_MKN[w_dtype]['mac'][1]
    shape_filter = list(shape_filter)
    shape_filter[1] = (
        (shape_filter[1] + w_block_k - 1) // w_block_k) * w_block_k
    w_block_n = CUBE_MKN[w_dtype]['mac'][2]
    shape_filter[0] = (
        (shape_filter[0] + w_block_n - 1) // w_block_n) * w_block_n

    tensor_list = _conv3d_compute(shape_fm, shape_filter, bias, stride_dhw,
                                  pads, fmp_dtype, w_dtype, res_dtype)

    with tvm.target.cce():
        sch = generic.auto_schedule(tensor_list[-1])

    config = {
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    te.lang.cce.cce_build_code(sch, config)
