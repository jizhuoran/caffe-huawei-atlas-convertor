#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd.

conv2d_backprop_input
"""

from __future__ import absolute_import
import te.lang.cce
from te import tvm
from topi import generic
from te.platform import CUBE_MKN
from te.platform.fusion_manager import fusion_manager
from topi.cce import util
import impl.util.util_deconv_comm as comm


# the dim of shape in conv2d_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4
# the dim of strides in conv2d_backprop must be 2
STRIDES_SHAPE_DIM = 2
# the dim of pads in conv2d_backprop must be 4
PADDING_SHAPE_DIM = 4

# fmap_H, fmap_W must be in [2,4096]
FMAP_HW_MIN = 2
FMAP_HW_MAX = 4096

# DeDy_H,DeDy_W must be in [2,4096]
DEDY_HW_MIN = 2
DEDY_HW_MAX = 4096

# filter_H, filter_W must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

# stride must be in [1,63] and h*w can't larger than 256
STRIDE_HW_MIN = 1
STRIDE_HW_MAX = 63
STRIDE_SIZE_MAX = 256

# pads must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# each axis of shape must less than 1000000
DEFAULT_MAX_SHAPE_NUM = 1000000

# the bytes length of several dtypes
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
# same as (2**63-1)
DATA_SIZE_MAX = 9223372036854775807

# If pads is string , only support "SAME" or "VALID"
PADDING_SUPPORT = ('SAME', 'VALID')
# pads valid mode is [0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0]


@util.check_input_type(dict, dict, dict, (tuple, list), (tuple, list),
                       (str, tuple, list), (tuple, list), int, str, str)
def conv2d_backprop_input_d(filter,  # pylint: disable=W0622,C0103,R0913,R0914
                            out_backprop, y, input_size, strides,
                            pads, dilations=(1, 1, 1, 1),
                            groups=None, data_format="NHWC",
                            kernel_name="conv2d_backprop_input"):
    """
    algorithm: conv2d_backprop_input

    Parameters
    ----------
    filter: dict with keys(shape and dtype)
            input weight tensor

    out_backprop: dict with keys(shape and dtype)
                  The shape of gradients.

    y: dict with keys(shape and dtype)
       conv2d_backprop_input output tensor, dtype must be assigned

    input_size: The shape of feature map.
                 4-D with shape [batch, channels, height, weight].

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_backprop_input
    groups: int
            param for group conv2d_backprop_input

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "conv2d_backprop_input"

    Returns
    -------
    None
    """

    ori_shape_filters = filter.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = y.get("ori_shape")

    filters_dtype = filter.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y.get("dtype")

    ori_format_filters = filter.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")
    ori_format_res = y.get("ori_format")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(ori_shape_filters,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(ori_shape_out_backprop,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(input_size,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(ori_shape_res,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(dilations,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)

    if len(strides) == 4:
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[h_index], strides[w_index]]

    shape_filters = comm.get_filter_shape(
        ori_format_filters, ori_shape_filters
    )

    shape_out_backprop = comm.get_shape_out_backprop(
        ori_format_out_backprop, ori_shape_out_backprop)

    shape_res = comm.get_shape_res(ori_format_res, ori_shape_res)

    dilations = comm.get_shape_dilation(ori_format_out_backprop, dilations)

    conv2d_backprop_input_cce(shape_filters,
                              shape_out_backprop,
                              shape_res,
                              strides,
                              pads,
                              dilations,
                              filters_dtype,
                              out_backprop_dtype,
                              res_dtype,
                              kernel_name)


@fusion_manager.register("conv2d_backprop_input_d")
def conv2d_backprop_input_d_compute(filter, out_backprop, y, input_size,
                                    strides, pads, dilations=(1, 1, 1, 1),
                                    groups=None, data_format="NHWC",
                                    kernel_name="conv2d_backprop_input"):
    """
    used for fusion
    Parameters
    ----------
    filter: Tensor
            input weight tensor

    out_backprop: Tensor
                  conv2d output gradients tenosr.

    y: dict with keys(shape and dtype)
       conv2d_backprop_input output tensor, dtype must be assigned

    input_size: The shape of feature map.
                 4-D with shape [batch, channels, height, weight].

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_backprop_input
    groups: int
            param for group conv2d_backprop_input
    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.
    kernel_name: str
                 kernel name, default value is "conv2d_backprop_input"

    Returns
    -------
    Tensor of conv2d_backprop_input
    """

    ori_shape_filters = [i.value for i in filter.op.attrs["ori_shape"]]
    ori_shape_out_backprop = \
        [i.value for i in out_backprop.op.attrs["ori_shape"]]
    ori_shape_res = [i for i in y["ori_shape"]]

    filters_dtype = filter.dtype
    out_backprop_dtype = out_backprop.dtype
    res_dtype = y["dtype"]

    ori_format_filters = filter.op.attrs["ori_format"]
    ori_format_out_backprop = out_backprop.op.attrs["ori_format"]
    ori_format_res = y["ori_format"]

    if len(strides) == 4:
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[h_index], strides[w_index]]

    shape_filters = comm.get_filter_shape(
        ori_format_filters, ori_shape_filters)
    shape_out_backprop = comm.get_shape_out_backprop(
        ori_format_out_backprop, ori_shape_out_backprop)
    shape_res = comm.get_shape_res(ori_format_res, ori_shape_res)
    dilations = comm.get_shape_dilation(ori_format_out_backprop, dilations)

    comm.check_conv2dbp_input_params(shape_filters, shape_out_backprop,
                                     shape_res, strides, pads, dilations,
                                     filters_dtype, out_backprop_dtype,
                                     res_dtype, kernel_name)

    pads = comm.get_padlist(pads, shape_res, strides, shape_filters, dilations)

    res = te.lang.cce.conv2d_backprop_input_compute(filter,
                                                    out_backprop,
                                                    shape_filters,
                                                    shape_res,
                                                    strides,
                                                    pads,
                                                    dilations,
                                                    res_dtype=res_dtype)
    return res


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple),
                       str,
                       str,
                       str,
                       str)
def conv2d_backprop_input_cce(shape_filter, shape_out_backprop, input_sizes,
                              strides, pads, dilations=(1, 1, 1, 1),
                              filter_dtype='float16',
                              out_backprop_dtype='float16',
                              res_dtype='float16',
                              kernel_name="conv2d_backprop_input_cce"):
    """
    Topi interface of conv2d backprop input

    Parameters:
    ----------
    shape_filter : The shape of filter.
                   4-D with shape [batch, channels, height, weight].

    shape_out_backprop : The shape of gradients.
                         4-D with shape [batch, channels, height, weight].

    input_sizes : The shape of feature map.
                  4-D with shape [batch, channels, height, weight].

    strides : A list of ints. The stride of the sliding window.

    pads : "SAME"or"VALID" indicating the type of pads algorithm to use,
           or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    filter_dtype : The dtype of filter data. Default value is float16.

    out_backprop_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    kernel_name : Cce kernel name. Default value is "conv2d_backprop_input_cce"

    Returns : None
    ----------
    """

    def _ceil(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2

    res = comm.check_conv2dbp_input_params(shape_filter, shape_out_backprop,
                                           input_sizes, strides, pads,
                                           dilations, filter_dtype,
                                           out_backprop_dtype, res_dtype,
                                           kernel_name)
    shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations, \
    filter_dtype, out_backprop_dtype, res_dtype, kernel_name = res

    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_channel, filter_h, filter_w = shape_filter

    _, dy_k0, _ = CUBE_MKN[out_backprop_dtype]['mac']
    _, w_k0, w_n0 = CUBE_MKN[filter_dtype]['mac']
    shape_dedy = (dedy_batch,
                  _ceil(dedy_channel, dy_k0), dedy_h, dedy_w, dy_k0)
    if filter_dtype == "int8" and out_backprop_dtype == "int8":
        filter_channel = comm.align(filter_channel, w_n0)
        shape_filter_frac = (
            _ceil(filter_batch, w_k0)*filter_h*filter_w,
            _ceil(filter_channel, w_n0), w_n0, w_k0)
    else:
        shape_filter_frac = (
            _ceil(filter_channel, w_n0)*filter_h*filter_w,
            _ceil(filter_batch, w_k0), w_k0, w_n0)
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)

    filter_frac = tvm.placeholder(shape_filter_frac,
                                  name="filter", dtype=filter_dtype)

    dedx = te.lang.cce.conv2d_backprop_input_compute(
        filters=filter_frac,
        out_backprop=dedy,
        filter_sizes=shape_filter,
        input_sizes=input_sizes,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype
    )
    tensor_list = [filter_frac, dedy, dedx]

    with tvm.target.cce():
        sch = generic.auto_schedule(dedx)

    config = {
        "name": kernel_name,
        "tensor_list": tensor_list
    }

    te.lang.cce.cce_build_code(sch, config)
