#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

conv3d_backprop_input
"""

from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform import cce_params
from topi import generic
from topi.cce import util

# the dim of shape in conv_backprop must be 5
CONV_BACKPROP_SHAPE_DIM = 5
# the dim of pads in conv3d_backprop must be 6
CONV_BACKPROP_PAD_SHAPE_DIM = 6
# the dim of strides in conv_backprop must be 3
STRIDES_SHAPE_DIM = 5

# fmapH, fmapW must be in [2,4096]
FMAP_HW_MIN = 2
FMAP_HW_MAX = 4096

# DeDy H,W must be in [2,4096]
DEDY_HW_MIN = 2
DEDY_HW_MAX = 4096

# filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255
FILTER_HW_SIZE = 256

# stride must be in [1,63] and h*w not lagger than 256
STRIDE_HW_MIN = 1
STRIDE_HW_MAX = 63
STRIDE_SIZE_MAX = 256
STRIDE_SIZE_HWD_MAX = 343

# pad must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# special num
STRID_D_COEFF = 126
KHW_COEFF = 2048
HW_COEEF = 1024
KHWD_COEFF = 343

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
# np.int64(2**63-1)
DATA_SIZE_MAX = 9223372036854775807
# pads valid mode to be [0, 0, 0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0, 0, 0]

@util.check_input_type(dict, dict, dict, (tuple, list), (tuple, list),
                       (str, tuple, list), str, (tuple, list), str)
def conv3d_backprop_input_d(filters, # pylint: disable=R0913,R0914
                            out_backprop, y_input, input_sizes, strides,
                            pads, data_format="NDHWC",
                            dilations=(1, 1, 1, 1, 1),
                            kernel_name="conv3d_backprop_input"):
    """
    algorithm: conv3d_backprop_input

    Parameters
    ----------
    filters: dict with keys(shape and dtype)
            input weight tensor

    out_backprop: dict with keys(shape and dtype)
                  The shape of gradients.

    y_input: dict with keys(shape and dtype)
       conv3d_backprop_input output tensor, dtype must be assigned

    input_sizes: The shape of feature map.
                 5-D with shape [batch, depth, height, weight, channels].

    strides: tuple/list of 5 integers
             filter move stride

    pads: tuple/list of 6 integers
             [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers
               filter expand size of dilated conv3d_backprop_input

    kernel_name: str
                 kernel name, default value is "conv3d_backprop_input"

    Returns
    -------
    None
    :param data_format:
    """
    def _ncdhw2ndhwc(shape1):
        shape2 = (shape1[0], shape1[2], shape1[3], shape1[4], shape1[1])
        return shape2

    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_sizes
    ori_shape_strides = strides
    ori_shape_dialtions = dilations

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    if ori_format_filters == "DHWNC":
        shape_filters = ori_shape_filters
    elif ori_format_filters == "NDHWC":
        shape_filters = (ori_shape_filters[1],
                         ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[0],
                         ori_shape_filters[4],
                        )
    elif ori_format_filters == "NCDHW":
        shape_filters = (ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[0],
                         ori_shape_filters[1],
                        )
    else:
        raise RuntimeError("filters's format should be DHWNC, NDHWC or NCDHW.")

    if ori_format_out_backprop == "NDHWC":
        shape_out_backprop = ori_shape_out_backprop
        shape_strides = ori_shape_strides
        shape_dilations = ori_shape_dialtions
    elif ori_format_out_backprop == "NCDHW":
        shape_out_backprop = _ncdhw2ndhwc(ori_shape_out_backprop)
        shape_strides = _ncdhw2ndhwc(ori_shape_strides)
        shape_dilations = _ncdhw2ndhwc(ori_shape_dialtions)
    else:
        raise RuntimeError("out_backprop's format should be NDHWC or NCDHW.")

    if ori_format_res == "NDHWC":
        shape_res = ori_shape_res
    elif ori_format_res == "NCDHW":
        shape_res = _ncdhw2ndhwc(ori_shape_res)
    else:
        raise RuntimeError("y's format should be NDHWC or NCDHW.")

    conv3d_backprop_input_cce(shape_filters,
                              shape_out_backprop,
                              shape_res,
                              shape_strides,
                              pads,
                              shape_dilations,
                              filters_dtype,
                              out_backprop_dtype,
                              res_dtype,
                              kernel_name)


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple),
                       str,
                       str,
                       str,
                       str)
def check_conv3dbp_input_params(shape_filter,# pylint:disable=R0913,R0914,R0915
                                shape_out_backprop,
                                input_sizes, strides, pads, dilations,
                                filter_dtype, out_backprop_dtype,
                                res_dtype, kernel_name):
    """
    The params check function of conv3d backprop input

    Parameters:
    -------------------------
    shape_filter : The shape of filter.
                   5-D with shape (depth, height, weight, batch, channels)

    shape_out_backprop : The shape of gradients.
                         5-D with shape[batch, depth, height, weight,channels].

    input_sizes : The shape of feature map.
                  5-D with shape [batch, depth, height, weight, channels].

    strides : A list of ints. The stride of the sliding window.

    pads : A list of ints.

    dilations : An optional list of ints. Only support [1, 1, 1, 1, 1] now.

    filter_dtype : The dtype of filter data. Default value is float16.

    out_backprop_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    kernel_name : Cce kernel name.
                  Default value is "conv3d_backprop_intput_cce"

    Returns : All transformed params.


    """

    def _ceil(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2

    def _align(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2 * x_2

    def _check_attr_range(attr_name, attr_value, attr_min=None, attr_max=None):
        if not attr_min and not attr_max:
            return
        if not attr_min:
            if attr_value > attr_max:
                raise RuntimeError(
                    "The {} exceed max_value limitation.".format(attr_name))
        elif not attr_max:
            if attr_value < attr_min:
                raise RuntimeError("The attr value is less than min_value.")
        elif attr_value < attr_min or attr_value > attr_max:
            raise RuntimeError(
                "The {} value must be in [min, max].".format(attr_name))

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype is None:
            bit_ratio = BIT_RATIO_DICT.get("float16")
        else:
            bit_ratio = BIT_RATIO_DICT.get(dtype)
        if attr_value * bit_ratio > DATA_SIZE_MAX:
            raise RuntimeError(
                "The {} exceed 64 bit limitations!".format(attr_name))

    def _check_shape_error():
        fmap_h_padding = fmap_h + pad_up + pad_down
        fmap_w_padding = fmap_w + pad_left + pad_right
        fmap_d_padding = fmap_deep + pad_head + pad_tail

        if fmap_channel != filter_batch:
            raise RuntimeError(
                "Shape error: Fmap's C must be equal to Filter'N.")
        if dedy_channel != filter_channel:
            raise RuntimeError(
                "Shape error: Dedy's C must be equal to Filter'C.")
        if fmap_batch != dedy_batch:
            raise RuntimeError(
                "Shape error: Fmap's N must be equal to Dedy'N.")
        if filter_h_dilation > fmap_h_padding:
            raise RuntimeError(
                "Shape error: Filter'H(after dilation) "
                "must be less than Fmap'H(after padding).")
        if filter_w_dilation > fmap_w_padding:
            raise RuntimeError("Shape error: Filter'W(after dilation) "
                               "must be less than Fmap'W(after padding).")
        if filter_d_dilation > fmap_d_padding:
            raise RuntimeError("Shape error: Filter'D(after dilation) "
                               "must be less than Fmap'D(after padding).")
        if ((fmap_h - filter_h_dilation +
             pad_up + pad_down) // stride_h + 1) != dedy_h:
            raise RuntimeError(
                "Shape error : "
                "The compute result of H under convolution rule cannot match")
        if ((fmap_w - filter_w_dilation
             + pad_left + pad_right) // stride_w + 1) != dedy_w:
            raise RuntimeError(
                "Shape error : "
                "The compute result of W under convolution rule cannot match")
        if ((fmap_deep - filter_d_dilation
             + pad_head + pad_tail) // stride_d + 1) != dedy_deep:
            raise RuntimeError(
                "Shape error : "
                "The compute result of D under convolution rule cannot match")
    # Base check, Mainly required by interface appearance
    # ===========================================================
    # util check
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_filter,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(shape_out_backprop,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(input_sizes,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(strides, STRIDES_SHAPE_DIM, STRIDES_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)

    # pads check
    if isinstance(pads, (tuple, list)) and \
            len(pads) != CONV_BACKPROP_PAD_SHAPE_DIM:
        raise RuntimeError("pads should be 6d list.")

    # dilations check
    util.check_shape_rule(dilations, CONV_BACKPROP_SHAPE_DIM,
                          CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    dilation_n, dilation_d, dilation_h, dilation_w, dilation_c = dilations
    if dilation_n != 1 or dilation_c != 1:
        raise RuntimeError(
            "Dilations in the batch and depth dimensions must be 1")

    # detype chek
    filter_dtype = filter_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    util.check_dtype_rule(filter_dtype, ['float16'])
    util.check_dtype_rule(out_backprop_dtype, ['float16'])
    util.check_dtype_rule(res_dtype, ['float16'])

    # the relation limits between shape
    shape_filter = list(shape_filter)
    shape_out_backprop = list(shape_out_backprop)
    input_sizes = list(input_sizes)
    strides = list(strides)
    fmap_batch, fmap_deep, fmap_h, fmap_w, fmap_channel = input_sizes
    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, \
    filter_w, filter_batch, filter_channel = shape_filter
    _, stride_d, stride_h, stride_w, _ = strides

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    filter_d_dilation = (filter_depth - 1) * dilation_d + 1

    if pads == 'SAME':
        pad_h = _align(fmap_h, stride_h) - stride_h + filter_h - fmap_h
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = _align(fmap_w, stride_w) - stride_w + filter_w - fmap_w
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_d = _align(fmap_deep, stride_d)\
                - stride_d + filter_depth - fmap_deep
        pad_d = max(pad_d, 0)
        pad_head = pad_d // 2
        pad_tail = pad_d - pad_head

        pads = [pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = PADDING_VAILD
    # pads compute
    pads = list(pads)
    pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right = pads

    fmap_h_padding = fmap_h + pad_up + pad_down
    fmap_w_padding = fmap_w + pad_left + pad_right

    # special cases
    dey_hw_min, fmap_hw_min = DEDY_HW_MIN, FMAP_HW_MIN
    # limitation by chip:
    # if kernel h,w in [1,11] and fmap h/w after padding equals to filter h/w
    # load3d support h,w is 1
    if (1 <= filter_h <= 11) and (1 <= filter_w <= 11) \
            and (fmap_h_padding == filter_h or fmap_w_padding == filter_w):
        dey_hw_min = 1
        fmap_hw_min = 1
    _check_shape_error()
    # Dedy value limit
    _check_attr_range("Dedy's H after expands", dedy_h * stride_h,
                      dey_hw_min, DEDY_HW_MAX)
    _check_attr_range("Dedy's W after expands", dedy_w * stride_w,
                      dey_hw_min, DEDY_HW_MAX)

    # filter value limit
    _check_attr_range("filter's H", filter_h, FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range("filter's W", filter_w, FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range("filter's D", filter_depth,
                      FILTER_HW_MIN, STRID_D_COEFF*stride_d + 1)

    _check_attr_range("filter H*W", filter_h * filter_w,
                      FILTER_HW_MIN, FILTER_HW_SIZE)
    _check_attr_range("filter H*W*D",
                      filter_h * filter_w * (2 + _ceil((filter_depth - 1),
                                                       stride_d)),
                      FILTER_HW_MIN, KHW_COEFF)
    _check_attr_range("filter H*W*D",
                      filter_h * filter_w * filter_depth,
                      FILTER_HW_MIN, KHWD_COEFF)

    # Fmap value limit
    _check_attr_range("Fmap's H", fmap_h, fmap_hw_min, FMAP_HW_MAX)
    _check_attr_range("Fmap's W", fmap_w, fmap_hw_min, FMAP_HW_MAX)

    # stride value limit
    _check_attr_range("stride's H", stride_h, STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range("stride's W", stride_w, STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range("stride's H*W",
                      stride_h*stride_w, STRIDE_HW_MIN, STRIDE_SIZE_MAX)
    _check_attr_range(
        "stride's H*W*D",
        stride_h*stride_w*stride_d, STRIDE_HW_MIN, STRIDE_SIZE_HWD_MAX)

    # check shape size, 64 bits limitation
    # ===========================================================
    c0_size = cce_params.C0_SIZE
    fmap_size = fmap_batch * _align(fmap_channel, c0_size) \
                * fmap_deep * fmap_h * fmap_w
    dedy_size = dedy_batch * _align(dedy_channel, c0_size) \
                * dedy_deep * dedy_h * dedy_w
    filter_size = _align(filter_batch, c0_size) * \
    _align(filter_channel, c0_size) * filter_depth * filter_h * filter_w
    _check_64bits_limitation("fmap_size", fmap_size, dtype=res_dtype)
    _check_64bits_limitation("dedy_size", dedy_size, dtype=out_backprop_dtype)
    _check_64bits_limitation("filter_size", filter_size, dtype=filter_dtype)

    result = (shape_filter, shape_out_backprop, input_sizes, strides,
              pads, dilations, filter_dtype, out_backprop_dtype,
              res_dtype, kernel_name)
    return result


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple),
                       str,
                       str,
                       str,
                       str)
def conv3d_backprop_input_cce(shape_filter, # pylint: disable=R0913,R0914
                              shape_out_backprop, input_sizes,
                              strides, pads, dilations=(1, 1, 1, 1, 1),
                              filter_dtype='float16',
                              out_backprop_dtype='float16',
                              res_dtype='float16',
                              kernel_name="conv3d_backprop_input_cce"):
    """
    Topi interface of conv3d backprop input

    Parameters:
    ----------
    shape_filter : The shape of filter.
                   5-D with shape [ depth, height, weight, batch, channels].

    shape_out_backprop : The shape of gradients.
                         5-D with shape [batch,
                                         depth, height, weight, channels].

    input_sizes : The shape of feature map.
                  5-D with shape [batch, depth, height, weight, channels].

    strides : A list of ints. The stride of the sliding window.

    pads : A list of ints.

    dilations : An optional list of ints. Only support [1, 1, 1, 1, 1] now.

    filter_dtype : The dtype of filter data. Default value is float16.

    out_backprop_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    kernel_name : Cce kernel name. Default value is "conv3d_backprop_cce"

    Returns : None
    ----------
    """

    def _ceil(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2

    def _align(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2 * x_2

    res = check_conv3dbp_input_params(shape_filter, shape_out_backprop,
                                      input_sizes, strides, pads, dilations,
                                      filter_dtype, out_backprop_dtype,
                                      res_dtype, kernel_name)
    shape_filter, shape_out_backprop, input_sizes, strides, pads, dilations,\
    filter_dtype, out_backprop_dtype, res_dtype, kernel_name = res

    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w,\
    filter_batch, filter_channel = shape_filter
    pads = list(pads)

    # ===========================================================
    c0_size = cce_params.C0_SIZE  # Channel axis should be align with 16
    shape_dedy = (dedy_batch,
                  dedy_deep,
                  _ceil(dedy_channel, c0_size), dedy_h, dedy_w, c0_size)

    shape_filter_frac = (filter_depth,
                         _ceil(filter_batch, c0_size) * filter_h * filter_w,
                         _ceil(filter_channel, c0_size), c0_size, c0_size)
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    shape_filter_ncdhw = [filter_batch,
                          filter_channel, filter_depth, filter_h, filter_w]

    filters = tvm.placeholder(shape_filter_frac,
                              name="filter", dtype=filter_dtype)

    dedx = te.lang.cce.conv3d_backprop_input_compute(
        filters=filters,
        out_backprop=dedy,
        filter_sizes=shape_filter_ncdhw,
        input_sizes=input_sizes,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype
    )
    tensor_list = [filters, dedy, dedx]

    with tvm.target.cce():
        sch = generic.auto_schedule(dedx)

    config = {
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    te.lang.cce.cce_build_code(sch, config)
