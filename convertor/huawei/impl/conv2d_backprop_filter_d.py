#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd.

conv2d_backprop_filter
"""
from __future__ import absolute_import
import te.lang.cce
from te import tvm
from te.platform import cce_params
from te.platform import get_soc_spec
from topi import generic
from topi.cce import util

# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4
# the dim of strides in conv_backprop must be 2
STRIDES_SHAPE_DIM = 2
# the dim of pads in conv_backprop must be 4
PADDING_SHAPE_DIM = 4
# the min x or y dim for cube mul
C0 = 16
# fmapH, fmapW must be in [1,4096]
FMAP_HW_MAX = 4096
FMAP_HW_MIN = 1

# DeDy H,W must be in [2,4096]
DEDY_HW_MAX = 4096
DEDY_HW_MIN = 2

# filterH, filterW must be in [1,255]
FILTER_HW_MAX = 255
FILTER_HW_MIN = 1

# stride must be in [1,63]
STRIDE_HW_MAX = 63
STRIDE_HW_MIN = 1

# pad must be in [0,255]
PAD_MAX = 255
PAD_MIN = 0

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000

# the max size is 2**63-1
DATA_SIZE_MAX = 9223372036854775807

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}

# pads valid mode to be [0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0]
# If pads is string , only support "SAME" or "VALID"
PADDING_SUPPORT = ('SAME', 'VALID')


@util.check_input_type(dict, dict, dict, (tuple, list), (tuple, list),
                       (str, tuple, list), (tuple, list), int, str, str)
def conv2d_backprop_filter_d(x, out_backprop, y, filter_size, strides, pads,
                             dilations=(1, 1, 1, 1), groups=None,
                             data_format='NHWC',
                             kernel_name="conv2d_backprop_filter"):
    """
    algorithm: conv2d_backprop_filter

    Parameters
    ----------
    x: dict with keys(shape and dtype)
       input feature map tensor

    out_backprop: dict with keys(shape and dtype)
                  input weight tensor

    y: dict with keys(shape and dtype)
       output tensor, dtype must be assigned

    filter_size: The shape of filter.
                  4-D with shape [batch, channels, height, weight].

    strides: tuple/list of 2 integers
             filter move stride

    pads: string of "SAME" or "VAILD"
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated conv2d_backprop_filter

    groups: int
            param for group conv2d_backprop_filter

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "conv2d_backprop_filter"

    Returns
    -------
    None
    """

    def _check_inputs_rules():
        if (not isinstance(ori_shape_out_backprop, (tuple, list))) \
                or len(ori_shape_out_backprop) != 4:
            raise RuntimeError("out_backprop's shape should be 4d list.")

        if (not isinstance(ori_shape_x, (tuple, list))) or \
                len(ori_shape_x) != 4:
            raise RuntimeError("x's shape should be 4d list.")

        if (not isinstance(ori_shape_res, (tuple, list))) \
                or len(ori_shape_res) != 4:
            raise RuntimeError("y's shape should be 4d list.")

        if len(strides) != 2:
            raise RuntimeError("strides should be 2d list.")

        if len(filter_size) != 4:
            raise RuntimeError("filter_size should be 4d list")

        if len(dilations) != 4:
            raise RuntimeError("dilations's shape should be 4d list.")

        if isinstance(pads, str) and pads not in PADDING_SUPPORT:
            raise RuntimeError("pads should be SAME or VALID,")

        if isinstance(pads, (tuple, list)) and len(pads) != 4:
            raise RuntimeError("pads should be 4d list.")

    def _calcute_input_shape():
        if ori_format_x == "NHWC":
            x_shape = (ori_shape_x[0], ori_shape_x[3],
                       ori_shape_x[1], ori_shape_x[2])
        elif ori_format_x == "NCHW":
            x_shape = ori_shape_x
        else:
            raise RuntimeError("x's format should be NCHW or NHWC.")

        if ori_format_out_backprop == "NCHW":
            shape_out = ori_shape_out_backprop
        elif ori_format_out_backprop == "NHWC":
            shape_out = (ori_shape_out_backprop[0],
                         ori_shape_out_backprop[3],
                         ori_shape_out_backprop[1],
                         ori_shape_out_backprop[2])
        else:
            raise RuntimeError("out_backprop's format should be NCHW or NHWC.")
        return x_shape, shape_out

    ori_shape_x = x.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = y.get("ori_shape")

    x_dtype = x.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y.get("dtype")

    ori_format_x = x.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")
    ori_format_res = y.get("ori_format")

    if len(strides) == 4:
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[h_index], strides[w_index]]

    _check_inputs_rules()

    shape_x, shape_out_backprop = _calcute_input_shape()

    if ori_format_res == "NCHW":
        shape_res = ori_shape_res
    elif ori_format_res == "NHWC":
        shape_res = (ori_shape_res[0], ori_shape_res[3],
                     ori_shape_res[1], ori_shape_res[2])
    elif ori_format_res == "HWCN":
        shape_res = (ori_shape_res[3], ori_shape_res[2],
                     ori_shape_res[0], ori_shape_res[1])
    else:
        raise RuntimeError("y's format should be NCHW, NHWC or HWCN,")

    dilations = get_shape_dilation(ori_format_out_backprop, dilations)

    conv2d_backprop_filter_cce(shape_x,
                               shape_out_backprop,
                               shape_res,
                               strides,
                               pads,
                               dilations,
                               x_dtype,
                               out_backprop_dtype,
                               res_dtype,
                               kernel_name)


def get_shape_dilation(ori_format_out_backprop, dilations):
    """
    Get result shape of NCHW from original shape
    :param ori_format_res:
    :param ori_shape_res:
    :return: result shape of NCHW
    """
    if ori_format_out_backprop == "NCHW":
        shape_dilations = dilations
    elif ori_format_out_backprop == "NHWC":
        shape_dilations = (dilations[0],
                           dilations[3],
                           dilations[1],
                           dilations[2])
    else:
        raise RuntimeError("dilations's format should be NCHW or NHWC")
    return shape_dilations


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple),
                       str,
                       str,
                       str,
                       str)
def check_conv2dbp_filter_params(shape_x, shape_out_backprop, filter_sizes,
                                 strides, pads, dilations,
                                 x_dtype,
                                 out_backprop_dtype,
                                 res_dtype,
                                 kernel_name):
    """
    The params check function of conv2d_backprop_filter

    Parameters:
    ----------
    shape_x : The shape of feature map,
              which is 4-D [batch, channels, height, weight].

    shape_out_backprop : The shape of gradients,
                         which is 4-D [batch, channels, height, weight].

    filter_sizes : The shape of filter.
                   which is 4-D [batch, channels, height, weight].

    strides : The stride of the sliding window. A list of ints.

    pads : "SAME"or"VALID",
           indicating the type of pads algorithm to use, or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    x_dtype : Fmeature map  data dtype. Default value is float16.

    out_backprop_dtype : Gradients data dtype. Default value is float16.

    res_dtype : Result(De/Dw) data dtype. Default value is float32.

    kernel_name : Kernel name of cce.
                  Default value is "conv2d_backprop_filter_cce"

    Returns : All transformed params.
    ----------
    """

    def _align(input_x, input_y):
        if input_y == 0:
            raise RuntimeError("Division by zero")
        return (input_x + input_y - 1) // input_y * input_y

    def _check_attr_range_dw(name, value, attr_min=None, attr_max=None):
        if not attr_min and not attr_max:
            return
        if not attr_min:
            if value > attr_max:
                raise RuntimeError(
                    "{} exceed max_value limitation. max_value={}."
                    .format(name, attr_max))
        elif not attr_max:
            if value < attr_min:
                raise RuntimeError("{} less than min_value. min_value={}."
                                   .format(name, attr_min))
        elif value > attr_max or value < attr_min:
            raise RuntimeError("{} must be in [{},{}]."
                               .format(name, attr_min, attr_max))

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype:
            bit_ratio = BIT_RATIO_DICT.get(dtype)
        else:
            bit_ratio = BIT_RATIO_DICT.get("float16")
        if attr_value * bit_ratio > DATA_SIZE_MAX:
            raise RuntimeError("{} exceed 64 bit limitations!"
                               .format(attr_name))

    # First : Base check, Mainly required by interface appearance
    # ===========================================================
    # util check
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(shape_out_backprop,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(filter_sizes,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(strides,
                          STRIDES_SHAPE_DIM, STRIDES_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)

    def _check_attr_pads():
        # pads check
        if isinstance(pads, (tuple, list)) and \
                len(pads) != CONV_BACKPROP_SHAPE_DIM:
            raise RuntimeError("pads should be 4d list.")

        if isinstance(pads, str) and pads not in PADDING_SUPPORT:
            raise RuntimeError(
                "only support pads model {}.".format(PADDING_SUPPORT))

    _check_attr_pads()

    # dilations check
    util.check_shape_rule(dilations,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    _check_attr_range_dw("dilations's H", dilation_h,
                         DILATION_MIN, DILATION_MAX)
    _check_attr_range_dw("dilations's W", dilation_w,
                         DILATION_MIN, DILATION_MAX)
    if dilation_n != 1 or dilation_c != 1:
        raise RuntimeError("Dilations in the batch and "
                           "depth dimensions must be 1")

    # detype chek
    x_dtype = x_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    util.check_dtype_rule(x_dtype, ['float16'])
    util.check_dtype_rule(out_backprop_dtype, ['float16'])
    util.check_dtype_rule(res_dtype, ['float32', 'float16'])

    # Second : Furture Check, Mainly required by SRS
    # ===========================================================
    # the relation limits between shape
    shape_x = list(shape_x)
    shape_out_backprop = list(shape_out_backprop)
    filter_sizes = list(filter_sizes)
    strides = list(strides)
    fmap_batch, fmap_channel, fmap_h, fmap_w = shape_x
    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_channel, filter_h, filter_w = filter_sizes
    stride_h, stride_w = strides

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    # pads compute
    if pads == 'SAME':
        pad_w = \
            _align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = \
            _align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pads = [pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = PADDING_VAILD
    pads = list(pads)
    pad_up, pad_down, pad_left, pad_right = pads
    if pad_up >= filter_h_dilation or pad_down >= filter_h_dilation:
        raise RuntimeError("Pads must be less than kernel in H.")
    if pad_left >= filter_w_dilation or pad_right >= filter_w_dilation:
        raise RuntimeError("Pads must be less than kernel in W.")

    fmap_w_padding = fmap_w + pad_left + pad_right
    fmap_h_padding = fmap_h + pad_up + pad_down

    # special cases
    fmap_hw_min, dey_hw_min = FMAP_HW_MIN, DEDY_HW_MIN
    # limitation by chip:
    # if kernel h,w in [1,11] and fmap h/w after padding equals to filter h/w
    # load3d support h,w is 1
    if (1 <= filter_w <= 11) and (1 <= filter_h <= 11) \
            and (fmap_w_padding == filter_w or fmap_h_padding == filter_h):
        fmap_hw_min = 1
        dey_hw_min = 1

    # Dedy value limit
    _check_attr_range_dw("Dedy's H", dedy_h, dey_hw_min, DEDY_HW_MAX)
    _check_attr_range_dw("Dedy's W", dedy_w, dey_hw_min, DEDY_HW_MAX)

    # filter value limit
    _check_attr_range_dw("filter's H", filter_h, FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range_dw("filter's W", filter_w, FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range_dw("filter H*W", filter_h * filter_w,
                         FILTER_HW_MIN, FILTER_HW_MAX)

    # Fmap value limit
    _check_attr_range_dw("Fmap's H", fmap_h, fmap_hw_min, FMAP_HW_MAX)
    _check_attr_range_dw("Fmap's W", fmap_w, fmap_hw_min, FMAP_HW_MAX)

    # stride value limit
    _check_attr_range_dw("stride's H", stride_h, STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range_dw("stride's W", stride_w, STRIDE_HW_MIN, STRIDE_HW_MAX)

    def _check_axis_hw():
        if fmap_batch != dedy_batch:
            raise RuntimeError("Shape error: "
                               "Fmap's N must be equal to Dedy'N.")
        if dedy_channel != filter_batch:
            raise RuntimeError("Shape error: "
                               "Dedy's C must be equal to Filter'N.")
        if fmap_channel != filter_channel:
            raise RuntimeError("Shape error: "
                               "Fmap's C must be equal to Filter'C.")
        if filter_w_dilation > fmap_w_padding:
            raise RuntimeError("Shape error: Filter'W(after dilation) "
                               "must be less than Fmap'W(after padding).")
        if filter_h_dilation > fmap_h_padding:
            raise RuntimeError("Shape error: Filter'H(after dilation) "
                               "must be less than Fmap'H(after padding).")

        # Third : value check, Mainly required by the convolution rule
        if ((fmap_w - filter_w_dilation + pad_left + pad_right)
                // stride_w + 1) != dedy_w:
            raise RuntimeError(
                "Shape error : The compute result of W"
                " under convolution rule cannot match")
        if ((fmap_h - filter_h_dilation + pad_up + pad_down)
                // stride_h + 1) != dedy_h:
            raise RuntimeError(
                "Shape error : The compute result of H"
                " under convolution rule cannot match")

    _check_axis_hw()

    def _min_l1_byte():
        # Forth : L1 limitation, Mainly required by chip
        al1_min_byte = C0 * C0 * 2

        if dedy_w % C0 == 0:
            bl1_min_byte = filter_h_dilation * fmap_w * C0 * 2
        else:
            bl1_min_byte = (filter_h_dilation + stride_h) * fmap_w * C0 * 2

        l1_size = get_soc_spec("L1_SIZE")  # L1 size
        if (al1_min_byte + bl1_min_byte) > l1_size:
            raise RuntimeError("Invalid input size due to large kernel size"
                               " and stride")

    _min_l1_byte()
    # Fifth : check shape size, 64 bits limitation
    c0_size = cce_params.C0_SIZE
    fmap_size = fmap_batch * _align(fmap_channel, c0_size) * fmap_h * fmap_w
    dedy_size = dedy_batch * _align(dedy_channel, c0_size) * dedy_h * dedy_w
    filter_size = \
        _align(filter_batch, c0_size) * _align(filter_channel, c0_size) \
        * filter_h * filter_w
    _check_64bits_limitation("fmap_size", fmap_size, dtype=x_dtype)
    _check_64bits_limitation("dedy_size", dedy_size, dtype=out_backprop_dtype)
    _check_64bits_limitation("filter_size", filter_size, dtype=res_dtype)

    result = (shape_x, shape_out_backprop, filter_sizes,
              strides, pads, dilations, x_dtype, out_backprop_dtype,
              res_dtype, kernel_name)
    return result


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple),
                       str,
                       str,
                       str,
                       str)
def conv2d_backprop_filter_cce(shape_x, shape_out_backprop, filter_sizes,
                               strides, pads, dilations=(1, 1, 1, 1),
                               x_dtype='float16',
                               out_backprop_dtype='float16',
                               res_dtype='float32',
                               kernel_name="conv2d_backprop_filter_cce"):
    """
    Topi interface of conv2d backprop filter

    Parameters:
    ----------
    shape_x : The shape of feature map.
              4-D with shape [batch, channels, height, weight].

    shape_out_backprop : The shape of gradients.
                         4-D with shape [batch, channels, height, weight].

    filter_sizes : The shape of filter.
                   4-D with shape [batch, channels, height, weight].

    strides : A list of ints. The stride of the sliding window.

    pads : "SAME"or"VALID",
           indicating the type of pads algorithm to use, or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    x_dtype : The dtype of feature map data. Default value is float16.

    out_backprop_dtype : The dtype of gradients data.
                         Default value is float16.

    res_dtype : The dtype of result(De/Dw) data. Default value is float32.

    kernel_name : Cce kernel name.
                  Default value is "conv2d_backprop_filter_cce"

    need_build : If need to build CCEC kernel. Default value is False.

    Returns : None
    ----------
    """

    def _ceil(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2

    res = check_conv2dbp_filter_params(shape_x, shape_out_backprop,
                                       filter_sizes, strides, pads,
                                       dilations, x_dtype, out_backprop_dtype,
                                       res_dtype, kernel_name)
    shape_x, shape_out_backprop, filter_sizes, strides, pads, dilations, \
    x_dtype, out_backprop_dtype, res_dtype, kernel_name = res

    fmap_batch, fmap_channel, fmap_h, fmap_w = shape_x
    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop

    c0_size = cce_params.C0_SIZE  # Channel axis should be align with 16
    shape_dedy = (dedy_batch,
                  _ceil(dedy_channel, c0_size), dedy_h, dedy_w, c0_size)
    shape_fmap = (fmap_batch,
                  _ceil(fmap_channel, c0_size), fmap_h, fmap_w, c0_size)
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    fmap = tvm.placeholder(shape_fmap, name="fmap", dtype=x_dtype)
    dedw = te.lang.cce.conv2d_backprop_filter_compute(
        input_x=fmap,
        out_backprop=dedy,
        filter_sizes=filter_sizes,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype
    )
    tensor_list_input = [fmap, dedy]

    with tvm.target.cce():
        sch = generic.auto_schedule(dedw)

    real_outs = sch.cce_special["real_out_tensor"]
    tensor_list = tensor_list_input + real_outs
    config = {
        "name": kernel_name,
        "tensor_list": tensor_list
    }

    te.lang.cce.cce_build_code(sch, config)
