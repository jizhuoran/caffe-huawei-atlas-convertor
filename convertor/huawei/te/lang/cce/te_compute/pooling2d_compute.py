#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache Licenses for more details at
http://www.apache.org/licenses/LICENSE-2.0

pooling2d compute
"""

import math

from te import platform as cceconf
from te import tvm
from te.platform import intrinsic_check_support
from te.platform import get_soc_spec
from te.lang.cce.te_compute.common import img2col, im2col_fractal
from te.platform.cce_conf import CceProductParams
from te.platform.cce_policy import get_L1_info

from .cast_compute import _cast
from .max_pool2d_3_3_2_2_compute import max_pool as max_pool_3_3_2_2
from .max_pool2d_3_3_1_1_compute import max_pool as max_pool_3_3_1_1

SIZE_OF_FP16 = 2
BLOCK_SIZE = 16

# pylint: disable=invalid-name
op_tag = "pooling2d_"


def reshape(tensor_in, new_shape):
    """
    :params:
    :input: tensor to be reshaped
    :new_shape: shape after input tensor reshaped
    :return: reshape tensor
    """

    def _img2col_compute(tensor, indices):
        # fmap_n, fmap_c1, fmap_h, fmap_w, fmap_c0
        axis_0, axis_1, axis_2, axis_3, axis_4, axis_5 = indices

        wh_ww = new_shape[3]

        return tensor(axis_0, axis_1, axis_2*wh_ww + axis_3, axis_4, axis_5)

    return tvm.compute(
        new_shape,
        lambda *indices: _img2col_compute(tensor_in, indices),
        name='reshape')


def check_fmap_shape(batch_size, c1_value, in_size_h, in_size_w, c_block_size):
    """
    check feature shape whether valid
    """
    if batch_size <= 0 or c1_value <= 0:
        raise RuntimeError("invalid featur map shape params, "
                           "shape must be uint format and each dim >=1, C0 must be equal to 16")

    if in_size_h <= 0 or in_size_w <= 0:
        raise RuntimeError("invalid featur map shape params, "
                           "shape must be uint format and each dim >=1, C0 must be equal to 16")

    if c_block_size <= 0 or (c_block_size != 16):
        raise RuntimeError("invalid featur map shape params, "
                           "shape must be uint format and each dim >=1, C0 must be equal to 16")


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
def pooling2d(tensor_in, window, stride, pooling_mode, padding_mode="SAME",
              pad=(0, 0, 0, 0), dilation=(1, 1), data_mode=1, ceil_mode=0,
              fusion_params={}):
    """
    :params:
    :tensor_in: input tensor
    :window: input window
    :pooling_mode: can be MAX, AVG, GAP, GMP
    :padding_mode: can be SAME, VALID
    :pad: padT, padB, padL, padR
    :dilation: params to be reserved, use default value
    :stride: window move steps in h or w dimension
    :data_mode: can be 0: CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    :ceil_mode : caffe round_mode params, 0:CEIL(default), 1:FLOOR
    :return: pooling result
    """
    l1_fusion_type = fusion_params.get("l1_fusion_type", -1)
    is_l1fusion = l1_fusion_type in (0, 1)
    is_l2fusion = get_L1_info("L2_fusion_enabled")
    in_select_read_flag = fusion_params.get("in_select_read_flag", False)
    is_del_fusion_params = not is_l1fusion and not is_l2fusion
    if is_del_fusion_params:
        fusion_params = {}

    check_attr_rule(tensor_in, window, stride, pooling_mode, padding_mode, pad,
                    dilation, data_mode, ceil_mode)

    # define dict to transfer pooling params
    pooling_params = {}

    # get shape info of feature map in NC1HWC0 format
    batch_size = tensor_in.shape[0].value
    c1_value = tensor_in.shape[1].value
    in_size_h = tensor_in.shape[2].value
    in_size_w = tensor_in.shape[3].value
    c_block_size = tensor_in.shape[4].value

    check_fmap_shape(batch_size, c1_value, in_size_h, in_size_w, c_block_size)

    # get window size
    window_h = window[0]
    window_w = window[1]

    # get stride size
    stride_h = stride[0]
    stride_w = stride[1]

    # get dilation size
    dilation_h = dilation[0]
    dilation_w = dilation[1]

    # get padding size for SAME mode
    # pad_left, pad_right, pad_top, pad_bottom is for pad on the left, right, top, bottom
    pad_top = pad[0]
    pad_bottom = pad[1]
    pad_left = pad[2]
    pad_right = pad[3]

    if data_mode == 1:
        pooling_mode = get_tensorflow_pooling_mode(padding_mode, pooling_mode, in_size_h, in_size_w,
                                                   window_h, window_w)

    def check_stride_window_rule(pooling_mode, stride, window):
        if pooling_mode in ["MAX", "AVG"]:
            is_stride_invalid = stride[0] > 2 * window[0] or \
                                stride[1] > 2 * window[1]
            if is_stride_invalid:
                raise RuntimeError("stride_h should be <= 2*window_h, "
                                   "stride_w should be <= 2*window_w.")
    check_stride_window_rule(pooling_mode, stride, window)

    # avg or max pooling
    if pooling_mode in ["AVG", "MAX"]:
        # only in AVG and MAX pooling related, img2col instrin nRepeat in [1,255]
        if window_h*window_w > 255:
            raise RuntimeError("invalid window params, window_h * window_w should be <= 255")

        if data_mode == 0:
            out_size_h, out_size_w, pad_top, pad_bottom, pad_left, pad_right \
                = get_caffe_out_size_and_pad(ceil_mode, in_size_h, in_size_w,
                                             window_h, window_w, stride_h,
                                             stride_w, dilation_h, dilation_w,
                                             pad_top, pad_bottom, pad_left,
                                             pad_right)
        elif data_mode == 1:
            out_size_h, out_size_w, pad_top, pad_bottom, pad_left, pad_right = \
                get_tensorflow_out_size_and_pad(padding_mode, in_size_h, in_size_w,
                                                window_h, window_w, stride_h,
                                                stride_w, dilation_h, dilation_w,
                                                pad_top, pad_bottom, pad_left,
                                                pad_right, fusion_params)

        # cloud out_size_h = 1 or out_size_w = 1, img2col does not act normally
        if get_soc_spec("SOC_VERSION") == "Ascend910":
            flag = out_size_h == 1 and out_size_w == 1
            if not flag:
                if in_size_h + pad_top + pad_bottom - window_h < stride_h:
                    raise RuntimeError("invalid params, must be in_size_h + "
                                       "pad_top + pad_bottom - window_h >= stride_h")

                if in_size_w + pad_left + pad_right - window_w < stride_w:
                    raise RuntimeError("invalid params, must be in_size_w + "
                                       "pad_left + pad_right - window_w >= stride_w")

        # get the min ub occupy size when out_size_h = 1 and c1 = 1
        ub_size = get_soc_spec("UB_SIZE")

        check_ub_tiling(data_mode, pooling_mode, padding_mode, out_size_w, window_h, window_w,
                        c_block_size, ub_size)

        # copy input fmap from gm to l1
        input_fmap_l1 = tvm.compute(tensor_in.shape,
                                    lambda *i: tensor_in[i],
                                    name="input_fmap_l1",
                                    tag=op_tag + "poolinginput")

        fmap_img2col_h = out_size_h*out_size_w
        pad = (pad_top, pad_bottom, pad_left, pad_right)
        stride = (stride_h, stride_w)

        # fmap img2col l1 -> ub in zZ format by fractal
        fmap_img2col_shape_ub = (batch_size, fmap_img2col_h, c1_value, window_h, window_w,
                                 c_block_size)
        padding_value = 0.0
        if pooling_mode == "MAX":
            padding_value = 0xFBFF
        fmap_img2col_ub = img2col(input_fmap_l1, fmap_img2col_shape_ub, window_h, window_w, pad,
                                  stride, tag='', padding_value=padding_value)

        fmap_img2col_m = out_size_h*out_size_w
        ho_wo = ((fmap_img2col_m + BLOCK_SIZE - 1)//BLOCK_SIZE)*BLOCK_SIZE

        fractal_shape = (batch_size, ho_wo//BLOCK_SIZE, window_h*window_w*c1_value, BLOCK_SIZE,
                         BLOCK_SIZE)
        fmap_fractal = im2col_fractal(fractal_shape, fmap_img2col_ub, tag='')

        fractal_shape_tmp_1 = (batch_size, ho_wo//BLOCK_SIZE, c1_value, window_h*window_w,
                               BLOCK_SIZE, BLOCK_SIZE)
        fmap_fractal_tmp_1 = reshape(fmap_fractal, fractal_shape_tmp_1)

        # output shape
        res_output_shape = (batch_size, ho_wo//BLOCK_SIZE, c1_value, BLOCK_SIZE, BLOCK_SIZE)

    # global avg or max pooling
    elif pooling_mode in ["GAP", "GMP"]:
        if data_mode == 1:
            not_global_flag = window_h < in_size_h or window_w < in_size_w
            if not_global_flag:
                raise RuntimeError("invalid window params in GAP or GMP mode, "
                                   "window size should be equal to input size")

            if padding_mode == "SAME":
                raise RuntimeError("invalid padding_mode params in GAP or GMP mode, "
                                   "padding_mode can only be VALID")

        stride_h = 1
        stride_w = 1
        out_size_h = 1
        out_size_w = 1
        res_output_shape = (batch_size, c1_value, 1, 1, c_block_size)

        # copy tensor in from gm to ub
        tensor_in_ub = tvm.compute(tensor_in.shape, lambda *i: tensor_in[i],
                                   name="tensor_in_ub",
                                   tag=op_tag + "poolinginput")

    # record pooling params
    pooling_params["pooling_mode"] = pooling_mode

    pooling_params["padding_mode"] = get_schedule_padding_mode(data_mode, padding_mode)

    pooling_params["data_mode"] = data_mode

    pooling_params["res_output_shape"] = res_output_shape

    pooling_params["batch_size"] = batch_size
    pooling_params["c1_value"] = c1_value
    pooling_params["window_h"] = window_h
    pooling_params["window_w"] = window_w
    pooling_params["c_block_size"] = c_block_size

    pooling_params["in_size_h"] = in_size_h
    pooling_params["in_size_w"] = in_size_w

    pooling_params["stride_h"] = stride_h
    pooling_params["stride_w"] = stride_w

    pooling_params["pad_top"] = pad_top
    pooling_params["pad_bottom"] = pad_bottom
    pooling_params["pad_left"] = pad_left
    pooling_params["pad_right"] = pad_right

    pooling_params["out_size_h"] = out_size_h
    pooling_params["out_size_w"] = out_size_w

    setfmatrix_dict = {"conv_kernel_h": window_h,
                       "conv_kernel_w": window_w,
                       "conv_padding_top": pad_top,
                       "conv_padding_bottom": pad_bottom,
                       "conv_padding_left": pad_left,
                       "conv_padding_right": pad_right,
                       "conv_stride_h": stride_h,
                       "conv_stride_w": stride_w,
                       "conv_fm_c": c1_value*c_block_size,
                       "conv_fm_h": in_size_h,
                       "conv_fm_w": in_size_w,
                       }

    if pooling_mode == "MAX":
        is_process_max_pool_3_3_2_2 = not is_l1fusion and \
                                      (not is_l2fusion or
                                       (is_l2fusion and
                                        not in_select_read_flag)) and \
                                      (window_h, window_w) == (3, 3) and \
                                      (stride_h, stride_w) == (2, 2)
        is_process_max_pool_3_3_1_1 = not is_l1fusion and \
                                      not is_l2fusion and \
                                      (window_h, window_w) == (3, 3) and \
                                      (stride_h, stride_w) == (1, 1)
        if is_process_max_pool_3_3_2_2:
            res = max_pool_3_3_2_2(tensor_in, (out_size_h, out_size_w),
                                   (pad_top, pad_bottom, pad_left, pad_right),
                                   pooling_params)
        elif is_process_max_pool_3_3_1_1:
            res = max_pool_3_3_1_1(tensor_in, (out_size_h, out_size_w),
                                   (pad_top, pad_bottom, pad_left, pad_right),
                                   pooling_params)
        else:
            res = pooling2d_max(fmap_fractal_tmp_1, res_output_shape, window_h,
                                window_w, pooling_params, setfmatrix_dict,
                                fusion_params)

    elif pooling_mode == "AVG":
        res = pooling2d_avg(fmap_fractal_tmp_1, res_output_shape, window_h,
                            window_w, pooling_params, setfmatrix_dict,
                            fusion_params)
    elif pooling_mode == "GMP":
        res = pooling2d_gmp(tensor_in_ub, res_output_shape, window_h, window_w,
                            pooling_params, setfmatrix_dict, fusion_params)
    elif pooling_mode == "GAP":
        res = pooling2d_gap(tensor_in_ub, res_output_shape, window_h, window_w,
                            pooling_params, setfmatrix_dict, fusion_params)

    return res


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
def check_attr_rule(tensor_in, window, stride, pooling_mode, padding_mode, pad=(0, 0, 0, 0),
                    dilation=(1, 1), data_mode=0, ceil_mode=0):
    """
    :params:
    :tensor_in: input tensor
    :window: input window
    :stride: window move steps in h or w dimension
    :data_mode: can be 0: DOMI_CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    :pooling_mode: can be MAX, AVG, GAP, GMP
    :padding_mode: can be SAME, VALID
    :pad: padT, padB, padL, padR
    :dilation: params to be reserved, use default value
    :data_mode: can be 0: CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    :ceil_mode : caffe round_mode params
    :return: pooling result
    """
    if not isinstance(tensor_in, tvm.tensor.Tensor):
        raise RuntimeError("invalid tensor_in params, type of tensor_in must be tvm.tensor.Tensor.")

    if not isinstance(window, tuple) and not isinstance(window, list):
        raise RuntimeError("invalid window params, type of window must be tuple or list.")

    if not isinstance(stride, tuple) and not isinstance(stride, list):
        raise RuntimeError("invalid stride params, type of stride must be tuple or list.")

    if not isinstance(padding_mode, str):
        raise RuntimeError("invalid padding_mode params, type of padding_mode must be str.")

    if not isinstance(pad, tuple) and not isinstance(pad, list):
        raise RuntimeError("invalid pad params, type of pad must be tuple or list.")

    if not isinstance(dilation, tuple) and not isinstance(dilation, list):
        raise RuntimeError("invalid dilation params, type of dilation must be tuple or list.")

    if len(tensor_in.shape) != 5:
        raise RuntimeError("invalid shape params, input feature map must be 5D format in kernel.")

    if len(window) != 2:
        raise RuntimeError("invalid window params, window dim must be 2.")

    if window[0] > 32768 or window[0] < 1:
        raise RuntimeError("invalid window params, window_h size must be [1, 32768].")

    if window[1] > 32768 or window[1] < 1:
        raise RuntimeError("invalid window params, window_w size must be [1, 32768].")

    if len(stride) != 2:
        raise RuntimeError("invalid stride params, stride dim must be 2.")

    if stride[0] > 63 or stride[0] < 1:
        raise RuntimeError("invalid stride params, stride_h size must be [1,63].")

    if stride[1] > 63 or stride[1] < 1:
        raise RuntimeError("invalid stride params, stride_w size must be [1,63].")

    if str(tensor_in.dtype) not in ["float16"]:
        raise RuntimeError("can only support float16 dtype of tensor_in.")

    # GAP is short for global avg pooling
    # GMP is short for global max pooling
    if pooling_mode not in ["AVG", "MAX", "GAP", "GMP"]:
        raise RuntimeError("can only support AVG or MAX or GAP or GMP pooling mode.")

    if len(pad) != 4 or pad[0] < 0 or pad[1] < 0 or pad[2] < 0 or pad[3] < 0:
        raise RuntimeError("invalid pad params, pad size must be 4 with uint format, each dim >= 0")

    if len(dilation) != 2:
        raise RuntimeError("invalid dilation params, dilation dim must be 2.")

    if dilation[0] > 255 or dilation[0] < 1:
        raise RuntimeError("invalid dilation params, dilation_h size must be [1,255].")

    if dilation[1] > 255 or dilation[1] < 1:
        raise RuntimeError("invalid dilation params, dilation_w size must be [1,255].")

    if data_mode not in [0, 1]:
        raise RuntimeError("data mode only support 0:CAFFE or 1:TENSORFLOW.")

    if data_mode == 0:
        check_caffe_attr_rule(ceil_mode, pad)

    if data_mode == 1:
        check_tensorflow_attr_rule(padding_mode)


def check_caffe_attr_rule(ceil_mode, pad):
    """
    :param ceil_mode: 0:PoolingParameter_RoundMode_CEIL or 1:PoolingParameter_RoundMode_FLOOR
    :param pad padT, padB, padL, padR
    :return:
    """
    if ceil_mode not in [0, 1]:
        raise RuntimeError("can only support 0:PCEIL or 1:RoundMode_FLOOR padding mode.")


def check_tensorflow_attr_rule(padding_mode):
    """
    :param padding_mode: "SAME", "VALID"
    :return:
    """
    if not isinstance(padding_mode, str):
        raise RuntimeError("invalid padding_mode params, type of padding_mode must be str.")
    if padding_mode not in ["SAME", "VALID"]:
        raise RuntimeError("can only support SAME or VALID padding mode.")


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
def get_tensorflow_pooling_mode(padding_mode, pooling_mode, in_size_h, in_size_w,
                                window_h, window_w):
    """
    :param padding_mode: can be SAME, VALID
    :param pooling_mode: can be MAX, AVG, GAP, GMP
    :param in_size_h: input tensor
    :param in_size_w: input tensor
    :param window_h: input window
    :param window_w: input window
    :return:
    """
    # in VALID mode window must be <= feature map
    # only in SAME mode [window > feature map] can be allowed
    # only [window = feature map && padding_mode = VALID]
    # is defined as GAP or GMP
    # that means there is no SAME mode in GAP and GMP
    # others are all handled as normal AVG or MAX pooling.
    # pylint: disable=no-else-raise
    if padding_mode == "VALID":
        if window_h > in_size_h or window_w > in_size_w:
            raise RuntimeError("invalid window params, "
                               "in VALID mode window must be <= feature map.")
        if window_h == in_size_h and window_w == in_size_w:
            if pooling_mode == "MAX":
                pooling_mode = "GMP"
            elif pooling_mode == "AVG":
                pooling_mode = "GAP"

    # redefine pooling_mode from global to normal max or avg
    elif padding_mode == "SAME":
        if window_h >= in_size_h or window_w >= in_size_w:
            if pooling_mode == "GMP":
                pooling_mode = "MAX"
            elif pooling_mode == "GAP":
                pooling_mode = "AVG"

    return pooling_mode


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
def check_ub_tiling(data_mode, pooling_mode, padding_mode, out_size_w, window_h, window_w,
                    c_block_size, ub_size):
    """
    :param data_mode: can be 0: CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    :param pooling_mode: can be MAX, AVG, GAP, GMP
    :param padding_mode: can be SAME, VALID
    :param out_size_w: output w
    :param window_h: window h
    :param window_w: window w
    :param c_block_size: channel block size
    :param ub_size: ub size
    :return:
    """
    if data_mode == 0:
        if pooling_mode == "MAX":
            data_size = get_ub_data_size(out_size_w, window_h, window_w, c_block_size, False)
        elif pooling_mode == "AVG":
            data_size = get_ub_data_size(out_size_w, window_h, window_w, c_block_size, True)

    elif data_mode == 1:
        if pooling_mode == "MAX" or (pooling_mode == "AVG" and padding_mode == "VALID"):
            data_size = get_ub_data_size(out_size_w, window_h, window_w, c_block_size, False)

        if pooling_mode == "AVG" and padding_mode == "SAME":
            data_size = get_ub_data_size(out_size_w, window_h, window_w, c_block_size, True)

    if data_size >= ub_size:
        raise RuntimeError("cutH and C1, can not find valid tiling params, cutW support needed")


# pylint: disable=too-many-locals, too-many-arguments
def get_ub_data_size(out_size_w, window_h, window_w, c_block_size, enable_avg_mean_factor):
    """
    :param out_size_w: output tensor
    :param window_h: input window
    :param window_w: input window
    :param c_block_size: channel block size
    :param enable_avg_mean_factor: avg calculate factor
    :return:
    """
    fmap_img2col_size = out_size_w*window_h*window_w*c_block_size*SIZE_OF_FP16
    res_size = out_size_w*c_block_size*SIZE_OF_FP16
    avg_mean_factor_size = get_avg_mean_factor_size(enable_avg_mean_factor, out_size_w,
                                                    c_block_size)
    data_size = fmap_img2col_size + avg_mean_factor_size + res_size

    return data_size


# pylint: disable=too-many-locals, too-many-arguments
def get_avg_mean_factor_size(enable_avg_mean_factor, out_size_w, c_block_size):
    """
    :param enable_avg_mean_factor: avg calculate factor
    :param out_size_w: output tensor
    :param c_block_size: channel block size
    :return:
    """
    if enable_avg_mean_factor:
        return out_size_w * c_block_size * SIZE_OF_FP16
    return 0


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
def get_caffe_out_size_and_pad(ceil_mode, in_size_h, in_size_w, window_h, window_w,
                               stride_h, stride_w, dilation_h, dilation_w, pad_top,
                               pad_bottom, pad_left, pad_right):
    """
    :param ceil_mode: caffe round_mode params, 0:CEIL(default), 1:FLOOR
    :param in_size_h: input h
    :param in_size_w: input w
    :param window_h: window h
    :param window_w: window w
    :param stride_h: stride h
    :param stride_w: stride w
    :param dilation_h: dilation h
    :param dilation_w: dilation w
    :param pad_top: pad top
    :param pad_bottom: pad bottom
    :param pad_left: pad left
    :param pad_right: pad right
    :return:
    """

    if ceil_mode == 0:
        out_size_h = math.ceil((in_size_h + pad_top + pad_bottom - window_h)/stride_h) + 1
        out_size_w = math.ceil((in_size_w + pad_left + pad_right - window_w)/stride_w) + 1

    if ceil_mode == 1:
        out_size_h = math.floor((in_size_h + pad_top + pad_bottom - window_h)/stride_h) + 1
        out_size_w = math.floor((in_size_w + pad_left + pad_right - window_w)/stride_w) + 1

    if pad_top != 0 or pad_left != 0:
        # If we have padding, ensure that the last pooling starts strictly
        # inside the image (instead of at the padding); otherwise clip the last.
        if (out_size_h - 1)*stride_h >= in_size_h + pad_top:
            out_size_h -= 1

        if (out_size_w - 1)*stride_w >= in_size_w + pad_left:
            out_size_w -= 1

        # CHECK_LT((out_size_h - 1) * stride_h, in_size_h + pad_top);
        # CHECK_LT((out_size_w - 1) * stride_w, in_size_w + pad_left);
        if (out_size_h - 1)*stride_h >= in_size_h + pad_top:
            raise RuntimeError("CHECK_LT((out_size_h - 1) * stride_h, in_size_h + pad_top)")
        if (out_size_w - 1)*stride_w >= in_size_w + pad_left:
            raise RuntimeError("CHECK_LT((out_size_w - 1) * stride_w, in_size_w + pad_left)")

    # floor mode modify davici pad
    if ceil_mode == 0:
        pad_rows = (out_size_h - 1)*stride_h + ((window_h - 1)*dilation_h + 1) - in_size_h
        pad_bottom = pad_rows - pad_top
        pad_cols = (out_size_w - 1)*stride_w + ((window_w - 1)*dilation_w + 1) - in_size_w
        pad_right = pad_cols - pad_left
    if pad_bottom < 0:
        pad_bottom = 0
    if pad_right < 0:
        pad_right = 0

    return out_size_h, out_size_w, pad_top, pad_bottom, pad_left, pad_right


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
def get_tensorflow_out_size_and_pad(padding_mode, in_size_h, in_size_w, window_h, window_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_top,
                                    pad_bottom, pad_left, pad_right,
                                    fusion_params={}):
    """
    :param padding_mode: can be SAME, VALID
    :param in_size_h: input tensor
    :param in_size_w: input tensor
    :param window_h: input window
    :param window_w: input window
    :param stride_h: stride
    :param stride_w: stride
    :param dilation_w: dilation
    :param dilation_h: dilation
    :param pad_top: pad top
    :param pad_bottom: pad bottom
    :param pad_left: pad left
    :param pad_right: pad right
    :return:
    """
    if padding_mode == "SAME":
        # caculate output size in SAME mode
        # Hout = ceil(Hi, Sh)
        # Wout = ceil(Wi, Sw)
        l1_fusion_type = fusion_params.get("l1_fusion_type")
        is_l1fusion = l1_fusion_type in (0, 1)
        is_l2fusion = get_L1_info("L2_fusion_enabled")
        in_split_index = fusion_params.get("in_split_index")
        is_use_out_shape = (is_l1fusion or is_l2fusion) and \
                           "out_shape" in fusion_params
        if is_use_out_shape:
            out_size_h = fusion_params["out_shape"][2]
            out_size_w = fusion_params["out_shape"][3]
        else:
            out_size_h = (in_size_h + stride_h - 1)//stride_h
            out_size_w = (in_size_w + stride_w - 1)//stride_w

        # Total padding on rows and cols is
        # Pr = (R' - 1) * S + (Kr - 1) * Dr + 1 - R
        # Pc = (C' - 1) * S + (Kc - 1) * Dc + 1 - C
        # where (R', C') are output dimensions, (R, C) are input dims.
        # S is stride, (Dr, Dc) are dilations, (Kr, Kc) are filter dims.
        # get total pad rows or pad columns
        pad_rows = (out_size_h - 1)*stride_h + \
                   ((window_h - 1)*dilation_h + 1) - in_size_h
        pad_cols = (out_size_w - 1)*stride_w + \
                   ((window_w - 1)*dilation_w + 1) - in_size_w

        # pad_rows and pad_columns is odd or even number
        # odd : half, half for pad_left, pad_right, pad_top, pad_bottom
        # even : less on pad_left and pad_top, more on pad_right and pad_bottom
        if in_split_index == 1:
            pad_top = pad_rows
            pad_bottom = 0
        elif in_split_index == 3:
            pad_top = 0
            pad_bottom = pad_rows
        else:
            pad_top = pad_rows // 2
            pad_bottom = pad_rows - pad_top

        pad_left = pad_cols//2
        pad_right = pad_cols - pad_left

        def get_corrected_pad(input_pad):
            if input_pad < 0:
                output_pad = 0
            else:
                output_pad = input_pad
            return output_pad

        pad_top = get_corrected_pad(pad_top)
        pad_bottom = get_corrected_pad(pad_bottom)
        pad_left = get_corrected_pad(pad_left)
        pad_right = get_corrected_pad(pad_right)

    # caculate output size in VALID mode
    elif padding_mode == "VALID":
        # caculate output size in VALID mode
        # Hout = ceil(Hi - Fh + 1, Sh)
        # Wout = ceil(Wi - Fw + 1, Sw)
        out_size_h = (in_size_h - window_h + 1 + (stride_h - 1))//stride_h
        out_size_w = (in_size_w - window_w + 1 + (stride_w - 1))//stride_w

    return out_size_h, out_size_w, pad_top, pad_bottom, pad_left, pad_right


def get_schedule_padding_mode(data_mode, padding_mode):
    """
    :param data_mode: can be 0: CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    :param padding_mode: can be SAME, VALID
    :return:
    """
    if data_mode == 0:
        # caffe after padding deal with tensorflow same
        padding_mode = "SAME"

    return padding_mode


# pylint: disable=too-many-locals, too-many-arguments
def pooling2d_max(fmap_img2col_ub, res_output_shape, window_h, window_w, pooling_params,
                  setfmatrix_dict, fusion_params):
    """
    :params:
    :fmap_img2col_ub: feature map after img2col in ub
    :res_output_shape: shape of max pooling of result
    :window_h: h dim of window
    :window_w: w dim of window
    :pooling_params: requantize pooling params
    :setfmatrix_dict: set_fmatrix params, it is a dictionary
    :return: res
    """
    # define reduce axis
    reduce_axis = tvm.reduce_axis((0, window_h*window_w), name="reduce_axis")

    # use emit_insn instead in the later procession
    pooling_out_ub = tvm.compute(
        res_output_shape,
        lambda i, j, k, m, n:
        tvm.max(fmap_img2col_ub[i, j, k, reduce_axis, m, n],
                axis=reduce_axis),
        name="pooling_out_ub"
    )

    batch_size = pooling_params["batch_size"]
    c1_size = pooling_params["c1_value"]
    oh_size = pooling_params["out_size_h"]
    ow_size = pooling_params["out_size_w"]
    c0_size = pooling_params["c_block_size"]

    pooling_ub_5hd_shape = (batch_size, c1_size, oh_size*ow_size, c0_size)
    pooling_ub_5hd = tvm.compute(
        pooling_ub_5hd_shape,
        lambda i, j, k, l: pooling_out_ub[i, k // 16, j, k % 16, l] + 0,
        name="pooling_ub_5hd"
    )

    res = tvm.compute(
        pooling_ub_5hd_shape,
        lambda *indices: pooling_ub_5hd[indices],
        name="pooling2d_res",
        tag=op_tag + 'max',
        attrs={'pooling_params': pooling_params,
               'setfmatrix_dict': setfmatrix_dict,
               'fusion_params': fusion_params}
    )

    return res


# pylint: disable=too-many-locals, too-many-arguments
def pooling2d_avg(fmap_img2col_ub, res_output_shape, window_h, window_w, pooling_params,
                  setfmatrix_dict, fusion_params):
    """
    :params:
    :fmap_img2col_ub: feature map after img2col in ub
    :res_output_shape: shape of avg pooling of result
    :window_h: h dim of window
    :window_w: w dim of window
    :pooling_params: requantize pooling params
    :setfmatrix_dict: set_fmatrix params, it is a dictionary
    :return: res
    """
    # define reduce axis
    reduce_axis = tvm.reduce_axis((0, window_h*window_w), name="reduce_axis")

    # not real compute here, need to vmul (1/area) after sum,
    # use emit_insn instead in the later procession
    pooling_out_ub = tvm.compute(
        res_output_shape,
        lambda i, j, k, m, n: tvm.sum(fmap_img2col_ub[i, j, k, reduce_axis, m, n],
                                      axis=reduce_axis),
        name="pooling2d_avg_sum"
    )

    pooling_out_ub_mul_factor = tvm.compute(
        res_output_shape,
        lambda *indices:
        pooling_out_ub[indices] * \
            (tvm.const(1.0/(window_h*window_w), "float16")),
        name="pooling2d_avg_mul_factor"
    )

    batch_size = pooling_params["batch_size"]
    c1_size = pooling_params["c1_value"]
    oh_size = pooling_params["out_size_h"]
    ow_size = pooling_params["out_size_w"]
    c0_size = pooling_params["c_block_size"]

    pooling_ub_5hd_shape = (batch_size, c1_size, oh_size*ow_size, c0_size)
    pooling_ub_5hd = tvm.compute(
        pooling_ub_5hd_shape,
        lambda i, j, k, l:
        pooling_out_ub_mul_factor[i, k // 16, j, k % 16, l] + 0,
        name="pooling_ub_5hd"
    )

    res = tvm.compute(
        pooling_ub_5hd_shape,
        lambda *indices: pooling_ub_5hd[indices],
        name="pooling2d_res",
        tag=op_tag + 'avg',
        attrs={'pooling_params': pooling_params,
               'setfmatrix_dict': setfmatrix_dict,
               'fusion_params': fusion_params}
    )

    return res


# pylint: disable=too-many-locals, too-many-arguments
def pooling2d_gmp(tensor_in_ub, res_output_shape, window_h, window_w, pooling_params,
                  setfmatrix_dict, fusion_params):
    """
    :params:
    :tensor_in_ub: feature map in ub
    :res_output_shape: shape of global max pooling of result
    :window_h: h dim of window
    :window_w: w dim of window
    :pooling_params: requantize pooling params
    :setfmatrix_dict: set_fmatrix params, it is a dictionary
    :return: res
    """
    # define reduce axis
    reduce_axis_h = tvm.reduce_axis((0, window_h), name="reduce_axis_h")
    reduce_axis_w = tvm.reduce_axis((0, window_w), name="reduce_axis_w")

    # use emit_insn instead in the later procession
    pooling_out_ub = tvm.compute(
        res_output_shape,
        lambda i, j, k, m, n: tvm.max(tensor_in_ub[i, j, reduce_axis_h, reduce_axis_w, n],
                                      axis=(reduce_axis_h, reduce_axis_w)),
        name="pooling_out_ub"
    )

    res_output_shape_new = (
        res_output_shape[0],
        res_output_shape[1],
        res_output_shape[2]*res_output_shape[3],
        res_output_shape[4]
    )

    res = tvm.compute(
        res_output_shape_new,
        lambda i, j, k, l: pooling_out_ub[i, j, k // 1, k % 1, l],
        name="pooling2d_res",
        tag=op_tag + 'gmp',
        attrs={'pooling_params': pooling_params,
               'setfmatrix_dict': setfmatrix_dict,
               'fusion_params': fusion_params}
    )

    return res


# pylint: disable=too-many-locals
def pooling2d_gap(tensor_in_ub, res_output_shape, window_h, window_w, pooling_params,
                  setfmatrix_dict, fusion_params):
    """
    :params:
    :tensor_in_ub: feature map in ub
    :res_output_shape: shape of global avg pooling of result
    :window_h: h dim of window
    :window_w: w dim of window
    :pooling_params: requantize pooling params
    :setfmatrix_dict: set_fmatrix params, it is a dictionary
    :return: res
    """
    # Check for vadd, vmul fp32 ability
    vconv_ability = intrinsic_check_support("Intrinsic_vconv", "f162f32")
    vadd_ability = intrinsic_check_support("Intrinsic_vadd", "float32")
    vmul_ability = intrinsic_check_support("Intrinsic_vmul", "float32")
    fp32_ability = vconv_ability and \
                   vadd_ability and \
                   vmul_ability and \
                   (not get_soc_spec("SOC_VERSION") in ("Ascend310",))
    # define reduce axis
    reduce_axis_h = tvm.reduce_axis((0, window_h), name="reduce_axis_h")
    reduce_axis_w = tvm.reduce_axis((0, window_w), name="reduce_axis_w")

    # need to vmul (1/area) after sum, use emit_insn instead in the later procession
    pooling_out_ub = tvm.compute(
        res_output_shape,
        lambda i, j, k, m, n: tvm.sum(tensor_in_ub[i, j, reduce_axis_h, reduce_axis_w, n],
                                      axis=(reduce_axis_h, reduce_axis_w)),
        name="pooling2d_avg_sum"
    )

    area_factor = 1.0/(window_h*window_w)

    pooling_out_ub_mul_factor = tvm.compute(
        res_output_shape,
        lambda *indices: pooling_out_ub[indices]*(tvm.const(area_factor, "float16")),
        name="pooling2d_avg_mul_factor"
    )

    res_output_shape_new = (
        res_output_shape[0],
        res_output_shape[1],
        res_output_shape[2] * res_output_shape[3],
        res_output_shape[4]
    )

    if fp32_ability:
        tensor_in_ub_f32 = _cast(tensor_in_ub, "float32", is_auto_cast=False)
        pooling_out_ub = tvm.compute(
            res_output_shape,
            lambda i, j, k, m, n: tvm.sum(tensor_in_ub_f32[i, j, reduce_axis_h, reduce_axis_w, n],
                                          axis=(reduce_axis_h, reduce_axis_w)),
            name="pooling2d_avg_sum"
        )

        area_factor = 1.0/(window_h*window_w)

        pooling_out_ub_mul_factor = tvm.compute(
            res_output_shape,
            lambda *indices: pooling_out_ub[indices]*(tvm.const(area_factor, "float32")),
            name="pooling2d_avg_mul_factor"
        )

        pooling_out_ub_mul_factor_f16 = _cast(pooling_out_ub_mul_factor, "float16",
                                              is_auto_cast=False)
        res = tvm.compute(
            res_output_shape_new,
            lambda i, j, k, l:
            pooling_out_ub_mul_factor_f16[i, j, k // 1, k % 1, l],
            name="res",
            tag=op_tag + 'gap',
            attrs={'pooling_params': pooling_params,
                   'setfmatrix_dict': setfmatrix_dict,
                   'fusion_params': fusion_params}
        )
    else:
        res = tvm.compute(
            res_output_shape_new,
            lambda i, j, k, l:
            pooling_out_ub_mul_factor[i, j, k // 1, k % 1, l],
            name="pooling2d_res",
            tag=op_tag + 'gap',
            attrs={'pooling_params': pooling_params,
                   'setfmatrix_dict': setfmatrix_dict,
                   'fusion_params': fusion_params}
        )

    return res
