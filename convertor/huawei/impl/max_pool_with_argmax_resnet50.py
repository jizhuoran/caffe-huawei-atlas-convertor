#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
maxpool_grad_with_argmax_resnet50
"""

import math
from te import tik
from impl import common_util
from impl import constant_util as constant
from te import platform as tbe_platform

# min value of fp16
MIN_VALUE_FP16 = -65504.0
# define dilation size
DILATION = 1
# parameters for vector instruct
MASK = 128
REPEAT_2 = 2
DSTSTRIDEM0 = 1
SRC0STRIDEM0 = 1
SRC1STRIDEM0 = 1
DSTSTRIDEM1 = 8
SRC0STRIDEM1 = 8
SRC1STRIDEM1 = 8
# get available ub size
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# get available l1 size
L1_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.L1_SIZE)


def _ceil_div(value, factor):
    """
    caculate ceil value of div

    Parameters
    ----------
    value: dtype of int or float
        original value
    factor: dtype of int or float
        dividend value

    Returns
    -------
    value: dtype of int or float
    """
    return math.ceil(value / factor)


# pylint: disable=locally-disabled, too-many-instance-attributes
# pylint: disable=too-few-public-methods
class MaxPoolWithargmaxResnet50():
    """
       Function: use to finish MaxPoolWithargmax main functions
    """

    def __init__(self, input_x, ksize, strides, padding):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        input_x: dict
            shape and datatype
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: str
            The type of padding algorithm to use.

        Returns
        -------
        None
        """
        self.input_shape = input_x.get("shape")
        self.input_dtype = input_x.get("dtype").lower()
        self.input_type_size = common_util.get_data_size(self.input_dtype)
        self.tik_instance = tik.Tik()

        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.batch_size = self.input_shape[0]
        self.c1_size = self.input_shape[1]
        self.in_size_h = self.input_shape[2]
        self.in_size_w = self.input_shape[3]
        self.c_block_size = self.input_shape[4]

        self.window_h = self.ksize[1]
        self.window_w = self.ksize[2]
        self.stride_h = self.strides[1]
        self.stride_w = self.strides[2]
        self.nc1 = self.batch_size * self.c1_size
        # scalar for load3d
        self.scalar_source_h = self.tik_instance.Scalar(dtype="int64")
        self.scalar_source_w = self.tik_instance.Scalar(dtype="int64")

        # caculate pad and output size
        self.pad, self.out_size_h, self.out_size_w = \
            self._calc_out_size_and_pad()
        # output_shape
        self.fmap_img2col_h = self.out_size_h * self.out_size_w
        self.fmap_img2col_w = self.window_h * self.window_w
        self.fmap_img2col_h_num = _ceil_div(self.fmap_img2col_h,
                                            self.c_block_size)
        if self.input_dtype == "float16":
            self.pad_value = MIN_VALUE_FP16
        # fmap is NC1HWC0 format
        fmap_gm_shape = (self.batch_size, self.c1_size, self.in_size_h,
                         self.in_size_w, self.c_block_size)

        output_gm_shape = (self.batch_size, self.c1_size, self.out_size_h,
                           self.out_size_w, self.c_block_size)
        output_mask_gm_shape = (
            self.batch_size, self.c1_size, self.fmap_img2col_w,
            (self.fmap_img2col_h_num + 1) * self.c_block_size)
        # input and output
        self.input_fmap_gm = self.tik_instance.Tensor(self.input_dtype,
                                                      fmap_gm_shape,
                                                      name="input_fmap_gm",
                                                      scope=tik.scope_gm)
        self.output_max_gm = self.tik_instance.Tensor(self.input_dtype,
                                                      output_gm_shape,
                                                      name="output_max_gm",
                                                      scope=tik.scope_gm)
        self.output_mask_gm = self.tik_instance.Tensor("uint16",
                                                       output_mask_gm_shape,
                                                       name="output_mask_gm",
                                                       scope=tik.scope_gm)

    def _calc_out_size_and_pad(self):
        """
        caculate output size and padding size
        -------
        pad: include pad_t, pad_b, pad_l, pad_r
        out_size_h: out_size in h direction
        out_size_w: out_size in w direction
        """
        # pad_l, pad_r, pad_t, pad_b is for pad on the left, right, top, bottom
        pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0

        if self.padding == "SAME":
            # Hout = ceil(Hi, Sh), Wout = ceil(Wi, Sw)
            out_size_h = (self.in_size_h + self.stride_h - 1) // self.stride_h
            out_size_w = (self.in_size_w + self.stride_w - 1) // self.stride_w

            # get total pad rows or pad columns
            pad_rows = (out_size_h - 1) * self.stride_h + \
                       ((self.window_h - 1) * DILATION + 1) - self.in_size_h
            pad_cols = (out_size_w - 1) * self.stride_w + \
                       ((self.window_w - 1) * DILATION + 1) - self.in_size_w

            # pad_rows and pad_columns is odd or even number
            if pad_rows % 2 == 0:
                pad_t = pad_rows // 2
                pad_b = pad_rows // 2
            else:
                pad_t = pad_rows // 2
                pad_b = pad_rows - pad_t

            if pad_cols % 2 == 0:
                pad_l = pad_cols // 2
                pad_r = pad_cols // 2
            else:
                pad_l = pad_cols // 2
                pad_r = pad_cols - pad_l

            if pad_t < 0:
                pad_t = 0

            if pad_b < 0:
                pad_b = 0

            if pad_l < 0:
                pad_l = 0

            if pad_r < 0:
                pad_r = 0

        # caculate output size in VALID mode
        if self.padding == "VALID":
            # Hout = ceil(Hi - Fh + 1, Sh), Wout = ceil(Wi - Fw + 1, Sw)
            out_size_h = (self.in_size_h - self.window_h + 1 +
                          (self.stride_h - 1)) // self.stride_h
            out_size_w = (self.in_size_w - self.window_w + 1 +
                          (self.stride_w - 1)) // self.stride_w
        pad = (pad_l, pad_r, pad_t, pad_b)

        return pad, out_size_h, out_size_w

    def _calc_max_fun(self, data_input, data_input_ub, index_w, index_h):
        """
        caculate max of data_input

        Parameters
        ----------
        data_input: input data
        data_input_ub: input data in ub
        index_w: input size in w direction
        index_h: input size in h direction

        Returns
        -------
        data_input: output tensor
        """
        self.tik_instance.vmax(
            MASK, data_input[index_h * 256], data_input[index_h * 256],
            data_input_ub[index_w * 256 + index_h * self.fmap_img2col_w * 256],
            REPEAT_2, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0, DSTSTRIDEM1,
            SRC0STRIDEM1, SRC1STRIDEM1)
        return data_input

    def _load3d_fm_to_ub(self, ub_buff, l1_buff, i_dim_w, i_dim_h):
        instance = self.tik_instance
        filter_size = self.window_h * self.window_w
        with instance.for_range(0, filter_size, thread_num=1) as loopk:
            k_dim_w = loopk % 3
            k_dim_h = loopk // 3
            instance.load3dv1(ub_buff[loopk * 4 * 56 * 16], l1_buff,
                              (self.pad[0], self.pad[1], 0, 0), self.in_size_h,
                              self.in_size_w,
                              0, k_dim_w, k_dim_h, i_dim_w, i_dim_h,
                              self.stride_w, self.stride_h,
                              self.window_w, self.window_h,
                              1, 1, 1, 1, 4 * 56 // 16, 0, self.pad_value)

    # pylint: disable=too-many-locals, too-many-statements
    def tik_instance_function(self, kernel_name):
        """
        implementation of max_pool_with_argmax and return the tik instance
        :param kernel_name: the kernel's name
        :return: tik instance
        """
        dtype = self.input_dtype
        input_shape = self.input_shape
        batch_size = input_shape[0]
        c1_dim = input_shape[1]
        c0_dim = 16
        window_h = self.window_h
        window_w = self.window_w
        filter_size = window_h * window_w
        output_h = self.out_size_h
        output_w = self.out_size_w
        input_h, input_w = input_shape[2:4]
        output_block_h = 4
        loop_h = output_h // output_block_h

        instance = self.tik_instance

        mask_one_window = ((output_h * output_w + 15) // 16 + 1) * 16

        mask_gap_element = (mask_one_window -
                            output_block_h * output_w)
        mask_gap = mask_gap_element * 2 // 32

        output_gm_shape = (batch_size, c1_dim, output_h, output_w, 16)
        output_mask_shape = (batch_size, c1_dim, filter_size,
                             mask_one_window // 16, 16)
        input_fmap_gm = instance.Tensor(dtype, input_shape,
                                        name="input_fmap_gm",
                                        scope=tik.scope_gm)
        output_max_gm = instance.Tensor(dtype, output_gm_shape,
                                        name="output_max_gm",
                                        scope=tik.scope_gm)
        output_mask_gm = instance.Tensor("uint16", output_mask_shape,
                                         name="output_mask_gm",
                                         scope=tik.scope_gm)

        l1_buff0_size = input_h * input_w * c0_dim + 32 * 1024
        l1_buff0 = instance.Tensor(dtype, (l1_buff0_size,), name="l1_buff0",
                                   scope=tik.scope_cbuf)

        ub_max_buff_size = self.stride_h * output_block_h * output_w * c0_dim
        ub_max_buff = instance.Tensor(dtype, (ub_max_buff_size,),
                                      name="ub_max_buff", scope=tik.scope_ubuf)

        ub_mask_buff_size = 8 * 1024
        ub_mask_buff = instance.Tensor("uint16", (ub_mask_buff_size,),
                                       name="ub_mask_buff",
                                       scope=tik.scope_ubuf)
        ub_mask_temp = instance.Tensor("uint16", (ub_mask_buff_size,),
                                       name="ub_mask_temp",
                                       scope=tik.scope_ubuf)
        ub_mask_or_buff = instance.Tensor("uint16", (ub_mask_buff_size,),
                                          name="ub_mask_or_buff",
                                          scope=tik.scope_ubuf)
        ub_mask_not_buff = instance.Tensor("uint16", (ub_mask_buff_size,),
                                           name="ub_mask_not_buff",
                                           scope=tik.scope_ubuf)

        ub_buff_size = output_block_h * output_w * c0_dim * filter_size
        ub_buff0 = instance.Tensor(dtype, (ub_buff_size,),
                                   name="ub_buff0", scope=tik.scope_ubuf)
        ub_buff1 = instance.Tensor(dtype, (ub_buff_size,),
                                   name="ub_buff1", scope=tik.scope_ubuf)

        with instance.for_range(0, batch_size, block_num=batch_size) as batch:
            with instance.for_range(0, c1_dim) as loopc:
                input_idx = instance.Scalar("uint64", name="input_idx")
                input_idx.set_as(batch * c1_dim * input_h * input_w * c0_dim +
                                 loopc * input_h * input_w * c0_dim)

                output_idx = instance.Scalar("uint64", name="output_idx")
                output_idx.set_as(batch*c1_dim*output_h*output_w*c0_dim +
                                  loopc*output_h*output_w*c0_dim)

                l1_buff0_idx = instance.Scalar("uint64", name="l1_buff0_idx")
                l1_buff0_idx.set_as(0)

                output_mask_idx = instance.Scalar("uint64",
                                                  name="output_mask_idx")
                output_mask_idx.set_as(batch*c1_dim*mask_one_window*filter_size
                                       + loopc*mask_one_window*filter_size)

                with instance.for_range(0, loop_h // 2) as looph:
                    # ping------------------------------------------------------
                    load_fm_size = instance.Scalar("uint64")
                    load_fm_size.set_as(0)
                    with instance.if_scope(looph == 0):
                        load_fm_size.set_as(
                            (output_block_h * self.stride_h + 1) *
                            input_w)
                    with instance.else_scope():
                        load_fm_size.set_as(
                            output_block_h * self.stride_h * input_w)
                    instance.data_move(l1_buff0[l1_buff0_idx],
                                       input_fmap_gm[input_idx],
                                       0, 1, load_fm_size, 0, 0)

                    ub_buff = ub_buff0
                    self._load3d_fm_to_ub(ub_buff, l1_buff0, 0,
                                          looph * 2 * output_block_h *
                                          self.stride_h)

                    repeat_times = (output_block_h * output_w * c0_dim //
                                    constant.MASK128)
                    repeat_stride = constant.MASK128 * 2 // 32
                    instance.vmax(constant.MASK128, ub_max_buff, ub_buff,
                                  ub_buff[output_block_h * output_w * c0_dim],
                                  repeat_times, 1, 1, 1,
                                  repeat_stride, repeat_stride, repeat_stride)

                    with instance.for_range(2, filter_size) as idx:
                        instance.vmax(constant.MASK128, ub_max_buff,
                                      ub_max_buff,
                                      ub_buff[output_block_h * output_w *
                                              c0_dim * idx],
                                      repeat_times, 1, 1, 1,
                                      repeat_stride, repeat_stride,
                                      repeat_stride)

                    output_idx_tmp = (output_idx +
                                      looph * 2 * output_block_h *
                                      output_w * c0_dim)
                    instance.data_move(output_max_gm[output_idx_tmp],
                                       ub_max_buff, 0, 1,
                                       output_block_h * output_w *
                                       c0_dim * 2 // 32,
                                       0, 0)

                    with instance.for_range(0, filter_size) as idx:
                        instance.vcmpv_eq(ub_mask_buff[idx * output_block_h *
                                                       output_w * c0_dim // 16],
                                          ub_buff[idx * output_block_h *
                                                  output_w * c0_dim],
                                          ub_max_buff, repeat_times,
                                          1, 1, repeat_stride, repeat_stride)

                    repeat_times = math.ceil(
                        output_block_h * output_w / constant.MASK128)
                    repeat_stride = constant.MASK128 * 2 // 32
                    instance.vnot(constant.MASK128, ub_mask_not_buff,
                                  ub_mask_buff, repeat_times, 1, 1,
                                  repeat_stride,
                                  repeat_stride)
                    instance.vor(constant.MASK128, ub_mask_or_buff,
                                 ub_mask_buff,
                                 ub_mask_buff[output_block_h * output_w],
                                 repeat_times, 1, 1, 1,
                                 repeat_stride, repeat_stride, repeat_stride)
                    instance.vand(constant.MASK128, ub_mask_temp,
                                  ub_mask_not_buff,
                                  ub_mask_buff[output_block_h * output_w],
                                  repeat_times, 1, 1, 1,
                                  repeat_stride, repeat_stride, repeat_stride)

                    with instance.for_range(2, filter_size) as idx:
                        instance.vnot(constant.MASK128, ub_mask_not_buff,
                                      ub_mask_or_buff, repeat_times, 1, 1,
                                      repeat_stride, repeat_stride)
                        instance.vor(constant.MASK128, ub_mask_or_buff,
                                     ub_mask_or_buff,
                                     ub_mask_buff[
                                         idx * output_block_h * output_w],
                                     repeat_times, 1, 1, 1,
                                     repeat_stride, repeat_stride,
                                     repeat_stride)
                        instance.vand(constant.MASK128,
                                      ub_mask_temp[(idx - 1) *
                                                   output_block_h * output_w],
                                      ub_mask_not_buff,
                                      ub_mask_buff[
                                          idx * output_block_h * output_w],
                                      repeat_times, 1, 1, 1,
                                      repeat_stride, repeat_stride,
                                      repeat_stride)
                    instance.data_move(
                        output_mask_gm[output_mask_idx],
                        ub_mask_buff, 0, 1, output_block_h * output_w // c0_dim,
                        0, 0)
                    instance.data_move(
                        output_mask_gm[output_mask_idx + mask_one_window],
                        ub_mask_temp,
                        0,
                        filter_size - 1,
                        output_block_h * output_w // c0_dim,
                        0, mask_gap)

                    input_idx.set_as(input_idx + load_fm_size * c0_dim)
                    output_mask_idx.set_as(output_mask_idx +
                                           output_block_h * output_w *
                                           c0_dim // 16)
                    l1_buff0_idx.set_as(l1_buff0_idx + load_fm_size * 16)
                    # pong------------------------------------------------------
                    load_fm_size = instance.Scalar("uint64")
                    load_fm_size.set_as(0)

                    with instance.if_scope(looph == loop_h // 2 - 1):
                        load_fm_size.set_as(
                            (output_block_h * self.stride_h - 1) * input_w)
                    with instance.else_scope():
                        load_fm_size.set_as(
                            output_block_h * self.stride_h * input_w)
                    instance.data_move(l1_buff0[l1_buff0_idx],
                                       input_fmap_gm[input_idx],
                                       0, 1, load_fm_size, 0, 0)

                    ub_buff = ub_buff1

                    self._load3d_fm_to_ub(ub_buff, l1_buff0, 0,
                                          (looph * 2 + 1) * output_block_h *
                                          self.stride_h)

                    repeat_times = (output_block_h * output_w * c0_dim //
                                    constant.MASK128)
                    repeat_stride = constant.MASK128 * 2 // 32
                    instance.vmax(constant.MASK128, ub_max_buff, ub_buff,
                                  ub_buff[output_block_h * output_w * c0_dim],
                                  repeat_times, 1, 1, 1,
                                  repeat_stride, repeat_stride, repeat_stride)

                    with instance.for_range(2, filter_size) as idx:
                        instance.vmax(constant.MASK128, ub_max_buff,
                                      ub_max_buff,
                                      ub_buff[output_block_h * output_w *
                                              c0_dim * idx],
                                      repeat_times, 1, 1, 1,
                                      repeat_stride, repeat_stride,
                                      repeat_stride)

                    output_idx_tmp = (output_idx +
                                      (looph * 2 + 1) * output_block_h *
                                      output_w * c0_dim)
                    instance.data_move(output_max_gm[output_idx_tmp],
                                       ub_max_buff, 0, 1, output_block_h *
                                       output_w * c0_dim * 2 // 32,
                                       0, 0)

                    with instance.for_range(0, filter_size) as idx:
                        instance.vcmpv_eq(ub_mask_buff[idx * output_block_h *
                                                       output_w * c0_dim // 16],
                                          ub_buff[idx * output_block_h *
                                                  output_w * c0_dim],
                                          ub_max_buff, repeat_times,
                                          1, 1, repeat_stride, repeat_stride)

                    repeat_times = math.ceil(
                        output_block_h * output_w / constant.MASK128)
                    repeat_stride = constant.MASK128 * 2 // 32
                    instance.vnot(constant.MASK128, ub_mask_not_buff,
                                  ub_mask_buff, repeat_times, 1, 1,
                                  repeat_stride,
                                  repeat_stride)
                    instance.vor(constant.MASK128, ub_mask_or_buff,
                                 ub_mask_buff,
                                 ub_mask_buff[output_block_h * output_w],
                                 repeat_times, 1, 1, 1,
                                 repeat_stride, repeat_stride, repeat_stride)
                    instance.vand(constant.MASK128, ub_mask_temp,
                                  ub_mask_not_buff,
                                  ub_mask_buff[output_block_h * output_w],
                                  repeat_times, 1, 1, 1,
                                  repeat_stride, repeat_stride, repeat_stride)

                    with instance.for_range(2, filter_size) as idx:
                        instance.vnot(constant.MASK128, ub_mask_not_buff,
                                      ub_mask_or_buff, repeat_times, 1, 1,
                                      repeat_stride, repeat_stride)
                        instance.vor(constant.MASK128, ub_mask_or_buff,
                                     ub_mask_or_buff,
                                     ub_mask_buff[
                                         idx * output_block_h * output_w],
                                     repeat_times, 1, 1, 1,
                                     repeat_stride, repeat_stride,
                                     repeat_stride)
                        instance.vand(constant.MASK128,
                                      ub_mask_temp[(idx - 1) * output_block_h *
                                                   output_w],
                                      ub_mask_not_buff,
                                      ub_mask_buff[
                                          idx * output_block_h * output_w],
                                      repeat_times, 1, 1, 1,
                                      repeat_stride, repeat_stride,
                                      repeat_stride)
                    instance.data_move(
                        output_mask_gm[output_mask_idx],
                        ub_mask_buff, 0, 1, output_block_h * output_w // c0_dim,
                        0, 0)
                    mask_gap_element = (mask_one_window -
                                        output_block_h * output_w)
                    mask_gap = mask_gap_element * 2 // 32
                    instance.data_move(
                        output_mask_gm[output_mask_idx + mask_one_window],
                        ub_mask_temp,
                        0,
                        filter_size - 1,
                        output_block_h * output_w // c0_dim,
                        0, mask_gap)

                    input_idx.set_as(input_idx + load_fm_size * c0_dim)
                    output_mask_idx.set_as(output_mask_idx +
                                           output_block_h * output_w *
                                           c0_dim // 16)
                    l1_buff0_idx.set_as(l1_buff0_idx + load_fm_size * 16)

        instance.BuildCCE(kernel_name=kernel_name,
                          inputs=(input_fmap_gm),
                          outputs=(output_max_gm,
                                   output_mask_gm))
        return instance


# pylint: disable=invalid-name
def is_max_pool_with_argmax_param(x, ksize, strides, padding):
    """
    test if the param suitable for this module to treat
    :param x: dict of shape and dtype of the input x
    :param ksize: value of ksize
    :param strides: value of strides
    :param padding: value of padding
    :return: Bool, if the param suitable for this module to treat return True,
             if not return False
    """
    resnet50_x = {"shape": (32, 4, 112, 112, 16), "dtype": "float16"}
    resnet50_ksize = [1, 3, 3, 1]
    resnet50_strides = [1, 2, 2, 1]
    resnet50_padding = "SAME"

    def is_valid_shape(resnet50shape, shape):
        if shape.get("dtype") != resnet50shape.get("dtype"):
            return False

        if len(shape.get("shape")) != len(resnet50shape.get("shape")):
            return False

        resnet50_last3dims = resnet50shape.get("shape")[2:]
        last3dims = shape.get("shape")[2:]

        return list(resnet50_last3dims) == list(last3dims)

    ksize = list(ksize)
    strides = list(strides)

    if (resnet50_ksize == ksize and resnet50_strides == strides and
            resnet50_padding == padding and
            is_valid_shape(resnet50_x, x)):
        return True

    return False


# pylint: disable=invalid-name
def max_pool_with_argmax(x, ksize, strides, padding, kernel_name):
    """
    implementation of max_pool_with_argmax and return the tik instance
    :param x: dict of shape and dtype of the input x
    :param ksize: value of strides
    :param strides: value of strides
    :param padding: value of padding
    :param kernel_name: the kernel's name
    :return: tik instance
    """
    max_pool_grad = MaxPoolWithargmaxResnet50(x, ksize, strides, padding)
    return max_pool_grad.tik_instance_function(kernel_name)
