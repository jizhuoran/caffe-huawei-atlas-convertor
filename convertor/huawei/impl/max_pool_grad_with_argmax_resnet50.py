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

from te import tik
from te import platform as tbe_platform
from impl import common_util
from impl import constant_util as constant

# size of vector calc one repeat
ONE_REPEAT = 256
# max repeat of vector calc
V_MAX_REPEAT = 255
# max num of fp16 in one repeat
FP16_MAX = 128
# max num of fp32 in one repeat
FP32_MAX = 64
# max num of fp16 mask handle one time
MASK_MAX = 8


# pylint: disable=locally-disabled,too-few-public-methods,
# pylint: disable=too-many-instance-attributes
class MaxpoolGradResnet50():
    """
    parameter for max_pool_grad_with_pool
    """

    # pylint: disable=locally-disabled,too-many-locals,too-many-arguments
    def __init__(self, grad, argmax, input_x, ksize, strides, padding):
        """
        init compare and bit pack base parameters
        Parameters
        ----------
        input_x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4,
                 just like [1, poolingStrideH, poolingStrideW, 1]
        padding: pad mode, just support "SANME" or "VALID"
        Returns
        -------
        None
        """
        self.blocknum = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

        self.input_gard_shape = grad.get("shape")
        self.argmax_shape = argmax.get("shape")
        self.y_shape = input_x.get("shape")
        self.dtype = grad.get("dtype").lower()
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.nc1 = 1
        self.block = self.input_gard_shape[0] * self.input_gard_shape[1]
        self.tik_instance = tik.Tik()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        dyh, dyw = self.input_gard_shape[2:4]
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        windowh, windoww = self.ksize[1:3]
        pad_top = 0
        pad_bottom = 0
        pad_right = 0
        pad_left = 0
        if padding == "SAME":
            padh = (dyh - 1) * strideh + windowh - dxh
            padw = (dyw - 1) * stridew + windoww - dxw
            padh = max([padh, 0])
            padw = max([padw, 0])
            pad_top = padh // 2
            pad_bottom = padh - pad_top
            pad_left = padw // 2
            pad_right = padw - pad_left
        self.pad = (pad_top, pad_bottom, pad_left, pad_right)

        self.hoverlap = 0
        if windowh > strideh:
            self.hoverlap = windowh - strideh
        self.woverlap = 0
        if windoww > stridew:
            self.woverlap = windoww - stridew

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals
    # pylint: disable=too-many-statements
    def tik_instance_cut_nc1_cut_h(self, kernel_name):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        batch_num = self.input_gard_shape[0]
        block_num = self.input_gard_shape[0]
        c0_dim = 16
        c1_dim = self.input_gard_shape[1]
        input_h = 56
        input_w = 56
        input_c = c0_dim * c1_dim
        output_h = 112
        output_w = 112
        output_c = c0_dim * c1_dim

        input_size = input_h * input_w * input_c
        output_size = output_h * output_w * output_c

        loop_c = self.input_gard_shape[1]
        block_h = 4
        loop_h = input_h // block_h
        windowh, windoww = self.ksize[1:3]
        filter_size = windowh * windoww
        input_block_size = block_h * input_w * c0_dim
        output_block_size = block_h * 2 * output_w * c0_dim
        mask_size = input_block_size // 16

        dtype = self.dtype
        batch, channel1, dyh, dyw, _ = self.input_gard_shape

        data_input = self.tik_instance.Tensor(dtype, self.input_gard_shape,
                                              name="data_input",
                                              scope=tik.scope_gm)

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16
        data_mask = self.tik_instance.Tensor("uint16",
                                             (batch * channel1 *
                                              filter_size * mask_one_window,),
                                             name="data_mask",
                                             scope=tik.scope_gm)
        data_output = self.tik_instance.Tensor(dtype, self.y_shape,
                                               name="data_output",
                                               scope=tik.scope_gm)
        data_input_origin = self.tik_instance.Tensor(dtype, self.y_shape,
                                                     name="data_input_origin",
                                                     scope=tik.scope_gm)

        grad_ub_size = 4 * 2 * 112 * 16
        ub_grad_buf0 = self.tik_instance.Tensor(dtype, (grad_ub_size,),
                                                name="ub_grad_buf0",
                                                scope=tik.scope_ubuf)
        ub_grad_buf1 = self.tik_instance.Tensor(dtype, (grad_ub_size,),
                                                name="ub_grad_buf1",
                                                scope=tik.scope_ubuf)

        select_ub_size = input_block_size
        ub_select_fp16_buf = self.tik_instance.Tensor(dtype, (select_ub_size,),
                                                      name="ub_select_fp16_buf",
                                                      scope=tik.scope_ubuf)

        max_pool_ub_size = input_block_size
        maxpool_ub_input_buf0 = self.tik_instance.Tensor(dtype,
                                                         (max_pool_ub_size,),
                                                         name="maxpool_ub0",
                                                         scope=tik.scope_ubuf)
        maxpool_ub_input_buf1 = self.tik_instance.Tensor(dtype,
                                                         (max_pool_ub_size,),
                                                         name="maxpool_ub1",
                                                         scope=tik.scope_ubuf)

        mask_ub_size = 9 * 4 * 56 * 16 // 16
        ub_loc_mask_buf0 = self.tik_instance.Tensor("uint16", (mask_ub_size,),
                                                    name="ub_loc_mask_buf0",
                                                    scope=tik.scope_ubuf)
        ub_loc_mask_buf1 = self.tik_instance.Tensor("uint16", (mask_ub_size,),
                                                    name="ub_loc_mask_buf1",
                                                    scope=tik.scope_ubuf)

        ub_zero_buf = self.tik_instance.Tensor(dtype, (128,),
                                               name="ub_zero_buf",
                                               scope=tik.scope_ubuf)

        ub_select_fp32_buf = self.tik_instance.Tensor("float32",
                                                      (select_ub_size,),
                                                      name="ub_select_fp32_buf",
                                                      scope=tik.scope_ubuf)

        fm_f32_ub_size = (4 * 2 + 1) * 112 * 16
        ub_fm_fp32_buf = self.tik_instance.Tensor("float32", (fm_f32_ub_size,),
                                                  name="ub_fm_fp32_buf",
                                                  scope=tik.scope_ubuf)

        fm_f32_tail_ub_size = 1 * 112 * 16
        ub_fm_fp32_tail_buf = self.tik_instance.Tensor("float32",
                                                       (fm_f32_tail_ub_size,),
                                                       name="ub_fm_fp32_tail",
                                                       scope=tik.scope_ubuf)

        self.tik_instance.data_move(ub_zero_buf[0],
                                    data_input_origin[0],
                                    constant.SID,
                                    constant.DEFAULT_NBURST,
                                    constant.DEFAULT_BURST_LEN,
                                    constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)

        with self.tik_instance.for_range(0, batch_num,
                                         block_num=block_num) as batch:
            # avoid tik take data_vsel_ub_zero as a input
            self.tik_instance.vector_dup(constant.MASK128, ub_zero_buf, 0.0, 1,
                                         1, 8)
            with self.tik_instance.for_range(0, loop_c) as loopc:
                with self.tik_instance.for_range(0, loop_h // 2) as looph:
                    # ping
                    data_max_pool_input_ub = maxpool_ub_input_buf0
                    data_mask_ub = ub_loc_mask_buf0
                    data_grad_ub = ub_grad_buf0

                    self.tik_instance.data_move(data_max_pool_input_ub[0],
                                                data_input[batch * input_size +
                                                           loopc * loop_h *
                                                           input_block_size +
                                                           looph * 2 *
                                                           input_block_size],
                                                0, 1, input_block_size // 16, 0,
                                                0)

                    self.tik_instance.data_move(data_mask_ub[0],
                                                data_mask[batch * loop_c *
                                                          mask_one_window *
                                                          filter_size +
                                                          loopc *
                                                          mask_one_window *
                                                          filter_size +
                                                          looph * 2 *
                                                          mask_size],
                                                0, filter_size,
                                                input_block_size // (8 * 32),
                                                (mask_one_window * c0_dim -
                                                 input_block_size) //
                                                (8 * 32),
                                                0)
                    self.tik_instance.vector_dup(constant.MASK64,
                                                 ub_fm_fp32_buf, 0.0,
                                                 ub_fm_fp32_buf.shape[0] //
                                                 constant.MASK64, 1, 8)

                    with self.tik_instance.if_scope(looph > 0):
                        self.tik_instance.vmuls(constant.MASK64,
                                                ub_fm_fp32_buf[0],
                                                ub_fm_fp32_tail_buf[0], 1.0, 14,
                                                2, 2, 16, 16)
                        self.tik_instance.vmuls(constant.MASK64,
                                                ub_fm_fp32_buf[8],
                                                ub_fm_fp32_tail_buf[8], 1.0, 14,
                                                2, 2, 16, 16)

                    with self.tik_instance.for_range(0, filter_size) as flt_idx:
                        with self.tik_instance.for_range(0,
                                                         input_block_size // 128
                                                         ) as r_idx:
                            cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                data_mask_ub[flt_idx * mask_size + r_idx * 8])
                            self.tik_instance.vsel(constant.MASK128, 0,
                                                   ub_select_fp16_buf[
                                                       r_idx * 128],
                                                   cmpmask,
                                                   data_max_pool_input_ub[
                                                       r_idx * 128],
                                                   ub_zero_buf[0],
                                                   1, 1, 1, 0, 8, 8, 0)
                        self.tik_instance.vconv(constant.MASK64, "",
                                                ub_select_fp32_buf,
                                                ub_select_fp16_buf,
                                                input_block_size // 64, 1, 1,
                                                8, 4)
                        with self.tik_instance.for_range(0, block_h) as h_idx:
                            fm_ub_idx = flt_idx // 3 * output_w * c0_dim + \
                                        flt_idx % 3 * c0_dim + \
                                        output_w * c0_dim * 2 * h_idx
                            select_ub_idx = input_w * c0_dim * h_idx
                            self.tik_instance.vadd(constant.MASK64,
                                                   ub_fm_fp32_buf[fm_ub_idx],
                                                   ub_fm_fp32_buf[fm_ub_idx],
                                                   ub_select_fp32_buf[
                                                       select_ub_idx],
                                                   7, 4, 4, 2, 32, 32, 16)
                            self.tik_instance.vadd(constant.MASK64,
                                                   ub_fm_fp32_buf[
                                                       fm_ub_idx + 8],
                                                   ub_fm_fp32_buf[
                                                       fm_ub_idx + 8],
                                                   ub_select_fp32_buf[
                                                       select_ub_idx + 8],
                                                   7, 4, 4, 2, 32, 32, 16)

                    self.tik_instance.vconv(constant.MASK64, "", data_grad_ub,
                                            ub_fm_fp32_buf,
                                            output_block_size // 64,
                                            1, 1, 4, 8)

                    data_output_idx = batch * output_size + loopc * loop_h * \
                                      output_block_size + \
                                      output_block_size * looph * 2
                    self.tik_instance.data_move(data_output[data_output_idx],
                                                data_grad_ub, 0, 1,
                                                output_block_size // 16, 0, 0)

                    self.tik_instance.vmuls(constant.MASK64,
                                            ub_fm_fp32_tail_buf,
                                            ub_fm_fp32_buf[output_block_size],
                                            1.0, 14, 2, 2, 16, 16)
                    self.tik_instance.vmuls(constant.MASK64,
                                            ub_fm_fp32_tail_buf[8],
                                            ub_fm_fp32_buf[
                                                output_block_size + 8],
                                            1.0, 14, 2, 2, 16, 16)

                    # pong
                    data_max_pool_input_ub = maxpool_ub_input_buf1
                    data_mask_ub = ub_loc_mask_buf1
                    data_grad_ub = ub_grad_buf1

                    self.tik_instance.data_move(data_max_pool_input_ub[0],
                                                data_input[batch * input_size +
                                                           loopc * loop_h *
                                                           input_block_size +
                                                           (looph * 2 + 1) *
                                                           input_block_size],
                                                0, 1, input_block_size // 16,
                                                0, 0)
                    self.tik_instance.data_move(data_mask_ub[0],
                                                data_mask[batch *
                                                          (loop_c *
                                                           filter_size *
                                                           mask_one_window) +
                                                          loopc *
                                                          mask_one_window *
                                                          filter_size +
                                                          (looph * 2 + 1) *
                                                          mask_size],
                                                0, filter_size,
                                                input_block_size // (8 * 32),
                                                (mask_one_window * c0_dim -
                                                 input_block_size) // (8 * 32),
                                                0)
                    self.tik_instance.vector_dup(constant.MASK64,
                                                 ub_fm_fp32_buf, 0.0,
                                                 ub_fm_fp32_buf.shape[0] //
                                                 constant.MASK64, 1, 8)

                    self.tik_instance.vmuls(constant.MASK64, ub_fm_fp32_buf[0],
                                            ub_fm_fp32_tail_buf[0], 1.0, 14,
                                            2, 2, 16, 16)
                    self.tik_instance.vmuls(constant.MASK64, ub_fm_fp32_buf[8],
                                            ub_fm_fp32_tail_buf[8], 1.0, 14,
                                            2, 2, 16, 16)

                    with self.tik_instance.for_range(0, filter_size) as flt_idx:
                        with self.tik_instance.for_range(0,
                                                         input_block_size //
                                                         128) as r_idx:
                            cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                data_mask_ub[flt_idx * mask_size + r_idx * 8])
                            self.tik_instance.vsel(constant.MASK128, 0,
                                                   ub_select_fp16_buf[
                                                       r_idx * 128],
                                                   cmpmask,
                                                   data_max_pool_input_ub[
                                                       r_idx * 128],
                                                   ub_zero_buf[0],
                                                   1, 1, 1, 0, 8, 8, 0)
                        self.tik_instance.vconv(constant.MASK64, "",
                                                ub_select_fp32_buf,
                                                ub_select_fp16_buf,
                                                input_block_size // 64, 1,
                                                1, 8, 4)
                        with self.tik_instance.for_range(0, block_h) as h_idx:
                            fm_ub_idx = flt_idx // 3 * output_w * c0_dim + \
                                        flt_idx % 3 * c0_dim + \
                                        output_w * c0_dim * 2 * h_idx
                            select_ub_idx = input_w * c0_dim * h_idx
                            self.tik_instance.vadd(constant.MASK64,
                                                   ub_fm_fp32_buf[fm_ub_idx],
                                                   ub_fm_fp32_buf[fm_ub_idx],
                                                   ub_select_fp32_buf[
                                                       select_ub_idx],
                                                   7, 4, 4, 2, 32, 32, 16)
                            self.tik_instance.vadd(constant.MASK64,
                                                   ub_fm_fp32_buf[
                                                       fm_ub_idx + 8],
                                                   ub_fm_fp32_buf[
                                                       fm_ub_idx + 8],
                                                   ub_select_fp32_buf[
                                                       select_ub_idx + 8],
                                                   7, 4, 4, 2, 32, 32, 16)

                    self.tik_instance.vconv(constant.MASK64, "", data_grad_ub,
                                            ub_fm_fp32_buf,
                                            output_block_size // 64,
                                            1, 1, 4, 8)

                    data_output_idx = batch * output_size + \
                                      loopc * loop_h * output_block_size + \
                                      output_block_size * (looph * 2 + 1)
                    self.tik_instance.data_move(data_output[data_output_idx],
                                                data_grad_ub, 0, 1,
                                                output_block_size // 16, 0, 0)

                    with self.tik_instance.if_scope(looph < loop_h // 2 - 1):
                        self.tik_instance.vmuls(constant.MASK64,
                                                ub_fm_fp32_tail_buf,
                                                ub_fm_fp32_buf[
                                                    output_block_size], 1.0,
                                                14, 2, 2, 16, 16)
                        self.tik_instance.vmuls(constant.MASK64,
                                                ub_fm_fp32_tail_buf[8],
                                                ub_fm_fp32_buf[
                                                    output_block_size + 8], 1.0,
                                                14, 2, 2, 16, 16)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(data_input_origin, data_input,
                                           data_mask),
                                   outputs=(data_output), enable_l2=False)
        return self.tik_instance


# pylint: disable=invalid-name, too-many-arguments
def is_max_pool_grad_with_argmax_param(grad, argmax, x, ksize, strides,
                                       padding):
    """
    test if the param suitable for this module to treat
    :param grad: dict of shape and dtype of the input grad
    :param argmax: dict of shape and dtype of the input argmax
    :param x: dict of shape and dtype of the input x
    :param ksize: value of ksize
    :param strides: value of strides
    :param padding: value of padding
    :return: Bool, if the param suitable for this module to treat return True,
             if not return False
    """
    resnet50_grad = {"shape": (32, 4, 56, 56, 16), "dtype": "float16"}
    resnet50_argmax = {"shape": (32, 4, 9, 197, 16), "dtype": "uint16"}
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

    if resnet50_padding != padding:
        return False

    if (resnet50_ksize == ksize and resnet50_strides == strides and
            is_valid_shape(resnet50_grad, grad) and
            is_valid_shape(resnet50_argmax, argmax) and
            is_valid_shape(resnet50_x, x)):
        return True

    return False


# pylint: disable=invalid-name, too-many-arguments
def max_pool_grad_with_argmax(grad, argmax, x, ksize, strides, padding,
                              kernel_name):
    """
    implementation of max_pool_with_argmax and return the tik instance
    :param grad: dict of shape and dtype of the input grad
    :param argmax: dict of shape and dtype of the input argmax
    :param x: dict of shape and dtype of the input x
    :param ksize: value of ksize
    :param strides: value of strides
    :param padding: value of padding
    :param kernel_name: kernel's name
    :return: tik instance
    """
    max_pool_grad = MaxpoolGradResnet50(grad, argmax, x, ksize, strides,
                                        padding)
    return max_pool_grad.tik_instance_cut_nc1_cut_h(kernel_name)
