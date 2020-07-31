#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
maxpool_grad_with_argmax
"""
import math
from te import tik
from impl import constant_util as constant
from impl.max_pool_grad_with_argmax_cut_w import MaxpoolGardObject

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
BLOCK_SIZE = 32


# pylint: disable=locally-disabled,too-few-public-methods,too-many-instance-attributes
class MaxpoolGradCustom(MaxpoolGardObject):
    """
    parameter for max_pool_grad_with_pool
    """
    # pylint: disable=locally-disabled,too-many-locals,too-many-arguments,useless-super-delegation
    def __init__(self, grad, argmax, input_x, ksize, strides, padding):
        """
        init compare and bit pack base parameters
        Parameters
        ----------
        input_x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
        padding: pad mode, just support "SANME" or "VALID"
        Returns
        -------
        None
        """
        super(MaxpoolGradCustom, self).__init__(grad, argmax, input_x, ksize, strides, padding)

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def tik_instance_cut_nc1_cut_one_h(self, kernel_name):
        """
        get vector instruct repeat times

        Parameters
        ----------
        kernel_name: cce kernel name, default value is "maxpoolGradWithArgmax"
        Returns
        -------
        None
        """
        batch, channel1, dyh, dyw, channel = self.input_gard_shape
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        if strideh > dxh:
            strideh = dxh

        if stridew > dxw:
            stridew = dxw

        dtype = self.dtype
        dtype_size = self.dtype_size
        windowh, windoww = self.ksize[1:3]
        block = self.block
        pad_top = self.pad[0]
        pad_left = self.pad[2]

        hoverlap = self.hoverlap
        col2img_h = windowh
        if col2img_h < strideh:
            col2img_h = strideh
        col2img_dyw = (dyw + 15) // 16 * 16
        if self.woverlap == 0:
            col2img_w = col2img_dyw * stridew
        else:
            col2img_w = (col2img_dyw - 1) * stridew + windoww

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16

        # vector_repeat_time
        v_rep_time = col2img_dyw * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % V_MAX_REPEAT

        # when every looph move data after, then dup col2img data
        v_rep_afmv = (windowh - hoverlap) * channel *\
                     col2img_w * dtype_size * 2 // ONE_REPEAT
        v_rep_afmv_cycle = v_rep_afmv // V_MAX_REPEAT
        v_rep_afmv_last = v_rep_afmv % V_MAX_REPEAT

        v_rep_time_col = (2 * col2img_w * channel * col2img_h * \
                          dtype_size + ONE_REPEAT - 1) // ONE_REPEAT
        v_rep_cycle_col = v_rep_time_col // V_MAX_REPEAT
        v_rep_last_col = v_rep_time_col % V_MAX_REPEAT

        data_input = self.tik_instance.Tensor(dtype, self.input_gard_shape, name="data_input",
                                              scope=tik.scope_gm)
        data_mask = self.tik_instance.Tensor("uint16", (batch * channel1 * windowh * windoww *
                                                        mask_one_window,),
                                             name="data_mask", scope=tik.scope_gm)
        if self.padding == "SAME":
            data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output",
                                                   scope=tik.scope_gm)
        else:
            data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output",
                                                   scope=tik.scope_gm, is_atomic_add=True)

        data_input_origin = self.tik_instance.Tensor(dtype, self.y_shape, name="data_input_origin",
                                                     scope=tik.scope_gm)

        real_block, block_cycle, block_index = self.get_block_param(block)
        with self.tik_instance.for_range(0, real_block, block_num=real_block) as block_id:
            real_cycle = self.tik_instance.Scalar("int32")
            block_base = self.tik_instance.Scalar("int32")
            block_num = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(block_id < block_index):
                real_cycle.set_as(block_cycle + 1)
                block_base.set_as(block_id * real_cycle)
            with self.tik_instance.else_scope():
                real_cycle.set_as(block_cycle)
                block_base.set_as(block_index + block_id * block_cycle)
            with self.tik_instance.for_range(0, real_cycle) as cycle_id:
                block_num.set_as(block_base + cycle_id)
                data_vsel_scalar = self.tik_instance.Scalar(dtype)
                data_vsel_scalar.set_as(0)
                data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,),
                                                             name="data_vsel_ub_zero",
                                                             scope=tik.scope_ubuf)
                self.tik_instance.data_move(data_vsel_ub_zero[0],
                                            data_input_origin[0],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            constant.DEFAULT_BURST_LEN,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)
                # vector_dup ub every time
                dxh_address_offset = self.tik_instance.Scalar("int32")
                dxh_address_offset.set_as(0)
                dxh_calcline = self.tik_instance.Scalar("int32")
                dxh_calcline.set_as(0)

                data_max_ub = self.tik_instance.Tensor(dtype, (col2img_dyw * channel,),
                                                       name="data_max_ub",
                                                       scope=tik.scope_ubuf)
                if self.woverlap > 0 and dyw % 16 != 0 and self.padding == "VALID":
                    self.clean_max_ub(data_max_ub, dtype)
                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32",
                                             (col2img_w * channel * col2img_h + 64,),
                                             name="data_vmul_ub_col2img_fp32",
                                             scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype,
                                             (col2img_w * channel * col2img_h + 128,),
                                             name="data_vmul_ub_col2img_fp16",
                                             scope=tik.scope_ubuf)
                self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                with self.tik_instance.for_range(0, dyh) as looph:
                    # dy copy gm to ub
                    self.tik_instance.data_move(data_max_ub,
                                                data_input[(block_num * dyh + looph) *
                                                           dyw * channel],
                                                constant.SID, constant.DEFAULT_NBURST,
                                                dyw * channel * dtype_size // BLOCK_SIZE,
                                                constant.STRIDE_ZERO,
                                                constant.STRIDE_ZERO)
                    # mask define
                    data_mask_ub = self.tik_instance.Tensor("uint16", (col2img_dyw,),
                                                            name="data_mask_ub",
                                                            scope=tik.scope_ubuf)
                    with self.tik_instance.for_range(0, windowh * windoww) as mask_id:
                        # mask copy gm to ub
                        self.tik_instance.data_move(data_mask_ub,
                                                    data_mask[block_num * mask_one_window *
                                                              windoww * windowh +
                                                              looph * dyw + mask_id *
                                                              mask_one_window],
                                                    constant.SID, 1,
                                                    col2img_dyw * dtype_size // BLOCK_SIZE,
                                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)
                        data_vsel_ub = self.tik_instance.Tensor(dtype, (col2img_dyw * channel,),
                                                                name="data_vsel_ub",
                                                                scope=tik.scope_ubuf)
                        data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (col2img_dyw *
                                                                                 channel,),
                                                                     name="data_vsel_ub_fp32",
                                                                     scope=tik.scope_ubuf)
                        if v_rep_time > 0:
                            with self.tik_instance.for_range(0, v_rep_time,
                                                             thread_num=1) as cycle:
                                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                    data_mask_ub[cycle * MASK_MAX])
                                self.tik_instance.vsel(constant.MASK128, 0,
                                                       data_vsel_ub[cycle * FP16_MAX],
                                                       cmpmask,
                                                       data_max_ub[cycle * FP16_MAX],
                                                       data_vsel_ub_zero[0],
                                                       constant.REPEAT_TIME_ONCE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT)

                        # fp16 to fp32
                        if v_rep_cycle_fp32 > 0:
                            with self.tik_instance.for_range(0, v_rep_cycle_fp32,
                                                             thread_num=1) as cycle:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[cycle * V_MAX_REPEAT *
                                                                          FP32_MAX],
                                                        data_vsel_ub[cycle * V_MAX_REPEAT *
                                                                     FP16_MAX],
                                                        V_MAX_REPEAT, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT,
                                                        constant.REPEAT_STRIDE_FOUR)
                        if v_rep_last_fp32 != 0:
                            self.tik_instance.vconv(constant.MASK64, "", data_vsel_ub_fp32[
                                v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX],
                                                    data_vsel_ub[
                                                        v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX],
                                                    v_rep_last_fp32, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_EIGHT,
                                                    constant.REPEAT_STRIDE_FOUR)
                        # col2img
                        fetch_filter_w = mask_id % windoww
                        fetch_filter_h = mask_id // windoww
                        left_top_w = 0
                        left_top_h = 0
                        self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],
                                                  data_vsel_ub_fp32[0],
                                                  (0, 0, 0, 0),
                                                  col2img_h, col2img_w, fetch_filter_w,
                                                  fetch_filter_h, left_top_w, left_top_h,
                                                  stridew, strideh,
                                                  windoww, windowh, 1, 1,
                                                  col2img_dyw // 16)

                    if v_rep_cycle_col > 0:
                        with self.tik_instance.for_range(0, v_rep_cycle_col,
                                                         thread_num=1) as cycle:
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vmul_ub_col2img_fp16[
                                                        cycle * V_MAX_REPEAT * FP32_MAX],
                                                    data_vmul_ub_col2img_fp32[
                                                        cycle * V_MAX_REPEAT * FP32_MAX],
                                                    V_MAX_REPEAT, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_FOUR,
                                                    constant.REPEAT_STRIDE_EIGHT)
                    if v_rep_last_col != 0:
                        self.tik_instance.vconv(constant.MASK64, "", data_vmul_ub_col2img_fp16[
                            v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                                                data_vmul_ub_col2img_fp32[
                                                    v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                                                v_rep_last_col, constant.STRIDE_ONE,
                                                constant.STRIDE_ONE,
                                                constant.REPEAT_STRIDE_FOUR,
                                                constant.REPEAT_STRIDE_EIGHT)

                    src_address = self.tik_instance.Scalar("int32")
                    dst_address = self.tik_instance.Scalar("int32")
                    nburst = self.tik_instance.Scalar("int32")
                    burst_len = self.tik_instance.Scalar("int32")
                    src_stride = self.tik_instance.Scalar("int32")
                    dst_stride = self.tik_instance.Scalar("int32")
                    if hoverlap == 0:
                        # move ub to gm
                        src_address.set_as(pad_left * channel)
                        dst_address.set_as(block_num * dxh * dxw * channel +
                                           (looph * col2img_h - pad_top) * dxw * channel)
                        nburst.set_as(col2img_h)
                        burst_len.set_as(self.offset_w)
                        src_stride.set_as(col2img_w - self.offset_w)
                        dst_stride.set_as(dxw - self.offset_w)
                        with self.tik_instance.if_scope(looph == 0):
                            src_address.set_as(src_address + pad_top * col2img_w * channel)
                            dst_address.set_as(block_num * dxh * dxw * channel)
                            nburst.set_as(nburst - pad_top)
                            with self.tik_instance.if_scope(looph == dyh - 1):
                                with self.tik_instance.if_scope(self.padding == "SAME"):
                                    nburst.set_as(dxh)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(looph == dyh - 1):
                                with self.tik_instance.if_scope(self.padding == "SAME"):
                                    nburst.set_as(dxh - col2img_h * looph + pad_top)
                                with self.tik_instance.else_scope():
                                    nburst.set_as(windowh)
                        self.tik_instance.data_move(data_output[dst_address],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, nburst, burst_len,
                                                    src_stride, dst_stride)
                        data_clean_scalar_fp32 = self.tik_instance.Scalar("float32")
                        data_clean_scalar_fp32.set_as(0)
                        if v_rep_cycle_col > 0:
                            with self.tik_instance.for_range(0, v_rep_cycle_col,
                                                             thread_num=1) as cycle:
                                self.tik_instance.vector_dup(constant.MASK64,
                                                             data_vmul_ub_col2img_fp32[
                                                                 cycle * V_MAX_REPEAT *
                                                                 FP32_MAX],
                                                             data_clean_scalar_fp32,
                                                             V_MAX_REPEAT,
                                                             constant.STRIDE_ONE,
                                                             constant.REPEAT_STRIDE_EIGHT)
                        if v_rep_last_col != 0:
                            self.tik_instance.vector_dup(constant.MASK64,
                                                         data_vmul_ub_col2img_fp32[
                                                             v_rep_cycle_col * \
                                                             V_MAX_REPEAT * FP32_MAX],
                                                         data_clean_scalar_fp32,
                                                         v_rep_last_col,
                                                         constant.STRIDE_ONE,
                                                         constant.REPEAT_STRIDE_EIGHT)
                    else:
                        with self.tik_instance.if_scope((looph + 1) * strideh > pad_top):
                            src_address.set_as(pad_left * channel)
                            dst_address.set_as(block_num * dxh * dxw * channel +
                                               dxh_address_offset)
                            nburst.set_as(strideh)
                            with self.tik_instance.if_scope(looph * strideh < pad_top):
                                nburst.set_as((looph + 1) * strideh - pad_top)
                                src_address.set_as(src_address +
                                                   (pad_top - looph * strideh) *
                                                   col2img_w * channel)
                            with self.tik_instance.if_scope(
                                    tik.all(dxh_calcline < dxh, looph == dyh - 1)):
                                with self.tik_instance.if_scope(self.padding == "SAME"):
                                    nburst.set_as(dxh - dxh_calcline)
                                with self.tik_instance.else_scope():
                                    nburst.set_as(windowh)
                            burst_len.set_as(self.offset_w)
                            src_stride.set_as(col2img_w - self.offset_w)
                            dst_stride.set_as(dxw - self.offset_w)
                            self.tik_instance.data_move(data_output[dst_address],
                                                        data_vmul_ub_col2img_fp16[src_address],
                                                        constant.SID, nburst,
                                                        burst_len, src_stride,
                                                        dst_stride)
                            dxh_address_offset.set_as(dxh_address_offset + \
                                                      nburst * dxw * channel)
                            dxh_calcline.set_as(dxh_calcline + nburst)

                        # dma_copy ub to ub
                        self.tik_instance.data_move(data_vmul_ub_col2img_fp32[0],
                                                    data_vmul_ub_col2img_fp32[
                                                        strideh * channel * col2img_w],
                                                    constant.SID, hoverlap, 2 * col2img_w,
                                                    constant.STRIDE_ZERO,
                                                    constant.STRIDE_ZERO)
                        data_clean_scalar_fp32 = self.tik_instance.Scalar("float32")
                        data_clean_scalar_fp32.set_as(0)
                        if v_rep_afmv_cycle > 0:
                            with self.tik_instance.for_range(0, v_rep_afmv_cycle,
                                                             thread_num=1) as cycle:
                                self.tik_instance.vector_dup(constant.MASK64,
                                                             data_vmul_ub_col2img_fp32[
                                                                 hoverlap * channel *
                                                                 col2img_w +
                                                                 cycle * V_MAX_REPEAT *
                                                                 FP32_MAX],
                                                             data_clean_scalar_fp32,
                                                             V_MAX_REPEAT,
                                                             constant.STRIDE_ONE,
                                                             constant.REPEAT_STRIDE_EIGHT)
                        if v_rep_afmv_last != 0:
                            self.tik_instance.vector_dup(constant.MASK64,
                                                         data_vmul_ub_col2img_fp32[
                                                             hoverlap * channel * \
                                                             col2img_w + \
                                                             v_rep_afmv_cycle * \
                                                             V_MAX_REPEAT * FP32_MAX],
                                                         data_clean_scalar_fp32,
                                                         v_rep_afmv_last,
                                                         constant.STRIDE_ONE,
                                                         constant.REPEAT_STRIDE_EIGHT)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(data_input_origin, data_input, data_mask),
                                   outputs=(data_output), enable_l2=False)
        return self.tik_instance

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def tik_instance_cut_nc1h_cut_one_h(self, kernel_name):
        """
        function for max_pool_grad_with_pool calc for normal shape
        """
        batch, channel1, dyh, dyw, channel = self.input_gard_shape
        dxh, dxw = self.y_shape[2:4]
        strideh, stridew = self.strides[1:3]
        if strideh > dxh:
            strideh = dxh

        if stridew > dxw:
            stridew = dxw

        dtype = self.dtype
        dtype_size = self.dtype_size
        windowh, windoww = self.ksize[1:3]
        pad_top, pad_bottom, pad_left = self.pad[0:3]
        hoverlap = self.hoverlap
        woverlap = self.woverlap

        ho_count = math.ceil(self.blocknum // (batch * channel1))
        if hoverlap == 0:
            ho_every = dyh // ho_count
            ho_last = dyh - ho_every * (ho_count - 1)
        else:
            ho_every = (dyh + ho_count - 1) // ho_count
            if ho_every == 1:
                ho_count = ho_count // 2
                ho_every = (dyh + ho_count - 1) // ho_count
            ho_last = dyh + ho_count - 1 - ho_every * (ho_count - 1)
        all_blocknum = ho_count * batch * channel1

        wo_max = math.ceil(dyw / 16) * 16
        col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + windoww
        ho_max = 1
        col2img_h = ho_max * strideh if hoverlap == 0 else (ho_max - 1) * strideh + windowh

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16
        mask_stride = (mask_one_window - wo_max) // 16

        # vector_repeat_time
        v_rep_time = wo_max * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % V_MAX_REPEAT
        v_rep_afmv = ((col2img_h - hoverlap) * channel * col2img_w + 64) * \
                     dtype_size * 2 // ONE_REPEAT
        v_rep_afmv_cycle = v_rep_afmv // V_MAX_REPEAT
        v_rep_afmv_last = v_rep_afmv % V_MAX_REPEAT

        data_input = self.tik_instance.Tensor(dtype, self.input_gard_shape, name="data_input",
                                              scope=tik.scope_gm)
        data_mask = self.tik_instance.Tensor("uint16", (batch * channel1 * windowh * windoww * \
                                                        mask_one_window,),
                                             name="data_mask", scope=tik.scope_gm)
        if self.padding == "SAME":
            data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output",
                                                   scope=tik.scope_gm)
        else:
            data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output",
                                                   scope=tik.scope_gm, is_atomic_add=True)

        data_input_origin = self.tik_instance.Tensor(dtype, self.y_shape, name="data_input_origin",
                                                     scope=tik.scope_gm)

        real_block, block_cycle, block_index = self.get_block_param(all_blocknum)
        with self.tik_instance.for_range(0, real_block, block_num=real_block) as block_id:
            real_cycle = self.tik_instance.Scalar("int32")
            block_base = self.tik_instance.Scalar("int32")
            block_num = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(block_id < block_index):
                real_cycle.set_as(block_cycle + 1)
                block_base.set_as(block_id * real_cycle)
            with self.tik_instance.else_scope():
                real_cycle.set_as(block_cycle)
                block_base.set_as(block_index + block_id * block_cycle)
            with self.tik_instance.for_range(0, real_cycle) as cycle_id:
                block_num.set_as(block_base + cycle_id)
                data_vsel_scalar = self.tik_instance.Scalar(dtype)
                data_vsel_scalar.set_as(0)
                data_vsel_ub_zero = self.tik_instance.Tensor(dtype, (128,),
                                                             name="data_vsel_ub_zero",
                                                             scope=tik.scope_ubuf)
                self.tik_instance.data_move(data_vsel_ub_zero[0],
                                            data_input_origin[0],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            constant.DEFAULT_BURST_LEN,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                self.clean_fp16_one_repeat(data_vsel_ub_zero, dtype)
                block_batch = self.tik_instance.Scalar("int32")
                block_batch.set_as(block_num // ho_count)
                block_h = self.tik_instance.Scalar("int32")
                block_h.set_as(block_num % ho_count)
                h_cycle = self.tik_instance.Scalar("int32")
                h_cycle.set_as(ho_every)
                with self.tik_instance.if_scope(block_h == ho_count - 1):
                    h_cycle.set_as(ho_last)
                dxh_address_offset = self.tik_instance.Scalar("int32")
                dxh_address_offset.set_as(block_batch * dxh * dxw * channel)
                if hoverlap == 0:
                    with self.tik_instance.if_scope(block_h != 0):
                        dxh_address_offset.set_as(dxh_address_offset +
                                                  (block_h * ho_every * windowh - pad_top) *
                                                  dxw * channel)
                else:
                    with self.tik_instance.if_scope(block_h != 0):
                        dxh_address_offset.set_as(dxh_address_offset +
                                                  ((block_h * (ho_every - 1) + 1) *
                                                   strideh - pad_top) * dxw * channel)
                # vector_dup ub every time
                data_max_ub = self.tik_instance.Tensor(dtype, (wo_max * channel,),
                                                       name="data_max_ub",
                                                       scope=tik.scope_ubuf)
                if self.woverlap > 0 and dyw % 16 != 0 and self.padding == "VALID":
                    self.clean_max_ub(data_max_ub, dtype)

                data_vmul_ub_col2img_fp32 = \
                    self.tik_instance.Tensor("float32",
                                             (col2img_w * channel * col2img_h + 64,),
                                             name="data_vmul_ub_col2img_fp32",
                                             scope=tik.scope_ubuf)
                data_vmul_ub_col2img_fp16 = \
                    self.tik_instance.Tensor(dtype,
                                             (col2img_w * channel * col2img_h + 128,),
                                             name="data_vmul_ub_col2img_fp16",
                                             scope=tik.scope_ubuf)
                self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)

                in_src_address = self.tik_instance.Scalar("int32")
                mask_address = self.tik_instance.Scalar("int32")
                with self.tik_instance.for_range(0, h_cycle) as looph:
                    # address  not  multiplex
                    in_src_address.set_as(block_batch * dyh * dyw * channel)
                    mask_address.set_as(block_batch * mask_one_window * windoww * windowh)
                    if hoverlap == 0:
                        in_src_address.set_as(in_src_address +
                                              (block_h * ho_every + looph) *
                                              dyw * channel)
                        mask_address.set_as(mask_address + (block_h * ho_every + looph) * dyw)
                    else:
                        in_src_address.set_as(in_src_address +
                                              (block_h * (ho_every - 1) + looph) *
                                              dyw * channel)
                        mask_address.set_as(mask_address +
                                            (block_h * (ho_every - 1) + looph) * dyw)
                    # dy copy gm to ub
                    self.tik_instance.data_move(data_max_ub,
                                                data_input[in_src_address],
                                                constant.SID, constant.DEFAULT_NBURST,
                                                dyw * channel * dtype_size // BLOCK_SIZE,
                                                constant.STRIDE_ZERO,
                                                constant.STRIDE_ZERO)
                    # mask define
                    data_mask_ub = self.tik_instance.Tensor("uint16", (wo_max *
                                                                       windowh * windoww,),
                                                            name="data_mask_ub",
                                                            scope=tik.scope_ubuf)
                    # mask copy gm to ub
                    self.tik_instance.data_move(data_mask_ub,
                                                data_mask[mask_address],
                                                constant.SID, windowh * windoww,
                                                wo_max * dtype_size // BLOCK_SIZE,
                                                mask_stride, constant.STRIDE_ZERO)

                    with self.tik_instance.for_range(0, windowh * windoww) as mask_id:
                        data_vsel_ub = self.tik_instance.Tensor(dtype, (wo_max * channel,),
                                                                name="data_vsel_ub",
                                                                scope=tik.scope_ubuf)
                        data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (wo_max *
                                                                                 channel,),
                                                                     name="data_vsel_ub_fp32",
                                                                     scope=tik.scope_ubuf)
                        if v_rep_time > 0:
                            with self.tik_instance.for_range(0, v_rep_time) as cycle:
                                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                    data_mask_ub[wo_max * mask_id +
                                                 cycle * MASK_MAX])
                                self.tik_instance.vsel(constant.MASK128, 0,
                                                       data_vsel_ub[cycle * FP16_MAX],
                                                       cmpmask,
                                                       data_max_ub[cycle * FP16_MAX],
                                                       data_vsel_ub_zero[0],
                                                       constant.REPEAT_TIME_ONCE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT)

                        # fp16 to fp32
                        if v_rep_cycle_fp32 > 0:
                            with self.tik_instance.for_range(0, v_rep_cycle_fp32) as cycle:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[cycle *
                                                                          V_MAX_REPEAT * FP32_MAX],
                                                        data_vsel_ub[cycle *
                                                                     V_MAX_REPEAT * FP32_MAX],
                                                        V_MAX_REPEAT, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_EIGHT,
                                                        constant.REPEAT_STRIDE_FOUR)
                        if v_rep_last_fp32 != 0:
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vsel_ub_fp32[v_rep_cycle_fp32 *
                                                                      V_MAX_REPEAT * FP32_MAX],
                                                    data_vsel_ub[v_rep_cycle_fp32 *
                                                                 V_MAX_REPEAT * FP32_MAX],
                                                    v_rep_last_fp32, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_EIGHT,
                                                    constant.REPEAT_STRIDE_FOUR)
                        # col2img
                        fetch_filter_w = mask_id % windoww
                        fetch_filter_h = mask_id // windoww
                        left_top_w = 0
                        left_top_h = 0
                        self.tik_instance.col2img(data_vmul_ub_col2img_fp32[0],
                                                  data_vsel_ub_fp32[0],
                                                  (0, 0, 0, 0),
                                                  col2img_h, col2img_w, fetch_filter_w,
                                                  fetch_filter_h, left_top_w, left_top_h,
                                                  stridew, strideh,
                                                  windoww, windowh, 1, 1,
                                                  wo_max // 16)
                    output_cuthline = self.tik_instance.Scalar("int32")
                    output_cuthline.set_as(strideh)
                    src_address = self.tik_instance.Scalar("int32")
                    src_address.set_as(pad_left * channel)
                    if hoverlap == 0:
                        with self.tik_instance.if_scope(block_h == 0):
                            with self.tik_instance.if_scope(looph == 0):
                                output_cuthline.set_as(output_cuthline - pad_top)
                        with self.tik_instance.if_scope(block_h == ho_count - 1):
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                output_cuthline.set_as(output_cuthline - pad_bottom)
                    else:
                        with self.tik_instance.if_scope(block_h == 0):
                            with self.tik_instance.if_scope(looph == 0):
                                output_cuthline.set_as(output_cuthline - pad_top)
                                src_address.set_as(src_address + pad_top * col2img_w * channel)
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(looph == 0):
                                output_cuthline.set_as(0)
                        with self.tik_instance.if_scope(block_h == ho_count - 1):
                            with self.tik_instance.if_scope(looph == h_cycle - 1):
                                output_cuthline.set_as(windowh - pad_bottom)
                    # fp32 to fp16
                    v_rep_time_col = 2 * (col2img_w * channel * col2img_h + 64) *\
                                     dtype_size // ONE_REPEAT
                    v_rep_cycle_col = v_rep_time_col // V_MAX_REPEAT
                    v_rep_last_col = v_rep_time_col % V_MAX_REPEAT
                    if v_rep_cycle_col > 0:
                        with self.tik_instance.for_range(0, v_rep_cycle_col,
                                                         thread_num=1) as cycle:
                            self.tik_instance.vconv(constant.MASK64, "",
                                                    data_vmul_ub_col2img_fp16[
                                                        cycle * V_MAX_REPEAT * FP32_MAX],
                                                    data_vmul_ub_col2img_fp32[
                                                        cycle * V_MAX_REPEAT * FP32_MAX],
                                                    V_MAX_REPEAT, constant.STRIDE_ONE,
                                                    constant.STRIDE_ONE,
                                                    constant.REPEAT_STRIDE_FOUR,
                                                    constant.REPEAT_STRIDE_EIGHT)
                    if v_rep_last_col != 0:
                        self.tik_instance.vconv(constant.MASK64, "", data_vmul_ub_col2img_fp16[
                            v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                                                data_vmul_ub_col2img_fp32[
                                                    v_rep_cycle_col * V_MAX_REPEAT * FP32_MAX],
                                                v_rep_last_col, constant.STRIDE_ONE,
                                                constant.STRIDE_ONE,
                                                constant.REPEAT_STRIDE_FOUR,
                                                constant.REPEAT_STRIDE_EIGHT)
                    with self.tik_instance.if_scope(output_cuthline != 0):
                        self.tik_instance.data_move(data_output[dxh_address_offset],
                                                    data_vmul_ub_col2img_fp16[src_address],
                                                    constant.SID, output_cuthline,
                                                    self.offset_w, col2img_w - self.offset_w,
                                                    dxw - self.offset_w)
                        dxh_address_offset.set_as(dxh_address_offset +
                                                  output_cuthline * dxw * channel)
                    if hoverlap != 0:
                        with self.tik_instance.if_scope(looph + 1 != h_cycle):
                            self.tik_instance.data_move(data_vmul_ub_col2img_fp32[0],
                                                        data_vmul_ub_col2img_fp32[
                                                            strideh * channel * col2img_w],
                                                        constant.SID, constant.DEFAULT_NBURST,
                                                        2 * hoverlap * col2img_w,
                                                        constant.STRIDE_ZERO,
                                                        constant.STRIDE_ZERO)

                            data_clean_scalar_fp32 = self.tik_instance.Scalar("float32")
                            data_clean_scalar_fp32.set_as(0)
                            if v_rep_afmv_cycle > 0:
                                with self.tik_instance.for_range(0, v_rep_afmv_cycle) as cycle:
                                    self.tik_instance.vector_dup(constant.MASK64,
                                                                 data_vmul_ub_col2img_fp32[
                                                                     hoverlap * channel *
                                                                     col2img_w +
                                                                     cycle * V_MAX_REPEAT *
                                                                     FP32_MAX],
                                                                 data_clean_scalar_fp32,
                                                                 V_MAX_REPEAT,
                                                                 constant.STRIDE_ONE,
                                                                 constant.REPEAT_STRIDE_EIGHT)
                            if v_rep_afmv_last != 0:
                                self.tik_instance.vector_dup(constant.MASK64,
                                                             data_vmul_ub_col2img_fp32[
                                                                 hoverlap * channel *
                                                                 col2img_w +
                                                                 v_rep_afmv_cycle *
                                                                 V_MAX_REPEAT * FP32_MAX],
                                                             data_clean_scalar_fp32,
                                                             v_rep_afmv_last,
                                                             constant.STRIDE_ONE,
                                                             constant.REPEAT_STRIDE_EIGHT)
                    else:
                        self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(data_input_origin, data_input, data_mask),
                                   outputs=(data_output), enable_l2=False)

        return self.tik_instance
