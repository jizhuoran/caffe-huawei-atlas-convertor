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
from impl.max_pool_grad_with_argmax_cut_h import MaxpoolGradBase

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


class MaxpoolGardObject(MaxpoolGradBase):
    """
    parameter for max_pool_grad_with_pool
    """
    # pylint: disable=locally-disabled,too-many-arguments,useless-super-delegation
    def __init__(self, grad, argmax, input_x, ksize, strides, padding):
        """
        init compare and bit pack base parameters
        Parameters
        ----------
        x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
        padding: pad mode, just support "SANME" or "VALID"
        Returns
        -------
        None
        """
        super(MaxpoolGardObject, self).__init__(grad, argmax, input_x, ksize, strides, padding)

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def tik_instance_cut_nc1_cut_w(self, kernel_name):
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
        block = self.block
        nc1 = self.nc1
        pad_top = self.pad[0]
        pad_left = self.pad[2]
        hoverlap = self.hoverlap
        woverlap = self.woverlap

        wo_max = math.ceil(dyw / 16) * 16
        ho_min = 1 if hoverlap == 0 else 2
        ho_max = ho_min
        col2img_h = ho_max * strideh if hoverlap == 0 else (ho_max - 1) * strideh + windowh
        col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + windoww
        while col2img_w * col2img_h * channel * dtype_size > self.ub_limit:
            wo_max -= 16
            col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + windoww
        if wo_max == 0:
            raise RuntimeError(
                "The shape or ksize or stride is too large and not supported, please check!")

        h_cycle = dyh
        if woverlap == 0:
            w_cycle = math.ceil(dyw / wo_max)
        else:
            w_cycle = math.ceil((dyw - 1) / (wo_max - 1))

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16
        # vector_repeat_time
        v_rep_time = ho_max * wo_max * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32 = 2 * v_rep_time // V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32 = 2 * v_rep_time % V_MAX_REPEAT

        v_rep_time_col = (2 * (col2img_w * channel * col2img_h + 64) * dtype_size) // ONE_REPEAT
        v_rep_cycle_col = v_rep_time_col // V_MAX_REPEAT

        v_rep_last_col = v_rep_time_col % V_MAX_REPEAT

        num_one_c0 = dxh * dxw * channel
        num_one_block = nc1 * num_one_c0

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

                with self.tik_instance.for_range(0, nc1) as loopc1:
                    # vector_dup ub every time
                    dxh_address_offset = self.tik_instance.Scalar("int32")
                    dxh_address_offset.set_as(0)
                    data_max_ub = self.tik_instance.Tensor(dtype, (ho_max * wo_max * channel,),
                                                           name="data_max_ub",
                                                           scope=tik.scope_ubuf)
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
                    # mask define
                    data_mask_ub = self.tik_instance.Tensor("uint16", (ho_max * wo_max,),
                                                            name="data_mask_ub",
                                                            scope=tik.scope_ubuf)
                    new_looph = self.tik_instance.Scalar("int32")
                    new_looph.set_as(0)
                    with self.tik_instance.for_range(0, h_cycle) as looph:
                        if hoverlap == 0:
                            new_looph.set_as(looph)
                        else:
                            with self.tik_instance.if_scope(looph != 0):
                                new_looph.set_as(looph - 1)
                        new_loopw = self.tik_instance.Scalar("int32")
                        new_loopw.set_as(0)
                        in_burstlen = self.tik_instance.Scalar("int32")
                        in_burstlen.set_as(wo_max)
                        with self.tik_instance.for_range(0, w_cycle) as loopw:
                            if woverlap == 0:
                                new_loopw.set_as(loopw * wo_max)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    in_burstlen.set_as(dyw - loopw * wo_max)
                            else:
                                with self.tik_instance.if_scope(loopw != 0):
                                    new_loopw.set_as(loopw * (wo_max - 1))
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    in_burstlen.set_as(dyw - loopw * (wo_max - 1))

                            self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                            if self.woverlap > 0 and dyw % 16 != 0 and self.padding == "VALID":
                                self.clean_max_ub(data_max_ub, dtype)

                            self.tik_instance.data_move(data_max_ub,
                                                        data_input[(block_num * nc1 * dyh + loopc1 *
                                                                    dyh + new_looph) * dyw *
                                                                   channel + new_loopw * channel],
                                                        constant.SID, ho_max,
                                                        in_burstlen, dyw - in_burstlen,
                                                        wo_max - in_burstlen)

                            with self.tik_instance.for_range(0, windowh * windoww) as mask_id:
                                with self.tik_instance.for_range(0, ho_max) as cycle:
                                    # mask copy gm to ub
                                    self.tik_instance.data_move(data_mask_ub[cycle * wo_max],
                                                                data_mask[block_num * nc1 *
                                                                          mask_one_window *
                                                                          windoww * windowh +
                                                                          loopc1 *
                                                                          mask_one_window *
                                                                          windoww * windowh +
                                                                          (new_looph + cycle) *
                                                                          dyw + mask_one_window *
                                                                          mask_id + new_loopw],
                                                                constant.SID, 1,
                                                                (in_burstlen + 15) // 16, 0, 0)
                                data_vsel_ub = self.tik_instance.Tensor(dtype, (ho_max *
                                                                                wo_max * channel,),
                                                                        name="data_vsel_ub",
                                                                        scope=tik.scope_ubuf)
                                data_vsel_ub_fp32 =\
                                    self.tik_instance.Tensor("float32",
                                                             (ho_max * wo_max * channel,),
                                                             name="data_vsel_ub_fp32",
                                                             scope=tik.scope_ubuf)
                                if v_rep_time > 0:
                                    with self.tik_instance.for_range(0, v_rep_time,
                                                                     thread_num=1) as cycle:
                                        cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                            data_mask_ub[cycle * MASK_MAX])
                                        self.tik_instance.vsel(constant.MASK128, 0,
                                                               data_vsel_ub[cycle * \
                                                                            FP16_MAX],
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
                                                                data_vsel_ub_fp32[
                                                                    cycle * V_MAX_REPEAT *
                                                                    FP32_MAX],
                                                                data_vsel_ub[
                                                                    cycle * V_MAX_REPEAT *
                                                                    FP32_MAX],
                                                                V_MAX_REPEAT, constant.STRIDE_ONE,
                                                                constant.STRIDE_ONE,
                                                                constant.REPEAT_STRIDE_EIGHT,
                                                                constant.REPEAT_STRIDE_FOUR)
                                if v_rep_last_fp32 != 0:
                                    self.tik_instance.vconv(constant.MASK64, "", data_vsel_ub_fp32[
                                        v_rep_cycle_fp32 * V_MAX_REPEAT * FP32_MAX],
                                                            data_vsel_ub[
                                                                v_rep_cycle_fp32 *
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
                                                          ho_max * wo_max // 16)
                            if v_rep_cycle_col > 0:
                                with self.tik_instance.for_range(0, v_rep_cycle_col) as cycle:
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
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vmul_ub_col2img_fp16[
                                                            v_rep_cycle_col * V_MAX_REPEAT *
                                                            FP32_MAX],
                                                        data_vmul_ub_col2img_fp32[
                                                            v_rep_cycle_col * V_MAX_REPEAT *
                                                            FP32_MAX],
                                                        v_rep_last_col, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_FOUR,
                                                        constant.REPEAT_STRIDE_EIGHT)

                            nburst = self.tik_instance.Scalar("int32")
                            nburst.set_as(strideh)
                            burst_len = self.tik_instance.Scalar("int32")
                            burst_len.set_as(1)
                            src_stride = self.tik_instance.Scalar("int32")
                            src_stride.set_as(0)
                            dst_stride = self.tik_instance.Scalar("int32")
                            dst_stride.set_as(0)
                            src_address = self.tik_instance.Scalar("int32")
                            src_address.set_as(0)
                            dst_address = self.tik_instance.Scalar("int32")
                            dst_address.set_as(block_num * num_one_block + loopc1 * num_one_c0)
                            if hoverlap != 0:
                                src_address.set_as(strideh * col2img_w * channel)
                            with self.tik_instance.if_scope(looph == 0):
                                nburst.set_as(strideh - pad_top)
                                src_address.set_as(pad_top * col2img_w * channel)
                            with self.tik_instance.else_scope():
                                dst_address.set_as(dst_address +
                                                   (looph * strideh - pad_top) * dxw * channel)
                                with self.tik_instance.if_scope(looph == h_cycle - 1):
                                    nburst.set_as(dxh - looph * strideh + pad_top)


                            if woverlap == 0:
                                with self.tik_instance.if_scope(loopw == 0):
                                    burst_len.set_as(col2img_w - pad_left)
                                    src_address.set_as(src_address + pad_left * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw)
                                with self.tik_instance.else_scope():
                                    burst_len.set_as(col2img_w)
                                    dst_address.set_as(dst_address +
                                                       (loopw * col2img_w - pad_left) * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw - col2img_w * loopw + pad_left)
                            else:
                                with self.tik_instance.if_scope(loopw == 0):
                                    burst_len.set_as(stridew * wo_max - pad_left)
                                    src_address.set_as(src_address + pad_left * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw)
                                with self.tik_instance.else_scope():
                                    burst_len.set_as(stridew * (wo_max - 1))
                                    src_address.set_as(src_address + stridew * channel)
                                    dst_address.set_as(dst_address +
                                                       ((loopw - 1) * stridew * (wo_max - 1) +
                                                        stridew * wo_max - pad_left) * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw - stridew * wo_max - (w_cycle - 2) *
                                                         stridew * (wo_max - 1) + pad_left)
                            src_stride.set_as(col2img_w - burst_len)
                            dst_stride.set_as(dxw - burst_len)
                            # move ub to gm
                            self.tik_instance.data_move(data_output[dst_address],
                                                        data_vmul_ub_col2img_fp16[src_address],
                                                        constant.SID, nburst, burst_len,
                                                        src_stride, dst_stride)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(data_input_origin, data_input, data_mask),
                                   outputs=(data_output), enable_l2=False)
        return self.tik_instance

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def tik_instance_cut_nc1h_cut_w(self, kernel_name):
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

        ho_max_every = 1 if hoverlap == 0 else 2
        col2img_h_every = ho_max_every * strideh if \
            hoverlap == 0 else (ho_max_every - 1) * strideh + windowh
        ho_max_last = ho_max_every
        col2img_h_last = col2img_h_every
        if hoverlap == 0:
            h_cycle_every = math.ceil(ho_every / ho_max_every)
            h_cycle_last = math.ceil(ho_last / ho_max_last)
        else:
            h_cycle_every = (ho_every - 1 - ho_max_every) // (ho_max_every - 1) + 2
            h_cycle_last = (ho_last - 1 - ho_max_last) // (ho_max_last - 1) + 2

        wo_max = math.ceil(dyw / 16) * 16
        col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + windoww
        while col2img_w * col2img_h_every * channel * dtype_size > self.ub_limit:
            wo_max -= 16
            col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + windoww
        if wo_max == 0:
            raise RuntimeError(
                "The shape or ksize or stride is too large and not supported, please check!")

        if woverlap == 0:
            w_cycle = math.ceil(dyw / wo_max)
        else:
            w_cycle = math.ceil((dyw - 1) / (wo_max - 1))

        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16
        # vector_repeat_time
        v_rep_time_last = ho_max_last * wo_max * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32_last = 2 * v_rep_time_last // V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32_last = 2 * v_rep_time_last % V_MAX_REPEAT

        v_rep_time_col_last = (2 * (col2img_w * channel * col2img_h_last + 64) *
                               dtype_size) // ONE_REPEAT
        v_rep_cycle_col_last = v_rep_time_col_last // V_MAX_REPEAT
        v_rep_last_col_last = v_rep_time_col_last % V_MAX_REPEAT

        # vector_repeat_time
        v_rep_time_every = ho_max_every * wo_max * channel * dtype_size // ONE_REPEAT
        v_rep_cycle_fp32_every = 2 * v_rep_time_every // V_MAX_REPEAT
        # v_rep_last
        v_rep_last_fp32_every = 2 * v_rep_time_every % V_MAX_REPEAT

        v_rep_time_col_every = (2 * (col2img_w * channel * col2img_h_every + 64) *
                                dtype_size) // ONE_REPEAT
        v_rep_cycle_col_every = v_rep_time_col_every // V_MAX_REPEAT
        v_rep_last_col_every = v_rep_time_col_every % V_MAX_REPEAT

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
                with self.tik_instance.if_scope(block_h == ho_count - 1):
                    data_max_ub = self.tik_instance.Tensor(dtype, (ho_max_last * wo_max * channel,),
                                                           name="data_max_ub",
                                                           scope=tik.scope_ubuf)
                    data_vmul_ub_col2img_fp32 = \
                        self.tik_instance.Tensor("float32",
                                                 (col2img_w * channel * col2img_h_last + 64,),
                                                 name="data_vmul_ub_col2img_fp32",
                                                 scope=tik.scope_ubuf)
                    data_vmul_ub_col2img_fp16 = \
                        self.tik_instance.Tensor(dtype,
                                                 (col2img_w * channel * col2img_h_last + 128,),
                                                 name="data_vmul_ub_col2img_fp16",
                                                 scope=tik.scope_ubuf)
                    # mask define
                    data_mask_ub = self.tik_instance.Tensor("uint16", (ho_max_last * wo_max,),
                                                            name="data_mask_ub",
                                                            scope=tik.scope_ubuf)
                    new_looph = self.tik_instance.Scalar("int32")
                    new_looph.set_as(0)
                    in_nburst = self.tik_instance.Scalar("int32")
                    in_nburst.set_as(ho_max_last)
                    in_src_address = self.tik_instance.Scalar("int32")
                    mask_address = self.tik_instance.Scalar("int32")
                    with self.tik_instance.for_range(0, h_cycle_last) as looph:
                        in_src_address.set_as(block_batch * dyh * dyw * channel)
                        mask_address.set_as(block_batch * mask_one_window * windoww * windowh)
                        if hoverlap == 0:
                            new_looph.set_as(looph * ho_max_last)
                            with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                in_nburst.set_as(ho_last - looph * ho_max_last)
                            in_src_address.set_as(in_src_address +
                                                  (block_h * ho_every + new_looph) *
                                                  dyw * channel)
                            mask_address.set_as(mask_address +
                                                (block_h * ho_every + new_looph) * dyw)
                        else:
                            with self.tik_instance.if_scope(looph != 0):
                                new_looph.set_as(looph * (ho_max_last - 1))
                            with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                in_nburst.set_as(ho_last - looph * (ho_max_last - 1))
                            in_src_address.set_as(in_src_address +
                                                  (block_h * (ho_every - 1) + new_looph) *
                                                  dyw * channel)
                            mask_address.set_as(mask_address + (block_h *
                                                                (ho_every - 1) + new_looph) * dyw)
                        new_loopw = self.tik_instance.Scalar("int32")
                        new_loopw.set_as(0)
                        in_burstlen = self.tik_instance.Scalar("int32")
                        in_burstlen.set_as(wo_max)
                        with self.tik_instance.for_range(0, w_cycle) as loopw:
                            if woverlap == 0:
                                new_loopw.set_as(loopw * wo_max)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    in_burstlen.set_as(dyw - loopw * wo_max)
                            else:
                                with self.tik_instance.if_scope(loopw != 0):
                                    new_loopw.set_as(loopw * (wo_max - 1))
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    in_burstlen.set_as(dyw - loopw * (wo_max - 1))

                            self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                            if self.woverlap > 0 and dyw % 16 != 0 and self.padding == "VALID":
                                self.clean_max_ub(data_max_ub, dtype)

                            self.tik_instance.data_move(data_max_ub,
                                                        data_input[in_src_address +
                                                                   new_loopw * channel],
                                                        constant.SID, in_nburst,
                                                        in_burstlen, dyw - in_burstlen,
                                                        wo_max - in_burstlen)

                            with self.tik_instance.for_range(0, windowh * windoww) as mask_id:
                                with self.tik_instance.for_range(0, in_nburst) as cycle:
                                    # mask copy gm to ub
                                    self.tik_instance.data_move(data_mask_ub[cycle * wo_max],
                                                                data_mask[mask_address +
                                                                          cycle * dyw +
                                                                          mask_one_window *
                                                                          mask_id + new_loopw],
                                                                constant.SID, 1,
                                                                (in_burstlen + 15) // 16, 0, 0)
                                data_vsel_ub = self.tik_instance.Tensor(dtype, (ho_max_last *
                                                                                wo_max * channel,),
                                                                        name="data_vsel_ub",
                                                                        scope=tik.scope_ubuf)
                                data_vsel_ub_fp32 =\
                                    self.tik_instance.Tensor("float32",
                                                             (ho_max_last * wo_max * channel,),
                                                             name="data_vsel_ub_fp32",
                                                             scope=tik.scope_ubuf)
                                if v_rep_time_last > 0:
                                    with self.tik_instance.for_range(0, v_rep_time_last,
                                                                     thread_num=1) as cycle:
                                        cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                            data_mask_ub[cycle * MASK_MAX])
                                        self.tik_instance.vsel(constant.MASK128, 0,
                                                               data_vsel_ub[cycle * \
                                                                            FP16_MAX],
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
                                if v_rep_cycle_fp32_last > 0:
                                    with self.tik_instance.for_range(0, v_rep_cycle_fp32_last,
                                                                     thread_num=1) as cycle:
                                        self.tik_instance.vconv(constant.MASK64, "",
                                                                data_vsel_ub_fp32[
                                                                    cycle * V_MAX_REPEAT *
                                                                    FP32_MAX],
                                                                data_vsel_ub[
                                                                    cycle * V_MAX_REPEAT *
                                                                    FP32_MAX],
                                                                V_MAX_REPEAT, constant.STRIDE_ONE,
                                                                constant.STRIDE_ONE,
                                                                constant.REPEAT_STRIDE_EIGHT,
                                                                constant.REPEAT_STRIDE_FOUR)
                                if v_rep_last_fp32_last != 0:
                                    self.tik_instance.vconv(constant.MASK64, "", data_vsel_ub_fp32[
                                        v_rep_cycle_fp32_last * V_MAX_REPEAT * FP32_MAX],
                                                            data_vsel_ub[
                                                                v_rep_cycle_fp32_last *
                                                                V_MAX_REPEAT * FP32_MAX],
                                                            v_rep_last_fp32_last,
                                                            constant.STRIDE_ONE,
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
                                                          col2img_h_last, col2img_w, fetch_filter_w,
                                                          fetch_filter_h, left_top_w, left_top_h,
                                                          stridew, strideh,
                                                          windoww, windowh, 1, 1,
                                                          ho_max_last * wo_max // 16)
                            if v_rep_cycle_col_last > 0:
                                with self.tik_instance.for_range(0, v_rep_cycle_col_last) as cycle:
                                    self.tik_instance.vconv(constant.MASK64, "",
                                                            data_vmul_ub_col2img_fp16[
                                                                cycle * V_MAX_REPEAT * FP32_MAX],
                                                            data_vmul_ub_col2img_fp32[
                                                                cycle * V_MAX_REPEAT * FP32_MAX],
                                                            V_MAX_REPEAT, constant.STRIDE_ONE,
                                                            constant.STRIDE_ONE,
                                                            constant.REPEAT_STRIDE_FOUR,
                                                            constant.REPEAT_STRIDE_EIGHT)
                            if v_rep_last_col_last != 0:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vmul_ub_col2img_fp16[
                                                            v_rep_cycle_col_last *
                                                            V_MAX_REPEAT * FP32_MAX],
                                                        data_vmul_ub_col2img_fp32[
                                                            v_rep_cycle_col_last *
                                                            V_MAX_REPEAT * FP32_MAX],
                                                        v_rep_last_col_last, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_FOUR,
                                                        constant.REPEAT_STRIDE_EIGHT)
                            # move ub to gm
                            output_cuthline = self.tik_instance.Scalar("int32")
                            output_cuthline.set_as(0)
                            src_address = self.tik_instance.Scalar("int32")
                            src_address.set_as(0)
                            dst_address = self.tik_instance.Scalar("int32")
                            dst_address.set_as(block_batch * dxh * dxw * channel)
                            burst_len = self.tik_instance.Scalar("int32")
                            burst_len.set_as(1)
                            if hoverlap == 0:
                                output_cuthline.set_as(col2img_h_last)
                                dst_address.set_as(dst_address +
                                                   (ho_count - 1) * ho_every * strideh *
                                                   dxw * channel + looph * ho_max_last *
                                                   strideh * dxw * channel -
                                                   pad_top * dxw * channel)
                                with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                    output_cuthline.set_as(dxh - ho_every *
                                                           (ho_count - 1) * strideh -
                                                           looph * ho_max_every * strideh +
                                                           pad_top)
                            else:
                                src_address.set_as(strideh * col2img_w * channel)
                                output_cuthline.set_as((ho_max_last - 1) * strideh)
                                with self.tik_instance.if_scope(looph == h_cycle_last - 1):
                                    output_cuthline.set_as((ho_last - looph *
                                                            (ho_max_last - 1) - 1) * strideh +
                                                           windowh - strideh - pad_bottom)
                                dst_address.set_as(dst_address +
                                                   ((block_h * (ho_every - 1) +
                                                     looph * (ho_max_last - 1) + 1) *
                                                    strideh - pad_top) * dxw * channel)
                            if woverlap == 0:
                                with self.tik_instance.if_scope(loopw == 0):
                                    src_address.set_as(src_address + pad_left * channel)
                                    burst_len.set_as(col2img_w - pad_left)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw)
                                with self.tik_instance.else_scope():
                                    burst_len.set_as(col2img_w)
                                    dst_address.set_as(dst_address +
                                                       (loopw * col2img_w - pad_left) * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw - col2img_w * loopw + pad_left)
                            else:
                                with self.tik_instance.if_scope(loopw == 0):
                                    burst_len.set_as(stridew * wo_max - pad_left)
                                    src_address.set_as(src_address + pad_left * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw)
                                with self.tik_instance.else_scope():
                                    burst_len.set_as(stridew * (wo_max - 1))
                                    src_address.set_as(src_address + stridew * channel)
                                    dst_address.set_as(dst_address + ((loopw - 1) * stridew *
                                                                      (wo_max - 1) + stridew *
                                                                      wo_max - pad_left) * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw - stridew * wo_max - (w_cycle - 2) *
                                                         stridew * (wo_max - 1) + pad_left)
                            self.tik_instance.data_move(data_output[dst_address],
                                                        data_vmul_ub_col2img_fp16[src_address],
                                                        constant.SID, output_cuthline, burst_len,
                                                        col2img_w - burst_len,
                                                        dxw - burst_len)
                with self.tik_instance.else_scope():
                    data_max_ub = self.tik_instance.Tensor(dtype,
                                                           (ho_max_every * wo_max * channel,),
                                                           name="data_max_ub",
                                                           scope=tik.scope_ubuf)
                    data_vmul_ub_col2img_fp32 = \
                        self.tik_instance.Tensor("float32",
                                                 (col2img_w * channel * col2img_h_every + 64,),
                                                 name="data_vmul_ub_col2img_fp32",
                                                 scope=tik.scope_ubuf)
                    data_vmul_ub_col2img_fp16 = \
                        self.tik_instance.Tensor(dtype,
                                                 (col2img_w * channel * col2img_h_every + 128,),
                                                 name="data_vmul_ub_col2img_fp16",
                                                 scope=tik.scope_ubuf)
                    # mask define
                    data_mask_ub = self.tik_instance.Tensor("uint16", (ho_max_every * wo_max,),
                                                            name="data_mask_ub",
                                                            scope=tik.scope_ubuf)
                    in_nburst = self.tik_instance.Scalar("int32")
                    in_nburst.set_as(ho_max_every)
                    in_src_address = self.tik_instance.Scalar("int32")
                    mask_address = self.tik_instance.Scalar("int32")
                    with self.tik_instance.for_range(0, h_cycle_every) as looph:
                        in_src_address.set_as(block_batch * dyh * dyw * channel)
                        mask_address.set_as(block_batch * mask_one_window * windoww * windowh)
                        if hoverlap == 0:
                            in_src_address.set_as(in_src_address +
                                                  (block_h * ho_every + looph * ho_max_every) *
                                                  dyw * channel)
                            mask_address.set_as(mask_address + (block_h * ho_every +
                                                                looph * ho_max_every) * dyw)
                            with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                                in_nburst.set_as(ho_every - looph * ho_max_every)
                        else:
                            in_src_address.set_as(in_src_address +
                                                  (block_h * (ho_every - 1) +
                                                   looph * (ho_max_every - 1)) * dyw * channel)
                            mask_address.set_as(mask_address +
                                                (block_h * (ho_every - 1) +
                                                 looph * (ho_max_every - 1)) * dyw)
                            with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                                in_nburst.set_as(ho_every - looph * (ho_max_every - 1))
                        new_loopw = self.tik_instance.Scalar("int32")
                        new_loopw.set_as(0)
                        in_burstlen = self.tik_instance.Scalar("int32")
                        in_burstlen.set_as(wo_max)
                        with self.tik_instance.for_range(0, w_cycle) as loopw:
                            if woverlap == 0:
                                new_loopw.set_as(loopw * wo_max)
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    in_burstlen.set_as(dyw - loopw * wo_max)
                            else:
                                with self.tik_instance.if_scope(loopw != 0):
                                    new_loopw.set_as(loopw * (wo_max - 1))
                                with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                    in_burstlen.set_as(dyw - loopw * (wo_max - 1))
                            self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                            if self.woverlap > 0 and dyw % 16 != 0 and self.padding == "VALID":
                                self.clean_max_ub(data_max_ub, dtype)

                            self.tik_instance.data_move(data_max_ub,
                                                        data_input[in_src_address +
                                                                   new_loopw * channel],
                                                        constant.SID, in_nburst,
                                                        in_burstlen, dyw - in_burstlen,
                                                        wo_max - in_burstlen)

                            with self.tik_instance.for_range(0, windowh * windoww) as mask_id:
                                with self.tik_instance.for_range(0, in_nburst) as cycle:
                                    # mask copy gm to ub
                                    self.tik_instance.data_move(data_mask_ub[cycle * wo_max],
                                                                data_mask[mask_address +
                                                                          cycle * dyw +
                                                                          mask_one_window *
                                                                          mask_id + new_loopw],
                                                                constant.SID, 1,
                                                                (in_burstlen + 15) // 16, 0, 0)
                                data_vsel_ub = self.tik_instance.Tensor(dtype, (ho_max_every *
                                                                                wo_max * channel,),
                                                                        name="data_vsel_ub",
                                                                        scope=tik.scope_ubuf)
                                data_vsel_ub_fp32 =\
                                    self.tik_instance.Tensor("float32",
                                                             (ho_max_every * wo_max * channel,),
                                                             name="data_vsel_ub_fp32",
                                                             scope=tik.scope_ubuf)
                                if v_rep_time_every > 0:
                                    with self.tik_instance.for_range(0, v_rep_time_every,
                                                                     thread_num=1) as cycle:
                                        cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                            data_mask_ub[cycle * MASK_MAX])
                                        self.tik_instance.vsel(constant.MASK128, 0,
                                                               data_vsel_ub[cycle * \
                                                                            FP16_MAX],
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
                                if v_rep_cycle_fp32_every > 0:
                                    with self.tik_instance.for_range(0, v_rep_cycle_fp32_every,
                                                                     thread_num=1) as cycle:
                                        self.tik_instance.vconv(constant.MASK64, "",
                                                                data_vsel_ub_fp32[
                                                                    cycle * V_MAX_REPEAT *
                                                                    FP32_MAX],
                                                                data_vsel_ub[
                                                                    cycle * V_MAX_REPEAT *
                                                                    FP32_MAX],
                                                                V_MAX_REPEAT, constant.STRIDE_ONE,
                                                                constant.STRIDE_ONE,
                                                                constant.REPEAT_STRIDE_EIGHT,
                                                                constant.REPEAT_STRIDE_FOUR)
                                if v_rep_last_fp32_every != 0:
                                    self.tik_instance.vconv(constant.MASK64, "",
                                                            data_vsel_ub_fp32[
                                                                v_rep_cycle_fp32_every *
                                                                V_MAX_REPEAT * FP32_MAX],
                                                            data_vsel_ub[
                                                                v_rep_cycle_fp32_every *
                                                                V_MAX_REPEAT * FP32_MAX],
                                                            v_rep_last_fp32_every,
                                                            constant.STRIDE_ONE,
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
                                                          col2img_h_every, col2img_w,
                                                          fetch_filter_w, fetch_filter_h,
                                                          left_top_w, left_top_h,
                                                          stridew, strideh,
                                                          windoww, windowh, 1, 1,
                                                          ho_max_every * wo_max // 16)
                            if v_rep_cycle_col_every > 0:
                                with self.tik_instance.for_range(0, v_rep_cycle_col_every) as cycle:
                                    self.tik_instance.vconv(constant.MASK64, "",
                                                            data_vmul_ub_col2img_fp16[
                                                                cycle * V_MAX_REPEAT * FP32_MAX],
                                                            data_vmul_ub_col2img_fp32[
                                                                cycle * V_MAX_REPEAT * FP32_MAX],
                                                            V_MAX_REPEAT, constant.STRIDE_ONE,
                                                            constant.STRIDE_ONE,
                                                            constant.REPEAT_STRIDE_FOUR,
                                                            constant.REPEAT_STRIDE_EIGHT)
                            if v_rep_last_col_every != 0:
                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vmul_ub_col2img_fp16[
                                                            v_rep_cycle_col_every *
                                                            V_MAX_REPEAT * FP32_MAX],
                                                        data_vmul_ub_col2img_fp32[
                                                            v_rep_cycle_col_every *
                                                            V_MAX_REPEAT * FP32_MAX],
                                                        v_rep_last_col_every, constant.STRIDE_ONE,
                                                        constant.STRIDE_ONE,
                                                        constant.REPEAT_STRIDE_FOUR,
                                                        constant.REPEAT_STRIDE_EIGHT)
                            # move ub to gm
                            output_cuthline = self.tik_instance.Scalar("int32")
                            output_cuthline.set_as(0)
                            src_address = self.tik_instance.Scalar("int32")
                            src_address.set_as(0)
                            dst_address = self.tik_instance.Scalar("int32")
                            dst_address.set_as(block_batch * dxh * dxw * channel)
                            burst_len = self.tik_instance.Scalar("int32")
                            burst_len.set_as(1)
                            if hoverlap == 0:
                                output_cuthline.set_as(col2img_h_every)
                                dst_address.set_as(dst_address +
                                                   ((block_h * ho_every +
                                                     looph * ho_max_every) *
                                                    strideh - pad_top) * dxw * channel)
                                with self.tik_instance.if_scope(block_h == 0):
                                    with self.tik_instance.if_scope(looph == 0):
                                        output_cuthline.set_as(output_cuthline - pad_top)
                                        src_address.set_as(pad_top * col2img_w * channel)
                                        dst_address.set_as(block_batch * dxh * dxw * channel)
                            else:
                                src_address.set_as(strideh * col2img_w * channel)
                                output_cuthline.set_as((ho_max_every - 1) * strideh)
                                with self.tik_instance.if_scope(looph == h_cycle_every - 1):
                                    output_cuthline.set_as((ho_every - looph *
                                                            (ho_max_every - 1) - 1) * strideh)
                                dst_address.set_as(dst_address +
                                                   ((block_h * (ho_every - 1) +
                                                     (looph - 1) * (ho_max_every - 1) +
                                                     ho_max_every) * strideh - pad_top) *
                                                   dxw * channel)
                                with self.tik_instance.if_scope(block_h == 0):
                                    with self.tik_instance.if_scope(looph == 0):
                                        output_cuthline.set_as(ho_max_every * strideh - pad_top)
                                        src_address.set_as(pad_top * col2img_w * channel)
                                        dst_address.set_as(block_batch * dxh * dxw * channel)
                            if woverlap == 0:
                                with self.tik_instance.if_scope(loopw == 0):
                                    burst_len.set_as(col2img_w - pad_left)
                                    src_address.set_as(src_address + pad_left * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw)
                                with self.tik_instance.else_scope():
                                    burst_len.set_as(col2img_w)
                                    dst_address.set_as(dst_address +
                                                       (loopw * col2img_w - pad_left) * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw - col2img_w * loopw + pad_left)
                            else:
                                with self.tik_instance.if_scope(loopw == 0):
                                    burst_len.set_as(stridew * wo_max - pad_left)
                                    src_address.set_as(src_address + pad_left * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw)
                                with self.tik_instance.else_scope():
                                    burst_len.set_as(stridew * (wo_max - 1))
                                    src_address.set_as(src_address + stridew * channel)
                                    dst_address.set_as(dst_address +
                                                       ((loopw - 1) * stridew * (wo_max - 1) +
                                                        stridew * wo_max - pad_left) * channel)
                                    with self.tik_instance.if_scope(loopw == w_cycle - 1):
                                        burst_len.set_as(dxw - stridew * wo_max - (w_cycle - 2) *
                                                         stridew * (wo_max - 1) + pad_left)
                            self.tik_instance.data_move(data_output[dst_address],
                                                        data_vmul_ub_col2img_fp16[src_address],
                                                        constant.SID, output_cuthline, burst_len,
                                                        col2img_w - burst_len,
                                                        dxw - burst_len)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(data_input_origin, data_input, data_mask),
                                   outputs=(data_output), enable_l2=False)
        return self.tik_instance
