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
from topi.cce import util
from impl import constant_util as constant
from impl.max_pool_grad_with_argmax_cut_one_h import MaxpoolGradCustom
from impl import max_pool_grad_with_argmax_resnet50 as resnet50

# size of 5HD format
DIM_5HD = 5
# size of c0 for fp16
C0 = 16
# min shape of attr
ATTR_SHAPE_MIN = 4
# size of useful UB buffer
USEFUL_UB_SIZE = 1024 * 240
# size of vector calc one repeat
ONE_REPEAT = 256
# size of one block
BLOCK_SIZE = 32
# max repeat of vector calc
V_MAX_REPEAT = 255
# max num of fp16 in one repeat
FP16_MAX = 128
# max num of fp32 in one repeat
FP32_MAX = 64
# max num of fp16 mask handle one time
MASK_MAX = 8

# pylint: disable=locally-disabled,too-many-arguments,invalid-name
@util.check_input_type(dict, dict, dict, dict, (list, tuple), (list, tuple), str, str)
def max_pool_grad_with_argmax(x, grad, argmax, y, ksize, strides, padding,
                              kernel_name="max_pool_grad_with_argmax"):
    """
    the main function of the maxpoolGradWithArgmax
    Parameters
    ----------
    x: input of maxpool, useless for maxpool gard
    grad: input of maxpoolgard or output of maxpool
    argmax:output of maxpool mask or index
    y: output of maxpoolgard
    ksize: kernel or windows size,minimum length is 4,
           just like [1, poolingWindowH, poolingWindowW, 1]
    strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
    padding: pad mode, just support "SAME" or "VALID"
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """
    check_param(x, grad, argmax, y, ksize, strides, padding, kernel_name)

    if resnet50.is_max_pool_grad_with_argmax_param(grad, argmax, x, ksize, strides, padding):
        return resnet50.max_pool_grad_with_argmax(grad, argmax, x, ksize,
                                                  strides, padding, kernel_name)
    maxpoolgard = MaxpoolGard(grad, argmax, x, ksize, strides, padding)
    return maxpoolgard.tik_instance_function(kernel_name)


class MaxpoolGard(MaxpoolGradCustom):
    """
    parameter for max_pool_grad_with_pool
    """
    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,useless-super-delegation
    def __init__(self, grad, argmax, input_x, ksize, strides, padding):
        """
        init compare and bit pack parameters
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
        super(MaxpoolGard, self).__init__(grad, argmax, input_x, ksize, strides, padding)

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,too-many-branches
    def tik_instance_shape_large_shape(self, kernel_name):
        """
        function for max_pool_grad_with_pool calc for large shape, ksize or windows special
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
        pad_top, pad_bottom, pad_left, pad_right = self.pad
        hoverlap = self.hoverlap
        woverlap = self.woverlap
        num_one_c0 = dxh * dxw * channel
        num_one_block = nc1 * num_one_c0
        h_ratio = 0
        h_stride = windowh
        if hoverlap != 0:
            h_ratio = 1
            h_stride = strideh
        w_ratio = 0
        w_stride = windoww
        if woverlap != 0:
            w_stride = stridew
            if dyw != 1:
                w_ratio = 1
        h_input = 1 + h_ratio
        col2img_dyw = 16
        col2img_h = windowh + strideh * h_ratio
        col2img_w = (col2img_dyw - 1) * stridew + windoww
        mask_one_window = ((dyh * dyw + 15) // 16 + 1) * 16
        mask_stride = (mask_one_window - col2img_dyw) // 16
        if col2img_w * channel * col2img_h * dtype_size > self.ub_limit:
            raise RuntimeError(
                "The shape or ksize or stride is too large, please check!")

        # fp32 to fp16
        v_rep_time_col = (2 * col2img_w * channel * col2img_h * \
                          dtype_size + (ONE_REPEAT - 1)) // ONE_REPEAT
        v_rep_cycle_col = v_rep_time_col // V_MAX_REPEAT
        v_rep_last_col = v_rep_time_col % V_MAX_REPEAT

        data_input = self.tik_instance.Tensor(dtype, self.input_gard_shape, name="data_input",
                                              scope=tik.scope_gm)
        data_mask = self.tik_instance.Tensor("uint16", (batch * channel1 * windowh * windoww * \
                                                        mask_one_window,),
                                             name="data_mask", scope=tik.scope_gm)
        data_output = self.tik_instance.Tensor(dtype, self.y_shape, name="data_output",
                                               scope=tik.scope_gm)
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
                self.clean_gm(dtype, num_one_block, block_num * num_one_block, data_output)
                with self.tik_instance.for_range(0, nc1, thread_num=1) as loopc1:
                    # vector_dup ub every time
                    dx_calc_line = self.tik_instance.Scalar("int32")
                    dx_calc_line.set_as(0)
                    dxh_address_offset = self.tik_instance.Scalar("int32")
                    dxh_address_offset.set_as(0)
                    data_max_ub = self.tik_instance.Tensor(dtype,
                                                           (h_input * col2img_dyw * channel,),
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
                    data_mask_ub = self.tik_instance.Tensor("uint16", (h_input * col2img_dyw *
                                                                       windowh * windoww,),
                                                            name="data_mask_ub",
                                                            scope=tik.scope_ubuf)
                    data_vsel_ub = self.tik_instance.Tensor(dtype,
                                                            (h_input * col2img_dyw *
                                                             channel,),
                                                            name="data_vsel_ub",
                                                            scope=tik.scope_ubuf)
                    data_vsel_ub_fp32 = self.tik_instance.Tensor("float32",
                                                                 (h_input * col2img_dyw *
                                                                  channel,),
                                                                 name="data_vsel_ub_fp32",
                                                                 scope=tik.scope_ubuf)
                    new_looph = self.tik_instance.Scalar("int32")
                    new_looph.set_as(0)
                    with self.tik_instance.for_range(0, dyh, thread_num=1) as looph:
                        if hoverlap == 0:
                            new_looph.set_as(looph)
                        else:
                            with self.tik_instance.if_scope(looph != 0):
                                new_looph.set_as(looph - 1)
                        new_loopw = self.tik_instance.Scalar("int32")
                        new_loopw.set_as(0)
                        with self.tik_instance.for_range(0, dyw) as loopw:
                            if woverlap == 0:
                                new_loopw.set_as(loopw)
                            else:
                                with self.tik_instance.if_scope(loopw != 0):
                                    new_loopw.set_as(loopw - 1)
                            self.clean_fp32_multi_repeat(data_vmul_ub_col2img_fp32, dtype_size * 2)
                            # dy copy gm to ub
                            self.tik_instance.data_move(data_max_ub,
                                                        data_input[block_num * nc1 * dyh * dyw *
                                                                   channel + loopc1 * dyh * dyw *
                                                                   channel + new_looph * dyw *
                                                                   channel + channel * new_loopw],
                                                        constant.SID, 1,
                                                        constant.DEFAULT_BURST_LEN + w_ratio,
                                                        0, 0)
                            if h_ratio != 0:
                                self.tik_instance.data_move(data_max_ub,
                                                            data_input[block_num * nc1 * dyh * dyw *
                                                                       channel + loopc1 * dyh *
                                                                       dyw * channel +
                                                                       (new_looph + 1) * dyw *
                                                                       channel +
                                                                       channel * new_loopw],
                                                            constant.SID, 1,
                                                            constant.DEFAULT_BURST_LEN + w_ratio,
                                                            0, 0)
                            # mask copy gm to ub
                            self.tik_instance.data_move(data_mask_ub,
                                                        data_mask[block_num * nc1 *
                                                                  mask_one_window *
                                                                  windoww * windowh + loopc1 *
                                                                  mask_one_window * windoww *
                                                                  windowh + new_looph * dyw +
                                                                  new_loopw],
                                                        constant.SID, windowh * windoww,
                                                        constant.DEFAULT_BURST_LEN,
                                                        mask_stride, constant.STRIDE_ZERO)
                            if hoverlap != 0:
                                self.tik_instance.data_move(data_mask_ub[col2img_dyw *
                                                                         windowh * windoww],
                                                            data_mask[block_num * nc1 *
                                                                      mask_one_window *
                                                                      windoww * windowh + loopc1 *
                                                                      mask_one_window * windoww *
                                                                      windowh + (new_looph + 1) *
                                                                      dyw + loopw],
                                                            constant.SID, windowh * windoww,
                                                            constant.DEFAULT_BURST_LEN,
                                                            mask_stride, constant.STRIDE_ZERO)
                            with self.tik_instance.for_range(0, windowh * windoww) as mask_id:
                                cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                    data_mask_ub[col2img_dyw * mask_id])
                                self.tik_instance.vsel(constant.MASK128, 0,
                                                       data_vsel_ub[0],
                                                       cmpmask,
                                                       data_max_ub[0],
                                                       data_vsel_ub_zero[0],
                                                       constant.REPEAT_TIME_ONCE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.STRIDE_ONE,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT,
                                                       constant.REPEAT_STRIDE_EIGHT)
                                if hoverlap != 0:
                                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                        data_mask_ub[col2img_dyw * mask_id + \
                                                     col2img_dyw * windowh * windoww])
                                    self.tik_instance.vsel(constant.MASK128, 0,
                                                           data_vsel_ub[col2img_dyw * channel],
                                                           cmpmask,
                                                           data_max_ub[col2img_dyw * channel],
                                                           data_vsel_ub_zero[0],
                                                           constant.REPEAT_TIME_ONCE,
                                                           constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE,
                                                           constant.STRIDE_ONE,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT,
                                                           constant.REPEAT_STRIDE_EIGHT)

                                self.tik_instance.vconv(constant.MASK64, "",
                                                        data_vsel_ub_fp32[0],
                                                        data_vsel_ub[0],
                                                        h_input * col2img_dyw * channel // FP32_MAX,
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
                                                          col2img_h, col2img_w, fetch_filter_w,
                                                          fetch_filter_h, left_top_w, left_top_h,
                                                          w_stride, h_stride,
                                                          windoww, windowh, 1, 1, 1 + h_ratio)

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
                            nburst.set_as(1)
                            burst_len = self.tik_instance.Scalar("int32")
                            burst_len.set_as(1)
                            src_stride = self.tik_instance.Scalar("int32")
                            src_stride.set_as(0)
                            dst_stride = self.tik_instance.Scalar("int32")
                            dst_stride.set_as(0)
                            src_address = self.tik_instance.Scalar("int32")
                            src_address.set_as(0)
                            dst_address = self.tik_instance.Scalar("int32")
                            dst_address.set_as(0)

                            self.calc_dma_param(block_num, burst_len, channel, col2img_w,
                                                dst_address, dst_stride, dxh, dxw, dyh,
                                                dyw, hoverlap, loopc1, looph, loopw,
                                                nburst, num_one_block, num_one_c0, pad_left,
                                                pad_top, src_address, src_stride, strideh,
                                                stridew, windowh, windoww, woverlap,
                                                pad_bottom, pad_right)

                            # move ub to gm
                            self.tik_instance.data_move(data_output[dst_address],
                                                        data_vmul_ub_col2img_fp16[src_address],
                                                        constant.SID, nburst,
                                                        burst_len, src_stride, dst_stride)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(data_input_origin, data_input, data_mask),
                                   outputs=(data_output), enable_l2=False)
        return self.tik_instance

    # pylint: disable=locally-disabled,too-many-statements
    def calc_dma_param(self, block_num, burst_len, channel, col2img_w, dst_address, dst_stride, dxh,
                       dxw, dyh, dyw, hoverlap, loopc1, looph, loopw, nburst, num_one_block,
                       num_one_c0, pad_left, pad_top, src_address,
                       src_stride, strideh, stridew, windowh, windoww, woverlap,
                       pad_bottom, pad_right):
        """
        function for max_pool_grad_with_pool calc for dma param
        """
        if hoverlap == 0:
            nburst.set_as(windowh)
            src_address.set_as(0)
            dst_address.set_as(block_num * num_one_block + loopc1 * num_one_c0 +
                               (looph * strideh - pad_top) * dxw * channel)
            with self.tik_instance.if_scope(looph == 0):
                nburst.set_as(windowh - pad_top)
                src_address.set_as(pad_top * col2img_w * channel)
                dst_address.set_as(block_num * num_one_block + loopc1 * num_one_c0)
            with self.tik_instance.if_scope(looph == dyh - 1):
                nburst.set_as(windowh - pad_bottom)
            if woverlap == 0:
                burst_len.set_as(windoww)
                with self.tik_instance.if_scope(loopw == 0):
                    burst_len.set_as(windoww - pad_left)
                    src_address.set_as(src_address + pad_left * channel)
                with self.tik_instance.else_scope():
                    dst_address.set_as(dst_address + (loopw * stridew - pad_left) * channel)
                    with self.tik_instance.if_scope(loopw == dyw - 1):
                        burst_len.set_as(windoww - pad_right)
            else:
                burst_len.set_as(stridew)
                with self.tik_instance.if_scope(loopw == 0):
                    burst_len.set_as(stridew - pad_left)
                    src_address.set_as(src_address + pad_left * channel)
                with self.tik_instance.else_scope():
                    src_address.set_as(src_address + stridew * channel)
                    dst_address.set_as(dst_address + (loopw * stridew - pad_left) * channel)
                    with self.tik_instance.if_scope(loopw == dyw - 1):
                        burst_len.set_as(dxw - (dyw - 1) * stridew + pad_left)
        else:
            burst_len.set_as(windoww)
            src_address.set_as(0)
            dst_address.set_as(
                block_num * num_one_block + loopc1 * num_one_c0 +
                (loopw * stridew - pad_left) * channel)
            with self.tik_instance.if_scope(loopw == 0):
                burst_len.set_as(windoww - pad_left)
                src_address.set_as(pad_left * channel)
                dst_address.set_as(block_num * num_one_block + loopc1 * num_one_c0)
            with self.tik_instance.if_scope(loopw == dyw - 1):
                burst_len.set_as(windoww - pad_right)
            with self.tik_instance.if_scope(looph == 0):
                nburst.set_as(strideh - pad_top)
                src_address.set_as(pad_top * col2img_w * channel + src_address)
            with self.tik_instance.else_scope():
                src_address.set_as(strideh * col2img_w * channel + src_address)
                dst_address.set_as(dst_address + (looph * strideh - pad_top) * dxw * channel)
                with self.tik_instance.if_scope(looph == dyh - 1):
                    nburst.set_as(dxh - (dyh - 1) * strideh + pad_top)
                with self.tik_instance.else_scope():
                    nburst.set_as(strideh)
        src_stride.set_as(col2img_w - burst_len)
        dst_stride.set_as(dxw - burst_len)

    # pylint: disable=locally-disabled,too-many-locals,too-many-statements,too-many-branches,too-many-return-statements
    def tik_instance_function(self, kernel_name):
        """
        get vector instruct repeat times
        Parameters
        ----------
        kernel_name: cce kernel name, default value is "maxpoolGradWithArgmax"
        Returns
        -------
        None
        """
        batch, c1, dyh, dyw, channel = self.input_gard_shape
        strideh, stridew = self.strides[1:3]
        windowh, windoww = self.ksize[1:3]
        # the minimum part can be dealed
        ho_min = 1 if self.hoverlap == 0 else 2
        hoverlap = self.hoverlap
        woverlap = self.woverlap
        dtype_size = self.dtype_size

        ho_min = 1 if hoverlap == 0 else 2
        ho_max = ho_min
        wo_max = math.ceil(dyw / 16) * 16
        col2img_one_h = strideh if hoverlap == 0 else windowh
        col2img_w = wo_max * stridew if woverlap == 0 else (wo_max - 1) * stridew + windoww
        col2img_h = ho_max * strideh if hoverlap == 0 else (ho_max - 1) * strideh + windowh

        if windowh > 2 * strideh or windoww > 2 * stridew:
            if col2img_w * col2img_one_h * channel * dtype_size > self.ub_limit:
                raise RuntimeError(
                    "The shape or ksize or stride is too large, please check!")
            return self.tik_instance_cut_nc1_cut_one_h(kernel_name)
        if batch * c1 >= self.blocknum or dyh <= self.blocknum:
            if col2img_w * col2img_h * channel * dtype_size > self.ub_limit:
                if col2img_w * col2img_one_h * channel * dtype_size < self.ub_limit:
                    return self.tik_instance_cut_nc1_cut_one_h(kernel_name)
                return self.tik_instance_cut_nc1_cut_w(kernel_name)
            return self.tik_instance_cut_nc1_cut_h(kernel_name)
        if batch * c1 * dyh < self.blocknum:
            self.change_blocknum(batch * c1 * dyh)
        if col2img_w * col2img_h * channel * dtype_size > self.ub_limit:
            if col2img_w * col2img_one_h * channel * dtype_size < self.ub_limit:
                return self.tik_instance_cut_nc1h_cut_one_h(kernel_name)
            return self.tik_instance_cut_nc1h_cut_w(kernel_name)
        return self.tik_instance_cut_nc1h_cut_h(kernel_name)


def check_shape_5hd(shape):
    """
    The common check rule for tensor shape, just for 5hd
    """
    util.check_shape_rule(shape)
    if len(shape) != DIM_5HD:
        raise RuntimeError(
            "The dim of tensor must be %d"
            ", actual dim is %d" % (DIM_5HD, len(shape)))

    if shape[DIM_5HD - 1] != C0:
        raise RuntimeError(
            "The value of C0 must be %d,"
            " actual input is (%d)" % (C0, shape[DIM_5HD - 1]))


def check_padding(padding, check_list):
    """
    The common check rule for padding
    """
    if padding not in check_list:
        raise RuntimeError("The padding only support SAME, VALID")


# pylint: disable=locally-disabled,too-many-locals
def check_output_dim_with_ksize_stride(padding, input_gard_shape, y_shape, ksize, strides):
    """
    The common check rule for output dim and ksize and strides
    """
    util.check_tensor_shape_size(ksize)
    util.check_tensor_shape_size(strides)
    if len(ksize) < ATTR_SHAPE_MIN or len(strides) < ATTR_SHAPE_MIN:
        raise RuntimeError(
            "The shape length of ksize or strides must be more than 4")
    if ksize[0] != 1 or ksize[3] != 1:
        raise RuntimeError("MaxPoolGradWithArgmax only supports pooling across width/height,"
                           "and other ksize dimension should be one")
    if strides[0] != 1 or strides[3] != 1:
        raise RuntimeError("MaxPoolGradWithArgmax only supports pooling across width/height,"
                           "and other strides dimension should be one")
    if ksize[1] * ksize[2] > 255:
        raise RuntimeError("invalid window params, window_h*window_w should be <=255")

    input_height = y_shape[2]
    input_weight = y_shape[3]
    input_batch = y_shape[0]
    xc1 = y_shape[1]
    xc0 = y_shape[4]
    output_height = input_gard_shape[2]
    output_weight = input_gard_shape[3]
    windowh = ksize[1]
    windoww = ksize[2]
    dyh = 0
    dyw = 0
    dyn = input_gard_shape[0]
    dyc1 = input_gard_shape[1]
    dyc0 = input_gard_shape[4]
    if padding == "SAME":
        dyh = (input_height + strides[1] - 1) // strides[1]
        dyw = (input_weight + strides[2] - 1) // strides[2]
    else:
        dyh = (input_height - windowh + strides[1]) // strides[1]
        dyw = (input_weight - windoww + strides[2]) // strides[2]

    if ksize[1] >= input_height or ksize[2] >= input_weight:
        raise RuntimeError("can not support global pooling now")

    if dyh != output_height or dyw != output_weight or \
            input_batch != dyn or xc1 != dyc1 or xc0 != dyc0:
        raise RuntimeError("dimentions of dx dy \
                padMode window stride is wrong,please check!")


def check_param(x, grad, argmax, y, ksize, strides, padding, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error
    Parameters
    ----------
    x: dict,shape and datatype
    grad: dict,shape and datatype
    argmax: dict,shape and datatype
    y: dict,shape and datatype
    ksize: kernel or windows size,minimum length is 4,
          just like [1, poolingWindowH, poolingWindowW, 1]
    strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
    padding: pad mode, just support "SANME" or "VALID"
    Returns
    -------
    None
    """
    y_shape = x.get("shape")
    y_dtype = x.get("dtype").lower()
    y_dtype_arg = y.get("dtype").lower()
    input_gard_shape = grad.get("shape")
    grad_dtype = grad.get("dtype").lower()
    argmax_shape = argmax.get("shape")
    argmax_dtype = argmax.get("dtype").lower()
    util.check_shape_rule(y_shape)
    util.check_shape_rule(input_gard_shape)
    util.check_shape_rule(argmax_shape)
    util.check_kernel_name(kernel_name)
    check_shape_5hd(y_shape)
    check_shape_5hd(input_gard_shape)
    util.check_tensor_shape_size(input_gard_shape)
    util.check_tensor_shape_size(argmax_shape)
    util.check_tensor_shape_size(y_shape)
    util.check_dtype_rule(grad_dtype, ("float16", "float32", "int32"))
    util.check_dtype_rule(argmax_dtype, ("uint16"))
    util.check_dtype_rule(y_dtype, ("float16", "float32", "int32"))

    if y_dtype != grad_dtype or y_dtype_arg != y_dtype:
        raise RuntimeError(
            "The dtype of tensor must be same")

    check_padding(padding, ("SAME", "VALID"))
    check_output_dim_with_ksize_stride(padding, input_gard_shape, y_shape, ksize, strides)
