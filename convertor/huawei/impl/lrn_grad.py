#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0. You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lrn_grad
"""

import math
from te import tik
from topi.cce import util
from te import platform as tbe_platform
from impl import common_util
from impl import constant_util as constant

MAX_REPEAT = 255
MAX_STRIDE = 65535
MORE_THREE_RADIUS = 3
MORE_TWO_RADIUS = 2


def _get_ub_segment_size(need_ub_segment_count):
    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

    valid_ub_size = ub_size

    segment_ub_size = math.floor(valid_ub_size / (need_ub_segment_count * 2))
    segment_ub_size = (segment_ub_size // constant.BLOCK_SIZE) * \
                      constant.BLOCK_SIZE

    return segment_ub_size


def _get_rep_stride(mask, dtype):
    return mask * common_util.get_data_size(dtype) // constant.BLOCK_SIZE


def _double_vector_func(func, dest, src0, src1, count):
    if count == 0:
        return
    remain = count
    index = 0

    element_count = constant.MASK64
    if dest.dtype == "float16":
        element_count = constant.MASK128

    dst_rep_stride = _get_rep_stride(element_count, dest.dtype)
    src_rep_stride = _get_rep_stride(element_count, src0.dtype)

    while remain > 0:
        if remain > element_count * MAX_REPEAT:
            func(element_count, dest[index],
                 src0[index], src1[index], MAX_REPEAT,
                 1, 1, 1, dst_rep_stride, src_rep_stride, src_rep_stride)
            handle_count = element_count * MAX_REPEAT
        elif remain > element_count:
            repeat_times = remain // element_count
            func(element_count, dest[index],
                 src0[index], src1[index], repeat_times,
                 1, 1, 1, dst_rep_stride, src_rep_stride, src_rep_stride)
            handle_count = element_count * repeat_times
        else:
            func(remain, dest[index],
                 src0[index], src1[index], 1,
                 1, 1, 1, dst_rep_stride, src_rep_stride, src_rep_stride)
            handle_count = remain
        index += handle_count
        remain -= handle_count


def _vector_scalar_func(func, dest, src0, scalar_value, count):
    if count == 0:
        return

    remain = count
    index = 0

    element_count = constant.MASK64
    if dest.dtype == "float16":
        element_count = constant.MASK128

    dst_rep_stride = _get_rep_stride(element_count, dest.dtype)
    src_rep_stride = _get_rep_stride(element_count, src0.dtype)

    while remain > 0:
        if remain > element_count * MAX_REPEAT:
            func(element_count, dest[index],
                 src0[index], scalar_value, MAX_REPEAT,
                 1, 1, dst_rep_stride, src_rep_stride)
            handle_count = element_count * MAX_REPEAT
        elif remain > element_count:
            repeat_times = remain // element_count
            func(element_count, dest[index],
                 src0[index], scalar_value, repeat_times,
                 1, 1, dst_rep_stride, src_rep_stride)
            handle_count = element_count * repeat_times
        else:
            func(remain, dest[index],
                 src0[index], scalar_value, 1,
                 1, 1, dst_rep_stride, src_rep_stride)
            handle_count = remain
        index += handle_count
        remain -= handle_count


def _single_vector_func(func, dest, src, count):
    if count == 0:
        return
    remain = count
    index = 0

    element_count = constant.MASK64
    if dest.dtype == "float16":
        element_count = constant.MASK128

    dst_rep_stride = _get_rep_stride(element_count, dest.dtype)
    src_rep_stride = _get_rep_stride(element_count, src.dtype)

    while remain > 0:
        if remain > element_count * MAX_REPEAT:
            func(element_count, dest[index],
                 src[index], MAX_REPEAT,
                 1, 1, dst_rep_stride, src_rep_stride)
            handle_count = element_count * MAX_REPEAT
        elif remain > element_count:
            repeat_times = remain // element_count
            func(element_count, dest[index],
                 src[index], repeat_times,
                 1, 1, dst_rep_stride, src_rep_stride)
            handle_count = element_count * repeat_times
        else:
            func(remain, dest[index],
                 src[index], 1,
                 1, 1, dst_rep_stride, src_rep_stride)
            handle_count = remain
        index += handle_count
        remain -= handle_count


# pylint: disable=too-many-instance-attributes, attribute-defined-outside-init
# pylint: disable=too-many-lines, too-few-public-methods
class LrnGrad:
    """
     implementation of lrn_grad
    """
    # pylint: disable=locally-disabled,too-many-locals,too-many-arguments
    def __init__(self, shape, dtype, depth_radius=5, bias=1, alpha=1, beta=0.5,
                 kernel_name="lrn_grad"):
        self.shape = shape
        dtype = dtype.lower()
        self.dtype = dtype

        self.batch = shape[0]
        self.channels = shape[1]
        self.height = shape[2]
        self.width = shape[3]
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.kernel_name = kernel_name

        tik_instance = tik.Tik()
        self.is_mini = True
        self.ub_dtype = "float16"
        if tbe_platform.cce_conf.api_check_support("tik.vln", "float32"):
            self.is_mini = False
            self.ub_dtype = "float32"

        self.ub_dtype_size = common_util.get_data_size(self.ub_dtype)
        self.tik_instance = tik_instance
        self.aicore_num = tik_instance.d_profiling.get_aicore_num()

        gm_size = 1
        for item in shape:
            gm_size *= item

        gm_shape = (gm_size,)

        self.data_input_grads = tik_instance.Tensor(dtype, gm_shape,
                                                    name="input_grads",
                                                    scope=tik.scope_gm)
        self.data_input_image = tik_instance.Tensor(dtype, gm_shape,
                                                    name="input_image",
                                                    scope=tik.scope_gm)
        self.data_output_image = tik_instance.Tensor(dtype, gm_shape,
                                                     name="output_image",
                                                     scope=tik.scope_gm)
        self.data_output = tik_instance.Tensor(dtype, gm_shape, name="output",
                                               scope=tik.scope_gm)

        need_ub_segment_count = 6
        if self.dtype != self.ub_dtype:
            if self.is_mini:
                # mini only support float16, need conv float32 to float16,
                # sizeof(float32) == 2 * sizeof(float16), so need another buffer
                # of size of 2 * buffer_float16
                need_ub_segment_count += 2
            else:
                # if not mini, need conv float16 to float32 to calculate,
                # sizeof(float32) == 2 * sizeof(float16), so need another buffer
                # of size of 0.5 * buffer_float32
                need_ub_segment_count += 0.5

        self.need_ub_segment_count = need_ub_segment_count
        self.ub_segment_size = _get_ub_segment_size(need_ub_segment_count)
        self.dtype_size = common_util.get_data_size(dtype)
        self.small_hw = False
        if self.width * self.height * self.dtype_size < constant.BLOCK_SIZE:
            self.small_hw = True

        self.ub_shape = (self.ub_segment_size // self.ub_dtype_size,)

    def lrn_grad(self):
        """
        implementation of lrn_grad and return the tik instance
        :return: tik instance
        """
        count_one_block = constant.BLOCK_SIZE // self.dtype_size
        ub_size = self.ub_shape[0]
        if self.channels * count_one_block > ub_size:
            return self._cut_channels()

        return self._cut_hw()

    def _get_32bytes_align_count(self, count):
        return math.ceil(count * self.dtype_size / constant.BLOCK_SIZE) * \
               constant.BLOCK_SIZE // self.dtype_size

    def _alloc_all_ub(self):
        tik_instance = self.tik_instance
        if self.dtype != self.ub_dtype:
            self.data_cov_ub = tik_instance.Tensor(self.dtype, self.ub_shape,
                                                   name="data_cov_ub",
                                                   scope=tik.scope_ubuf)

        dtype = self.ub_dtype
        self.input_grads_ub = tik_instance.Tensor(dtype, self.ub_shape,
                                                  name="input_grads_ub",
                                                  scope=tik.scope_ubuf)
        self.input_image_ub = tik_instance.Tensor(dtype, self.ub_shape,
                                                  name="input_image_ub",
                                                  scope=tik.scope_ubuf)
        self.output_image_ub = tik_instance.Tensor(dtype, self.ub_shape,
                                                   name="output_image_ub",
                                                   scope=tik.scope_ubuf)
        self.pre_norm_ub = tik_instance.Tensor(dtype, self.ub_shape,
                                               name="pre_norm_ub",
                                               scope=tik.scope_ubuf)
        self.norm_ub = tik_instance.Tensor(dtype, self.ub_shape,
                                           name="norm_ub", scope=tik.scope_ubuf)
        self.norm_beta_ub = tik_instance.Tensor(dtype, self.ub_shape,
                                                name="norm_beta_ub",
                                                scope=tik.scope_ubuf)
        self.data_output_ub = self.norm_beta_ub

    def _alloc_all_ub_cut_channel(self):
        factor = self.hw_factor4cut_channels
        channels_out = self.channels_out_one_loop
        radius = self.depth_radius
        ub_output_count = channels_out * factor

        ub_input_shape = (ub_output_count + MORE_THREE_RADIUS*radius*factor,)
        ub_pre_norm_shape = (ub_output_count +
                             (MORE_THREE_RADIUS * radius + 1) * factor,)
        ub_norm_shape = (ub_output_count + MORE_TWO_RADIUS * radius * factor,)
        ub_pre_norm2_shape = (ub_output_count +
                              (MORE_TWO_RADIUS * radius + 1) * factor,)
        ub_norm2_shape = (ub_output_count + radius * factor,)
        ub_output_shape = (ub_output_count,)

        tik_instance = self.tik_instance
        if self.dtype != self.ub_dtype:
            self.data_cov_ub = tik_instance.Tensor(self.dtype, ub_input_shape,
                                                   name="data_cov_ub",
                                                   scope=tik.scope_ubuf)

        dtype = self.ub_dtype
        self.input_grads_ub = tik_instance.Tensor(dtype, ub_input_shape,
                                                  name="input_grads_ub",
                                                  scope=tik.scope_ubuf)
        self.input_image_ub = tik_instance.Tensor(dtype, ub_input_shape,
                                                  name="input_image_ub",
                                                  scope=tik.scope_ubuf)
        self.output_image_ub = tik_instance.Tensor(dtype, ub_input_shape,
                                                   name="output_image_ub",
                                                   scope=tik.scope_ubuf)
        self.pre_norm_ub = tik_instance.Tensor(dtype, ub_pre_norm_shape,
                                               name="pre_norm_ub",
                                               scope=tik.scope_ubuf)
        self.norm_ub = tik_instance.Tensor(dtype, ub_norm_shape,
                                           name="norm_ub", scope=tik.scope_ubuf)
        self.norm_beta_ub = tik_instance.Tensor(dtype, ub_norm_shape,
                                                name="norm_beta_ub",
                                                scope=tik.scope_ubuf)
        self.pre_norm2_ub = tik_instance.Tensor(dtype, ub_pre_norm2_shape,
                                                name="pre_norm2_ub",
                                                scope=tik.scope_ubuf)
        self.norm2_ub = tik_instance.Tensor(dtype, ub_norm2_shape,
                                            name="norm2_ub",
                                            scope=tik.scope_ubuf)
        self.data_output_ub = tik_instance.Tensor(dtype, ub_output_shape,
                                                  name="output_ub",
                                                  scope=tik.scope_ubuf)

    def _gm2ub(self, dest, src, count, channels_count):
        tik_instance = self.tik_instance
        width = self.width
        height = self.height
        dtype_size = self.dtype_size
        data_size = count * dtype_size
        burst = math.ceil(data_size / constant.BLOCK_SIZE)

        repeats = channels_count
        data_move_ub = dest
        if dest.dtype != src.dtype:
            data_move_ub = self.data_cov_ub

        if self.tail * dtype_size % constant.BLOCK_SIZE == 0:
            src_stride = (height * width - count) * dtype_size // \
                         constant.BLOCK_SIZE
            if src_stride <= MAX_STRIDE:
                tik_instance.data_move(data_move_ub, src, 0, repeats, burst,
                                       src_stride, 0)
            else:
                count_one_burst = burst * constant.BLOCK_SIZE // self.dtype_size
                with tik_instance.for_range(0, repeats) as i:
                    ub_index = i * count_one_burst
                    tik_instance.data_move(data_move_ub[ub_index],
                                           src[i * width * height], 0, 1, burst,
                                           0, 0)
        elif self.tail != width * height:
            count_one_burst = burst * constant.BLOCK_SIZE // self.dtype_size
            with tik_instance.for_range(0, repeats) as i:
                ub_index = i * count_one_burst
                tik_instance.data_move(data_move_ub[ub_index],
                                       src[i * width * height], 0, 1, burst, 0,
                                       0)
        else:
            count_one_block = constant.BLOCK_SIZE // self.dtype_size
            head_count = self.tail % count_one_block
            count_one_burst = burst * constant.BLOCK_SIZE // self.dtype_size
            with tik_instance.for_range(0, repeats) as i:
                ub_index = i * count_one_burst
                tik_instance.data_move(data_move_ub[ub_index],
                                       src[i * width * height], 0, 1, 1, 0, 0)
                if burst - 1 > 0:
                    tik_instance.data_move(data_move_ub[ub_index +
                                                        count_one_block],
                                           src[i * width * height + head_count],
                                           0, 1, burst - 1, 0, 0)

        if dest.dtype != src.dtype:
            conv_count = self._get_32bytes_align_count(count) * repeats
            self._vconv(dest, self.data_cov_ub, conv_count)

    def _ub2gm(self, gm_index, count, channels):
        dest = self.data_output[gm_index]
        src = self.data_output_ub
        tik_instance = self.tik_instance
        width = self.width
        height = self.height
        dtype_size = self.dtype_size
        data_size = count * dtype_size
        burst = math.ceil(data_size / constant.BLOCK_SIZE)
        repeats = channels
        data_move_ub = src
        if dest.dtype != src.dtype:
            data_move_ub = self.data_cov_ub
            conv_count = self._get_32bytes_align_count(count) * repeats
            self._vconv(self.data_cov_ub, src, conv_count)

        if self.tail * dtype_size % constant.BLOCK_SIZE == 0:
            desc_stride = (width * height - count) * dtype_size // \
                          constant.BLOCK_SIZE
            if desc_stride <= MAX_STRIDE:
                tik_instance.data_move(dest, data_move_ub,
                                       0, repeats, burst, 0, desc_stride)
            else:
                count_one_burst = burst * constant.BLOCK_SIZE // self.dtype_size
                with tik_instance.for_range(0, repeats) as i:
                    ub_index = i * count_one_burst
                    tik_instance.data_move(dest[i * width * height],
                                           data_move_ub[ub_index], 0, 1, burst,
                                           0, 0)
        elif self.tail != width * height:
            count_one_burst = burst * constant.BLOCK_SIZE // self.dtype_size
            with tik_instance.for_range(0, repeats) as i:
                ub_index = i * count_one_burst
                tik_instance.data_move(dest[i * width * height],
                                       data_move_ub[ub_index], 0, 1, burst, 0,
                                       0)
        else:
            count_one_block = constant.BLOCK_SIZE // self.dtype_size
            count_one_burst = burst * constant.BLOCK_SIZE // self.dtype_size
            head_count = self.tail % count_one_block
            with tik_instance.for_range(0, repeats) as i:
                ub_index = i * count_one_burst
                tik_instance.data_move(dest[i * width * height],
                                       data_move_ub[ub_index], 0, 1, 1, 0,
                                       0)
                if burst - 1 > 0:
                    tik_instance.data_move(dest[i * width * height +
                                                head_count],
                                           data_move_ub[ub_index +
                                                        count_one_block],
                                           0, 1, burst - 1, 0, 0)

    def _vconv(self, dest, src, count):
        tik_instance = self.tik_instance
        remain = count
        index = 0

        dst_rep_stride = _get_rep_stride(constant.MASK64, dest.dtype)
        src_rep_stride = _get_rep_stride(constant.MASK64, src.dtype)

        while remain > 0:
            if remain > constant.MASK64 * MAX_REPEAT:
                tik_instance.vconv(constant.MASK64, "",
                                   dest[index], src[index], MAX_REPEAT,
                                   1, 1, dst_rep_stride, src_rep_stride)
                handle_count = constant.MASK64 * MAX_REPEAT
            elif remain > constant.MASK64:
                repeat_times = remain // constant.MASK64
                tik_instance.vconv(constant.MASK64, "",
                                   dest[index], src[index], repeat_times,
                                   1, 1, dst_rep_stride, src_rep_stride)
                handle_count = constant.MASK64 * repeat_times
            else:
                tik_instance.vconv(remain, "",
                                   dest[index], src[index], 1,
                                   1, 1, dst_rep_stride, src_rep_stride)
                handle_count = remain

            index += handle_count
            remain -= handle_count

    def _vmul(self, dest, src0, src1, count):
        tik_instance = self.tik_instance
        _double_vector_func(tik_instance.vmul, dest, src0, src1, count)

    def _vadd(self, dest, src0, src1, count):
        tik_instance = self.tik_instance
        _double_vector_func(tik_instance.vadd, dest, src0, src1, count)

    def _vsub(self, dest, src0, src1, count):
        tik_instance = self.tik_instance
        _double_vector_func(tik_instance.vsub, dest, src0, src1, count)

    def _vdiv(self, dest, src0, src1, count):
        if not self.is_mini:
            tik_instance = self.tik_instance
            _double_vector_func(tik_instance.vdiv, dest, src0, src1, count)
        else:
            self._vrec(dest, src1, count)
            self._vmul(dest, dest, src0, count)

    def _vadds(self, dest, src0, scalar_value, count):
        tik_instance = self.tik_instance
        _vector_scalar_func(tik_instance.vadds, dest, src0, scalar_value, count)

    def _vmuls(self, dest, src0, scalar_value, count):
        tik_instance = self.tik_instance
        _vector_scalar_func(tik_instance.vmuls, dest, src0, scalar_value, count)

    def _vln(self, dest, src, count):
        tik_instance = self.tik_instance
        _single_vector_func(tik_instance.vln, dest, src, count)

    def _vexp(self, dest, src, count):
        tik_instance = self.tik_instance
        _single_vector_func(tik_instance.vexp, dest, src, count)

    def _vrec(self, dest, src, count):
        tik_instance = self.tik_instance
        _single_vector_func(tik_instance.vrec, dest, src, count)

    # pylint: disable=invalid-name
    def _vpow(self, dest, x, y, count):
        """
        pow compute
        calculating data pow, res =x ^ y,
        x > 0: use exp(y*ln(x))
        x < 0: use [-2*(|y|%2)+1]*exp(y*ln|x|)
        x = 0: 0^0=1 & 0^y=0
        """
        if self.alpha > 0 and self.bias > 0:
            self._vln(dest, x, count)
            self._vmuls(dest, dest, y, count)
            self._vexp(dest, dest, count)
        elif self.alpha < 0 and self.bias < 0:
            raise RuntimeError("alpha and bias should be greater than 0")
        else:
            raise RuntimeError("alpha and bias should be greater than 0")

    def _sum4windows_cut_hw(self, dest, src, count_one_window):
        tik_instance = self.tik_instance
        radius = self.depth_radius
        channels = self.channels
        last_begin_idx = tik_instance.Scalar("int64")
        last_end_idx = tik_instance.Scalar("int64")
        last_idx = tik_instance.Scalar("int64")

        last_idx.set_as(0)
        last_begin_idx.set_as(0)
        last_end_idx.set_as(0)

        count_one_window = self._get_32bytes_align_count(count_one_window)
        with tik_instance.for_range(0, channels) as i:
            begin_idx = tik_instance.Scalar("int64")
            end_idx = tik_instance.Scalar("int64")
            begin_idx.set_as(i - radius)
            end_idx.set_as(i + radius)
            tik_instance.scalar_max(begin_idx, 0, begin_idx)
            tik_instance.scalar_min(end_idx, channels - 1, end_idx)

            with tik_instance.if_scope(i != 0):
                self._vadds(dest[i * count_one_window],
                            dest[last_idx * count_one_window],
                            0.0, count_one_window)
                with tik_instance.for_range(last_end_idx + 1, end_idx + 1) as j:
                    self._vadd(dest[i * count_one_window],
                               dest[i * count_one_window],
                               src[j * count_one_window], count_one_window)

                with tik_instance.for_range(last_begin_idx, begin_idx) as j:
                    self._vsub(dest[i * count_one_window],
                               dest[i * count_one_window],
                               src[j * count_one_window], count_one_window)

            with tik_instance.else_scope():
                self._vadds(dest[i * count_one_window],
                            src[end_idx * count_one_window],
                            0.0, count_one_window)
                with tik_instance.for_range(begin_idx, end_idx) as j:
                    self._vadd(dest[i * count_one_window],
                               dest[i * count_one_window],
                               src[j * count_one_window],
                               count_one_window)

            last_begin_idx.set_as(begin_idx)
            last_end_idx.set_as(end_idx)
            last_idx.set_as(i)

    def _sum4windows_cut_channels(self, dest, src, count_one_window,
                                  channel_idx, channel_count):
        tik_instance = self.tik_instance
        radius = self.depth_radius
        channels = self.channels
        last_begin_idx = tik_instance.Scalar("int64")
        last_end_idx = tik_instance.Scalar("int64")
        last_idx = tik_instance.Scalar("int64")

        sub_idx = tik_instance.Scalar(dtype="int64", name="sub_idx")
        with tik_instance.if_scope(channel_idx == 0):
            last_idx.set_as(0)
            last_begin_idx.set_as(0)
            last_end_idx.set_as(0)
            sub_idx.set_as(0)
        with tik_instance.else_scope():
            last_idx.set_as(-1)
            last_begin_idx.set_as(channel_idx - 1 - radius)
            last_end_idx.set_as(channel_idx - 1 + radius)
            tik_instance.scalar_max(last_begin_idx, 0, last_begin_idx)
            tik_instance.scalar_min(last_end_idx, channels - 1, last_end_idx)
            sub_idx.set_as(0 - radius - 1)

        count_one_window = self._get_32bytes_align_count(count_one_window)
        begin = 0
        end = begin + channel_count
        with tik_instance.for_range(begin, end) as i:
            with tik_instance.if_scope(channel_idx + i < channels):
                begin_idx = tik_instance.Scalar("int64")
                end_idx = tik_instance.Scalar("int64")
                begin_idx.set_as(channel_idx + i - radius)
                end_idx.set_as(channel_idx + i + radius)
                tik_instance.scalar_max(begin_idx, 0, begin_idx)
                tik_instance.scalar_min(end_idx, channels - 1, end_idx)

                is_first = tik_instance.Scalar(dtype="int64", name="is_first")
                is_first.set_as(0)
                with tik_instance.if_scope(channel_idx == 0):
                    with tik_instance.if_scope(i == 0):
                        is_first.set_as(1)

                with tik_instance.if_scope(is_first == 1):
                    self._vadds(dest[i * count_one_window],
                                src[end_idx * count_one_window],
                                0.0, count_one_window)
                    with tik_instance.for_range(begin_idx, end_idx) as j:
                        self._vadd(dest[i * count_one_window],
                                   dest[i * count_one_window],
                                   src[j * count_one_window],
                                   count_one_window)
                with tik_instance.else_scope():
                    self._vadds(dest[i * count_one_window],
                                dest[last_idx * count_one_window],
                                0.0, count_one_window)
                    with tik_instance.if_scope(last_end_idx != end_idx):
                        self._vadd(dest[i * count_one_window],
                                   dest[i * count_one_window],
                                   src[(i + radius) * count_one_window],
                                   count_one_window)
                    with tik_instance.if_scope(last_begin_idx != begin_idx):
                        self._vsub(dest[i * count_one_window],
                                   dest[i * count_one_window],
                                   src[sub_idx * count_one_window],
                                   count_one_window)
                        sub_idx.set_as(sub_idx + 1)

                last_begin_idx.set_as(begin_idx)
                last_end_idx.set_as(end_idx)
                last_idx.set_as(i)

    def _cut_hw_compute(self, count_one_loop, count_one_window):
        to_square_ub = self.input_image_ub
        pre_norm_ub = self.pre_norm_ub
        self._vmul(pre_norm_ub, to_square_ub, to_square_ub, count_one_loop)
        norm_ub = self.norm_ub
        self._sum4windows_cut_hw(norm_ub, self.pre_norm_ub, count_one_window)

        self._vmuls(self.norm_ub, self.norm_ub, self.alpha, count_one_loop)
        self._vadds(self.norm_ub, self.norm_ub, self.bias, count_one_loop)

        self._vpow(self.norm_beta_ub, self.norm_ub, -1 * self.beta,
                   count_one_loop)
        self._vmul(self.norm_beta_ub, self.norm_beta_ub, self.input_grads_ub,
                   count_one_loop)

        self._vmul(self.pre_norm_ub, self.output_image_ub,
                   self.input_grads_ub, count_one_loop)
        self._vdiv(self.pre_norm_ub, self.pre_norm_ub, self.norm_ub,
                   count_one_loop)

        norm_ub = self.norm_ub
        pre_norm_ub = self.pre_norm_ub
        self._sum4windows_cut_hw(norm_ub, pre_norm_ub, count_one_window)

        self._vmul(norm_ub, norm_ub,
                   self.input_image_ub, count_one_loop)
        self._vmuls(norm_ub, norm_ub,
                    -2.0 * self.alpha * self.beta, count_one_loop)
        data_output_ub = self.data_output_ub
        self._vadd(data_output_ub, self.norm_beta_ub,
                   self.norm_ub, count_one_loop)

    def _cut_hw_one_batch(self, batch_idx):
        tik_instance = self.tik_instance
        width = self.width
        height = self.height
        loop, factor, tail = self._tilling_hw(self.channels)
        self.tail = tail
        batch_size = width * height * self.channels
        thread_num = max([min([loop, 2]), 1])
        if loop == 0:
            if tail != 0:
                self._alloc_all_ub()
                gm_index = batch_idx * batch_size
                self._treat_cut_hw_one_part(gm_index, tail)
        else:
            with tik_instance.for_range(0, loop,
                                        thread_num=thread_num) as idx:
                self._alloc_all_ub()
                if tail != 0:
                    with tik_instance.if_scope(idx == 0):
                        gm_index = batch_idx * batch_size
                        self._treat_cut_hw_one_part(gm_index, tail)

                gm_index = batch_idx * batch_size + tail + idx * factor
                self._treat_cut_hw_one_part(gm_index, factor)

    def _cut_hw(self):
        batch = self.batch
        aicore_num = self.aicore_num
        if self.small_hw:
            aicore_num = 1

        loop_b = batch // aicore_num
        if loop_b == 0:
            loop_b = 1
            aicore_num = batch % aicore_num
        elif batch % aicore_num != 0:
            loop_b += 1

        tik_instance = self.tik_instance
        with tik_instance.for_range(0, loop_b) as loop_idx:
            with tik_instance.for_range(0, aicore_num,
                                        block_num=aicore_num) as block_idx:
                batch_idx = block_idx + loop_idx * aicore_num
                with tik_instance.if_scope(batch_idx < batch):
                    self._cut_hw_one_batch(block_idx + loop_idx * aicore_num)

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.data_input_grads,
                                      self.data_input_image,
                                      self.data_output_image],
                              outputs=[self.data_output])
        return tik_instance

    def _all_gm2ub_cut_hw(self, gm_index, factor):
        self._gm2ub(self.input_image_ub,
                    self.data_input_image[gm_index], factor, self.channels)
        self._gm2ub(self.input_grads_ub,
                    self.data_input_grads[gm_index], factor, self.channels)
        self._gm2ub(self.output_image_ub,
                    self.data_output_image[gm_index], factor, self.channels)

    def _cut_channels_one_batch(self, batch_idx):
        radius = self.depth_radius
        if radius == 0:
            channels_out_one_loop = 16
        else:
            channels_out_one_loop = 6 * radius
        cut_hw_factor_size = 0
        while cut_hw_factor_size < self.ub_dtype_size \
                and channels_out_one_loop > 1:
            im_seg_count = channels_out_one_loop + \
                           MORE_THREE_RADIUS * radius
            ig_seg_count = im_seg_count
            om_seg_count = im_seg_count
            pre_norm_seg_count = im_seg_count + 1
            norm_seg_count = channels_out_one_loop + \
                             MORE_TWO_RADIUS * radius
            norm_beta_seg_count = norm_seg_count
            pre_norm2_seg_count = norm_seg_count + 1
            norm2_seg_count = channels_out_one_loop + radius
            output_seg_count = channels_out_one_loop
            convent_ub_seg_count = 0
            if self.ub_dtype_size != self.dtype_size:
                convent_ub_seg_count = im_seg_count * self.ub_dtype_size / \
                                       self.dtype_size

            segment_count = (im_seg_count +
                             ig_seg_count +
                             om_seg_count +
                             pre_norm_seg_count +
                             norm_seg_count +
                             norm_beta_seg_count +
                             pre_norm2_seg_count +
                             norm2_seg_count +
                             output_seg_count +
                             convent_ub_seg_count) * 2

            cut_hw_factor_size = _get_ub_segment_size(segment_count)
            channels_out_one_loop -= 1
        hw_factor = cut_hw_factor_size // self.ub_dtype_size
        hw_factor = (hw_factor * self.dtype_size // constant.BLOCK_SIZE) * \
                    constant.BLOCK_SIZE // self.dtype_size
        width = self.width
        height = self.height
        loop = height * width // hw_factor
        tail = height * width % hw_factor

        self.channels_out_one_loop = channels_out_one_loop
        self.hw_factor4cut_channels = hw_factor
        self.tail = tail
        tik_instance = self.tik_instance
        batch_size = width * height * self.channels
        thread_num = max([min([loop, 2]), 1])
        if loop == 0:
            if tail != 0:
                self._alloc_all_ub_cut_channel()
                gm_index = batch_idx * batch_size
                self._treat_cut_channels_one_part(gm_index, tail,
                                                  channels_out_one_loop)
        else:
            with tik_instance.for_range(0, loop,
                                        thread_num=thread_num) as idx:
                self._alloc_all_ub_cut_channel()
                if tail != 0:
                    with tik_instance.if_scope(idx == 0):
                        gm_index = batch_idx * batch_size
                        self._treat_cut_channels_one_part(gm_index, tail,
                                                          channels_out_one_loop)

                gm_index = batch_idx * batch_size + tail + idx * hw_factor
                self._treat_cut_channels_one_part(gm_index, hw_factor,
                                                  channels_out_one_loop)

    def _cut_channels(self):
        batch = self.batch
        aicore_num = self.aicore_num
        if self.small_hw:
            aicore_num = 1
        loop_b = batch // aicore_num
        if loop_b == 0:
            loop_b = 1
            aicore_num = batch % aicore_num
        elif batch % aicore_num != 0:
            loop_b += 1

        tik_instance = self.tik_instance
        with tik_instance.for_range(0, loop_b) as loop_idx:
            with tik_instance.for_range(0, aicore_num,
                                        block_num=aicore_num) as block_idx:
                batch_idx = block_idx + loop_idx * aicore_num
                with tik_instance.if_scope(batch_idx < batch):
                    self._cut_channels_one_batch(block_idx +
                                                 loop_idx * aicore_num)

        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.data_input_grads,
                                      self.data_input_image,
                                      self.data_output_image],
                              outputs=[self.data_output])
        return tik_instance

    def _tilling_hw(self, channels_count):
        channels_size = channels_count * self.ub_dtype_size
        factor = math.floor(self.ub_segment_size / channels_size)
        factor = (factor * self.ub_dtype_size // constant.BLOCK_SIZE) * \
                 constant.BLOCK_SIZE // self.ub_dtype_size
        factor = (factor * self.dtype_size // constant.BLOCK_SIZE) * \
                 constant.BLOCK_SIZE // self.dtype_size
        tail = (self.width * self.height) % factor
        loop = self.width * self.height // factor

        return loop, factor, tail

    def _treat_cut_hw_one_part(self, gm_index, count):
        count_one_window = self._get_32bytes_align_count(count)
        count_one_loop = self.channels * count_one_window
        self._all_gm2ub_cut_hw(gm_index, count)
        self._cut_hw_compute(count_one_loop, count_one_window)
        self._ub2gm(gm_index, count, self.channels)

    def _treat_cut_channels_one_part(self, gm_index, count, channels_count):
        tik_instance = self.tik_instance
        radius = self.depth_radius
        loop_c = self.channels // channels_count
        tail_c = self.channels % channels_count
        count_one_window = self._get_32bytes_align_count(count)
        width = self.width
        height = self.height
        if loop_c == 0:
            self._all_gm2ub_cut_channels(gm_index, count, 0, tail_c)
            self._cut_channels_compute(count_one_window, 0, tail_c, 0)
            self._ub2gm(gm_index, count, tail_c)
        else:
            channels = channels_count + MORE_THREE_RADIUS * radius
            self._all_gm2ub_cut_channels(gm_index, count, 0, channels)
            self._cut_channels_compute(count_one_window, 0, channels, 0)
            self._ub2gm(gm_index, count, channels_count)
            if tail_c != 0 or loop_c > 1:
                self._backup_ub_cut_channels(count_one_window, True)
            if loop_c > 1:
                channels = channels_count
                ub_idx = MORE_THREE_RADIUS * radius * count_one_window
                with tik_instance.for_range(1, loop_c) as c_idx:
                    gm_in_idx = gm_index + width * height * (MORE_THREE_RADIUS *
                                                             radius +
                                                             channels_count *
                                                             c_idx)
                    gm_out_idx = gm_index + (channels_count * c_idx *
                                             width * height)
                    if tail_c > MORE_THREE_RADIUS * radius:
                        self._all_gm2ub_cut_channels(gm_in_idx, count, ub_idx,
                                                     channels)
                    else:
                        with tik_instance.if_scope(c_idx != loop_c - 1):
                            self._all_gm2ub_cut_channels(gm_in_idx, count,
                                                         ub_idx, channels)
                        with tik_instance.else_scope():
                            if channels + tail_c - MORE_THREE_RADIUS*radius > 0:
                                self._all_gm2ub_cut_channels(gm_in_idx, count,
                                                             ub_idx,
                                                             channels + tail_c -
                                                             MORE_THREE_RADIUS *
                                                             radius)
                    self._cut_channels_compute(count_one_window, ub_idx,
                                               channels, channels_count * c_idx)
                    self._ub2gm(gm_out_idx,
                                count, channels)

                    self._backup_ub_cut_channels(count_one_window, False)
            if tail_c > 0:
                gm_in_idx = gm_index + width * height * (MORE_THREE_RADIUS *
                                                         radius +
                                                         channels_count *
                                                         loop_c)
                gm_out_idx = gm_index + (channels_count * loop_c *
                                         width * height)
                ub_idx = MORE_THREE_RADIUS * radius * count_one_window

                if tail_c > MORE_THREE_RADIUS * radius:
                    tail_channels = tail_c - MORE_THREE_RADIUS * radius
                    self._all_gm2ub_cut_channels(gm_in_idx, count, ub_idx,
                                                 tail_channels)
                self._cut_channels_compute(count_one_window, ub_idx, tail_c,
                                           channels_count * loop_c)
                self._ub2gm(gm_out_idx,
                            count, tail_c)

    def _all_gm2ub_cut_channels(self, gm_index, count, ub_idx, channels):
        self._gm2ub(self.input_image_ub[ub_idx:],
                    self.data_input_image[gm_index], count,
                    channels)
        self._gm2ub(self.input_grads_ub[ub_idx:],
                    self.data_input_grads[gm_index], count,
                    channels)
        self._gm2ub(self.output_image_ub[ub_idx:],
                    self.data_output_image[gm_index], count,
                    channels)

    # pylint: disable=too-many-statements
    def _cut_channels_compute(self, count_one_window, ub_idx, channels,
                              channel_idx):
        radius = self.depth_radius
        count_one_loop = channels * count_one_window

        if ub_idx == 0:
            norm_ub_idx = 0
            norm_ub2_idx = 0
            input_ub_idx = 0
            pre_norm_ub_idx = 0
            pre_norm2_ub_idx = 0
            channels_count = channels - radius
            second_channels_out = channels_count - radius
            norm_count = count_one_loop - radius * count_one_window
            output_count = count_one_loop - MORE_THREE_RADIUS * radius * \
                           count_one_window
            sum_first_start_idx = 0
            sum_second_start_idx = 0
            square_value_ub_idx = 0
        else:
            norm_ub_idx = MORE_TWO_RADIUS * radius * count_one_window
            norm_ub2_idx = radius * count_one_window
            input_ub_idx = MORE_THREE_RADIUS * radius * count_one_window
            pre_norm_ub_idx = norm_ub_idx + count_one_window
            pre_norm2_ub_idx = norm_ub2_idx + count_one_window
            square_value_ub_idx = input_ub_idx + count_one_window
            channels_count = channels
            second_channels_out = channels
            norm_count = count_one_loop
            output_count = count_one_loop
            sum_first_start_idx = channel_idx + MORE_TWO_RADIUS * radius
            sum_second_start_idx = channel_idx + radius

        to_square_ub = self.input_image_ub[input_ub_idx:]
        square_value_ub = self.pre_norm_ub[square_value_ub_idx:]
        self._vmul(square_value_ub, to_square_ub, to_square_ub, count_one_loop)

        norm_ub = self.norm_ub[norm_ub_idx:]
        pre_norm_ub = self.pre_norm_ub[pre_norm_ub_idx:]
        self._sum4windows_cut_channels(norm_ub, pre_norm_ub, count_one_window,
                                       sum_first_start_idx, channels_count)

        norm_value = self.norm_beta_ub[norm_ub_idx:]
        self._vmuls(norm_value, norm_ub, self.alpha, norm_count)
        self._vadds(norm_value, norm_value, self.bias, norm_count)

        pre_norm2_ub = self.pre_norm2_ub[pre_norm_ub_idx:]
        output_image_ub = self.output_image_ub[norm_ub_idx:]
        input_grads_ub = self.input_grads_ub[norm_ub_idx:]
        self._vmul(pre_norm2_ub, output_image_ub, input_grads_ub, norm_count)
        self._vdiv(pre_norm2_ub, pre_norm2_ub, norm_value, norm_count)

        norm_beta_ub = self.norm_beta_ub[norm_ub_idx:]
        self._vpow(norm_beta_ub, norm_beta_ub, -1 * self.beta, norm_count)
        self._vmul(norm_beta_ub, norm_beta_ub, input_grads_ub, norm_count)

        pre_norm2_ub = self.pre_norm2_ub[pre_norm2_ub_idx:]
        norm_ub2 = self.norm2_ub[norm_ub2_idx:]
        self._sum4windows_cut_channels(norm_ub2, pre_norm2_ub, count_one_window,
                                       sum_second_start_idx,
                                       second_channels_out)
        self._vmul(self.data_output_ub, self.norm2_ub,
                   self.input_image_ub, output_count)
        self._vmuls(self.data_output_ub, self.data_output_ub,
                    -2.0 * self.alpha * self.beta, output_count)

        self._vadd(self.data_output_ub, self.norm_beta_ub,
                   self.data_output_ub, output_count)

    def _backup_ub_cut_channels(self, count_one_window, first_backup):
        src_idx = self.channels_out_one_loop * count_one_window

        if first_backup:
            pre_norm_idx = src_idx - count_one_window
        else:
            pre_norm_idx = src_idx

        radius = self.depth_radius
        if radius != 0:
            self._vadds(self.input_image_ub[0:], self.input_image_ub[src_idx:],
                        0.0, MORE_THREE_RADIUS * radius * count_one_window)

            self._vadds(self.input_grads_ub[0:], self.input_grads_ub[src_idx:],
                        0.0, MORE_THREE_RADIUS * radius * count_one_window)

            self._vadds(self.output_image_ub[0:],
                        self.output_image_ub[src_idx:],
                        0.0, MORE_THREE_RADIUS * radius * count_one_window)

            self._vadds(self.norm_ub[0:], self.norm_ub[src_idx:], 0.0,
                        MORE_TWO_RADIUS * radius * count_one_window)

            self._vadds(self.norm_beta_ub[0:], self.norm_beta_ub[src_idx:], 0.0,
                        MORE_TWO_RADIUS * radius * count_one_window)

            norm2_ub = self.norm2_ub
            self._vadds(norm2_ub, norm2_ub[src_idx:], 0.0,
                        radius * count_one_window)

        self._vadds(self.pre_norm_ub[0:], self.pre_norm_ub[pre_norm_idx:], 0.0,
                    MORE_THREE_RADIUS*radius*count_one_window +
                    count_one_window)

        self._vadds(self.pre_norm2_ub[0:], self.pre_norm2_ub[pre_norm_idx:],
                    0.0, MORE_TWO_RADIUS*radius*count_one_window +
                    count_one_window)


def _check_param(input_grads, input_image, output_image, depth_radius,
                 kernel_name):
    input_grads_shape = input_grads.get("shape")
    input_grads_dtype = input_grads.get("dtype")

    input_image_shape = input_image.get("shape")
    input_image_dtype = input_image.get("dtype")

    output_image_shape = output_image.get("shape")
    output_image_dtype = output_image.get("dtype")

    util.check_tensor_shape_size(input_grads_shape)
    util.check_tensor_shape_size(input_image_shape)
    util.check_tensor_shape_size(output_image_shape)

    util.check_dtype_rule(input_grads_dtype, ("float16", "float32"))
    util.check_dtype_rule(input_image_dtype, ("float16", "float32"))
    util.check_dtype_rule(output_image_dtype, ("float16", "float32"))

    util.check_kernel_name(kernel_name)

    if len(input_grads_shape) != 4:
        raise RuntimeError("The shape of tensor must be 4D.")

    if (input_grads_dtype != input_image_dtype or
            input_grads_dtype != output_image_dtype):
        raise RuntimeError(
            "The dtype of input_grads,input_image,output_image must be same")

    if (input_grads_shape != input_image_shape or
            input_grads_shape != output_image_shape):
        raise RuntimeError(
            "The shape of input_grads,input_image,output_image must be same")

    if depth_radius > 48 or depth_radius < 0:
        raise RuntimeError("depth_radius should not between 0 and 48.")


# pylint: disable=too-many-arguments, unused-argument, invalid-name
@util.check_input_type(dict, dict, dict, dict, int, (float, int), (float, int),
                       (float, int), str)
def lrn_grad(grads, x, y, z, depth_radius=5,
             bias=1.0, alpha=1.0, beta=0.5, kernel_name="lrn_grad"):
    """
    calculating data

    Parameters
    ----------
    grads : dict
        shape and dtype of input, with shape [batch, channels, height, width]
    x : dict
        shape and dtype of input, with shape [batch, channels, height, width]
    y : dict
        shape and dtype of input, with shape [batch, channels, height, width]
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    depth_radius : int
        A depth radius, defaults to 5, max value is 48
    bias : float
        An offset (usually > 0 to avoid dividing by 0), defaults to 1
    alpha : float
        A scale factor, usually positive, defaults to 1
    beta : beta
        An exponent, defaults to 0.5
    kernel_name : str
        kernel name, default value is "lrn_grad"

    Returns
    -------
    tik_instance : Tik
    """

    _check_param(grads, x, y, depth_radius, kernel_name)

    lrngrad = LrnGrad(grads.get("shape"), grads.get("dtype"),
                      depth_radius, bias, alpha, beta, kernel_name)
    return lrngrad.lrn_grad()
