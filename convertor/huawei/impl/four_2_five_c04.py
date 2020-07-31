#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

four2five_c04
"""
# pylint: disable=too-many-lines,import-error
from te import platform as tbe_platform
from te import tik
from topi.cce import util
import math

# available ub size: split double ub
TOTAL_UB_MEMORY = (tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - 1024) // 2
# available number of cores
MAX_CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# bytes of type float16
SIZE_TWO_BYTES = 2
# bytes of type float32
SIZE_FOUR_BYTES = 4
# minimum unit of data_move: 32Bytes
DATA_MOVE_MIN_UNIT = 32

# Maximum Stride
MAX_STRIDE = 65535

MASK = 128

MASK_FP32 = 64

MASK_FP16 = 128

MAX_REPEAT = 255


def _cal_core(tik_instance, total_core_loop_num, num_core, core_number):
    """
    calculate the loop number on each core
    """
    core_loop = tik_instance.Scalar("uint64")
    sum_core = tik_instance.Scalar("uint64")
    with tik_instance.if_scope(num_core < total_core_loop_num %
                               MAX_CORE_NUM):
        core_loop.set_as((total_core_loop_num + core_number - 1)//
                         core_number)
        sum_core.set_as(core_loop*num_core)
    with tik_instance.else_scope():
        core_loop.set_as(total_core_loop_num//core_number)
        sum_core.set_as((core_loop + 1)*(total_core_loop_num % MAX_CORE_NUM) +
                        core_loop*(num_core - total_core_loop_num %
                                   MAX_CORE_NUM))
    return core_loop, sum_core


def factorization(number, core_num, hwc0, step_move):

    number_0 = 1
    number_1 = number
    n0n1=[number_0, number_1]

    i = 2
    while i**2 <= number:
        if number % i == 0:
            number_0 = int(number / i)
            number_1 = i
            n0n1.append(number_0)
            n0n1.append(number_1)
        i += 1

    combination = int(len(n0n1)/2)
    core_list = []
    for i in range(combination):
        n0 = n0n1[i*2]
        n1 = n0n1[i*2+1]
        n1hwc0 = hwc0 * n1
        if n1hwc0 >= step_move:
            core_list.append(n0)
        n1hwc0 = hwc0 * n0
        if n1hwc0 >= step_move:
            core_list.append(n1)

    # only support single core
    if core_list == []:
        core_list.append(1)

    core_list_copy = [(i - core_num) for i in core_list]
    min_value = min(core_list_copy)
    max_value = max(core_list_copy)
    if min_value >= 0:
        n0 = core_list[core_list_copy.index(min(core_list_copy))]
    elif max_value <= 0:
        n0 = core_list[core_list_copy.index(max(core_list_copy))]
    else:
        core_list_copy_2 = [i for i in core_list_copy if i >=0]
        n0 = min(core_list_copy_2) + core_num

    return n0


class four2fiveCompute(object):

    def __init__(self, input_shape, output_shape, dtype, src_format, kernel_name):

        self.input_shape = list(input_shape).copy()
        self.output_shape = list(output_shape).copy()

        self.input_sum_elements = self.calc_element(self.input_shape)
        self.output_sum_elements = self.calc_element(self.output_shape)

        self.kernel_name = kernel_name
        self.dtype = dtype
        self.format = src_format

        self.mask, self.num_bit, self.maximum_size_ub = self.params_pattern(self.dtype)
        self.step_move = DATA_MOVE_MIN_UNIT // self.num_bit
        self.maximum_gm = math.ceil(self.input_sum_elements / self.step_move) * self.step_move

    def params_pattern(self, dtype):

        num_bit = int(SIZE_TWO_BYTES)
        mask = int(MASK_FP16)
        if dtype == "float32":
            num_bit = int(SIZE_FOUR_BYTES)
            mask = int(MASK_FP32)
        maximum_size_ub = int(TOTAL_UB_MEMORY // num_bit)

        return mask, num_bit, maximum_size_ub

    def calc_element(self, shape):

        sum_result = 1
        if len(shape) == 0:
            return sum_result
        for i in range(len(shape)):
            sum_result = sum_result * shape[i]
        return sum_result

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        self.input_x_gm = tik_instance.Tensor(self.dtype,
                                              [self.input_sum_elements, ],
                                              name="input_x_gm",
                                              scope=tik.scope_gm)
        self.output_y_gm = tik_instance.Tensor(self.dtype,
                                               [self.output_sum_elements, ],
                                               name="output_y_gm",
                                               scope=tik.scope_gm)

    def set_ub_tensor(self, tik_instance, psm_out, psm_in):

        input_x_ub = tik_instance.Tensor(self.dtype, [psm_in,], name="input_x_ub",
                                         scope=tik.scope_ubuf)
        input_y_ub = tik_instance.Tensor(self.dtype, [psm_out,], name="input_y_ub",
                                         scope=tik.scope_ubuf)
        input_z_ub = tik_instance.Tensor(self.dtype, [16,], name="input_z_ub",
                                         scope=tik.scope_ubuf)

        return input_x_ub, input_y_ub, input_z_ub

    def set_vector_dup(self, tik_instance, psm, dst, number):

        dup_psm = MAX_REPEAT * self.mask
        dup_repeat_merchant = psm // dup_psm
        dup_repeat_remainder = psm % dup_psm
        dst_blk_stride = 1
        dst_rep_stride = 8

        with tik_instance.for_range(0, dup_repeat_merchant) as i:
            tik_instance.vector_dup(self.mask,
                                    dst[0 + i * dup_psm],
                                    number,
                                    MAX_REPEAT,
                                    dst_blk_stride,
                                    dst_rep_stride)

        if dup_repeat_remainder != 0:
            repeats = dup_repeat_remainder // self.mask
            dup_remainder = dup_repeat_remainder % self.mask
            if repeats != 0:
                tik_instance.vector_dup(self.mask,
                                        dst[dup_repeat_merchant * dup_psm],
                                        number,
                                        repeats,
                                        dst_blk_stride,
                                        dst_rep_stride)
            if dup_remainder != 0:
                tik_instance.vector_dup(dup_remainder,
                                        dst[dup_repeat_merchant * dup_psm + repeats * self.mask],
                                        number,
                                        1,
                                        dst_blk_stride,
                                        dst_rep_stride)

    def set_move_in_zero(self, tik_instance, psm, src_x_offset, dst, src, src_x_index):

        nburst_gm2ub = 1
        burstlen_gm2ub = int(psm * self.num_bit / 32)
        srcstride_gm2ub = 0
        dststride_gm2ub = 0
        tik_instance.data_move(dst[0],
                               src[src_x_index - src_x_offset],
                               0,
                               nburst_gm2ub,
                               burstlen_gm2ub,
                               srcstride_gm2ub,
                               dststride_gm2ub)

    def set_move_in_one(self, tik_instance, psm, dst, src, src_x_index):

        nburst_gm2ub = 1
        burstlen_gm2ub = int(psm * self.num_bit / 32)
        srcstride_gm2ub = 0
        dststride_gm2ub = 0
        tik_instance.data_move(dst[0],
                               src[src_x_index],
                               0,
                               nburst_gm2ub,
                               burstlen_gm2ub,
                               srcstride_gm2ub,
                               dststride_gm2ub)

    def set_move_in_two(self, tik_instance, n_new, len_hw,
                        src_gm_tensor, src_gm_index, dst_ub_tensor):
        n = self.input_shape[-1]
        c = self.input_shape[-2]
        nburst_gm2ub = len_hw * c
        burstlen_gm2ub = int(n_new * self.num_bit / 32)
        srcstride_gm2ub_ori = n - n_new
        srcstride_gm2ub = srcstride_gm2ub_ori * self.num_bit // 32
        dststride_gm2ub = 0

        condition_0 = srcstride_gm2ub_ori % self.step_move == 0
        condition_1 = srcstride_gm2ub <= MAX_STRIDE

        if condition_0 and condition_1:
            tik_instance.data_move(dst_ub_tensor[0],
                                   src_gm_tensor[src_gm_index],
                                   0,
                                   nburst_gm2ub,
                                   burstlen_gm2ub,
                                   srcstride_gm2ub,
                                   dststride_gm2ub)
        else:
            src_gm_gap = n
            dst_ub_gap = n_new
            dst_ub_index = 0
            with tik_instance.for_range(0, nburst_gm2ub) as i:
                src = src_gm_index + src_gm_gap * i
                dst = dst_ub_index + dst_ub_gap * i
                tik_instance.data_move(dst_ub_tensor[dst],
                                       src_gm_tensor[src],
                                       0,
                                       1,
                                       burstlen_gm2ub,
                                       0,
                                       0)

    def set_move_out_zero(self, tik_instance, n_new, len_hw,
                          dst_gm_tensor, dst_gm_index,
                          src_ub_tensor, reg, reg_z_ub, tail_hw):

        c0 = 4
        srcstride_ub2gm = 0
        nburst_ub2gm = n_new
        burstlen_ub2gm = c0 * len_hw
        burstlen_ub2gm_ori = math.ceil(burstlen_ub2gm / self.step_move) * self.step_move
        dststride_ub2gm = self.input_shape[0] * self.input_shape[1] * c0 - c0 * len_hw
        dststride_ub2gm_ori = math.ceil(dststride_ub2gm / self.step_move) * self.step_move

        condition_0 = dststride_ub2gm_ori == dststride_ub2gm
        condition_1 = burstlen_ub2gm == burstlen_ub2gm_ori
        condition_2 = (dststride_ub2gm * self.num_bit // 32) <= MAX_STRIDE

        if condition_0 and condition_1 and condition_2:
            burstlen_ub2gm = burstlen_ub2gm * self.num_bit // 32
            dststride_ub2gm = dststride_ub2gm * self.num_bit // 32
            tik_instance.data_move(dst_gm_tensor[dst_gm_index],
                                   src_ub_tensor[0],
                                   0,
                                   nburst_ub2gm,
                                   burstlen_ub2gm,
                                   srcstride_ub2gm,
                                   dststride_ub2gm)
        else:
            dst_gm_gap = self.input_shape[0] * self.input_shape[1] * c0
            src_y_gap = burstlen_ub2gm_ori
            if burstlen_ub2gm >= self.step_move:
                offset = burstlen_ub2gm % self.step_move
                burstlen_ub2gm = (burstlen_ub2gm - offset) * self.num_bit // 32
                dst_gm_align = dst_gm_index

                with tik_instance.for_range(0, n_new) as i:
                    dst_gm_align = dst_gm_align + i * dst_gm_gap
                    tik_instance.data_move(dst_gm_tensor[dst_gm_align],
                                           src_ub_tensor[i * src_y_gap],
                                           0,
                                           1,
                                           burstlen_ub2gm,
                                           0,
                                           0)

                if offset != 0:
                    dst_gm_n_align = dst_gm_index
                    with tik_instance.for_range(0, n_new) as i:
                        with tik_instance.for_range(0, self.step_move) as j:
                            index_y_ub = i * src_y_gap + c0 * len_hw - self.step_move + j
                            reg.set_as(src_ub_tensor[index_y_ub])
                            reg_z_ub[j].set_as(reg)
                        dst_gm_n_align = dst_gm_n_align + i * dst_gm_gap + c0 * len_hw - self.step_move
                        tik_instance.data_move(dst_gm_tensor[dst_gm_n_align],
                                               reg_z_ub[0],
                                               0,
                                               1,
                                               1,
                                               0,
                                               0)
            else:
                if tail_hw != 0:
                    offset = self.step_move - burstlen_ub2gm
                    dst_gm_n_align = dst_gm_index
                    with tik_instance.for_range(0, n_new) as i:
                        with tik_instance.for_range(0, burstlen_ub2gm) as j:
                            index_y_ub = i * src_y_gap + j
                            reg.set_as(src_ub_tensor[index_y_ub])
                            reg_z_ub[offset+j].set_as(reg)

                        dst_gm_n_align = dst_gm_n_align + i * dst_gm_gap - offset
                        tik_instance.data_move(dst_gm_tensor[dst_gm_n_align],
                                               reg_z_ub[0],
                                               0,
                                               1,
                                               1,
                                               0,
                                               0)
                else:
                    nburst_ub2gm = 1
                    burstlen_ub2gm = math.ceil(len_hw * c0 * n_new / self.step_move)
                    burstlen_ub2gm = burstlen_ub2gm * self.step_move * self.num_bit //32
                    srcstride_ub2gm = 0
                    dststride_ub2gm = 0
                    tik_instance.data_move(dst_gm_tensor[dst_gm_index],
                                           src_ub_tensor[0],
                                           0,
                                           nburst_ub2gm,
                                           burstlen_ub2gm,
                                           srcstride_ub2gm,
                                           dststride_ub2gm)

    def set_move_out_one(self, tik_instance, n_new, len_hw,
                         dst_gm_tensor, dst_gm_index, src_ub_tensor):
        c0 = 4
        nburst_ub2gm = n_new
        srcstride_ub2gm = 0
        burstlen_ub2gm = c0 * len_hw * self.num_bit // 32
        dststride_ub2gm = (self.input_shape[0] * self.input_shape[1] * c0 - c0 * len_hw) * self.num_bit
        condition_0 = dststride_ub2gm <= MAX_STRIDE
        condition_1 = dststride_ub2gm % 32 ==0
        if condition_0 and condition_1:
            dststride_ub2gm = dststride_ub2gm // 32
            tik_instance.data_move(dst_gm_tensor[dst_gm_index],
                                   src_ub_tensor[0],
                                   0,
                                   nburst_ub2gm,
                                   burstlen_ub2gm,
                                   srcstride_ub2gm,
                                   dststride_ub2gm)
        else:
            dst_gm_gap = self.input_shape[0] * self.input_shape[1] * c0
            src_y_gap = len_hw * c0
            with tik_instance.for_range(0,n_new) as i:
                tik_instance.data_move(dst_gm_tensor[dst_gm_index + i * dst_gm_gap],
                                       src_ub_tensor[i * src_y_gap],
                                       0,
                                       1,
                                       burstlen_ub2gm,
                                       0,
                                       0)

    def four2five_nchw_case0(self, tik_instance, param):

        core_number = param.get("core_num")
        total_core_loop_num = param.get("total_core_loop_num")
        tiling_shape = param.get("tiling_shape")
        psm_in_ac = param.get("psm_in_ac")
        psm_out_ac = param.get("psm_out_ac")
        psm_move_in = param.get("psm_move_in")
        psm_move_out = param.get("psm_move_out")
        hw = tiling_shape[-1] * tiling_shape[-2]
        split_hw = self.step_move
        c_axis = tiling_shape[0]

        if psm_move_out <= self.maximum_size_ub:
            psm_out = psm_move_out
            psm_in = psm_move_in
            tail = 0
            tail_block = hw
        else:
            merchant_hw = hw // self.step_move
            remainder_hw = hw % self.step_move

            for i in range(merchant_hw):
                split_hw = self.step_move * (i+1)
                psm_out = 4 * split_hw
                psm_in = c_axis * split_hw
                if psm_out > self.maximum_size_ub:
                    split_hw = self.step_move * (i)
                    psm_out = 4 * split_hw
                    psm_in = c_axis * split_hw
                    break

            tail = hw // split_hw
            tail_block = hw % split_hw

        input_x_ub = tik_instance.Tensor(self.dtype, [psm_in,], name="input_x_ub",
                                         scope=tik.scope_ubuf)
        input_y_ub = tik_instance.Tensor(self.dtype, [psm_out,], name="input_y_ub",
                                         scope=tik.scope_ubuf)
        input_z_ub = tik_instance.Tensor(self.dtype, [16,], name="input_z_ub",
                                         scope=tik.scope_ubuf)
        reg = tik_instance.Scalar(dtype=self.dtype)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:

                merchant = sum_core + num_core_loop
                if tail_block != 0:
                    src_x_index = psm_in_ac * merchant + tail * split_hw
                    if tail == 0:
                        nburst_gm2ub = 1
                        burstlen_gm2ub = int(psm_move_in * self.num_bit / 32)
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 0
                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)

                        with tik_instance.for_range(0, c_axis) as i:
                            with tik_instance.for_range(0, tail_block) as j:
                                src_x_ub_index = i * tail_block + j
                                dst_y_ub_index = j * 4 + i
                                reg.set_as(input_x_ub[src_x_ub_index])
                                input_y_ub[dst_y_ub_index].set_as(reg)

                        reg.set_as(0)
                        with tik_instance.for_range(c_axis, 4) as i:
                            with tik_instance.for_range(0, tail_block) as j:
                                dst_y_ub_index = j * 4 + i
                                input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        dst_gm = merchant * psm_out_ac
                        offset = psm_out_ac % self.step_move
                        nburst_ub2gm = 1
                        burstlen_ub2gm_ori = (psm_out_ac - offset) * self.num_bit // 32
                        srcstride_ub2gm = 0
                        dststride_ub2gm = 0

                        if burstlen_ub2gm_ori == 0:
                            burstlen_ub2gm = 1
                        else:
                            burstlen_ub2gm = burstlen_ub2gm_ori

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

                        if burstlen_ub2gm_ori != 0 and offset != 0:
                            with tik_instance.for_range(0, self.step_move) as i:

                                index_y_ub = psm_out_ac - self.step_move + i
                                reg.set_as(input_y_ub[index_y_ub])
                                input_z_ub[i].set_as(reg)

                            dst_gm_extra = dst_gm + psm_out_ac - self.step_move

                            tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                   input_z_ub[0],
                                                   0,
                                                   1,
                                                   1,
                                                   0,
                                                   0)

                    else:
                        src_x_offset = tail_block % self.step_move
                        srcstride_gm2ub = int(tail * split_hw * self.num_bit / 32)
                        if src_x_offset == 0 and srcstride_gm2ub <= MAX_STRIDE:
                            nburst_gm2ub = c_axis
                            burstlen_gm2ub = int(tail_block * self.num_bit / 32)
                            dststride_gm2ub = 0
                            move_forward = 0

                            tik_instance.data_move(input_x_ub[0],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub)

                        else:
                            srcstride_gm2ub = 0
                            move_forward = self.step_move - src_x_offset
                            with tik_instance.for_range(0, c_axis) as i:

                                src_x_gm_gap = hw
                                dst_x_ub_gap = tail_block + move_forward
                                nburst_gm2ub = 1
                                burstlen_gm2ub = dst_x_ub_gap * self.num_bit // 32
                                dststride_gm2ub = 0
                                tik_instance.data_move(input_x_ub[0 + i * dst_x_ub_gap],
                                                       self.input_x_gm[src_x_index - move_forward + i * src_x_gm_gap],
                                                       0,
                                                       nburst_gm2ub,
                                                       burstlen_gm2ub,
                                                       srcstride_gm2ub,
                                                       dststride_gm2ub)
                        # reorder
                        with tik_instance.for_range(0, c_axis) as i:
                            with tik_instance.for_range(0, tail_block + move_forward) as j:
                                src_x_ub_index = i * (tail_block + move_forward) + j
                                dst_y_ub_index = j * 4 + i
                                reg.set_as(input_x_ub[src_x_ub_index])
                                input_y_ub[dst_y_ub_index].set_as(reg)

                        reg.set_as(0)
                        with tik_instance.for_range(c_axis, 4) as i:
                            with tik_instance.for_range(0, tail_block + move_forward) as j:
                                dst_y_ub_index = j * 4 + i
                                input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out serial
                        dst_gm = merchant * psm_out_ac + (tail * split_hw - move_forward) * 4
                        nburst_ub2gm = 1
                        burstlen_ub2gm = (tail_block + move_forward) * 4 * self.num_bit // 32
                        srcstride_ub2gm = 0
                        dststride_ub2gm = 0

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

                if tail != 0:
                    with tik_instance.for_range(0, tail) as num_ub_loop:

                        src_x_index = psm_in_ac * merchant + num_ub_loop * split_hw
                        srcstride_gm2ub = (hw - split_hw) * self.num_bit / 32
                        srcstirde_gm2ub_ceil = math.ceil(srcstride_gm2ub)

                        if srcstride_gm2ub == srcstirde_gm2ub_ceil and srcstride_gm2ub <= MAX_STRIDE:
                            nburst_gm2ub = c_axis
                            burstlen_gm2ub = int(split_hw * self.num_bit / 32)
                            srcstride_gm2ub = int(srcstride_gm2ub)
                            dststride_gm2ub = 0

                            tik_instance.data_move(input_x_ub[0],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub)

                        else:
                            srcstride_gm2ub = 0
                            with tik_instance.for_range(0, c_axis) as i:
                                src_x_gm_gap = hw
                                dst_x_ub_gap = split_hw
                                nburst_gm2ub = 1
                                burstlen_gm2ub = int(dst_x_ub_gap * self.num_bit / 32)
                                dststride_gm2ub = 0
                                tik_instance.data_move(input_x_ub[0 + i * dst_x_ub_gap],
                                                       self.input_x_gm[src_x_index + i * src_x_gm_gap],
                                                       0,
                                                       nburst_gm2ub,
                                                       burstlen_gm2ub,
                                                       srcstride_gm2ub,
                                                       dststride_gm2ub)

                        # reorder
                        with tik_instance.for_range(0, c_axis) as i:
                            with tik_instance.for_range(0, split_hw) as j:
                                src_x_ub_index = i * split_hw + j
                                dst_y_ub_index = j * 4 + i
                                reg.set_as(input_x_ub[src_x_ub_index])
                                input_y_ub[dst_y_ub_index].set_as(reg)

                        reg.set_as(0)
                        with tik_instance.for_range(c_axis, 4) as i:
                            with tik_instance.for_range(0, split_hw) as j:
                                dst_y_ub_index = j * 4 + i
                                input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out serial
                        dst_gm = merchant * psm_out_ac + (num_ub_loop * split_hw) * 4
                        nburst_ub2gm = 1
                        burstlen_ub2gm = split_hw * 4 * self.num_bit // 32
                        srcstride_ub2gm = 0
                        dststride_ub2gm = 0

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

    def four2five_nchw_casex(self, tik_instance, param):
        """
        in this case, hw must be small, but n0 can so many,
        so don't care about hw too large to save in ub
        """
        core_number = param.get("core_num")
        total_core_loop_num = param.get("total_core_loop_num")
        tiling_shape = param.get("tiling_shape")
        psm_in_ac = param.get("psm_in_ac")
        psm_out_ac = param.get("psm_out_ac")
        psm_move_in = param.get("psm_move_in")
        psm_move_out = param.get("psm_move_out")
        hw = tiling_shape[-1] * tiling_shape[-2]
        n1 = tiling_shape[0]
        c0, c1 = 4, 1
        c = tiling_shape[1]

        if psm_move_out <= self.maximum_size_ub:
            psm_out = psm_move_out
            psm_in = psm_move_in
            tail_n1 = 0
            tail_block_n1 = n1
            # not use
            split_n1 = -1

        else:
            for split_n1 in range(n1):
                split_n1 = split_n1 + 1
                psm_out = math.ceil(c0 * hw * split_n1 / self.step_move) * self.step_move
                psm_in = math.ceil(c * hw * split_n1 / self.step_move) * self.step_move
                if psm_out > self.maximum_size_ub:
                    split_n1 -= 1
                    psm_out = math.ceil(c0 * hw * split_n1 / self.step_move) * self.step_move
                    psm_in = math.ceil(c * hw * split_n1 / self.step_move) * self.step_move
                    break

            if split_n1 < 1:
                raise RuntimeError("In this case, split_n1 must >= 1")
            tail_n1 = n1 // split_n1
            tail_block_n1 = n1 % split_n1

        input_x_ub = tik_instance.Tensor(self.dtype, [psm_in,], name="input_x_ub",
                                         scope=tik.scope_ubuf)
        input_y_ub = tik_instance.Tensor(self.dtype, [psm_out,], name="input_y_ub",
                                         scope=tik.scope_ubuf)
        input_z_ub = tik_instance.Tensor(self.dtype, [16,], name="input_z_ub",
                                         scope=tik.scope_ubuf)
        reg = tik_instance.Scalar(dtype=self.dtype)
        src_x_offset = tik_instance.Scalar(dtype="int64")
        src_x_offset.set_as(0)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:

                merchant = sum_core + num_core_loop
                if tail_block_n1 != 0:
                    src_x_index = psm_in_ac * merchant + tail_n1 * split_n1 * c * hw

                    if tail_n1 == 0:
                        with tik_instance.if_scope(src_x_index + psm_in > self.maximum_gm):
                            src_x_offset.set_as(src_x_index + psm_in - self.maximum_gm)

                        nburst_gm2ub = 1
                        burstlen_gm2ub = int(psm_in * self.num_bit / 32)
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 0
                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index - src_x_offset],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)

                        # reorder
                        with tik_instance.for_range(0, tail_block_n1) as k:
                            with tik_instance.for_range(0, c) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    src_x_ub_index = (i * hw + j) + k * c * hw
                                    dst_y_ub_index = (j * 4 + i) + k * c0 * hw
                                    reg.set_as(input_x_ub[src_x_offset + src_x_ub_index])
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        reg.set_as(0)
                        with tik_instance.for_range(0, tail_block_n1) as k:
                            with tik_instance.for_range(c, 4) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    dst_y_ub_index = (j * 4 + i) + k * c0 * hw
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        dst_gm = merchant * psm_out_ac
                        offset = psm_out_ac % self.step_move
                        nburst_ub2gm = 1
                        burstlen_ub2gm_ori = (psm_out_ac - offset) * self.num_bit // 32
                        srcstride_ub2gm = 0
                        dststride_ub2gm = 0

                        if burstlen_ub2gm_ori == 0:
                            burstlen_ub2gm = 1
                        else:
                            burstlen_ub2gm = burstlen_ub2gm_ori

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

                        if burstlen_ub2gm_ori != 0 and offset != 0:
                            with tik_instance.for_range(0, self.step_move) as i:

                                index_y_ub = psm_out_ac - self.step_move + i
                                reg.set_as(input_y_ub[index_y_ub])
                                input_z_ub[i].set_as(reg)

                            dst_gm_extra = dst_gm + psm_out_ac - self.step_move

                            tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                   input_z_ub[0],
                                                   0,
                                                   1,
                                                   1,
                                                   0,
                                                   0)

                    else:
                        psm_in_tail_block_n1_ac = tail_block_n1 * c * hw
                        psm_in_tail_block_n1 = math.ceil(psm_in_tail_block_n1_ac / self.step_move) * self.step_move
                        with tik_instance.if_scope(src_x_index + psm_in_tail_block_n1 > self.maximum_gm):
                            src_x_offset.set_as(src_x_index + psm_in_tail_block_n1 - self.maximum_gm)

                        nburst_gm2ub = 1
                        burstlen_gm2ub = psm_in_tail_block_n1 * self.num_bit // 32
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 0

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index - src_x_offset],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)

                        # reorder
                        with tik_instance.for_range(0, tail_block_n1) as k:
                            with tik_instance.for_range(0, c) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    src_x_ub_index = (i * hw + j) + k * c * hw
                                    dst_y_ub_index = (j * 4 + i) + k * c0 * hw
                                    reg.set_as(input_x_ub[src_x_offset + src_x_ub_index])
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        reg.set_as(0)
                        with tik_instance.for_range(0, tail_block_n1) as k:
                            with tik_instance.for_range(c, 4) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    dst_y_ub_index = (j * 4 + i) + k * c0 * hw
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        psm_out_tail_block_n1_ac = tail_block_n1 * c0 * hw
                        psm_out_tail_block_n1 = math.ceil(psm_out_tail_block_n1_ac / self.step_move) * self.step_move
                        dst_gm = merchant * psm_out_ac + tail_n1 * split_n1 * c0 * hw

                        if psm_out_tail_block_n1_ac >= 32 // self.num_bit:
                            offset = psm_out_tail_block_n1_ac % self.step_move
                            nburst_ub2gm = 1
                            burstlen_ub2gm = (psm_out_tail_block_n1_ac - offset) * self.num_bit // 32
                            srcstride_ub2gm = 0
                            dststride_ub2gm = 0

                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   srcstride_ub2gm,
                                                   dststride_ub2gm)

                            if offset != 0:
                                with tik_instance.for_range(0, self.step_move) as i:
                                    index_y_ub = psm_out_tail_block_n1_ac - self.step_move + i
                                    reg.set_as(input_y_ub[index_y_ub])
                                    input_z_ub[i].set_as(reg)

                                dst_gm_extra = dst_gm + psm_out_tail_block_n1_ac - self.step_move
                                tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       1,
                                                       0,
                                                       0)
                        else:
                            offset = psm_out_tail_block_n1 - psm_out_tail_block_n1_ac
                            if psm_out_tail_block_n1_ac >= 32 // self.num_bit:
                                raise RuntimeError("In the branch, acture of datamove < 32B")
                            with tik_instance.for_range(0, psm_out_tail_block_n1_ac) as i:
                                reg.set_as(input_y_ub[i])
                                input_z_ub[i + offset].set_as(reg)

                            dst_gm_extra = dst_gm - offset
                            tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                   input_z_ub[0],
                                                   0,
                                                   1,
                                                   1,
                                                   0,
                                                   0)


                if tail_n1 != 0:
                    with tik_instance.for_range(0, tail_n1) as num_ub_loop:

                        src_x_index = psm_in_ac * merchant + num_ub_loop * split_n1 * hw * c
                        nburst_gm2ub = 1
                        burstlen_gm2ub = psm_in * self.num_bit // 32
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 0

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)
                        # reorder
                        with tik_instance.for_range(0, split_n1) as k:
                            with tik_instance.for_range(0, c) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    src_x_ub_index = (i * hw + j) + k * c * hw
                                    dst_y_ub_index = (j * 4 + i) + k * c0 * hw
                                    reg.set_as(input_x_ub[src_x_ub_index])
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        reg.set_as(0)
                        with tik_instance.for_range(0, split_n1) as k:
                            with tik_instance.for_range(c, 4) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    dst_y_ub_index = (j * 4 + i) + k * c0 * hw
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        dst_gm = merchant * psm_out_ac + num_ub_loop * split_n1 * hw * c0
                        offset = (split_n1 * hw * c0) % self.step_move
                        nburst_ub2gm = 1
                        burstlen_ub2gm = (split_n1 * hw * c0 - offset) * self.num_bit // 32
                        srcstride_ub2gm = 0
                        dststride_ub2gm = 0

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

                        if offset != 0:
                            with tik_instance.for_range(0, self.step_move) as i:
                                index_y_ub = split_n1 * hw * c0 - self.step_move + i
                                reg.set_as(input_y_ub[index_y_ub])
                                input_z_ub[i].set_as(reg)

                            dst_gm_extra = dst_gm + split_n1 * hw * c0 - self.step_move
                            tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                   input_z_ub[0],
                                                   0,
                                                   1,
                                                   1,
                                                   0,
                                                   0)

    def four2five_nhwc_case0(self, tik_instance, param):

        core_number = param.get("core_num")
        total_core_loop_num = param.get("total_core_loop_num")
        tiling_shape = param.get("tiling_shape")
        psm_in_ac = param.get("psm_in_ac")
        psm_out_ac = param.get("psm_out_ac")
        psm_move_in = param.get("psm_move_in")
        psm_move_out = param.get("psm_move_out")
        hw = tiling_shape[0] * tiling_shape[1]
        split_hw = self.step_move
        c = tiling_shape[2]
        c0 = 4

        merchant_hw = hw // self.step_move
        for i in range(merchant_hw):
            split_hw = self.step_move * (i+1)
            psm_out = c0 * split_hw
            psm_in = c * split_hw
            if psm_out > self.maximum_size_ub:
                split_hw = self.step_move * i
                psm_out = c0 * split_hw
                psm_in = c * split_hw
                break

        tail = hw // split_hw
        tail_block = hw % split_hw

        input_x_ub = tik_instance.Tensor(self.dtype, [psm_in,], name="input_x_ub",
                                         scope=tik.scope_ubuf)
        input_y_ub = tik_instance.Tensor(self.dtype, [psm_out,], name="input_y_ub",
                                         scope=tik.scope_ubuf)
        input_z_ub = tik_instance.Tensor(self.dtype, [16,], name="input_z_ub",
                                         scope=tik.scope_ubuf)
        reg = tik_instance.Scalar(dtype=self.dtype)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:

                merchant = sum_core + num_core_loop
                if tail_block != 0:
                    src_x_index = psm_in_ac * merchant + tail * split_hw * c
                    nburst_gm2ub = 1
                    burstlen_gm2ub = math.ceil(tail_block * c / self.step_move) * self.step_move * self.num_bit // 32
                    srcstride_gm2ub = 0
                    dststride_gm2ub = 0

                    tik_instance.data_move(input_x_ub[0],
                                           self.input_x_gm[src_x_index],
                                           0,
                                           nburst_gm2ub,
                                           burstlen_gm2ub,
                                           srcstride_gm2ub,
                                           dststride_gm2ub)

                    psm_out_tail_block_ac = tail_block * c0
                    psm_out_tail_block = math.ceil(psm_out_tail_block_ac / self.step_move) * self.step_move
                    if c != c0:
                        self.set_vector_dup(tik_instance, psm_out_tail_block, input_y_ub, 0)

                    # reorder
                    with tik_instance.for_range(0, c) as i:
                        with tik_instance.for_range(0, tail_block) as j:
                            src_x_ub_index = i + j * c
                            dst_y_ub_index = i + j * c0
                            reg.set_as(input_x_ub[src_x_ub_index])
                            input_y_ub[dst_y_ub_index].set_as(reg)


                    # data move out serial
                    dst_gm = merchant * psm_out_ac + tail * split_hw * c0
                    nburst_ub2gm = 1
                    burstlen_ub2gm_ori = tail_block * c0
                    srcstride_ub2gm = 0
                    dststride_ub2gm = 0

                    offset = burstlen_ub2gm_ori % self.step_move
                    burstlen_ub2gm = (burstlen_ub2gm_ori - offset) * self.num_bit // 32
                    dst_gm_extra = dst_gm + burstlen_ub2gm_ori - self.step_move

                    if burstlen_ub2gm != 0:
                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

                        if offset != 0:
                            with tik_instance.for_range(0, self.step_move) as i:
                                reg.set_as(input_y_ub[burstlen_ub2gm_ori - self.step_move + i])
                                input_z_ub[i].set_as(reg)
                            tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                   input_z_ub[0],
                                                   0,
                                                   1,
                                                   1,
                                                   0,
                                                   0)

                    else:
                        offset = self.step_move - burstlen_ub2gm_ori
                        with tik_instance.for_range(0, burstlen_ub2gm_ori) as i:
                            reg.set_as(input_y_ub[i])
                            input_z_ub[offset + i].set_as(reg)
                        tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                               input_z_ub[0],
                                               0,
                                               1,
                                               1,
                                               0,
                                               0)

                if tail != 0:
                    with tik_instance.for_range(0, tail) as num_ub_loop:
                        src_x_index = psm_in_ac * merchant + num_ub_loop * split_hw * c
                        nburst_gm2ub = 1
                        burstlen_gm2ub = split_hw * c * self.num_bit // 32
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 0

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)

                        if c != c0:
                            self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                        # reorder
                        with tik_instance.for_range(0, c) as i:
                            with tik_instance.for_range(0, split_hw) as j:
                                src_x_ub_index = i + j * c
                                dst_y_ub_index = i + j * c0
                                reg.set_as(input_x_ub[src_x_ub_index])
                                input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out serial
                        dst_gm = merchant * psm_out_ac + num_ub_loop * split_hw * c0
                        nburst_ub2gm = 1
                        burstlen_ub2gm = split_hw * c0 * self.num_bit // 32
                        srcstride_ub2gm = 0
                        dststride_ub2gm = 0

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

    def four2five_nhwc_casex(self, tik_instance, param):
        """
        in this case, hw must be small, but n0 can so many,
        so don't care about hw too large to save in ub
        """
        core_number = param.get("core_num")
        total_core_loop_num = param.get("total_core_loop_num")
        tiling_shape = param.get("tiling_shape")
        psm_in_ac = param.get("psm_in_ac")
        psm_out_ac = param.get("psm_out_ac")
        psm_move_in = param.get("psm_move_in")
        psm_move_out = param.get("psm_move_out")
        hw = tiling_shape[1] * tiling_shape[2]
        n1 = tiling_shape[0]
        c0, c1 = 4, 1
        c = tiling_shape[3]

        if psm_move_out <= self.maximum_size_ub:
            psm_out = psm_move_out
            psm_in = psm_move_in
            tail_n1 = 0
            tail_block_n1 = n1
            # not use
            split_n1 = -1

        else:
            for split_n1 in range(n1):
                split_n1 = split_n1 + 1
                psm_out = math.ceil(c0 * hw * split_n1 / self.step_move) * self.step_move
                psm_in = math.ceil(c * hw * split_n1 / self.step_move) * self.step_move
                if psm_out > self.maximum_size_ub:
                    split_n1 -= 1
                    psm_out = math.ceil(c0 * hw * split_n1 / self.step_move) * self.step_move
                    psm_in = math.ceil(c * hw * split_n1 / self.step_move) * self.step_move
                    break

            if split_n1 < 1:
                raise RuntimeError("In this case, split_n1 must >= 1")
            tail_n1 = n1 // split_n1
            tail_block_n1 = n1 % split_n1

        input_x_ub = tik_instance.Tensor(self.dtype, [psm_in,], name="input_x_ub",
                                         scope=tik.scope_ubuf)
        input_y_ub = tik_instance.Tensor(self.dtype, [psm_out,], name="input_y_ub",
                                         scope=tik.scope_ubuf)
        input_z_ub = tik_instance.Tensor(self.dtype, [16,], name="input_z_ub",
                                         scope=tik.scope_ubuf)
        reg = tik_instance.Scalar(dtype=self.dtype)
        src_x_offset = tik_instance.Scalar(dtype="int64")
        src_x_offset.set_as(0)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:

                merchant = sum_core + num_core_loop
                if tail_block_n1 != 0:
                    src_x_index = psm_in_ac * merchant + tail_n1 * split_n1 * c * hw

                    if tail_n1 == 0:
                        with tik_instance.if_scope(src_x_index + psm_in > self.maximum_gm):
                            src_x_offset.set_as(src_x_index + psm_in - self.maximum_gm)

                        self.set_move_in_zero(tik_instance, psm_in, src_x_offset,
                                              input_x_ub, self.input_x_gm, src_x_index)
                        if c != c0:
                            self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                        # reorder
                        with tik_instance.for_range(0, tail_block_n1) as k:
                            with tik_instance.for_range(0, c) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    src_x_ub_index = (i + j * c) + k * c * hw
                                    dst_y_ub_index = (i + j * c0) + k * c0 * hw
                                    reg.set_as(input_x_ub[src_x_offset + src_x_ub_index])
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        dst_gm = merchant * psm_out_ac
                        offset = psm_out_ac % self.step_move
                        nburst_ub2gm = 1
                        burstlen_ub2gm_ori = (psm_out_ac - offset) * self.num_bit // 32
                        srcstride_ub2gm = 0
                        dststride_ub2gm = 0

                        if burstlen_ub2gm_ori == 0:
                            burstlen_ub2gm = 1
                        else:
                            burstlen_ub2gm = burstlen_ub2gm_ori

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

                        if burstlen_ub2gm_ori != 0 and offset != 0:
                            with tik_instance.for_range(0, self.step_move) as i:

                                index_y_ub = psm_out_ac - self.step_move + i
                                reg.set_as(input_y_ub[index_y_ub])
                                input_z_ub[i].set_as(reg)

                            dst_gm_extra = dst_gm + psm_out_ac - self.step_move

                            tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                   input_z_ub[0],
                                                   0,
                                                   1,
                                                   1,
                                                   0,
                                                   0)

                    else:
                        psm_in_tail_block_n1_ac = tail_block_n1 * hw * c
                        psm_in_tail_block_n1 = math.ceil(psm_in_tail_block_n1_ac / self.step_move) * self.step_move
                        with tik_instance.if_scope(src_x_index + psm_in_tail_block_n1 > self.maximum_gm):
                            src_x_offset.set_as(src_x_index + psm_in_tail_block_n1 - self.maximum_gm)

                        self.set_move_in_zero(tik_instance, psm_in_tail_block_n1, src_x_offset,
                                              input_x_ub, self.input_x_gm, src_x_index)
                        psm_out_tail_block_n1_ac = tail_block_n1 * c0 * hw
                        psm_out_tail_block_n1 = math.ceil(psm_out_tail_block_n1_ac / self.step_move) * self.step_move

                        if c != c0:
                            self.set_vector_dup(tik_instance, psm_out_tail_block_n1, input_y_ub, 0)

                        # reorder
                        with tik_instance.for_range(0, tail_block_n1) as k:
                            with tik_instance.for_range(0, c) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    src_x_ub_index = (i + j * c) + k * c * hw
                                    dst_y_ub_index = (i + j * c0) + k * c0 * hw
                                    reg.set_as(input_x_ub[src_x_offset + src_x_ub_index])
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        dst_gm = merchant * psm_out_ac + tail_n1 * split_n1 * c0 * hw

                        if psm_out_tail_block_n1_ac >= 32 // self.num_bit:
                            offset = psm_out_tail_block_n1_ac % self.step_move
                            nburst_ub2gm = 1
                            burstlen_ub2gm = (psm_out_tail_block_n1_ac - offset) * self.num_bit // 32
                            srcstride_ub2gm = 0
                            dststride_ub2gm = 0

                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   srcstride_ub2gm,
                                                   dststride_ub2gm)

                            if offset != 0:
                                with tik_instance.for_range(0, self.step_move) as i:
                                    index_y_ub = psm_out_tail_block_n1_ac - self.step_move + i
                                    reg.set_as(input_y_ub[index_y_ub])
                                    input_z_ub[i].set_as(reg)

                                dst_gm_extra = dst_gm + psm_out_tail_block_n1_ac - self.step_move
                                tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       1,
                                                       0,
                                                       0)
                        else:
                            offset = psm_out_tail_block_n1 - psm_out_tail_block_n1_ac
                            with tik_instance.for_range(0, psm_out_tail_block_n1_ac) as i:
                                reg.set_as(input_y_ub[i])
                                input_z_ub[i + offset].set_as(reg)

                            dst_gm_extra = dst_gm - offset
                            tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                   input_z_ub[0],
                                                   0,
                                                   1,
                                                   1,
                                                   0,
                                                   0)

                if tail_n1 != 0:
                    with tik_instance.for_range(0, tail_n1) as num_ub_loop:

                        src_x_index = psm_in_ac * merchant + num_ub_loop * split_n1 * hw * c
                        self.set_move_in_zero(tik_instance, psm_in, 0, input_x_ub,
                                              self.input_x_gm, src_x_index)

                        if c != c0:
                            self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                        # reorder
                        with tik_instance.for_range(0, split_n1) as k:
                            with tik_instance.for_range(0, c) as i:
                                with tik_instance.for_range(0, hw) as j:
                                    src_x_ub_index = (i + j * c) + k * c * hw
                                    dst_y_ub_index = (i + j * c0) + k * c0 * hw
                                    reg.set_as(input_x_ub[src_x_ub_index])
                                    input_y_ub[dst_y_ub_index].set_as(reg)


                        # data move out
                        dst_gm = merchant * psm_out_ac + num_ub_loop * split_n1 * hw * c0
                        offset = (split_n1 * hw * c0) % self.step_move
                        nburst_ub2gm = 1
                        burstlen_ub2gm = (split_n1 * hw * c0 - offset) * self.num_bit // 32
                        srcstride_ub2gm = 0
                        dststride_ub2gm = 0

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)

                        if offset != 0:
                            with tik_instance.for_range(0, self.step_move) as i:
                                index_y_ub = split_n1 * hw * c0 - self.step_move + i
                                reg.set_as(input_y_ub[index_y_ub])
                                input_z_ub[i].set_as(reg)

                            dst_gm_extra = dst_gm + split_n1 * hw * c0 - self.step_move
                            tik_instance.data_move(self.output_y_gm[dst_gm_extra],
                                                   input_z_ub[0],
                                                   0,
                                                   1,
                                                   1,
                                                   0,
                                                   0)

    def four2five_hwcn_case00(self, tik_instance, param):
        """
        split hw as core
        """
        core_number = param.get("core_num")
        total_core_loop_num = param.get("total_core_loop_num")
        tiling_shape = param.get("tiling_shape")
        psm_in_ac = param.get("psm_in_ac")
        psm_out_ac = param.get("psm_out_ac")
        psm_move_in = param.get("psm_move_in")
        psm_move_out = param.get("psm_move_out")

        hw1, c, n = tiling_shape[0], tiling_shape[1], tiling_shape[2]
        c0, c1 = 4, 1
        split_hw = self.step_move // c0

        if psm_move_out <= self.maximum_size_ub:
            psm_out = psm_move_out
            psm_in = psm_move_in
            tail_hw = 0
            tail_block_hw = hw1
        else:
            raise RuntimeError("In hwcn-00-branch, ub can save tiling_shape")

        input_x_ub, input_y_ub, \
        input_z_ub = self.set_ub_tensor(tik_instance, psm_out, psm_in)
        reg = tik_instance.Scalar(dtype=self.dtype)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:

                merchant = sum_core + num_core_loop
                if tail_block_hw != 0:
                    src_x_index = psm_in_ac * merchant + tail_hw * split_hw * c * n
                    if tail_hw == 0:
                        self.set_move_in_one(tik_instance, psm_in, input_x_ub,
                                             self.input_x_gm, src_x_index)
                        if c != c0:
                            self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                        # reorder
                        if tail_block_hw * c0 >= self.step_move:
                            n_gap = math.ceil(tail_block_hw*c0/self.step_move) * self.step_move
                            with tik_instance.for_range(0, tail_block_hw) as k:
                                with tik_instance.for_range(0, c) as i:
                                    with tik_instance.for_range(0, n) as j:
                                        src_x_ub_index = j + i*n + k*c*n
                                        dst_y_ub_index = j * n_gap + i + k*c0
                                        reg.set_as(input_x_ub[src_x_ub_index])
                                        input_y_ub[dst_y_ub_index].set_as(reg)
                        else:
                            with tik_instance.for_range(0, tail_block_hw) as k:
                                with tik_instance.for_range(0, c) as i:
                                    with tik_instance.for_range(0, n) as j:
                                        src_x_ub_index = j + i*n + k*c*n
                                        dst_y_ub_index = j*tail_block_hw*c0 + i + k*c0
                                        reg.set_as(input_x_ub[src_x_ub_index])
                                        input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        dst_gm = merchant * (psm_out_ac//n)
                        nburst_ub2gm = n
                        srcstride_ub2gm = 0
                        burstlen_ub2gm = c0 * tail_block_hw
                        burstlen_ub2gm_ori = math.ceil(burstlen_ub2gm / self.step_move) * self.step_move
                        dststride_ub2gm = self.input_shape[0] * self.input_shape[1] * c0 - c0 * tail_block_hw
                        dststride_ub2gm_ori = math.ceil(dststride_ub2gm / self.step_move) * self.step_move

                        if dststride_ub2gm_ori == dststride_ub2gm and burstlen_ub2gm == burstlen_ub2gm_ori:
                            burstlen_ub2gm = burstlen_ub2gm * self.num_bit // 32
                            dststride_ub2gm = dststride_ub2gm * self.num_bit // 32
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   srcstride_ub2gm,
                                                   dststride_ub2gm)
                        else:
                            if burstlen_ub2gm >= self.step_move:
                                offset = burstlen_ub2gm % self.step_move
                                burstlen_ub2gm = (burstlen_ub2gm - offset) * self.num_bit // 32
                                dst_gm_gap = self.input_shape[0] * self.input_shape[1] * c0
                                src_y_gap = burstlen_ub2gm_ori

                                with tik_instance.for_range(0, n) as i:
                                    dst_gm = dst_gm + i * dst_gm_gap
                                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                                           input_y_ub[i * src_y_gap],
                                                           0,
                                                           1,
                                                           burstlen_ub2gm,
                                                           0,
                                                           0)

                                if offset != 0:
                                    dst_gm = merchant * (psm_out_ac//n)
                                    dst_gm_gap = self.input_shape[0] * self.input_shape[1] * c0
                                    with tik_instance.for_range(0, n) as i:
                                        with tik_instance.for_range(0, self.step_move) as j:
                                            index_y_ub = i * src_y_gap + c0 * tail_block_hw - self.step_move + j
                                            reg.set_as(input_y_ub[index_y_ub])
                                            input_z_ub[j].set_as(reg)
                                        dst_gm = dst_gm + i * dst_gm_gap + c0 *tail_block_hw - self.step_move
                                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                                               input_z_ub[0],
                                                               0,
                                                               1,
                                                               1,
                                                               0,
                                                               0)
                            else:
                                nburst_ub2gm = 1
                                burstlen_ub2gm = math.ceil(tail_block_hw * c0 * n / self.step_move)
                                burstlen_ub2gm = burstlen_ub2gm * self.step_move * self.num_bit //32
                                srcstride_ub2gm = 0
                                dststride_ub2gm = 0
                                tik_instance.data_move(self.output_y_gm[dst_gm],
                                                       input_y_ub[0],
                                                       0,
                                                       nburst_ub2gm,
                                                       burstlen_ub2gm,
                                                       srcstride_ub2gm,
                                                       dststride_ub2gm)

                    else:
                        raise RuntimeError("In hwcn-00-branch,tail_hw must be 0")

    def four2five_hwcn_case01(self, tik_instance, param):
        """
        split hw as core
        tiling_shape mabe stored in ub
        but self.step_move*n > maximum_size_ub
        """
        core_number = param.get("core_num")
        total_core_loop_num = param.get("total_core_loop_num")
        tiling_shape = param.get("tiling_shape")
        psm_in_ac = param.get("psm_in_ac")
        psm_out_ac = param.get("psm_out_ac")
        psm_move_in = param.get("psm_move_in")
        psm_move_out = param.get("psm_move_out")

        hw1, c, n = tiling_shape[0], tiling_shape[1], tiling_shape[2]
        c0, c1 = 4, 1
        if psm_move_out <= self.maximum_size_ub:
            raise RuntimeError("In hwcn-01-branch, ub can't save tiling_shape")
        if self.step_move*n <= self.maximum_size_ub:
            raise RuntimeError("In hwcn-01-branch, self.step_move*n must > maximum_size_ub")

        split_hw = self.step_move // c0
        tail_hw = hw1 // split_hw
        tail_block_hw = hw1 % split_hw

        split_n = self.step_move
        merchant_n = n // split_n

        for i in range(merchant_n):
            split_n = self.step_move * (i+1)
            psm_move_out = split_hw*c0*split_n
            psm_move_in = split_hw*c*split_n
            if psm_move_out > self.maximum_size_ub:
                split_n = self.step_move * i
                psm_move_out = split_hw*c0*split_n
                psm_move_in = split_hw*c*split_n
                break

        if split_n == 0:
            raise RuntimeError("In hwcn-01 case, split_n can't be 0")

        tail_n = n // split_n
        tail_block_n = n % split_n
        if tail_block_n != 0:
            offset_n = self.step_move - (tail_block_n % self.step_move)
        else:
            offset_n = 0
        input_x_ub, input_y_ub, \
        input_z_ub = self.set_ub_tensor(tik_instance, psm_move_out, psm_move_in)
        reg = tik_instance.Scalar(dtype=self.dtype)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:

                merchant = sum_core + num_core_loop

                if tail_block_n != 0:
                    n_new = tail_block_n + offset_n
                    if tail_block_hw != 0:
                        # data move in
                        src_x_index = psm_in_ac * merchant + tail_hw * split_hw * c * n
                        src_x_index = src_x_index + tail_n * split_n - offset_n
                        self.set_move_in_two(tik_instance, n_new, tail_block_hw,
                                             self.input_x_gm, src_x_index, input_x_ub)

                        # vector_dup
                        if c != c0:
                            psm_out = math.ceil(tail_block_hw*c0/self.step_move) * \
                                      self.step_move * n_new
                            self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                        # reorder
                        if tail_hw != 0:
                            n_gap = math.ceil(tail_block_hw*c0/self.step_move) \
                                    * self.step_move
                        else:
                            n_gap = tail_block_hw*c0
                        with tik_instance.for_range(0, tail_block_hw) as k:
                            with tik_instance.for_range(0, c) as i:
                                with tik_instance.for_range(0, n_new) as j:
                                    src_x_ub_index = j + i*n_new + k*c*n_new
                                    dst_y_ub_index = j * n_gap + i + k*c0
                                    reg.set_as(input_x_ub[src_x_ub_index])
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        dst_gm = (tail_n * split_n - offset_n) * \
                                 self.input_shape[0] * \
                                 self.input_shape[1] * c0

                        dst_gm = dst_gm + merchant*(psm_out_ac//n) \
                                 + tail_hw * split_hw * c0

                        self.set_move_out_zero(tik_instance, n_new, tail_block_hw,
                                               self.output_y_gm, dst_gm,
                                               input_y_ub, reg,
                                               input_z_ub, tail_hw)

                    if tail_hw != 0:
                        with tik_instance.for_range(0, tail_hw) as num_ub_loop:
                            # data move in
                            src_x_index = psm_in_ac * merchant + \
                                          num_ub_loop * split_hw * c * n

                            src_x_index = src_x_index + \
                                          tail_n * split_n - \
                                          offset_n

                            self.set_move_in_two(tik_instance, n_new,
                                                 split_hw, self.input_x_gm,
                                                 src_x_index, input_x_ub)

                            # vector_dup
                            if c != c0:
                                psm_out = split_hw * c0 * n_new
                                self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                            # reorder
                            n_gap = split_hw * c0
                            with tik_instance.for_range(0, split_hw) as k:
                                with tik_instance.for_range(0, c) as i:
                                    with tik_instance.for_range(0, n_new) as j:
                                        src_x_ub_index = j + i*n_new + k*c*n_new
                                        dst_y_ub_index = j * n_gap + i + k*c0
                                        reg.set_as(input_x_ub[src_x_ub_index])
                                        input_y_ub[dst_y_ub_index].set_as(reg)

                            # data move out
                            dst_gm = (tail_n * split_n - offset_n) * \
                                     self.input_shape[0] * \
                                     self.input_shape[1] * c0

                            dst_gm = dst_gm + merchant * \
                                     (psm_out_ac//n) + \
                                     num_ub_loop * split_hw * c0

                            self.set_move_out_one(tik_instance, n_new,
                                                  split_hw, self.output_y_gm,
                                                  dst_gm, input_y_ub)

                if tail_n != 0:
                    n_new = split_n
                    if tail_block_hw != 0:
                        with tik_instance.for_range(0, tail_n) as n_num_ub_loop:
                            # data move in
                            src_x_index = psm_in_ac * merchant + \
                                          tail_hw * split_hw * \
                                          c * n

                            src_x_index = src_x_index + n_num_ub_loop * split_n

                            self.set_move_in_two(tik_instance, n_new,
                                                 tail_block_hw,
                                                 self.input_x_gm,
                                                 src_x_index,
                                                 input_x_ub)

                            # vector_dup
                            if c != c0:
                                psm_out = math.ceil(tail_block_hw*c0 / self.step_move) * \
                                          self.step_move * n_new
                                self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                            # reorder
                            if tail_hw != 0:
                                n_gap = math.ceil(tail_block_hw*c0/self.step_move) * self.step_move
                            else:
                                n_gap = tail_block_hw*c0
                            with tik_instance.for_range(0, tail_block_hw) as k:
                                with tik_instance.for_range(0, c) as i:
                                    with tik_instance.for_range(0, n_new) as j:
                                        src_x_ub_index = j + i*n_new + k*c*n_new
                                        dst_y_ub_index = j * n_gap + i + k*c0
                                        reg.set_as(input_x_ub[src_x_ub_index])
                                        input_y_ub[dst_y_ub_index].set_as(reg)

                            # data move out
                            dst_gm = n_num_ub_loop * split_n * \
                                     self.input_shape[0] * \
                                     self.input_shape[1] * c0

                            dst_gm = dst_gm + merchant * \
                                     (psm_out_ac//n) + \
                                     tail_hw * split_hw * c0

                            self.set_move_out_zero(tik_instance, n_new,
                                                   tail_block_hw,
                                                   self.output_y_gm,
                                                   dst_gm, input_y_ub,
                                                   reg, input_z_ub,
                                                   tail_hw)

                    if tail_hw != 0:
                        with tik_instance.for_range(0, tail_n) as n_num_ub_loop:
                            with tik_instance.for_range(0, tail_hw) as num_ub_loop:
                                src_x_index = psm_in_ac * merchant + \
                                              num_ub_loop * \
                                              split_hw * c * n

                                src_x_index = src_x_index + \
                                              n_num_ub_loop * split_n

                                self.set_move_in_two(tik_instance, n_new,
                                                     split_hw,
                                                     self.input_x_gm,
                                                     src_x_index,
                                                     input_x_ub)

                                # vector_dup
                                if c != c0:
                                    psm_out = split_hw * c0 * n_new
                                    self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                                # reorder
                                n_gap = split_hw * c0
                                with tik_instance.for_range(0, split_hw) as k:
                                    with tik_instance.for_range(0, c) as i:
                                        with tik_instance.for_range(0, n_new) as j:
                                            src_x_ub_index = j + i*n_new + k*c*n_new
                                            dst_y_ub_index = j * n_gap + i + k*c0
                                            reg.set_as(input_x_ub[src_x_ub_index])
                                            input_y_ub[dst_y_ub_index].set_as(reg)

                                # data move out
                                dst_gm = n_num_ub_loop * split_n * \
                                         self.input_shape[0] * \
                                         self.input_shape[1] * c0

                                dst_gm = dst_gm + merchant * \
                                         (psm_out_ac//n) + \
                                         num_ub_loop * split_hw * c0

                                self.set_move_out_one(tik_instance, n_new,
                                                      split_hw,
                                                      self.output_y_gm,
                                                      dst_gm,
                                                      input_y_ub)

    def four2five_hwcn_case02(self, tik_instance, param):
        """
        split hw as core
        tiling_shape can't be stored in ub
        split tiling_shape[0] for tiling
        """
        core_number = param.get("core_num")
        total_core_loop_num = param.get("total_core_loop_num")
        tiling_shape = param.get("tiling_shape")
        psm_in_ac = param.get("psm_in_ac")
        psm_out_ac = param.get("psm_out_ac")
        psm_move_in = param.get("psm_move_in")
        psm_move_out = param.get("psm_move_out")

        hw1, c, n = tiling_shape[0], tiling_shape[1], tiling_shape[2]
        c0, c1 = 4, 1
        split_hw = self.step_move // c0

        if psm_move_out <= self.maximum_size_ub:
            raise RuntimeError("In hwcn-02-branch, ub can't save tiling_shape")
        else:
            merchant_hw1 = hw1 // split_hw
            for i in range(merchant_hw1):
                split_hw = self.step_move // c0 * (i+1)
                psm_move_out = split_hw * c0 * n
                psm_move_in = math.ceil(split_hw*c*n/self.step_move) * self.step_move
                if psm_move_out > self.maximum_size_ub:
                    split_hw = self.step_move // c0 * i
                    psm_move_out = split_hw * c0 * n
                    psm_move_in = math.ceil(split_hw*c*n/self.step_move) * self.step_move
                    break
            # in this case:
            # hw1*c0*n > maximum_ub
            # because the scale of n can't be controlled
            # split_hw*c0*n will be bigger than maximum_ub
            # while split_hw = self.step_move // c0
            if split_hw == 0:
                raise RuntimeError("in this case split_hw can't be zero")

        input_x_ub, input_y_ub, \
        input_z_ub = self.set_ub_tensor(tik_instance, psm_move_out, psm_move_in)
        reg = tik_instance.Scalar(dtype=self.dtype)
        tail_block_hw = hw1 % split_hw
        tail_hw = hw1 // split_hw
        move_mark = 0
        move_MARK = 0

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                merchant = sum_core + num_core_loop

                if tail_block_hw != 0:
                    src_x_index = psm_in_ac * merchant + tail_hw * split_hw * c * n
                    psm_in = math.ceil(tail_block_hw*c*n/self.step_move) * self.step_move
                    self.set_move_in_one(tik_instance, psm_in, input_x_ub,
                                         self.input_x_gm, src_x_index)

                    # vector_dup
                    psm_out = math.ceil(tail_block_hw*c0 / self.step_move) * self.step_move * n
                    if c != c0:
                        self.set_vector_dup(tik_instance, psm_out, input_y_ub, 0)

                    # reorder
                    n_gap = math.ceil(tail_block_hw*c0/self.step_move) * self.step_move
                    with tik_instance.for_range(0, tail_block_hw) as k:
                        with tik_instance.for_range(0, c) as i:
                            with tik_instance.for_range(0, n) as j:
                                src_x_ub_index = j + i*n + k*c*n
                                dst_y_ub_index = j * n_gap + i + k*c0
                                reg.set_as(input_x_ub[src_x_ub_index])
                                input_y_ub[dst_y_ub_index].set_as(reg)

                    # data move out
                    dst_gm = merchant * (psm_out_ac//n) + tail_hw * split_hw * c0
                    nburst_ub2gm = n
                    srcstride_ub2gm = 0
                    burstlen_ub2gm = c0 * tail_block_hw
                    burstlen_ub2gm_ori = math.ceil(burstlen_ub2gm / self.step_move) * self.step_move
                    dststride_ub2gm = self.input_shape[0] * self.input_shape[1] * c0 - c0 * tail_block_hw
                    dststride_ub2gm_ori = math.ceil(dststride_ub2gm / self.step_move) * self.step_move

                    condition_0 = dststride_ub2gm_ori == dststride_ub2gm
                    condition_1 = burstlen_ub2gm == burstlen_ub2gm_ori
                    condition_2 = (dststride_ub2gm * self.num_bit // 32) <= MAX_STRIDE

                    if condition_0 and condition_1 and condition_2:
                        move_mark = 1
                        burstlen_ub2gm = burstlen_ub2gm * self.num_bit // 32
                        dststride_ub2gm = dststride_ub2gm * self.num_bit // 32
                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               srcstride_ub2gm,
                                               dststride_ub2gm)
                    else:
                        move_mark = 0
                        dst_gm_gap = self.input_shape[0] * self.input_shape[1] * c0
                        src_y_gap = burstlen_ub2gm_ori
                        if burstlen_ub2gm >= self.step_move:
                            offset = burstlen_ub2gm % self.step_move
                            burstlen_ub2gm = (burstlen_ub2gm - offset) * self.num_bit // 32
                            dst_gm_align = dst_gm

                            with tik_instance.for_range(0, n) as i:
                                dst_gm_align = dst_gm_align + i * dst_gm_gap
                                tik_instance.data_move(self.output_y_gm[dst_gm_align],
                                                       input_y_ub[i * src_y_gap],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                            if offset != 0:
                                dst_gm_n_align = dst_gm
                                with tik_instance.for_range(0, n) as i:
                                    with tik_instance.for_range(0, self.step_move) as j:
                                        index_y_ub = i * src_y_gap + c0 * tail_block_hw - self.step_move + j
                                        reg.set_as(input_y_ub[index_y_ub])
                                        input_z_ub[j].set_as(reg)
                                    dst_gm_n_align = dst_gm_n_align + i * dst_gm_gap + c0 *tail_block_hw - self.step_move
                                    tik_instance.data_move(self.output_y_gm[dst_gm_n_align],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           1,
                                                           0,
                                                           0)
                        else:
                            offset = self.step_move - burstlen_ub2gm
                            dst_gm_n_align = dst_gm
                            with tik_instance.for_range(0, n) as i:
                                with tik_instance.for_range(0, burstlen_ub2gm) as j:
                                    index_y_ub = i * src_y_gap + j
                                    reg.set_as(input_y_ub[index_y_ub])
                                    input_z_ub[offset+j].set_as(reg)

                                dst_gm_n_align = dst_gm_n_align + i * dst_gm_gap - offset
                                tik_instance.data_move(self.output_y_gm[dst_gm_n_align],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       1,
                                                       0,
                                                       0)

                if tail_hw != 0:
                    with tik_instance.for_range(0, tail_hw) as num_ub_loop:
                        src_x_index = psm_in_ac * merchant + num_ub_loop * split_hw * c * n
                        self.set_move_in_one(tik_instance, psm_move_in, input_x_ub,
                                             self.input_x_gm, src_x_index)

                        if c != c0:
                            self.set_vector_dup(tik_instance, psm_move_out, input_y_ub, 0)

                        # reorder
                        n_gap = split_hw * c0
                        with tik_instance.for_range(0, split_hw) as k:
                            with tik_instance.for_range(0, c) as i:
                                with tik_instance.for_range(0, n) as j:
                                    src_x_ub_index = j + i*n + k*c*n
                                    dst_y_ub_index = j * n_gap + i + k*c0
                                    reg.set_as(input_x_ub[src_x_ub_index])
                                    input_y_ub[dst_y_ub_index].set_as(reg)

                        # data move out
                        dst_gm = merchant * (psm_out_ac//n) + num_ub_loop * split_hw * c0
                        nburst_ub2gm = n
                        srcstride_ub2gm = 0
                        burstlen_ub2gm = c0 * split_hw * self.num_bit // 32
                        dststride_ub2gm = (self.input_shape[0] * self.input_shape[1] * c0 - c0 * split_hw) * self.num_bit

                        condition_0 = dststride_ub2gm <= MAX_STRIDE
                        condition_1 = dststride_ub2gm % 32 ==0
                        if condition_0 and condition_1:
                            dststride_ub2gm = dststride_ub2gm // 32
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   srcstride_ub2gm,
                                                   dststride_ub2gm)
                        else:
                            dst_gm_gap = self.input_shape[0] * self.input_shape[1] * c0
                            src_y_gap = split_hw * c0
                            with tik_instance.for_range(0,n) as i:
                                tik_instance.data_move(self.output_y_gm[dst_gm + i * dst_gm_gap],
                                                       input_y_ub[i * src_y_gap],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

    def pattern_case(self, tik_instance, param):

        src_format = param.get("format")
        pattern = param.get("pattern")

        if src_format == "NCHW":
            if pattern == 0:
                self.four2five_nchw_case0(tik_instance, param)
            elif pattern == 1:
                self.four2five_nchw_casex(tik_instance, param)

        elif src_format == "NHWC":
            if pattern == 0:
                self.four2five_nhwc_case0(tik_instance, param)
            elif pattern == 1:
                self.four2five_nhwc_casex(tik_instance, param)

        elif src_format == "HWCN":
            if pattern == "00":
                self.four2five_hwcn_case00(tik_instance, param)
            elif pattern == "01":
                self.four2five_hwcn_case01(tik_instance, param)
            elif pattern == "02":
                self.four2five_hwcn_case02(tik_instance, param)
        else:
            raise RuntimeError("please check the src_format !!!")

    def split_shape_nchw_0(self, shape):

        n = shape[0]
        hw = shape[-1] * shape[-2]
        chw_in = hw * shape[-3]
        chw_out = chw_in + (4 - shape[-3]) * hw

        chw_move_out = math.ceil(chw_out / self.step_move) * self.step_move
        chw_move_in = math.ceil(chw_in / self.step_move) * self.step_move

        total_core_loop_num = n
        psm_in_ac = chw_in
        psm_out_ac = chw_out
        psm_move_out = chw_move_out
        psm_move_in = chw_move_in
        tiling_shape = [shape[1], shape[2], shape[3]]

        if total_core_loop_num < MAX_CORE_NUM:
            core_num = total_core_loop_num
        else:
            core_num = MAX_CORE_NUM

        param_split_shape = {"core_num": core_num,"total_core_loop_num": total_core_loop_num,
                             "tiling_shape":tiling_shape,"psm_in_ac":psm_in_ac,
                             "psm_out_ac":psm_out_ac,"psm_move_out":psm_move_out,"psm_move_in":psm_move_in}

        return param_split_shape

    def split_shape_nchw_nhwc_x(self, shape, src_format):
        """
        situation:
        input=[n,c,h,w], output=[n,c1,h,w,c0], c1=1, c0=4, hwc0*self.num_bit<32B
        way:
        split n as [n0,n1], make n1c1hwc0*self.num_bit>32B, n0 as core_num
        while n is a prime, n0 = 1
        """
        if src_format == "NHWC":
            n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        elif src_format == "NCHW":
            n, c, h, w = shape[0], shape[1], shape[2], shape[3]
        c1, c0 = 1, 4
        hwc0 = h * w * c0

        n0 = factorization(n, MAX_CORE_NUM, hwc0, self.step_move)
        n1 = n // n0
        if n % n0 != 0:
            raise RuntimeError("n0 is illegal in the branch")
        if src_format == "NHWC":
            tiling_shape = [n1,h,w,c]
        elif src_format == "NCHW":
            tiling_shape = [n1,c,h,w]

        chw_in = self.calc_element(tiling_shape)
        chw_out = n1 * h * w * c0
        chw_move_out = math.ceil(chw_out / self.step_move) * self.step_move
        chw_move_in = math.ceil(chw_in / self.step_move) * self.step_move

        total_core_loop_num = n0
        psm_in_ac = chw_in
        psm_out_ac = chw_out
        psm_move_out = chw_move_out
        psm_move_in = chw_move_in

        if total_core_loop_num < MAX_CORE_NUM:
            core_num = total_core_loop_num
        else:
            core_num = MAX_CORE_NUM

        param_split_shape = {"core_num": core_num,"total_core_loop_num": total_core_loop_num,
                             "tiling_shape":tiling_shape,"psm_in_ac":psm_in_ac,
                             "psm_out_ac":psm_out_ac,"psm_move_out":psm_move_out,"psm_move_in":psm_move_in}

        return param_split_shape

    def split_shape_nhwc_0(self, shape, src_format):

        n = shape[0]
        c0 = 4
        c1 = 1
        maximum_size_ub = self.maximum_size_ub

        if src_format == "NCHW":
            n,c,h,w = shape[0], shape[1], shape[2],shape[3]
        elif src_format == "NHWC":
            n,h,w,c = shape[0], shape[1], shape[2],shape[3]

        hw = h * w
        chw_in = hw * c
        chw_out = chw_in + (c0 - c) * hw

        chw_move_out = math.ceil(chw_out / self.step_move) * self.step_move
        chw_move_in = math.ceil(chw_in / self.step_move) * self.step_move

        total_core_loop_num = n
        psm_in_ac = chw_in
        psm_out_ac = chw_out
        psm_move_out = chw_move_out
        psm_move_in = chw_move_in
        tiling_shape = [shape[1], shape[2], shape[3]]

        if total_core_loop_num < MAX_CORE_NUM:
            core_num = total_core_loop_num
        else:
            core_num = MAX_CORE_NUM

        param_split_shape = {"core_num": core_num,"total_core_loop_num": total_core_loop_num,
                             "tiling_shape":tiling_shape,"psm_in_ac":psm_in_ac,
                             "psm_out_ac":psm_out_ac,"psm_move_out":psm_move_out,"psm_move_in":psm_move_in}

        return param_split_shape

    def split_shape_hwcn_0(self, shape):
        """
        split hw for cn as core
        """
        h, w, c, n = shape[0], shape[1], shape[2], shape[3]
        hw = h*w
        c1, c0 = 1, 4
        hw0 = factorization(hw, MAX_CORE_NUM, c0, self.step_move)
        hw1 = hw // hw0
        tiling_shape = [hw1, c, n]

        chw_in = self.calc_element(tiling_shape)
        chw_out = hw1*c0
        chw_move_out = math.ceil(chw_out / self.step_move) * self.step_move * n
        chw_move_in = math.ceil(chw_in / self.step_move) * self.step_move

        total_core_loop_num = hw0
        psm_in_ac = chw_in
        psm_out_ac = hw1*c0*n
        psm_move_out = chw_move_out
        psm_move_in = chw_move_in

        if total_core_loop_num < MAX_CORE_NUM:
            core_num = total_core_loop_num
        else:
            core_num = MAX_CORE_NUM

        param_split_shape = {"core_num": core_num,"total_core_loop_num": total_core_loop_num,
                             "tiling_shape":tiling_shape,"psm_in_ac":psm_in_ac,
                             "psm_out_ac":psm_out_ac,"psm_move_out":psm_move_out,"psm_move_in":psm_move_in}

        return param_split_shape

    def split_core(self, params):
        src_format = params.get("format")
        pattern = params.get("pattern")

        if src_format == "NCHW":
            if pattern == 0:
                param_split_shape = self.split_shape_nchw_0(self.input_shape)
            elif pattern == 1:
                param_split_shape = self.split_shape_nchw_nhwc_x(self.input_shape, src_format)
            else:
                raise RuntimeError("not complete pattern = ", params.get("pattern"))

        elif src_format == "NHWC":
            if pattern == 0:
                param_split_shape = self.split_shape_nhwc_0(self.input_shape, src_format)
            elif pattern == 1:
                param_split_shape = self.split_shape_nchw_nhwc_x(self.input_shape, src_format)

        elif src_format == "HWCN":
            if pattern == "00":
                param_split_shape = self.split_shape_hwcn_0(self.input_shape)
            elif pattern == "01":
                param_split_shape = self.split_shape_hwcn_0(self.input_shape)
            elif pattern == "02":
                param_split_shape = self.split_shape_hwcn_0(self.input_shape)
            else:
                raise RuntimeError("pattern is illegal")

        params.update(param_split_shape)

        return params

    def pattern_format(self):

        input_shape = self.input_shape.copy()
        src_format = self.format
        if src_format == "NCHW":
            n = input_shape[0]
            hw = input_shape[-1] * input_shape[-2]
            hwc0 = hw * 4
            psm_move_out = math.ceil(hwc0 / self.step_move) * self.step_move

            if psm_move_out <= self.maximum_size_ub:
                pattern = 1
            else:
                if n >= 32:
                    pattern = 0
                else:
                    if hw > 2 * self.step_move and hw % self.step_move == 0:
                        pattern = 0
                    else:
                        pattern = 0

        elif src_format == "NHWC":
            # reuse NCHW->5HD_4
            n = input_shape[0]
            hw = input_shape[1] * input_shape[2]
            c0 = 4
            hwc0 = hw * c0

            psm_move_out = math.ceil(hwc0 / self.step_move) * self.step_move
            if psm_move_out <= self.maximum_size_ub:
                pattern = 1
            else:
                if n >= 32:
                    pattern = 0
                else:
                    pattern = 0

        else:
            # HWCN->5HD_4
            h, w, c, n = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
            hw = h*w
            c0 = 4
            hw0 = factorization(hw, MAX_CORE_NUM, c0, self.step_move)
            hw1 = hw // hw0

            # The output is maximum ub tensor in core
            core_deal = hw1*c0
            core_deal = math.ceil(core_deal / self.step_move) * self.step_move * n
            # use self.step_move*n: make ub begin position of 32B align
            if self.step_move*n > self.maximum_size_ub:
                pattern = "01"
            else:
                if core_deal > self.maximum_size_ub:
                    pattern = "02"
                else:
                    pattern = "00"

        params = {"format": src_format, "pattern": pattern}

        return params

    def four2five_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        params = self.pattern_format()
        params = self.split_core(params)
        self.pattern_case(tik_instance, params)

        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.four2five_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.input_x_gm],
                              outputs=[self.output_y_gm])

        return tik_instance


def check_format_rule(format, check_list):
    """
    The common check rule for tensor dtype
    """
    if format is None:
        raise RuntimeError("format is None")

    if format.upper() not in check_list:
        raise RuntimeError("only support %s while format is %s" % (",".join(check_list), format))

def check_c_axis_rule(format, shape):
    """
    The common check rule for tensor dtype
    """
    if format == 'NCHW':
        c = shape[1]
        if c > 4 or c < 1 :
            raise RuntimeError("value of C axis is illeagl")

# pylint: disable=invalid-name,unused-argument
@util.check_input_type(dict, dict, str, str, str)
def four_2_five_c04(src, dst, src_format, dst_format, kernel_name="four_2_five_c04"):
    """
    algorithm: four_2_five_c04

    Parameters
    ----------
    src : dict
        dict with keys(shape, dtype) of src
    dst : dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src
    dst_format: str
        data format of dst
    kernel_name: str
        kernel name, default value is "four_2_five_c04"

    Returns
    -------
    tik_instance : tik_instance
    """
    input_shape = src.get("shape")
    if src_format.upper() == "NCHW":
        n,c,h,w = input_shape[0], input_shape[1],input_shape[2],input_shape[3]
        c1 = 1
        c0 = 4
    elif src_format.upper() == "NHWC":
        n,h,w,c = input_shape[0], input_shape[1],input_shape[2],input_shape[3]
        c1 = 1
        c0 = 4
    else:
        h,w,c,n = input_shape[0], input_shape[1],input_shape[2],input_shape[3]
        c1 = 1
        c0 = 4
    output_shape = [n,c1,h,w,c0]
    dtype = src.get("dtype").lower()
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)
    check_list_dtype = ("float16","float32")
    util.check_dtype_rule(dtype, check_list_dtype)
    src_format = src_format.upper()
    check_list_format = ("NCHW", "NHWC","HWCN")
    check_format_rule(src_format, check_list_format)
    check_c_axis_rule(src_format, input_shape)
    result = four2fiveCompute(input_shape, output_shape, dtype, src_format, kernel_name)

    return result.get_tik_instance()
