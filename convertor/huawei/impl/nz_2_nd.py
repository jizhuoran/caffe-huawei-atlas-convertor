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

nz_2_nd
"""
# pylint: disable=too-many-lines,import-error,too-many-branches,too-many-arguments
# pylint: disable=too-many-statements,too-many-locals,missing-function-docstring
from functools import reduce as functools_reduce
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
# maximum mask
MASK = 128
# vector_fp32_mask
MASK_FP32 = 64
# vector_fp16_mask
MASK_FP16 = 128
# maximum rep stride
MAX_REPEAT = 240


def _cal_core(tik_instance, total_core_loop_num, num_core, core_number):
    """
    calculate the loop number on each core
    """
    if total_core_loop_num % core_number == 0:
        core_loop = total_core_loop_num // core_number
        sum_core = core_loop * num_core
    else:
        core_loop = tik_instance.Scalar("uint64")
        sum_core = tik_instance.Scalar("uint64")
        with tik_instance.if_scope(num_core < total_core_loop_num %
                                   MAX_CORE_NUM):
            core_loop.set_as((total_core_loop_num + core_number - 1) //
                             core_number)
            sum_core.set_as(core_loop * num_core)
        with tik_instance.else_scope():
            core_loop.set_as(total_core_loop_num // core_number)
            sum_core.set_as((core_loop + 1) * (total_core_loop_num % MAX_CORE_NUM) +
                            core_loop * (num_core - total_core_loop_num %
                                         MAX_CORE_NUM))
    return core_loop, sum_core


def _check_db(total_loop, core_number):
    # check whether the conditions meet double_buffer
    if total_loop // core_number >= 2:
        thread_num = 2
    else:
        thread_num = 1

    return thread_num


class Nz2NDCompute(object):
    """
    the main of Nz2ND
    """
    def __init__(self, input_shape, output_shape, dtype, kernel_name):

        self.input_shape = list(input_shape)
        self.output_shape = list(output_shape)
        if len(self.output_shape) == 1:
            self.output_shape.append(1)

        self.input_sum_elements = self.calc_element(self.input_shape)
        self.output_sum_elements = self.calc_element(self.output_shape)

        self.kernel_name = kernel_name
        self.dtype = dtype
        self.num_bit = int(SIZE_TWO_BYTES)
        self.mask = int(MASK_FP16)
        if self.dtype == "float32" or self.dtype == "int32":
            self.num_bit = int(SIZE_FOUR_BYTES)
            self.mask = int(MASK_FP32)

        self.maximum_size_ub = int(TOTAL_UB_MEMORY // self.num_bit)

    def calc_element(self, shape):

        sum_result = 1
        if len(shape) == 0:
            return sum_result
        for i, _ in enumerate(range(len(shape))):
            sum_result = sum_result * shape[i]
        return sum_result

    def _division_nearest(self, number, p_val):
        number_split = 1
        number_d_value = number / number_split
        for i, _ in enumerate(range(number)):
            i = i + 1
            number_split_remainder = number % i
            if number_split_remainder == 0:
                number_split = i * p_val
                if number_split >= 32:
                    number_d_value = int(number * p_val / number_split)
                    break

        return number_split, number_d_value

    def split_shape_nzs(self, shape):
        """
        situation: not zero suppression
        set core and tiling shape
        """
        all_ele = functools_reduce(lambda x1, x2: x1 * x2, shape[0:])
        patch_ele = functools_reduce(lambda x1, x2: x1 * x2, shape[-4:])
        # because of db, ub_maximum // 2
        if self.dtype in ["float16", "float32"]:
            ub_maximum = self.maximum_size_ub // 2
        else:
            ub_maximum = self.maximum_size_ub

        # the case of shape like [A,...B, D, C, 16, 16]
        # D*16*16 > ub_maximum  pattern = 1
        # D*16*16 <= ub_maximum and A*B >= 32 pattern = 0
        # else pattern = 1
        patch_ele_ub_maximum = patch_ele / shape[-3] / 16 * 17
        if (all_ele // patch_ele) >= 32 and patch_ele_ub_maximum <= ub_maximum:
            tiling_shape = [shape[-4], shape[-3], shape[-2], shape[-1]]
            pattern = 0
        else:
            tiling_shape = [shape[-4], shape[-2], shape[-1]]
            pattern = 1

        psm = self.calc_element(tiling_shape)
        total_core_loop_num = int(all_ele / psm)

        # core_num
        if total_core_loop_num < MAX_CORE_NUM:
            core_num = total_core_loop_num
        else:
            core_num = MAX_CORE_NUM

        params = {"core_num": core_num,
                  "total_core_loop_num": total_core_loop_num,
                  "tiling_shape": tiling_shape,
                  "psm": psm,
                  "pattern": pattern}

        return params

    def split_shape(self, shape):
        """
        set core and tiling shape
        """
        tiling_shape = [shape[-4], shape[-2], shape[-1]]
        psm = self.calc_element(tiling_shape)
        total_core_loop_num = int(functools_reduce(lambda x1, x2: x1 * x2, shape[0:]) / psm)

        # core_num
        if total_core_loop_num < MAX_CORE_NUM:
            core_num = total_core_loop_num
        else:
            core_num = MAX_CORE_NUM

        params = {"core_num": core_num,
                  "total_core_loop_num": total_core_loop_num,
                  "tiling_shape": tiling_shape,
                  "psm": psm}
        return params

    def split_shape_zs(self, shape):
        """
        set core and tiling shape [A,B,...,D,C,16,16]:
        1. A*B>=32,tiling shape=[D,C,16,16]====>not complete in zs
           a. [D,16,16]<=maximum_ub [D,C,16,16]
           b. [D,C,16,16]>maximum_ub [D,16,16]
        2. A*B<32 ====>complete in zs
           a. A*B*C and A*B*D >= 32  split C, tilingshape=[D,16,16]
           b. A*B*C and A*B*D < 32   split C, tilingshape=[D,16,16]
           c. A*B*C <32 and A*B*D >= 32   split D,
           tilingshape=[D/d_split,C,16,16]
        """
        all_ele = functools_reduce(lambda x1, x2: x1 * x2, shape[0:])
        patch_ele = functools_reduce(lambda x1, x2: x1 * x2, shape[-4:])
        ub_maximum = self.maximum_size_ub
        # p_one:A*B*C p_two:A*B*D
        p_zero = all_ele // patch_ele
        p_one = p_zero * shape[-3]
        p_two = p_zero * shape[-4]

        if self.dtype == "int32":
            if p_zero < 32:
                if p_one >= 32 and p_two >= 32:
                    tiling_shape = [shape[-4], shape[-2], shape[-1]]
                    pattern = 1
                elif p_one < 32 and p_two < 32:
                    tiling_shape = [shape[-4], shape[-2], shape[-1]]
                    pattern = 1
                elif p_one < 32 and p_two >= 32:
                    # [2,c,16,16]: assure c can be saved in ub
                    if shape[-3] * 16 * 16 * 2 >= self.maximum_size_ub:
                        tiling_shape = [shape[-4], shape[-2], shape[-1]]
                        pattern = 1
                    else:
                        d_split, d_value = \
                            self._division_nearest(shape[-4], p_zero)
                        if d_value == 1:
                            tiling_shape = [shape[-4], shape[-2], shape[-1]]
                            pattern = 1
                        else:
                            tiling_shape = \
                                [d_value, shape[-3], shape[-2], shape[-1]]
                            pattern = 0
                else:
                    tiling_shape = [shape[-4], shape[-2], shape[-1]]
                    pattern = 1

            else:
                # wait for optimizer
                tiling_shape = [shape[-4], shape[-2], shape[-1]]
                pattern = 1
        else:
            tiling_shape = [shape[-4], shape[-2], shape[-1]]
            pattern = 1

        psm = self.calc_element(tiling_shape)
        total_core_loop_num = int(all_ele / psm)

        # core_num
        if total_core_loop_num < MAX_CORE_NUM:
            core_num = total_core_loop_num
        else:
            core_num = MAX_CORE_NUM

        params = {"core_num": core_num,
                  "total_core_loop_num": total_core_loop_num,
                  "tiling_shape": tiling_shape,
                  "psm": psm,
                  "pattern": pattern}

        return params

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

    def set_ub_tensor(self, tik_instance, num_gm, num_gm2ub,
                      tiling_shape, tiling_shape_gm2ub, tail, tail_block):
        """
        set ub tensor
        num_gm: the input size of gm
        num_gm2ub: the maximum capacity of ub
        tiling_shape: the storage space after sorting
        tiling_shape_gm2ub: the storage space before sorting
        input_z_ub: reg_move
        """
        if num_gm < num_gm2ub:
            input_x_ub = tik_instance.Tensor(self.dtype,
                                             tiling_shape_gm2ub,
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub = tik_instance.Tensor(self.dtype,
                                             tiling_shape, name="input_y_ub",
                                             scope=tik.scope_ubuf)
        else:
            tail = num_gm // num_gm2ub
            tail_block = num_gm % num_gm2ub

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             [num_gm2ub, 17, 16],
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub = tik_instance.Tensor(self.dtype,
                                             [num_gm2ub, 16, 16],
                                             name="input_y_ub",
                                             scope=tik.scope_ubuf)

        input_z_ub = tik_instance.Tensor(self.dtype, [64, ], name="input_z_ub",
                                         scope=tik.scope_ubuf)

        return input_x_ub, input_y_ub, input_z_ub, tail, tail_block

    def set_ub_tensor_int32(self, tik_instance, num_gm, num_gm2ub,
                            tiling_shape, tiling_shape_gm2ub, tail, tail_block):
        """
        set ub tensor
        num_gm: the input size of gm
        num_gm2ub: the maximum capacity of ub
        tiling_shape: the storage space after sorting
        tiling_shape_gm2ub: the storage space before sorting
        input_z_ub: reg_move
        """
        if num_gm < num_gm2ub:

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             tiling_shape_gm2ub,
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub = tik_instance.Tensor("float32",
                                             tiling_shape,
                                             name="input_y_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub_vconv = tik_instance.Tensor(self.dtype,
                                                   tiling_shape,
                                                   name="input_y_ub_vconv",
                                                   scope=tik.scope_ubuf)
            psm_vconv = self.calc_element(tiling_shape)

        else:
            tail = num_gm // num_gm2ub
            tail_block = num_gm % num_gm2ub

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             [num_gm2ub, 17, 16],
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub = tik_instance.Tensor("float32",
                                             [num_gm2ub, 16, 16],
                                             name="input_y_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub_vconv = tik_instance.Tensor(self.dtype,
                                                   [num_gm2ub, 16, 16],
                                                   name="input_y_ub_vconv",
                                                   scope=tik.scope_ubuf)

            psm_vconv = self.calc_element([num_gm2ub, 16, 16])

        input_z_ub = tik_instance.Tensor(self.dtype, [64, ], name="input_z_ub",
                                         scope=tik.scope_ubuf)

        return input_x_ub, input_y_ub, input_y_ub_vconv, \
               input_z_ub, tail, tail_block, psm_vconv

    def set_ub_tensor_int32_mini(self, tik_instance, num_gm, num_gm2ub,
                                 tiling_shape, tiling_shape_gm2ub,
                                 tail, tail_block):
        """
        set ub tensor
        num_gm: the input size of gm
        num_gm2ub: the maximum capacity of ub
        tiling_shape: the storage space after sorting
        tiling_shape_gm2ub: the storage space before sorting
        input_z_ub: reg_move
        """
        if num_gm < num_gm2ub:

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             tiling_shape_gm2ub,
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub = tik_instance.Tensor(self.dtype,
                                             tiling_shape,
                                             name="input_y_ub_vconv",
                                             scope=tik.scope_ubuf)

        else:
            tail = num_gm // num_gm2ub
            tail_block = num_gm % num_gm2ub

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             [num_gm2ub, 17, 16],
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub = tik_instance.Tensor(self.dtype,
                                             [num_gm2ub, 16, 16],
                                             name="input_y_ub_vconv",
                                             scope=tik.scope_ubuf)

        input_z_ub = tik_instance.Tensor(self.dtype,
                                         [16, ], name="input_z_ub",
                                         scope=tik.scope_ubuf)

        input_v_ub = tik_instance.Tensor(self.dtype,
                                         [16, ], name="input_v_ub",
                                         scope=tik.scope_ubuf)

        return input_x_ub, input_y_ub, input_z_ub, input_v_ub, tail, tail_block

    def data_move_gm_ub_int32(self, tik_instance, dst_tensor_ub, src_tensor_gm,
                              nburst, burstlen, srcstride, dststride,
                              src_index, dst_index):

        if srcstride <= MAX_STRIDE:

            tik_instance.data_move(dst_tensor_ub[dst_index],
                                   src_tensor_gm[src_index],
                                   0,
                                   nburst,
                                   burstlen,
                                   srcstride,
                                   dststride)
        else:
            src_x_index_gap = self.input_shape[-3] * 16 * 16
            dst_x_index_gap = 16 * 17
            dst_x_index = 0
            src_x_gm_index = src_index
            with tik_instance.for_range(0, nburst) as i:
                src_x_gm_index = src_x_gm_index + i * src_x_index_gap
                dst_x_index = dst_x_index + i * dst_x_index_gap
                tik_instance.data_move(dst_tensor_ub[dst_x_index],
                                       src_tensor_gm[src_x_gm_index],
                                       0,
                                       1,
                                       burstlen,
                                       0,
                                       0)

    def reorder_s322f32(self, tik_instance, vector_repeat_merchant,
                        vector_repeat_remainder,
                        mask, dst_tensor_ub, src_tensor_ub,
                        dst_ub_gap, src_ub_gap,
                        repeats, dst_blk_stride,
                        src_blk_stride, dst_rep_stride, src_rep_stride):
        # reorder
        with tik_instance.for_range(0, vector_repeat_merchant) as i:

            tik_instance.vconv(mask,
                               'none',
                               dst_tensor_ub[i * dst_ub_gap],
                               src_tensor_ub[i * src_ub_gap],
                               repeats,
                               dst_blk_stride,
                               src_blk_stride,
                               dst_rep_stride,
                               src_rep_stride)
            # fp32 need twice
            if mask == 64:
                tik_instance.vconv(mask,
                                   'none',
                                   dst_tensor_ub[i * dst_ub_gap + 8],
                                   src_tensor_ub[i * src_ub_gap + 8],
                                   repeats,
                                   dst_blk_stride,
                                   src_blk_stride,
                                   dst_rep_stride,
                                   src_rep_stride)

        if vector_repeat_remainder != 0:
            vconv_mask = int(vector_repeat_remainder / MASK * mask)
            tik_instance.vconv(
                vconv_mask,
                'none',
                dst_tensor_ub[vector_repeat_merchant * dst_ub_gap],
                src_tensor_ub[vector_repeat_merchant * src_ub_gap],
                repeats,
                dst_blk_stride,
                src_blk_stride,
                dst_rep_stride,
                src_rep_stride)

            if self.mask == 64:
                tik_instance.vconv(
                    vconv_mask,
                    'none',
                    dst_tensor_ub[vector_repeat_merchant * dst_ub_gap + 8],
                    src_tensor_ub[vector_repeat_merchant * src_ub_gap + 8],
                    repeats,
                    dst_blk_stride,
                    src_blk_stride,
                    dst_rep_stride,
                    src_rep_stride)

    def reorder_vadd_mini(self, tik_instance, vector_repeat_merchant,
                          vector_repeat_remainder,
                          mask, dst_tensor_ub, src_tensor_ub,
                          input_v_ub, dst_ub_gap, src_ub_gap,
                          repeats, dst_blk_stride, src_blk_stride,
                          dst_rep_stride, src_rep_stride):

        src0_blk_stride = src_blk_stride
        src0_rep_stride = src_rep_stride
        src1_blk_stride = 0
        src1_rep_stride = 0

        # reorder
        with tik_instance.for_range(0, vector_repeat_merchant) as i:

            tik_instance.vadd(mask,
                              dst_tensor_ub[i * dst_ub_gap],
                              src_tensor_ub[i * src_ub_gap],
                              input_v_ub[0],
                              repeats,
                              dst_blk_stride,
                              src0_blk_stride,
                              src1_blk_stride,
                              dst_rep_stride,
                              src0_rep_stride,
                              src1_rep_stride, )

            # fp32 need twice
            if mask == 64:
                tik_instance.vadd(mask,
                                  dst_tensor_ub[i * dst_ub_gap + 8],
                                  src_tensor_ub[i * src_ub_gap + 8],
                                  input_v_ub[0],
                                  repeats,
                                  dst_blk_stride,
                                  src0_blk_stride,
                                  src1_blk_stride,
                                  dst_rep_stride,
                                  src0_rep_stride,
                                  src1_rep_stride, )

        if vector_repeat_remainder != 0:
            vconv_mask = int(vector_repeat_remainder / MASK * mask)
            tik_instance.vadd(
                vconv_mask,
                dst_tensor_ub[vector_repeat_merchant * dst_ub_gap],
                src_tensor_ub[vector_repeat_merchant * src_ub_gap],
                input_v_ub[0],
                repeats,
                dst_blk_stride,
                src0_blk_stride,
                src1_blk_stride,
                dst_rep_stride,
                src0_rep_stride,
                src1_rep_stride, )

            if self.mask == 64:
                tik_instance.vadd(
                    vconv_mask,
                    dst_tensor_ub[vector_repeat_merchant * dst_ub_gap + 8],
                    src_tensor_ub[vector_repeat_merchant * src_ub_gap + 8],
                    input_v_ub[0],
                    repeats,
                    dst_blk_stride,
                    src0_blk_stride,
                    src1_blk_stride,
                    dst_rep_stride,
                    src0_rep_stride,
                    src1_rep_stride, )

    def reorder_s322f32_pattern_0(self, tik_instance, vector_repeat_merchant_d,
                                  vector_repeat_remainder_d,
                                  vector_repeat_merchant_c,
                                  vector_repeat_remainder_c,
                                  mask, dst_tensor_ub, src_tensor_ub,
                                  dst_ub_gap, src_ub_gap,
                                  repeats, dst_blk_stride, src_blk_stride,
                                  dst_rep_stride, src_rep_stride, nburst):

        # reorder
        with tik_instance.for_range(0, vector_repeat_merchant_c) as j:
            with tik_instance.for_range(0, vector_repeat_merchant_d) as i:

                tik_instance.vconv(
                    mask,
                    'none',
                    dst_tensor_ub[i * dst_ub_gap +
                                  j * nburst * 16 * MAX_REPEAT],
                    src_tensor_ub[i * src_ub_gap + j * MAX_REPEAT * 16],
                    repeats,
                    dst_blk_stride,
                    src_blk_stride,
                    dst_rep_stride,
                    src_rep_stride)
                # fp32 need twice
                if self.mask == 64:
                    tik_instance.vconv(
                        mask,
                        'none',
                        dst_tensor_ub[i * dst_ub_gap + 8 +
                                      j * nburst * 16 * MAX_REPEAT],
                        src_tensor_ub[i * src_ub_gap + 8 +
                                      j * MAX_REPEAT * 16],
                        repeats,
                        dst_blk_stride,
                        src_blk_stride,
                        dst_rep_stride,
                        src_rep_stride)

            if vector_repeat_remainder_d != 0:
                vconv_mask = int(vector_repeat_remainder_d / MASK * self.mask)
                tik_instance.vconv(
                    vconv_mask,
                    'none',
                    dst_tensor_ub[vector_repeat_merchant_d *
                                  dst_ub_gap + j * nburst * 16 * MAX_REPEAT],
                    src_tensor_ub[vector_repeat_merchant_d *
                                  src_ub_gap + j * MAX_REPEAT * 16],
                    repeats,
                    dst_blk_stride,
                    src_blk_stride,
                    dst_rep_stride,
                    src_rep_stride)

                if self.mask == 64:
                    tik_instance.vconv(
                        vconv_mask,
                        'none',
                        dst_tensor_ub[vector_repeat_merchant_d *
                                      dst_ub_gap + 8 +
                                      j * nburst * 16 * MAX_REPEAT],
                        src_tensor_ub[vector_repeat_merchant_d *
                                      src_ub_gap + 8 + j * MAX_REPEAT * 16],
                        repeats,
                        dst_blk_stride,
                        src_blk_stride,
                        dst_rep_stride,
                        src_rep_stride)

        if vector_repeat_remainder_c != 0:
            repeats = vector_repeat_remainder_c
            with tik_instance.for_range(0, vector_repeat_merchant_d) as i:

                tik_instance.vconv(mask,
                                   'none',
                                   dst_tensor_ub[i * dst_ub_gap +
                                                 vector_repeat_merchant_c *
                                                 nburst * 16 * MAX_REPEAT],
                                   src_tensor_ub[i * src_ub_gap +
                                                 vector_repeat_merchant_c *
                                                 MAX_REPEAT * 16],
                                   repeats,
                                   dst_blk_stride,
                                   src_blk_stride,
                                   dst_rep_stride,
                                   src_rep_stride)

                # fp32 need twice
                if self.mask == 64:
                    tik_instance.vconv(mask,
                                       'none',
                                       dst_tensor_ub[i * dst_ub_gap + 8 +
                                                     vector_repeat_merchant_c *
                                                     nburst * 16 * MAX_REPEAT],
                                       src_tensor_ub[i * src_ub_gap + 8 +
                                                     vector_repeat_merchant_c *
                                                     MAX_REPEAT * 16],
                                       repeats,
                                       dst_blk_stride,
                                       src_blk_stride,
                                       dst_rep_stride,
                                       src_rep_stride)

            if vector_repeat_remainder_d != 0:
                vconv_mask = int(vector_repeat_remainder_d / MASK * self.mask)
                tik_instance.vconv(vconv_mask,
                                   'none',
                                   dst_tensor_ub[vector_repeat_merchant_d *
                                                 dst_ub_gap +
                                                 vector_repeat_merchant_c *
                                                 nburst * 16 * MAX_REPEAT],
                                   src_tensor_ub[vector_repeat_merchant_d *
                                                 src_ub_gap +
                                                 vector_repeat_merchant_c *
                                                 MAX_REPEAT * 16],
                                   repeats,
                                   dst_blk_stride,
                                   src_blk_stride,
                                   dst_rep_stride,
                                   src_rep_stride)

                if self.mask == 64:
                    tik_instance.vconv(vconv_mask,
                                       'none',
                                       dst_tensor_ub[vector_repeat_merchant_d *
                                                     dst_ub_gap + 8 +
                                                     vector_repeat_merchant_c *
                                                     nburst * 16 * MAX_REPEAT],
                                       src_tensor_ub[vector_repeat_merchant_d *
                                                     src_ub_gap + 8 +
                                                     vector_repeat_merchant_c *
                                                     MAX_REPEAT * 16],
                                       repeats,
                                       dst_blk_stride,
                                       src_blk_stride,
                                       dst_rep_stride,
                                       src_rep_stride)

    def reorder_vadd_pattern_0_mini(self, tik_instance,
                                    vector_repeat_merchant_d,
                                    vector_repeat_remainder_d,
                                    vector_repeat_merchant_c,
                                    vector_repeat_remainder_c,
                                    mask, dst_tensor_ub, src_tensor_ub,
                                    input_v_ub, dst_ub_gap, src_ub_gap,
                                    repeats, dst_blk_stride,
                                    src_blk_stride, dst_rep_stride,
                                    src_rep_stride, nburst):

        src0_blk_stride = src_blk_stride
        src0_rep_stride = src_rep_stride
        src1_blk_stride = 0
        src1_rep_stride = 0

        # reorder
        with tik_instance.for_range(0, vector_repeat_merchant_c) as j:
            with tik_instance.for_range(0, vector_repeat_merchant_d) as i:

                tik_instance.vadd(
                    mask,
                    dst_tensor_ub[i * dst_ub_gap + j *
                                  nburst * 16 * MAX_REPEAT],
                    src_tensor_ub[i * src_ub_gap + j * MAX_REPEAT * 16],
                    input_v_ub[0],
                    repeats,
                    dst_blk_stride,
                    src0_blk_stride,
                    src1_blk_stride,
                    dst_rep_stride,
                    src0_rep_stride,
                    src1_rep_stride, )

                # fp32 need twice
                if self.mask == 64:
                    tik_instance.vadd(
                        mask,
                        dst_tensor_ub[i * dst_ub_gap + 8 +
                                      j * nburst * 16 * MAX_REPEAT],
                        src_tensor_ub[i * src_ub_gap + 8 +
                                      j * MAX_REPEAT * 16],
                        input_v_ub[0],
                        repeats,
                        dst_blk_stride,
                        src0_blk_stride,
                        src1_blk_stride,
                        dst_rep_stride,
                        src0_rep_stride,
                        src1_rep_stride, )

            if vector_repeat_remainder_d != 0:
                vconv_mask = int(vector_repeat_remainder_d / MASK * self.mask)
                tik_instance.vadd(
                    vconv_mask,
                    dst_tensor_ub[vector_repeat_merchant_d *
                                  dst_ub_gap + j * nburst * 16 * MAX_REPEAT],
                    src_tensor_ub[vector_repeat_merchant_d *
                                  src_ub_gap + j * MAX_REPEAT * 16],
                    input_v_ub[0],
                    repeats,
                    dst_blk_stride,
                    src0_blk_stride,
                    src1_blk_stride,
                    dst_rep_stride,
                    src0_rep_stride,
                    src1_rep_stride, )

                if self.mask == 64:
                    tik_instance.vadd(
                        vconv_mask,
                        dst_tensor_ub[vector_repeat_merchant_d *
                                      dst_ub_gap + 8 +
                                      j * nburst * 16 * MAX_REPEAT],
                        src_tensor_ub[vector_repeat_merchant_d *
                                      src_ub_gap + 8 + j *
                                      MAX_REPEAT * 16],
                        input_v_ub[0],
                        repeats,
                        dst_blk_stride,
                        src0_blk_stride,
                        src1_blk_stride,
                        dst_rep_stride,
                        src0_rep_stride,
                        src1_rep_stride, )

        if vector_repeat_remainder_c != 0:
            repeats = vector_repeat_remainder_c
            with tik_instance.for_range(0, vector_repeat_merchant_d) as i:
                tik_instance.vadd(
                    mask,
                    dst_tensor_ub[i * dst_ub_gap +
                                  vector_repeat_merchant_c *
                                  nburst * 16 * MAX_REPEAT],
                    src_tensor_ub[i * src_ub_gap +
                                  vector_repeat_merchant_c *
                                  MAX_REPEAT * 16],
                    input_v_ub[0],
                    repeats,
                    dst_blk_stride,
                    src0_blk_stride,
                    src1_blk_stride,
                    dst_rep_stride,
                    src0_rep_stride,
                    src1_rep_stride, )

                # fp32 need twice
                if self.mask == 64:
                    tik_instance.vadd(
                        mask,
                        dst_tensor_ub[i * dst_ub_gap + 8 +
                                      vector_repeat_merchant_c *
                                      nburst * 16 * MAX_REPEAT],
                        src_tensor_ub[i * src_ub_gap + 8 +
                                      vector_repeat_merchant_c *
                                      MAX_REPEAT * 16],
                        input_v_ub[0],
                        repeats,
                        dst_blk_stride,
                        src0_blk_stride,
                        src1_blk_stride,
                        dst_rep_stride,
                        src0_rep_stride,
                        src1_rep_stride, )

            if vector_repeat_remainder_d != 0:
                vconv_mask = int(vector_repeat_remainder_d / MASK * self.mask)

                tik_instance.vadd(
                    vconv_mask,
                    dst_tensor_ub[vector_repeat_merchant_d *
                                  dst_ub_gap + vector_repeat_merchant_c *
                                  nburst * 16 * MAX_REPEAT],
                    src_tensor_ub[vector_repeat_merchant_d *
                                  src_ub_gap + vector_repeat_merchant_c *
                                  MAX_REPEAT * 16],
                    input_v_ub[0],
                    repeats,
                    dst_blk_stride,
                    src0_blk_stride,
                    src1_blk_stride,
                    dst_rep_stride,
                    src0_rep_stride,
                    src1_rep_stride, )

                if self.mask == 64:
                    tik_instance.vadd(
                        vconv_mask,
                        dst_tensor_ub[vector_repeat_merchant_d *
                                      dst_ub_gap + 8 +
                                      vector_repeat_merchant_c * nburst
                                      * 16 * MAX_REPEAT],
                        src_tensor_ub[vector_repeat_merchant_d *
                                      src_ub_gap + 8 +
                                      vector_repeat_merchant_c *
                                      MAX_REPEAT * 16],
                        input_v_ub[0],
                        repeats,
                        dst_blk_stride,
                        src0_blk_stride,
                        src1_blk_stride,
                        dst_rep_stride,
                        src0_rep_stride,
                        src1_rep_stride, )

    def f322s32(self, tik_instance, vconv_repeat_merchant,
                vconv_repeat_remainder, mask,
                dst_tensor_ub, src_tensor_ub, dst_ub_gap, src_ub_gap,
                repeats, dst_blk_stride_vconv, src_blk_stride_vconv,
                dst_rep_stride_vconv, src_rep_stride_vconv):

        # cast
        with tik_instance.for_range(0, vconv_repeat_merchant) as i:
            tik_instance.vconv(mask,
                               'round',
                               dst_tensor_ub[i * dst_ub_gap],
                               src_tensor_ub[i * src_ub_gap],
                               repeats,
                               dst_blk_stride_vconv,
                               src_blk_stride_vconv,
                               dst_rep_stride_vconv,
                               src_rep_stride_vconv)

        if vconv_repeat_remainder != 0:
            tik_instance.vconv(mask,
                               'round',
                               dst_tensor_ub[vconv_repeat_merchant *
                                             dst_ub_gap],
                               src_tensor_ub[vconv_repeat_merchant *
                                             src_ub_gap],
                               vconv_repeat_remainder,
                               dst_blk_stride_vconv,
                               src_blk_stride_vconv,
                               dst_rep_stride_vconv,
                               src_rep_stride_vconv)

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
                                        dst[dup_repeat_merchant * dup_psm +
                                            repeats * self.mask],
                                        number,
                                        1,
                                        dst_blk_stride,
                                        dst_rep_stride)

    def pattern_case_zero(self, tik_instance, params, dtype):
        pattern = params.get("pattern")
        if dtype == "int32":
            if not tbe_platform.cce_conf.api_check_support("tik.vconv", "s322f32"):
                if pattern == 0:
                    self.nz2nd_normal_case0_int32_mini(tik_instance, params)
                else:
                    self.nz2nd_normal_case1_int32_mini(tik_instance, params)
            else:
                if pattern == 0:
                    self.nz2nd_normal_case0_int32(tik_instance, params)
                else:
                    self.nz2nd_normal_case1_int32(tik_instance, params)
        else:
            if pattern == 0:
                self.nz2nd_normal_case0(tik_instance, params)
            else:
                self.nz2nd_normal_case1(tik_instance, params)

    def pattern_case_one(self, tik_instance, params, dtype):
        if dtype == "int32":
            if not tbe_platform.cce_conf.api_check_support("tik.vconv", "s322f32"):
                self.nz2nd_special_case0_int32_mini(tik_instance, params)
            else:
                self.nz2nd_special_case0_int32(tik_instance, params)
        else:
            self.nz2nd_special_case0(tik_instance, params)

    def pattern_case_two(self, tik_instance, params):
        self.nz2nd_special_case1(tik_instance, params)

    def pattern_case_three(self, tik_instance, params, dtype):
        pattern = params.get("pattern")
        if dtype == "int32":
            if pattern == 0:
                if not tbe_platform.cce_conf.api_check_support("tik.vconv", "s322f32"):
                    self.nz2nd_special_case2_int32_pattern_zero_mini(
                        tik_instance, params)
                else:
                    self.nz2nd_special_case2_int32_pattern_zero(
                        tik_instance, params)
            else:
                if not tbe_platform.cce_conf.api_check_support("tik.vconv", "s322f32"):
                    self.nz2nd_special_case2_int32_mini(tik_instance, params)
                else:
                    self.nz2nd_special_case2_int32(tik_instance, params)
        else:
            if pattern == 1:
                self.nz2nd_special_case2(tik_instance, params)
            else:
                raise RuntimeError(
                    "fp32 or fp16 just only support pattern==1 currently")

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.nz_2_nd_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.input_x_gm],
                              outputs=[self.output_y_gm])

        return tik_instance

    def nz2nd_normal_case0(self, tik_instance, params):
        """
        no padding
        [32,16,16,16,16,16]
        deal [16,16,16,16] == [D,C,16,16]
        [32,16,256,256]
        total_num = 32*16
        """
        # D-axis optimization
        # base shape: [D,1,16,16]
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        core_number = core_num
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        c_value = tiling_shape[1]
        d_value = tiling_shape[0]
        for c_split in range(c_value):
            c_split = c_split + 1
            psm_ub_in = d_value * 16 * (c_split * 16 + 1)
            psm_ub_out = d_value * 16 * c_split * 16
            # because of db, self.maximum_size_ub // 2
            if psm_ub_in > self.maximum_size_ub // 2:
                c_split = c_split - 1
                psm_ub_in = d_value * 16 * (c_split * 16 + 1)
                psm_ub_out = d_value * 16 * c_split * 16
                break
        if c_split == 0:
            # psm exceed space of ub and [D,1,16,16] exceed space of ub
            # should return case1
            raise RuntimeError("not support c_split == 0")
        else:
            # psm exceed space of ub  while [D,1,16,16] not exceed space of ub
            tail_d = 0
            tail_block_d = d_value
            tail_c = c_value // c_split
            tail_block_c = c_value % c_split

        with tik_instance.for_range(0, core_number,
                                    block_num=core_number) as num_core:
            # calculate the loop number on each core
            core_loop, sum_core = _cal_core(tik_instance,
                                            total_core_loop_num,
                                            num_core, core_number)

            # calculate serial data until tail_block
            def case0_main_tail(loop, x_ub, y_ub):
                num_core_loop = loop
                input_x_ub = x_ub
                input_y_ub = y_ub

                # deal [0,tail_c) data
                merchant = sum_core + num_core_loop // tail_c
                src_x_index = psm * merchant + \
                              num_core_loop % tail_c * 16 * 16 * c_split

                nburst_gm2ub = tail_block_d
                burstlen_gm2ub = c_split * 16 * 16 * self.num_bit // 32
                srcstride_gm2ub = (tail_block_c + (tail_c - 1) * c_split) * \
                                  16 * 16 * self.num_bit // 32
                dststride_gm2ub = 16 * self.num_bit // 32

                if srcstride_gm2ub <= MAX_STRIDE:

                    tik_instance.data_move(input_x_ub[0],
                                           self.input_x_gm[src_x_index],
                                           0,
                                           nburst_gm2ub,
                                           burstlen_gm2ub,
                                           srcstride_gm2ub,
                                           dststride_gm2ub)
                else:
                    src_x_index_gap = self.input_shape[-3] * 16 * 16
                    dst_x_index_gap = (tail_block_c * 16 + 1) * 16
                    dst_x_index = 0
                    with tik_instance.for_range(0, nburst_gm2ub) as i:
                        src_x_index = src_x_index + i * src_x_index_gap
                        dst_x_index = dst_x_index + i * dst_x_index_gap
                        tik_instance.data_move(
                            input_x_ub[dst_x_index],
                            self.input_x_gm[src_x_index],
                            0,
                            1,
                            burstlen_gm2ub,
                            0,
                            0)

                # vector maxcompute 128
                vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                vector_repeat_merchant_c = c_split * 16 // MAX_REPEAT
                vector_repeat_remainder_c = c_split * 16 % MAX_REPEAT

                src_ub_gap = MASK * (c_split * 16 + 1)
                dst_ub_gap = MASK
                src_blk_stride = 16 * (c_split * 16 + 1) * \
                                 self.num_bit // DATA_MOVE_MIN_UNIT
                src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                dst_rep_stride = tail_block_d * 16 * \
                                 self.num_bit // DATA_MOVE_MIN_UNIT
                repeat_times = MAX_REPEAT

                # order
                with tik_instance.for_range(
                        0, vector_repeat_merchant_c) as j:
                    with tik_instance.for_range(
                            0, vector_repeat_merchant_d) as i:

                        tik_instance.vadds(
                            self.mask,
                            input_y_ub[i * dst_ub_gap +
                                       j * tail_block_d *
                                       16 * MAX_REPEAT],
                            input_x_ub[i * src_ub_gap +
                                       j * MAX_REPEAT * 16],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)
                        # fp32 need twice
                        if self.mask == 64:
                            tik_instance.vadds(
                                self.mask,
                                input_y_ub[i * dst_ub_gap +
                                           8 + j * tail_block_d *
                                           16 * MAX_REPEAT],
                                input_x_ub[i * src_ub_gap +
                                           8 + j * MAX_REPEAT * 16],
                                0,
                                repeat_times,
                                dst_blk_stride,
                                src_blk_stride,
                                dst_rep_stride,
                                src_rep_stride)

                    if vector_repeat_remainder_d != 0:
                        vadds_mask = int(vector_repeat_remainder_d /
                                         MASK * self.mask)
                        tik_instance.vadds(
                            vadds_mask,
                            input_y_ub[vector_repeat_merchant_d *
                                       dst_ub_gap + j *
                                       tail_block_d * 16 * MAX_REPEAT],
                            input_x_ub[vector_repeat_merchant_d *
                                       src_ub_gap + j *
                                       MAX_REPEAT * 16],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)

                        if self.mask == 64:
                            tik_instance.vadds(
                                vadds_mask,
                                input_y_ub[vector_repeat_merchant_d *
                                           dst_ub_gap + 8 +
                                           j * tail_block_d *
                                           16 * MAX_REPEAT],
                                input_x_ub[vector_repeat_merchant_d *
                                           src_ub_gap + 8 +
                                           j * MAX_REPEAT * 16],
                                0,
                                repeat_times,
                                dst_blk_stride,
                                src_blk_stride,
                                dst_rep_stride,
                                src_rep_stride)

                if vector_repeat_remainder_c != 0:
                    repeat_times = vector_repeat_remainder_c
                    with tik_instance.for_range(
                            0, vector_repeat_merchant_d) as i:

                        tik_instance.vadds(
                            self.mask,
                            input_y_ub[i * dst_ub_gap +
                                       vector_repeat_merchant_c *
                                       tail_block_d * 16 * MAX_REPEAT],
                            input_x_ub[i * src_ub_gap +
                                       vector_repeat_merchant_c *
                                       MAX_REPEAT * 16],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)

                        # fp32 need twice
                        if self.mask == 64:
                            tik_instance.vadds(
                                self.mask,
                                input_y_ub[i * dst_ub_gap +
                                           8 +
                                           vector_repeat_merchant_c *
                                           tail_block_d * 16 *
                                           MAX_REPEAT],
                                input_x_ub[i * src_ub_gap + 8 +
                                           vector_repeat_merchant_c *
                                           MAX_REPEAT * 16],
                                0,
                                repeat_times,
                                dst_blk_stride,
                                src_blk_stride,
                                dst_rep_stride,
                                src_rep_stride)

                    if vector_repeat_remainder_d != 0:
                        vadds_mask = int(vector_repeat_remainder_d / MASK * self.mask)
                        tik_instance.vadds(
                            vadds_mask,
                            input_y_ub[vector_repeat_merchant_d *
                                       dst_ub_gap +
                                       vector_repeat_merchant_c *
                                       tail_block_d * 16 * MAX_REPEAT],
                            input_x_ub[vector_repeat_merchant_d *
                                       src_ub_gap +
                                       vector_repeat_merchant_c *
                                       MAX_REPEAT * 16],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)

                        if self.mask == 64:
                            tik_instance.vadds(
                                vadds_mask,
                                input_y_ub[vector_repeat_merchant_d *
                                           dst_ub_gap + 8 +
                                           vector_repeat_merchant_c *
                                           tail_block_d * 16 *
                                           MAX_REPEAT],
                                input_x_ub[vector_repeat_merchant_d *
                                           src_ub_gap + 8 +
                                           vector_repeat_merchant_c *
                                           MAX_REPEAT * 16],
                                0,
                                repeat_times,
                                dst_blk_stride,
                                src_blk_stride,
                                dst_rep_stride,
                                src_rep_stride)

                # out
                dst_gm = acture_memory * merchant
                dst_gm = dst_gm + num_core_loop % tail_c * psm_ub_out

                nburst_ub2gm = 16 * c_split
                burstlen_ub2gm = int(tail_block_d * 16 * self.num_bit /
                                     DATA_MOVE_MIN_UNIT)
                src_stride = 0
                dst_stride = int((self.output_shape[-1] -
                                  tail_block_d * 16) *
                                 self.num_bit / DATA_MOVE_MIN_UNIT)
                if dst_stride <= MAX_STRIDE:

                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                           input_y_ub[0],
                                           0,
                                           nburst_ub2gm,
                                           burstlen_ub2gm,
                                           src_stride,
                                           dst_stride)
                else:
                    src_y_ub_index = 0
                    src_y_ub_gap = tail_block_d * 16
                    dst_gm_gap = self.output_shape[-1]
                    with tik_instance.for_range(0, nburst_ub2gm) as i:
                        src_y_ub_index = src_y_ub_index + \
                                         i * src_y_ub_gap
                        dst_gm = dst_gm + i * dst_gm_gap
                        tik_instance.data_move(
                            self.output_y_gm[dst_gm],
                            input_y_ub[src_y_ub_index],
                            0,
                            1,
                            burstlen_ub2gm,
                            0,
                            0)

            # calculate tail_block data
            def case0_main_tail_block(loop, x_ub, y_ub):
                num_core_loop = loop
                input_x_ub = x_ub
                input_y_ub = y_ub

                # deal tail_block firstly
                merchant = sum_core + num_core_loop
                src_x_index = psm * merchant + tail_c * 16 * 16 * c_split

                nburst_gm2ub = tail_block_d
                burstlen_gm2ub = tail_block_c * 16 * 16 * self.num_bit // 32
                srcstride_gm2ub = (self.input_shape[-3] - tail_block_c) * \
                                  16 * 16 * self.num_bit // 32
                dststride_gm2ub = 16 * self.num_bit // 32

                if srcstride_gm2ub <= MAX_STRIDE:

                    tik_instance.data_move(input_x_ub[0],
                                           self.input_x_gm[src_x_index],
                                           0,
                                           nburst_gm2ub,
                                           burstlen_gm2ub,
                                           srcstride_gm2ub,
                                           dststride_gm2ub)
                else:
                    src_x_index_gap = self.input_shape[-3] * 16 * 16
                    dst_x_index_gap = (tail_block_c * 16 + 1) * 16
                    dst_x_index = 0
                    with tik_instance.for_range(0, nburst_gm2ub) as i:
                        src_x_index = src_x_index + i * src_x_index_gap
                        dst_x_index = dst_x_index + i * dst_x_index_gap
                        tik_instance.data_move(input_x_ub[dst_x_index],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               1,
                                               burstlen_gm2ub,
                                               0,
                                               0)

                # vector maxcompute 128
                vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                vector_repeat_merchant_c = tail_block_c * 16 // MAX_REPEAT
                vector_repeat_remainder_c = tail_block_c * 16 % MAX_REPEAT

                src_ub_gap = MASK * (tail_block_c * 16 + 1)
                dst_ub_gap = MASK
                src_blk_stride = 16 * (tail_block_c * 16 + 1) * \
                                 self.num_bit // DATA_MOVE_MIN_UNIT
                src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                # in this branch, tail_block_d == tiling_shape[0]
                dst_rep_stride = tail_block_d * 16 * \
                                 self.num_bit // DATA_MOVE_MIN_UNIT
                repeat_times = MAX_REPEAT

                # order
                with tik_instance.for_range(
                        0, vector_repeat_merchant_c) as j:
                    with tik_instance.for_range(
                            0, vector_repeat_merchant_d) as i:

                        tik_instance.vadds(self.mask,
                                           input_y_ub[i * dst_ub_gap +
                                                      j * tail_block_d *
                                                      16 * MAX_REPEAT],
                                           input_x_ub[i * src_ub_gap +
                                                      j * MAX_REPEAT * 16],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)
                        # fp32 need twice
                        if self.mask == 64:
                            tik_instance.vadds(
                                self.mask,
                                input_y_ub[i * dst_ub_gap +
                                           8 + j * tail_block_d *
                                           16 * MAX_REPEAT],
                                input_x_ub[i * src_ub_gap +
                                           8 + j * MAX_REPEAT * 16],
                                0,
                                repeat_times,
                                dst_blk_stride,
                                src_blk_stride,
                                dst_rep_stride,
                                src_rep_stride)

                    if vector_repeat_remainder_d != 0:
                        vadds_mask = int(vector_repeat_remainder_d /
                                         MASK * self.mask)
                        tik_instance.vadds(
                            vadds_mask,
                            input_y_ub[vector_repeat_merchant_d *
                                       dst_ub_gap + j * tail_block_d *
                                       16 * MAX_REPEAT],
                            input_x_ub[vector_repeat_merchant_d *
                                       src_ub_gap + j * MAX_REPEAT * 16],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)

                        if self.mask == 64:
                            tik_instance.vadds(
                                vadds_mask,
                                input_y_ub[vector_repeat_merchant_d *
                                           dst_ub_gap + 8 + j *
                                           tail_block_d * 16 * MAX_REPEAT],
                                input_x_ub[vector_repeat_merchant_d *
                                           src_ub_gap + 8 + j *
                                           MAX_REPEAT * 16],
                                0,
                                repeat_times,
                                dst_blk_stride,
                                src_blk_stride,
                                dst_rep_stride,
                                src_rep_stride)

                if vector_repeat_remainder_c != 0:
                    repeat_times = vector_repeat_remainder_c
                    with tik_instance.for_range(
                            0, vector_repeat_merchant_d) as i:

                        tik_instance.vadds(
                            self.mask,
                            input_y_ub[i * dst_ub_gap +
                                       vector_repeat_merchant_c *
                                       tail_block_d * 16 * MAX_REPEAT],
                            input_x_ub[i * src_ub_gap +
                                       vector_repeat_merchant_c *
                                       MAX_REPEAT * 16],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)

                        # fp32 need twice
                        if self.mask == 64:
                            tik_instance.vadds(
                                self.mask,
                                input_y_ub[i * dst_ub_gap + 8 +
                                           vector_repeat_merchant_c *
                                           tail_block_d * 16 * MAX_REPEAT],
                                input_x_ub[i * src_ub_gap + 8 +
                                           vector_repeat_merchant_c *
                                           MAX_REPEAT * 16],
                                0,
                                repeat_times,
                                dst_blk_stride,
                                src_blk_stride,
                                dst_rep_stride,
                                src_rep_stride)

                    if vector_repeat_remainder_d != 0:
                        vadds_mask = int(vector_repeat_remainder_d /
                                         MASK * self.mask)
                        tik_instance.vadds(
                            vadds_mask,
                            input_y_ub[vector_repeat_merchant_d *
                                       dst_ub_gap +
                                       vector_repeat_merchant_c *
                                       tail_block_d * 16 * MAX_REPEAT],
                            input_x_ub[vector_repeat_merchant_d *
                                       src_ub_gap +
                                       vector_repeat_merchant_c *
                                       MAX_REPEAT * 16],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)

                        if self.mask == 64:
                            tik_instance.vadds(
                                vadds_mask,
                                input_y_ub[vector_repeat_merchant_d *
                                           dst_ub_gap + 8 +
                                           vector_repeat_merchant_c *
                                           tail_block_d * 16 * MAX_REPEAT],
                                input_x_ub[vector_repeat_merchant_d *
                                           src_ub_gap + 8 +
                                           vector_repeat_merchant_c *
                                           MAX_REPEAT * 16],
                                0,
                                repeat_times,
                                dst_blk_stride,
                                src_blk_stride,
                                dst_rep_stride,
                                src_rep_stride)

                # out
                # in this branch, tail_d = 0,  nburst_ub2gm  also can be 1
                dst_gm = acture_memory * merchant
                dst_gm = dst_gm + tail_c * psm_ub_out

                nburst_ub2gm = 16 * tail_block_c
                burstlen_ub2gm = int(tail_block_d * 16 * self.num_bit /
                                     DATA_MOVE_MIN_UNIT)
                src_stride = 0
                dst_stride = int((self.output_shape[-1] -
                                  tail_block_d * 16) *
                                 self.num_bit / DATA_MOVE_MIN_UNIT)

                if dst_stride <= MAX_STRIDE:

                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                           input_y_ub[0],
                                           0,
                                           nburst_ub2gm,
                                           burstlen_ub2gm,
                                           src_stride,
                                           dst_stride)
                else:
                    src_y_ub_index = 0
                    src_y_ub_gap = tail_block_d * 16
                    dst_gm_gap = self.output_shape[-1]
                    with tik_instance.for_range(0, nburst_ub2gm) as i:
                        src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                        dst_gm = dst_gm + i * dst_gm_gap
                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[src_y_ub_index],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

            # set ub tensor in case0
            def set_ub_tensor():
                input_x_ub = tik_instance.Tensor(self.dtype,
                                                 [psm_ub_in, ],
                                                 name="input_x_ub",
                                                 scope=tik.scope_ubuf)

                input_y_ub = tik_instance.Tensor(self.dtype,
                                                 [psm_ub_out, ],
                                                 name="input_y_ub",
                                                 scope=tik.scope_ubuf)
                return input_x_ub, input_y_ub

            # check whether the conditions meet double_buffer
            # core_loop must not be zero
            # tail_c mabe zero
            if tail_c != 0:
                thread = \
                    _check_db(total_core_loop_num * tail_c, core_number)
                loop_num = core_loop * tail_c
                with tik_instance.new_stmt_scope():
                    with tik_instance.for_range(0, loop_num,
                                                thread_num=thread) as loop:
                            x_ub, y_ub = set_ub_tensor()
                            case0_main_tail(loop, x_ub, y_ub)

            if tail_block_c != 0:
                thread = \
                    _check_db(total_core_loop_num, core_number)
                loop_num = core_loop
                with tik_instance.for_range(0, loop_num,
                                            thread_num=thread) as num_loop:
                    x_ub, y_ub = set_ub_tensor()
                    case0_main_tail_block(num_loop, x_ub, y_ub)

    def nz2nd_normal_case0_int32(self, tik_instance, params):
        """
        no padding
        [32,16,16,16,16,16]
        deal [16,16,16,16] == [D,C,16,16]
        [32,16,256,256]
        total_num = 32*16
        """
        # D-axis optimization
        # base shape: [D,1,16,16]
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        core_number = core_num
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        c_value = tiling_shape[1]
        d_value = tiling_shape[0]
        for c_split in range(c_value):
            c_split = c_split + 1
            psm_ub_in = d_value * 16 * (c_split * 16 + 1)
            psm_ub_out = d_value * 16 * c_split * 16
            if psm_ub_in > self.maximum_size_ub:
                c_split = c_split - 1
                psm_ub_in = d_value * 16 * (c_split * 16 + 1)
                psm_ub_out = d_value * 16 * c_split * 16
                break

        if c_split == 0:
            # psm exceed space of ub and [D,1,16,16] exceed space of ub
            # should return case1
            raise RuntimeError("not support c_split = 0")
        else:
            # psm exceed space of ub  while [D,1,16,16] not exceed space of ub
            tail_d = 0
            tail_block_d = d_value
            tail_c = c_value // c_split
            tail_block_c = c_value % c_split

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             [psm_ub_in, ], name="input_x_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub = tik_instance.Tensor("float32",
                                             [psm_ub_out, ], name="input_y_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub_vconv = tik_instance.Tensor(self.dtype,
                                                   [psm_ub_out, ],
                                                   name="input_y_ub",
                                                   scope=tik.scope_ubuf)

        with tik_instance.for_range(
                0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:

                if tail_block_c != 0:
                    # deal tail_block firstly
                    merchant = sum_core + num_core_loop
                    src_x_index = psm * merchant + tail_c * 16 * 16 * c_split

                    nburst_gm2ub = tail_block_d
                    burstlen_gm2ub = tail_block_c * 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - tail_block_c) * \
                                      16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    if srcstride_gm2ub <= MAX_STRIDE:

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)
                    else:
                        src_x_index_gap = self.input_shape[-3] * 16 * 16
                        dst_x_index_gap = (tail_block_c * 16 + 1) * 16
                        dst_x_index = 0
                        with tik_instance.for_range(0, nburst_gm2ub) as i:
                            src_x_index = src_x_index + i * src_x_index_gap
                            dst_x_index = dst_x_index + i * dst_x_index_gap
                            tik_instance.data_move(input_x_ub[dst_x_index],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   1,
                                                   burstlen_gm2ub,
                                                   0,
                                                   0)

                    # -----------------Vconv int32 to fp32--------------------
                    vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                    vector_repeat_merchant_c = tail_block_c * 16 // MAX_REPEAT
                    vector_repeat_remainder_c = tail_block_c * 16 % MAX_REPEAT

                    src_ub_gap = MASK * (tail_block_c * 16 + 1)
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * (tail_block_c * 16 + 1) * \
                                     self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = tail_block_d * 16 * \
                                     self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = MAX_REPEAT

                    self.reorder_s322f32_pattern_0(tik_instance,
                                                   vector_repeat_merchant_d,
                                                   vector_repeat_remainder_d,
                                                   vector_repeat_merchant_c,
                                                   vector_repeat_remainder_c,
                                                   self.mask, input_y_ub,
                                                   input_x_ub, dst_ub_gap,
                                                   src_ub_gap,
                                                   repeat_times,
                                                   dst_blk_stride,
                                                   src_blk_stride,
                                                   dst_rep_stride,
                                                   src_rep_stride, nburst_gm2ub)

                    # -----------------Vconv fp32 to int32--------------------
                    all_rep = nburst_gm2ub * 16 * tail_block_c * 16 // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant,
                                 vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv,
                                 input_y_ub, dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # out
                    # in this branch, tail_d = 0,  nburst_ub2gm  also can be 1
                    dst_gm = acture_memory * merchant
                    dst_gm = dst_gm + tail_c * psm_ub_out

                    nburst_ub2gm = 16 * tail_block_c
                    burstlen_ub2gm = int(tail_block_d * 16 * self.num_bit /
                                         DATA_MOVE_MIN_UNIT)
                    src_stride = 0
                    dst_stride = int((self.output_shape[-1] -
                                      tail_block_d * 16) *
                                     self.num_bit / DATA_MOVE_MIN_UNIT)

                    if dst_stride <= MAX_STRIDE:

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub_vconv[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               src_stride,
                                               dst_stride)
                    else:
                        src_y_ub_index = 0
                        src_y_ub_gap = tail_block_d * 16
                        dst_gm_gap = self.output_shape[-1]
                        with tik_instance.for_range(0, nburst_ub2gm) as i:
                            src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                            dst_gm = dst_gm + i * dst_gm_gap
                            tik_instance.data_move(
                                self.output_y_gm[dst_gm],
                                input_y_ub_vconv[src_y_ub_index],
                                0,
                                1,
                                burstlen_ub2gm,
                                0,
                                0)

                if tail_c != 0:
                    # deal [0,tail_c) data
                    with tik_instance.for_range(0, tail_c) as num_ub_loop:

                        merchant = sum_core + num_core_loop
                        src_x_index = psm * merchant + num_ub_loop * \
                                      16 * 16 * c_split

                        nburst_gm2ub = tail_block_d
                        burstlen_gm2ub = c_split * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (tail_block_c + (tail_c - 1) *
                                           c_split) * 16 * 16 * \
                                          self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        if srcstride_gm2ub <= MAX_STRIDE:

                            tik_instance.data_move(input_x_ub[0],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub)
                        else:
                            src_x_index_gap = self.input_shape[-3] * 16 * 16
                            dst_x_index_gap = (tail_block_c * 16 + 1) * 16
                            dst_x_index = 0
                            with tik_instance.for_range(0, nburst_gm2ub) as i:
                                src_x_index = src_x_index + i * src_x_index_gap
                                dst_x_index = dst_x_index + i * dst_x_index_gap
                                tik_instance.data_move(
                                    input_x_ub[dst_x_index],
                                    self.input_x_gm[src_x_index],
                                    0,
                                    1,
                                    burstlen_gm2ub,
                                    0,
                                    0)

                        vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                        vector_repeat_merchant_c = c_split * 16 // MAX_REPEAT
                        vector_repeat_remainder_c = c_split * 16 % MAX_REPEAT

                        src_ub_gap = MASK * (c_split * 16 + 1)
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * (c_split * 16 + 1) * \
                                         self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = tail_block_d * 16 * \
                                         self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = MAX_REPEAT

                        # order
                        self.reorder_s322f32_pattern_0(
                            tik_instance, vector_repeat_merchant_d,
                            vector_repeat_remainder_d, vector_repeat_merchant_c,
                            vector_repeat_remainder_c, self.mask, input_y_ub,
                            input_x_ub, dst_ub_gap, src_ub_gap,
                            repeat_times, dst_blk_stride, src_blk_stride,
                            dst_rep_stride, src_rep_stride, nburst_gm2ub)

                        all_rep = psm_ub_out // self.mask
                        vconv_repeat_merchant = all_rep // 255
                        vconv_repeat_remainder = all_rep % 255

                        src_ub_gap_vconv = 255 * self.mask
                        dst_ub_gap_vconv = 255 * self.mask
                        src_blk_stride_vconv = 8 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        src_rep_stride_vconv = 64 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        dst_blk_stride_vconv = 8 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        dst_rep_stride_vconv = 64 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        repeat_times_vconv = 255

                        self.f322s32(tik_instance, vconv_repeat_merchant,
                                     vconv_repeat_remainder,
                                     self.mask, input_y_ub_vconv, input_y_ub,
                                     dst_ub_gap_vconv,
                                     src_ub_gap_vconv, repeat_times_vconv,
                                     dst_blk_stride_vconv, src_blk_stride_vconv,
                                     dst_rep_stride_vconv, src_rep_stride_vconv)
                        # out
                        dst_gm = acture_memory * merchant
                        dst_gm = dst_gm + num_ub_loop * psm_ub_out

                        nburst_ub2gm = 16 * c_split
                        burstlen_ub2gm = int(tail_block_d * 16 *
                                             self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride = 0
                        dst_stride = int((self.output_shape[-1] -
                                          tail_block_d * 16) *
                                         self.num_bit / DATA_MOVE_MIN_UNIT)
                        if dst_stride <= MAX_STRIDE:

                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub_vconv[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   src_stride,
                                                   dst_stride)
                        else:
                            src_y_ub_index = 0
                            src_y_ub_gap = tail_block_d * 16
                            dst_gm_gap = self.output_shape[-1]
                            with tik_instance.for_range(0, nburst_ub2gm) as i:
                                src_y_ub_index = src_y_ub_index + \
                                                 i * src_y_ub_gap
                                dst_gm = dst_gm + i * dst_gm_gap
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm],
                                    input_y_ub_vconv[src_y_ub_index],
                                    0,
                                    1,
                                    burstlen_ub2gm,
                                    0,
                                    0)

    def nz2nd_normal_case0_int32_mini(self, tik_instance, params):
        """
        no padding
        [32,16,16,16,16,16]
        deal [16,16,16,16] == [D,C,16,16]
        [32,16,256,256]
        total_num = 32*16
        """
        # D-axis optimization
        # base shape: [D,1,16,16]
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        core_number = core_num
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        c_value = tiling_shape[1]
        d_value = tiling_shape[0]
        for c_split in range(c_value):
            c_split = c_split + 1
            psm_ub_in = d_value * 16 * (c_split * 16 + 1)
            psm_ub_out = d_value * 16 * c_split * 16
            if psm_ub_in > self.maximum_size_ub:
                c_split = c_split - 1
                psm_ub_in = d_value * 16 * (c_split * 16 + 1)
                psm_ub_out = d_value * 16 * c_split * 16
                break

        if c_split == 0:
            # psm exceed space of ub and [D,1,16,16] exceed space of ub
            # should return case1
            raise RuntimeError("not support c_split = 0")
        else:
            # psm exceed space of ub  while [D,1,16,16] not exceed space of ub
            tail_d = 0
            tail_block_d = d_value
            tail_c = c_value // c_split
            tail_block_c = c_value % c_split

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             [psm_ub_in, ], name="input_x_ub",
                                             scope=tik.scope_ubuf)

            input_y_ub = tik_instance.Tensor(self.dtype,
                                             [psm_ub_out, ], name="input_y_ub",
                                             scope=tik.scope_ubuf)

            input_v_ub = tik_instance.Tensor(self.dtype,
                                             [16, ], name="input_v_ub",
                                             scope=tik.scope_ubuf)

        with tik_instance.for_range(
                0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                self.set_vector_dup(tik_instance, 16, input_v_ub, 0)
                if tail_block_c != 0:
                    # deal tail_block firstly
                    merchant = sum_core + num_core_loop
                    src_x_index = psm * merchant + tail_c * 16 * 16 * c_split

                    nburst_gm2ub = tail_block_d
                    burstlen_gm2ub = tail_block_c * 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - tail_block_c) * \
                                      16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    if srcstride_gm2ub <= MAX_STRIDE:

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)
                    else:
                        src_x_index_gap = self.input_shape[-3] * 16 * 16
                        dst_x_index_gap = (tail_block_c * 16 + 1) * 16
                        dst_x_index = 0
                        with tik_instance.for_range(0, nburst_gm2ub) as i:
                            src_x_index = src_x_index + i * src_x_index_gap
                            dst_x_index = dst_x_index + i * dst_x_index_gap
                            tik_instance.data_move(input_x_ub[dst_x_index],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   1,
                                                   burstlen_gm2ub,
                                                   0,
                                                   0)

                    # -----------------Vconv int32 to fp32--------------------
                    vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                    vector_repeat_merchant_c = tail_block_c * 16 // MAX_REPEAT
                    vector_repeat_remainder_c = tail_block_c * 16 % MAX_REPEAT

                    src_ub_gap = MASK * (tail_block_c * 16 + 1)
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * (tail_block_c * 16 + 1) * \
                                     self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = tail_block_d * 16 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    repeat_times = MAX_REPEAT

                    self.reorder_vadd_pattern_0_mini(
                        tik_instance, vector_repeat_merchant_d,
                        vector_repeat_remainder_d,
                        vector_repeat_merchant_c, vector_repeat_remainder_c,
                        self.mask, input_y_ub, input_x_ub,
                        input_v_ub, dst_ub_gap, src_ub_gap,
                        repeat_times, dst_blk_stride, src_blk_stride,
                        dst_rep_stride, src_rep_stride, nburst_gm2ub)

                    # out
                    # in this branch, tail_d = 0,  nburst_ub2gm  also can be 1
                    dst_gm = acture_memory * merchant
                    dst_gm = dst_gm + tail_c * psm_ub_out

                    nburst_ub2gm = 16 * tail_block_c
                    burstlen_ub2gm = int(tail_block_d * 16 * self.num_bit /
                                         DATA_MOVE_MIN_UNIT)
                    src_stride = 0
                    dst_stride = int((self.output_shape[-1] -
                                      tail_block_d * 16) *
                                     self.num_bit / DATA_MOVE_MIN_UNIT)

                    if dst_stride <= MAX_STRIDE:

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               src_stride,
                                               dst_stride)
                    else:
                        src_y_ub_index = 0
                        src_y_ub_gap = tail_block_d * 16
                        dst_gm_gap = self.output_shape[-1]
                        with tik_instance.for_range(0, nburst_ub2gm) as i:
                            src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                            dst_gm = dst_gm + i * dst_gm_gap
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[src_y_ub_index],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

                if tail_c != 0:
                    # deal [0,tail_c) data
                    with tik_instance.for_range(0, tail_c) as num_ub_loop:

                        merchant = sum_core + num_core_loop
                        src_x_index = psm * merchant + num_ub_loop * \
                                      16 * 16 * c_split

                        nburst_gm2ub = tail_block_d
                        burstlen_gm2ub = c_split * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (tail_block_c +
                                           (tail_c - 1) * c_split) * \
                                          16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        if srcstride_gm2ub <= MAX_STRIDE:

                            tik_instance.data_move(input_x_ub[0],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub)
                        else:
                            src_x_index_gap = self.input_shape[-3] * 16 * 16
                            dst_x_index_gap = (tail_block_c * 16 + 1) * 16
                            dst_x_index = 0
                            with tik_instance.for_range(0, nburst_gm2ub) as i:
                                src_x_index = src_x_index + i * src_x_index_gap
                                dst_x_index = dst_x_index + i * dst_x_index_gap
                                tik_instance.data_move(
                                    input_x_ub[dst_x_index],
                                    self.input_x_gm[src_x_index],
                                    0,
                                    1,
                                    burstlen_gm2ub,
                                    0,
                                    0)

                        vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                        vector_repeat_merchant_c = c_split * 16 // MAX_REPEAT
                        vector_repeat_remainder_c = c_split * 16 % MAX_REPEAT

                        src_ub_gap = MASK * (c_split * 16 + 1)
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * (c_split * 16 + 1) * \
                                         self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = tail_block_d * 16 * self.num_bit // \
                                         DATA_MOVE_MIN_UNIT
                        repeat_times = MAX_REPEAT

                        # order
                        self.reorder_vadd_pattern_0_mini(
                            tik_instance, vector_repeat_merchant_d,
                            vector_repeat_remainder_d,
                            vector_repeat_merchant_c, vector_repeat_remainder_c,
                            self.mask, input_y_ub, input_x_ub,
                            input_v_ub, dst_ub_gap, src_ub_gap,
                            repeat_times, dst_blk_stride, src_blk_stride,
                            dst_rep_stride, src_rep_stride, nburst_gm2ub)

                        # out
                        dst_gm = acture_memory * merchant
                        dst_gm = dst_gm + num_ub_loop * psm_ub_out

                        nburst_ub2gm = 16 * c_split
                        burstlen_ub2gm = int(tail_block_d * 16 *
                                             self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride = 0
                        dst_stride = int((self.output_shape[-1] -
                                          tail_block_d * 16) *
                                         self.num_bit / DATA_MOVE_MIN_UNIT)
                        if dst_stride <= MAX_STRIDE:

                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   src_stride,
                                                   dst_stride)
                        else:
                            src_y_ub_index = 0
                            src_y_ub_gap = tail_block_d * 16
                            dst_gm_gap = self.output_shape[-1]
                            with tik_instance.for_range(0, nburst_ub2gm) as i:
                                src_y_ub_index = src_y_ub_index + \
                                                 i * src_y_ub_gap
                                dst_gm = dst_gm + i * dst_gm_gap
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm],
                                    input_y_ub[src_y_ub_index],
                                    0,
                                    1,
                                    burstlen_ub2gm,
                                    0,
                                    0)

    def nz2nd_normal_case1(self, tik_instance, params):
        """
        no padding
        [5,32,32,16,16]
        total_num = 160
        output 5 512 512
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num
        # because of db,self.maximum_size_ub // 2
        num_gm2ub = self.maximum_size_ub // 2 // (17 * 16)
        num_gm = tiling_shape[-3]
        tail = num_gm // num_gm2ub
        tail_block = num_gm % num_gm2ub

        with tik_instance.for_range(0, core_number,
                                    block_num=core_number) as num_core:
            # calculate the loop number on each core
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            # calculate serial data until tail_block
            def case1_main_tail(loop, x_ub, y_ub):
                # deal tail data
                num_core_loop = loop
                input_x_ub = x_ub
                input_y_ub = y_ub
                merchant = (sum_core + num_core_loop // tail) // \
                           self.input_shape[-3]
                remainder = (sum_core + num_core_loop // tail) % \
                            self.input_shape[-3]

                src_x_index = psm * self.input_shape[-3] * \
                              merchant + remainder * 16 * 16
                src_x_index_temp = src_x_index
                src_x_index = src_x_index_temp + num_core_loop % tail * \
                              num_gm2ub * 16 * 16 * self.input_shape[-3]

                nburst_gm2ub = num_gm2ub
                burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                  16 * 16 * self.num_bit // 32
                dststride_gm2ub = 16 * self.num_bit // 32

                if srcstride_gm2ub <= MAX_STRIDE:

                    tik_instance.data_move(input_x_ub[0],
                                           self.input_x_gm[src_x_index],
                                           0,
                                           nburst_gm2ub,
                                           burstlen_gm2ub,
                                           srcstride_gm2ub,
                                           dststride_gm2ub)
                else:
                    src_x_index_gap = self.input_shape[-3] * 16 * 16
                    dst_x_index_gap = 16 * 17
                    dst_x_index = 0
                    with tik_instance.for_range(0, nburst_gm2ub) as i:
                        src_x_index = src_x_index + i * src_x_index_gap
                        dst_x_index = dst_x_index + i * dst_x_index_gap
                        tik_instance.data_move(
                            input_x_ub[dst_x_index],
                            self.input_x_gm[src_x_index],
                            0,
                            1,
                            burstlen_gm2ub,
                            0,
                            0)

                # tail_block * 16
                # vector maximum compute 128
                vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                src_ub_gap = MASK * 17
                dst_ub_gap = MASK
                src_blk_stride = 16 * 17 * self.num_bit // \
                                 DATA_MOVE_MIN_UNIT
                src_rep_stride = 16 * self.num_bit // \
                                 DATA_MOVE_MIN_UNIT
                dst_blk_stride = 16 * self.num_bit // \
                                 DATA_MOVE_MIN_UNIT
                dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // \
                                 DATA_MOVE_MIN_UNIT
                repeat_times = 16

                # order
                with tik_instance.for_range(
                        0, vector_repeat_merchant) as i:

                    tik_instance.vadds(self.mask,
                                       input_y_ub[i * dst_ub_gap],
                                       input_x_ub[i * src_ub_gap],
                                       0,
                                       repeat_times,
                                       dst_blk_stride,
                                       src_blk_stride,
                                       dst_rep_stride,
                                       src_rep_stride)
                    # fp32 need twice
                    if self.mask == 64:
                        tik_instance.vadds(
                            self.mask,
                            input_y_ub[i * dst_ub_gap + 8],
                            input_x_ub[i * src_ub_gap + 8],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)

                if vector_repeat_remainder != 0:
                    vadds_mask = int(vector_repeat_remainder /
                                     MASK * self.mask)
                    tik_instance.vadds(
                        vadds_mask,
                        input_y_ub[vector_repeat_merchant * dst_ub_gap],
                        input_x_ub[vector_repeat_merchant * src_ub_gap],
                        0,
                        repeat_times,
                        dst_blk_stride,
                        src_blk_stride,
                        dst_rep_stride,
                        src_rep_stride)

                    if self.mask == 64:
                        dst_y_ub = vector_repeat_merchant * \
                                   dst_ub_gap + 8
                        src_x_ub = vector_repeat_merchant * \
                                   src_ub_gap + 8
                        tik_instance.vadds(vadds_mask,
                                           input_y_ub[dst_y_ub],
                                           input_x_ub[src_x_ub],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)

                # out
                dst_gm = acture_memory * merchant + \
                         remainder * self.output_shape[-1] * 16
                dst_gm = dst_gm + num_core_loop % tail * num_gm2ub * 16
                nburst_ub2gm = 16
                burstlen_ub2gm = int(num_gm2ub * 16 *
                                     self.num_bit / DATA_MOVE_MIN_UNIT)
                src_stride = 0
                dst_stride = int((self.output_shape[-1] -
                                  num_gm2ub * 16) *
                                 self.num_bit / DATA_MOVE_MIN_UNIT)

                if dst_stride <= MAX_STRIDE:
                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                           input_y_ub[0],
                                           0,
                                           nburst_ub2gm,
                                           burstlen_ub2gm,
                                           src_stride,
                                           dst_stride)
                else:
                    src_y_ub_index = 0
                    src_y_ub_gap = num_gm2ub * 16
                    dst_gm_gap = self.output_shape[-1]
                    with tik_instance.for_range(0, nburst_ub2gm) as i:
                        src_y_ub_index = src_y_ub_index + \
                                         i * src_y_ub_gap
                        dst_gm = dst_gm + i * dst_gm_gap
                        tik_instance.data_move(
                            self.output_y_gm[dst_gm],
                            input_y_ub[src_y_ub_index],
                            0,
                            1,
                            burstlen_ub2gm,
                            0,
                            0)

            # calculate tail_block data
            def case1_main_tail_block(loop, x_ub, y_ub):
                # deal tail_block firstly
                num_core_loop = loop
                input_x_ub = x_ub
                input_y_ub = y_ub
                merchant = (sum_core + num_core_loop) // \
                           self.input_shape[-3]
                remainder = (sum_core + num_core_loop) % \
                            self.input_shape[-3]
                src_x_index = psm * self.input_shape[-3] * \
                              merchant + remainder * 16 * 16
                src_x_index_temp = src_x_index
                src_x_index = src_x_index_temp + tail * \
                              num_gm2ub * 16 * 16 * self.input_shape[-3]

                nburst_gm2ub = tail_block
                burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                  16 * 16 * self.num_bit // 32
                dststride_gm2ub = 16 * self.num_bit // 32

                if srcstride_gm2ub <= MAX_STRIDE:

                    tik_instance.data_move(input_x_ub[0],
                                           self.input_x_gm[src_x_index],
                                           0,
                                           nburst_gm2ub,
                                           burstlen_gm2ub,
                                           srcstride_gm2ub,
                                           dststride_gm2ub)
                else:
                    src_x_index_gap = self.input_shape[-3] * 16 * 16
                    dst_x_index_gap = 16 * 17
                    dst_x_index = 0
                    with tik_instance.for_range(0, nburst_gm2ub) as i:
                        src_x_index = src_x_index + i * src_x_index_gap
                        dst_x_index = dst_x_index + i * dst_x_index_gap
                        tik_instance.data_move(input_x_ub[dst_x_index],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               1,
                                               burstlen_gm2ub,
                                               0,
                                               0)

                # tail_block * 16
                # vector maxcompute 128
                vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                src_ub_gap = MASK * 17
                dst_ub_gap = MASK
                src_blk_stride = 16 * 17 * self.num_bit // \
                                 DATA_MOVE_MIN_UNIT
                src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                dst_rep_stride = tail_block * 16 * self.num_bit // \
                                 DATA_MOVE_MIN_UNIT
                repeat_times = 16

                # order
                with tik_instance.for_range(0, vector_repeat_merchant) as i:

                    tik_instance.vadds(self.mask,
                                       input_y_ub[i * dst_ub_gap],
                                       input_x_ub[i * src_ub_gap],
                                       0,
                                       repeat_times,
                                       dst_blk_stride,
                                       src_blk_stride,
                                       dst_rep_stride,
                                       src_rep_stride)
                    # fp32 need twice
                    if self.mask == 64:
                        tik_instance.vadds(self.mask,
                                           input_y_ub[i * dst_ub_gap + 8],
                                           input_x_ub[i * src_ub_gap + 8],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)

                if vector_repeat_remainder != 0:
                    vadds_mask = int(vector_repeat_remainder /
                                     MASK * self.mask)
                    tik_instance.vadds(vadds_mask,
                                       input_y_ub[vector_repeat_merchant *
                                                  dst_ub_gap],
                                       input_x_ub[vector_repeat_merchant *
                                                  src_ub_gap],
                                       0,
                                       repeat_times,
                                       dst_blk_stride,
                                       src_blk_stride,
                                       dst_rep_stride,
                                       src_rep_stride)

                    if self.mask == 64:
                        tik_instance.vadds(
                            vadds_mask,
                            input_y_ub[vector_repeat_merchant *
                                       dst_ub_gap + 8],
                            input_x_ub[vector_repeat_merchant *
                                       src_ub_gap + 8],
                            0,
                            repeat_times,
                            dst_blk_stride,
                            src_blk_stride,
                            dst_rep_stride,
                            src_rep_stride)

                # out
                dst_gm = acture_memory * merchant + \
                         remainder * self.output_shape[-1] * 16
                dst_gm = dst_gm + tail * num_gm2ub * 16

                nburst_ub2gm = 16
                burstlen_ub2gm = int(tail_block * 16 *
                                     self.num_bit / DATA_MOVE_MIN_UNIT)
                src_stride = 0
                dst_stride = int((self.output_shape[-1] - tail_block * 16) *
                                 self.num_bit / DATA_MOVE_MIN_UNIT)
                if dst_stride <= MAX_STRIDE:

                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                           input_y_ub[0],
                                           0,
                                           nburst_ub2gm,
                                           burstlen_ub2gm,
                                           src_stride,
                                           dst_stride)
                else:
                    src_y_ub_index = 0
                    src_y_ub_gap = tail_block * 16
                    dst_gm_gap = self.output_shape[-1]
                    with tik_instance.for_range(0, nburst_ub2gm) as i:
                        src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                        dst_gm = dst_gm + i * dst_gm_gap
                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[src_y_ub_index],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

            # set ub tensor in case1
            def set_ub_tensor():
                if num_gm < num_gm2ub:
                    input_x_ub = tik_instance.Tensor(self.dtype,
                                                     tiling_shape_gm2ub, name="input_x_ub",
                                                     scope=tik.scope_ubuf)

                    input_y_ub = tik_instance.Tensor(self.dtype,
                                                     tiling_shape, name="input_y_ub",
                                                     scope=tik.scope_ubuf)
                else:
                    input_x_ub = tik_instance.Tensor(self.dtype,
                                                     [num_gm2ub, 17, 16],
                                                     name="input_x_ub",
                                                     scope=tik.scope_ubuf)

                    input_y_ub = tik_instance.Tensor(self.dtype,
                                                     [num_gm2ub, 16, 16],
                                                     name="input_y_ub",
                                                     scope=tik.scope_ubuf)
                return input_x_ub, input_y_ub

            # check whether the conditions meet double_buffer
            # core_loop must not be zero
            # tail mabe zero
            if tail != 0:
                thread = \
                    _check_db(total_core_loop_num * tail, core_number)
                loop_num = core_loop * tail
                with tik_instance.new_stmt_scope():
                    with tik_instance.for_range(0, loop_num,
                                                thread_num=thread) as loop:
                        x_ub, y_ub = set_ub_tensor()
                        case1_main_tail(loop, x_ub, y_ub)

            if tail_block != 0:
                thread = \
                    _check_db(total_core_loop_num, core_number)
                loop_num = core_loop
                with tik_instance.for_range(0, loop_num,
                                            thread_num=thread) as num_loop:
                    x_ub, y_ub = set_ub_tensor()
                    case1_main_tail_block(num_loop, x_ub, y_ub)

    def nz2nd_normal_case1_int32(self, tik_instance, params):
        """
        no padding
        [5,32,32,16,16]
        total_num = 160
        output 5 512 512
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        tail = 0
        tail_block = 0
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num

        num_gm2ub = self.maximum_size_ub // (17 * 16)
        num_gm = tiling_shape[-3]

        if num_gm < num_gm2ub:
            input_x_ub = tik_instance.Tensor(self.dtype,
                                             tiling_shape_gm2ub,
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)
            input_y_ub = tik_instance.Tensor("float32",
                                             tiling_shape, name="input_y_ub",
                                             scope=tik.scope_ubuf)
            input_y_ub_vconv = tik_instance.Tensor(self.dtype,
                                                   tiling_shape,
                                                   name="input_y_ub",
                                                   scope=tik.scope_ubuf)
            psm_vconv = self.calc_element(tiling_shape)
        else:
            tail = num_gm // num_gm2ub
            tail_block = num_gm % num_gm2ub

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             [num_gm2ub, 17, 16],
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)
            input_y_ub = tik_instance.Tensor("float32",
                                             [num_gm2ub, 16, 16],
                                             name="input_y_ub",
                                             scope=tik.scope_ubuf)
            input_y_ub_vconv = tik_instance.Tensor(self.dtype,
                                                   [num_gm2ub, 16, 16],
                                                   name="input_y_ub",
                                                   scope=tik.scope_ubuf)
            psm_vconv = self.calc_element([num_gm2ub, 16, 16])

        with tik_instance.for_range(
                0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                if tail == 0:
                    merchant = (sum_core + num_core_loop) // \
                               self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % \
                                self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * \
                                  merchant + remainder * 16 * 16

                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                      16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub,
                                               self.input_x_gm,
                                               num_gm, burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    vector_repeat_merchant = num_gm * 16 // MASK
                    vector_repeat_remainder = num_gm * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm * 16 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                         vector_repeat_remainder,
                                         self.mask, input_y_ub, input_x_ub,
                                         dst_ub_gap, src_ub_gap,
                                         repeat_times, dst_blk_stride,
                                         src_blk_stride,
                                         dst_rep_stride, src_rep_stride)

                    all_rep = psm_vconv // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant,
                                 vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub,
                                 dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # out
                    dst_gm = acture_memory * merchant + \
                             remainder * self.output_shape[-1] * 16
                    burstlen_ub2gm = int(psm * self.num_bit /
                                         DATA_MOVE_MIN_UNIT)

                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                           input_y_ub_vconv[0],
                                           0,
                                           1,
                                           burstlen_ub2gm,
                                           0,
                                           0)

                elif (tail != 0) and (tail_block != 0):

                    # deal tail_block firstly
                    merchant = (sum_core + num_core_loop) // \
                               self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % \
                                self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * \
                                  merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + tail * \
                                  num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = tail_block
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                      16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub,
                                               self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # ----------------Vconv int32 to fp32---------------------
                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                         vector_repeat_remainder,
                                         self.mask, input_y_ub, input_x_ub,
                                         dst_ub_gap, src_ub_gap,
                                         repeat_times, dst_blk_stride,
                                         src_blk_stride,
                                         dst_rep_stride, src_rep_stride)

                    all_rep = tail_block * 16 * 16 // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant,
                                 vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub,
                                 dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # out
                    dst_gm = acture_memory * merchant + remainder * \
                             self.output_shape[-1] * 16
                    dst_gm = dst_gm + tail * num_gm2ub * 16

                    nburst_ub2gm = 16
                    burstlen_ub2gm = int(tail_block * 16 *
                                         self.num_bit / DATA_MOVE_MIN_UNIT)
                    src_stride = 0
                    dst_stride = int((self.output_shape[-1] - tail_block * 16) *
                                     self.num_bit / DATA_MOVE_MIN_UNIT)
                    if dst_stride <= MAX_STRIDE:

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub_vconv[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               src_stride,
                                               dst_stride)
                    else:
                        src_y_ub_index = 0
                        src_y_ub_gap = tail_block * 16
                        dst_gm_gap = self.output_shape[-1]
                        with tik_instance.for_range(0, nburst_ub2gm) as i:
                            src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                            dst_gm = dst_gm + i * dst_gm_gap
                            tik_instance.data_move(
                                self.output_y_gm[dst_gm],
                                input_y_ub_vconv[src_y_ub_index],
                                0,
                                1,
                                burstlen_ub2gm,
                                0,
                                0)

                    # deal tail data
                    with tik_instance.for_range(0, tail) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * \
                                      merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                          16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance,
                                                   input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub,
                                                   src_x_index, 0)

                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // \
                                         DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // \
                                         DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        self.reorder_s322f32(tik_instance,
                                             vector_repeat_merchant,
                                             vector_repeat_remainder,
                                             self.mask, input_y_ub,
                                             input_x_ub, dst_ub_gap, src_ub_gap,
                                             repeat_times, dst_blk_stride,
                                             src_blk_stride,
                                             dst_rep_stride, src_rep_stride)

                        all_rep = psm_vconv // self.mask
                        vconv_repeat_merchant = all_rep // 255
                        vconv_repeat_remainder = all_rep % 255

                        src_ub_gap_vconv = 255 * self.mask
                        dst_ub_gap_vconv = 255 * self.mask
                        src_blk_stride_vconv = 8 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        src_rep_stride_vconv = 64 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        dst_blk_stride_vconv = 8 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        dst_rep_stride_vconv = 64 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        repeat_times_vconv = 255

                        self.f322s32(tik_instance, vconv_repeat_merchant,
                                     vconv_repeat_remainder,
                                     self.mask, input_y_ub_vconv, input_y_ub,
                                     dst_ub_gap_vconv,
                                     src_ub_gap_vconv, repeat_times_vconv,
                                     dst_blk_stride_vconv, src_blk_stride_vconv,
                                     dst_rep_stride_vconv, src_rep_stride_vconv)

                        # out
                        dst_gm = acture_memory * merchant + \
                                 remainder * self.output_shape[-1] * 16
                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        nburst_ub2gm = 16
                        burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit /
                                             DATA_MOVE_MIN_UNIT)
                        src_stride = 0
                        dst_stride = int((self.output_shape[-1] -
                                          num_gm2ub * 16) *
                                         self.num_bit / DATA_MOVE_MIN_UNIT)

                        if dst_stride <= MAX_STRIDE:
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub_vconv[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   src_stride,
                                                   dst_stride)
                        else:
                            src_y_ub_index = 0
                            src_y_ub_gap = num_gm2ub * 16
                            dst_gm_gap = self.output_shape[-1]
                            with tik_instance.for_range(0, nburst_ub2gm) as i:
                                src_y_ub_index = src_y_ub_index + i * \
                                                 src_y_ub_gap
                                dst_gm = dst_gm + i * dst_gm_gap
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm],
                                    input_y_ub_vconv[src_y_ub_index],
                                    0,
                                    1,
                                    burstlen_ub2gm,
                                    0,
                                    0)

                else:
                    # deal (tail-1) block firstly
                    merchant = (sum_core + num_core_loop) // \
                               self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % \
                                self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * \
                                  merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + (tail - 1) * \
                                  num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = num_gm2ub
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                      16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub,
                                               self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                         vector_repeat_remainder,
                                         self.mask, input_y_ub, input_x_ub,
                                         dst_ub_gap, src_ub_gap,
                                         repeat_times, dst_blk_stride,
                                         src_blk_stride,
                                         dst_rep_stride, src_rep_stride)

                    all_rep = psm_vconv // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // \
                                           DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant,
                                 vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub,
                                 dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # out
                    dst_gm = acture_memory * merchant + remainder * \
                             self.output_shape[-1] * 16
                    dst_gm = dst_gm + (tail - 1) * num_gm2ub * 16
                    nburst_ub2gm = 16
                    burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit /
                                         DATA_MOVE_MIN_UNIT)
                    src_stride = 0
                    dst_stride = int((self.output_shape[-1] - num_gm2ub * 16) *
                                     self.num_bit / DATA_MOVE_MIN_UNIT)

                    if dst_stride <= MAX_STRIDE:

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub_vconv[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               src_stride,
                                               dst_stride)

                    else:
                        src_y_ub_index = 0
                        src_y_ub_gap = num_gm2ub * 16
                        dst_gm_gap = self.output_shape[-1]
                        with tik_instance.for_range(0, nburst_ub2gm) as i:
                            src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                            dst_gm = dst_gm + i * dst_gm_gap
                            tik_instance.data_move(
                                self.output_y_gm[dst_gm],
                                input_y_ub_vconv[src_y_ub_index],
                                0,
                                1,
                                burstlen_ub2gm,
                                0,
                                0)

                    # deal tail-1 data
                    with tik_instance.for_range(0, tail - 1) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * \
                                      merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                          16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub,
                                                   self.input_x_gm,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub,
                                                   src_x_index, 0)

                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // \
                                         DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // \
                                         DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        self.reorder_s322f32(tik_instance,
                                             vector_repeat_merchant,
                                             vector_repeat_remainder,
                                             self.mask, input_y_ub,
                                             input_x_ub, dst_ub_gap, src_ub_gap,
                                             repeat_times, dst_blk_stride,
                                             src_blk_stride,
                                             dst_rep_stride, src_rep_stride)

                        all_rep = psm_vconv // self.mask
                        vconv_repeat_merchant = all_rep // 255
                        vconv_repeat_remainder = all_rep % 255

                        src_ub_gap_vconv = 255 * self.mask
                        dst_ub_gap_vconv = 255 * self.mask
                        src_blk_stride_vconv = 8 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        src_rep_stride_vconv = 64 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        dst_blk_stride_vconv = 8 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        dst_rep_stride_vconv = 64 * self.num_bit // \
                                               DATA_MOVE_MIN_UNIT
                        repeat_times_vconv = 255

                        self.f322s32(tik_instance, vconv_repeat_merchant,
                                     vconv_repeat_remainder,
                                     self.mask, input_y_ub_vconv,
                                     input_y_ub, dst_ub_gap_vconv,
                                     src_ub_gap_vconv, repeat_times_vconv,
                                     dst_blk_stride_vconv, src_blk_stride_vconv,
                                     dst_rep_stride_vconv, src_rep_stride_vconv)

                        # out
                        dst_gm = acture_memory * merchant + \
                                 remainder * self.output_shape[-1] * 16
                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        nburst_ub2gm = 16
                        burstlen_ub2gm = int(num_gm2ub * 16 *
                                             self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride = 0
                        dst_stride = int((self.output_shape[-1] -
                                          num_gm2ub * 16) *
                                         self.num_bit / DATA_MOVE_MIN_UNIT)

                        if dst_stride <= MAX_STRIDE:
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub_vconv[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   src_stride,
                                                   dst_stride)
                        else:
                            src_y_ub_index = 0
                            src_y_ub_gap = num_gm2ub * 16
                            dst_gm_gap = self.output_shape[-1]
                            with tik_instance.for_range(0, nburst_ub2gm) as i:
                                src_y_ub_index = src_y_ub_index + \
                                                 i * src_y_ub_gap
                                dst_gm = dst_gm + i * dst_gm_gap
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm],
                                    input_y_ub_vconv[src_y_ub_index],
                                    0,
                                    1,
                                    burstlen_ub2gm,
                                    0,
                                    0)

    def nz2nd_normal_case1_int32_mini(self, tik_instance, params):
        """
        no padding
        [5,32,32,16,16]
        total_num = 160
        output 5 512 512
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        tail = 0
        tail_block = 0
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num

        num_gm2ub = self.maximum_size_ub // (17 * 16)
        num_gm = tiling_shape[-3]

        if num_gm < num_gm2ub:
            input_x_ub = tik_instance.Tensor(self.dtype,
                                             tiling_shape_gm2ub,
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)
            input_y_ub = tik_instance.Tensor(self.dtype,
                                             tiling_shape, name="input_y_ub",
                                             scope=tik.scope_ubuf)
        else:
            tail = num_gm // num_gm2ub
            tail_block = num_gm % num_gm2ub

            input_x_ub = tik_instance.Tensor(self.dtype,
                                             [num_gm2ub, 17, 16],
                                             name="input_x_ub",
                                             scope=tik.scope_ubuf)
            input_y_ub = tik_instance.Tensor(self.dtype,
                                             [num_gm2ub, 16, 16],
                                             name="input_y_ub",
                                             scope=tik.scope_ubuf)

        input_v_ub = tik_instance.Tensor(self.dtype, [16, ], name="input_v_ub",
                                         scope=tik.scope_ubuf)

        with tik_instance.for_range(
                0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                self.set_vector_dup(tik_instance, 16, input_v_ub, 0)

                if tail == 0:
                    merchant = (sum_core + num_core_loop) // \
                               self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % \
                                self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * \
                                  merchant + remainder * 16 * 16

                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                      16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub,
                                               self.input_x_gm,
                                               num_gm, burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # ----------------Vconv int32 to fp32---------------------
                    vector_repeat_merchant = num_gm * 16 // MASK
                    vector_repeat_remainder = num_gm * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm * 16 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_vadd_mini(
                        tik_instance, vector_repeat_merchant,
                        vector_repeat_remainder,
                        self.mask, input_y_ub, input_x_ub,
                        input_v_ub, dst_ub_gap, src_ub_gap,
                        repeat_times, dst_blk_stride, src_blk_stride,
                        dst_rep_stride, src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + \
                             remainder * self.output_shape[-1] * 16
                    burstlen_ub2gm = int(psm * self.num_bit /
                                         DATA_MOVE_MIN_UNIT)

                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                           input_y_ub[0],
                                           0,
                                           1,
                                           burstlen_ub2gm,
                                           0,
                                           0)

                elif (tail != 0) and (tail_block != 0):

                    # deal tail_block firstly
                    merchant = (sum_core + num_core_loop) // \
                               self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % \
                                self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * \
                                  merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + tail * \
                                  num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = tail_block
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                      16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub,
                                               self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # ----------------Vconv int32 to fp32---------------------
                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // \
                                     DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_vadd_mini(tik_instance,
                                           vector_repeat_merchant,
                                           vector_repeat_remainder,
                                           self.mask, input_y_ub,
                                           input_x_ub, input_v_ub,
                                           dst_ub_gap, src_ub_gap,
                                           repeat_times, dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride, src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + \
                             remainder * self.output_shape[-1] * 16
                    dst_gm = dst_gm + tail * num_gm2ub * 16

                    nburst_ub2gm = 16
                    burstlen_ub2gm = int(tail_block * 16 *
                                         self.num_bit / DATA_MOVE_MIN_UNIT)
                    src_stride = 0
                    dst_stride = int((self.output_shape[-1] - tail_block * 16) *
                                     self.num_bit / DATA_MOVE_MIN_UNIT)
                    if dst_stride <= MAX_STRIDE:

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               src_stride,
                                               dst_stride)
                    else:
                        src_y_ub_index = 0
                        src_y_ub_gap = tail_block * 16
                        dst_gm_gap = self.output_shape[-1]
                        with tik_instance.for_range(0, nburst_ub2gm) as i:
                            src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                            dst_gm = dst_gm + i * dst_gm_gap
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[src_y_ub_index],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

                    # deal tail data
                    with tik_instance.for_range(0, tail) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * \
                                      merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * \
                                          16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub,
                                                   self.input_x_gm,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub,
                                                   src_x_index, 0)

                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // \
                                         DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // \
                                         DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        self.reorder_vadd_mini(tik_instance,
                                               vector_repeat_merchant,
                                               vector_repeat_remainder,
                                               self.mask, input_y_ub,
                                               input_x_ub, input_v_ub,
                                               dst_ub_gap, src_ub_gap,
                                               repeat_times, dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride, src_rep_stride)

                        # out
                        dst_gm = acture_memory * merchant + remainder * \
                                 self.output_shape[-1] * 16
                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        nburst_ub2gm = 16
                        burstlen_ub2gm = int(num_gm2ub * 16 *
                                             self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride = 0
                        dst_stride = int((self.output_shape[-1] -
                                          num_gm2ub * 16) *
                                         self.num_bit / DATA_MOVE_MIN_UNIT)

                        if dst_stride <= MAX_STRIDE:
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   src_stride,
                                                   dst_stride)
                        else:
                            src_y_ub_index = 0
                            src_y_ub_gap = num_gm2ub * 16
                            dst_gm_gap = self.output_shape[-1]
                            with tik_instance.for_range(0, nburst_ub2gm) as i:
                                src_y_ub_index = src_y_ub_index + \
                                                 i * src_y_ub_gap
                                dst_gm = dst_gm + i * dst_gm_gap
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm],
                                    input_y_ub[src_y_ub_index],
                                    0,
                                    1,
                                    burstlen_ub2gm,
                                    0,
                                    0)

                else:
                    # deal (tail-1) block firstly
                    merchant = (sum_core + num_core_loop) // \
                               self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % \
                                self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * \
                                  merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + (tail - 1) * \
                                  num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = num_gm2ub
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_vadd_mini(tik_instance, vector_repeat_merchant,
                                           vector_repeat_remainder,
                                           self.mask, input_y_ub, input_x_ub, input_v_ub,
                                           dst_ub_gap, src_ub_gap,
                                           repeat_times, dst_blk_stride, src_blk_stride,
                                           dst_rep_stride, src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm = dst_gm + (tail - 1) * num_gm2ub * 16
                    nburst_ub2gm = 16
                    burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    src_stride = 0
                    dst_stride = int((self.output_shape[-1] - num_gm2ub * 16) *
                                     self.num_bit / DATA_MOVE_MIN_UNIT)

                    if dst_stride <= MAX_STRIDE:

                        tik_instance.data_move(self.output_y_gm[dst_gm],
                                               input_y_ub[0],
                                               0,
                                               nburst_ub2gm,
                                               burstlen_ub2gm,
                                               src_stride,
                                               dst_stride)

                    else:
                        src_y_ub_index = 0
                        src_y_ub_gap = num_gm2ub * 16
                        dst_gm_gap = self.output_shape[-1]
                        with tik_instance.for_range(0, nburst_ub2gm) as i:
                            src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                            dst_gm = dst_gm + i * dst_gm_gap
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[src_y_ub_index],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

                    # deal tail-1 data
                    with tik_instance.for_range(0, tail - 1) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * \
                                      merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                                   dststride_gm2ub, src_x_index, 0)

                        # -------------------Vconv fp32 to int32---------------------
                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        self.reorder_vadd_mini(tik_instance, vector_repeat_merchant,
                                               vector_repeat_remainder,
                                               self.mask, input_y_ub, input_x_ub, input_v_ub,
                                               dst_ub_gap, src_ub_gap,
                                               repeat_times, dst_blk_stride, src_blk_stride,
                                               dst_rep_stride, src_rep_stride)

                        # out
                        dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        nburst_ub2gm = 16
                        burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride = 0
                        dst_stride = int((self.output_shape[-1] - num_gm2ub * 16) *
                                         self.num_bit / DATA_MOVE_MIN_UNIT)

                        if dst_stride <= MAX_STRIDE:
                            tik_instance.data_move(self.output_y_gm[dst_gm],
                                                   input_y_ub[0],
                                                   0,
                                                   nburst_ub2gm,
                                                   burstlen_ub2gm,
                                                   src_stride,
                                                   dst_stride)
                        else:
                            src_y_ub_index = 0
                            src_y_ub_gap = num_gm2ub * 16
                            dst_gm_gap = self.output_shape[-1]
                            with tik_instance.for_range(0, nburst_ub2gm) as i:
                                src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                                dst_gm = dst_gm + i * dst_gm_gap
                                tik_instance.data_move(self.output_y_gm[dst_gm],
                                                       input_y_ub[src_y_ub_index],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

    def nz2nd_special_case0(self, tik_instance, params):
        """
        padding
        [5,1,1,16,16]
        total_num = 5
        output = 5, 14, 14
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        tail = 0
        tail_block = 0
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num

        num_gm2ub = self.maximum_size_ub // (17 * 16)
        num_gm = tiling_shape[-3]
        use_time = tik_instance.Scalar(dtype="int64")
        use_time_merchant = tik_instance.Scalar(dtype="int64")
        use_time_remainder = tik_instance.Scalar(dtype="int64")
        utmerchant = tik_instance.Scalar(dtype="int64")

        input_x_ub, input_y_ub, input_z_ub, tail, \
        tail_block = self.set_ub_tensor(tik_instance, num_gm, num_gm2ub,
                                        tiling_shape, tiling_shape_gm2ub, tail, tail_block)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                if tail == 0:
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16

                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32
                    if srcstride_gm2ub <= MAX_STRIDE:

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               num_gm,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)

                    else:
                        src_x_index_gap = self.input_shape[-3] * 16 * 16
                        dst_x_index_gap = 16 * 17
                        dst_x_index = 0
                        with tik_instance.for_range(0, num_gm) as i:
                            src_x_index = src_x_index + i * src_x_index_gap
                            dst_x_index = dst_x_index + i * dst_x_index_gap
                            tik_instance.data_move(input_x_ub[dst_x_index],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   1,
                                                   burstlen_gm2ub,
                                                   0,
                                                   0)

                    # num_gm * 16
                    # vector maximum compute 128
                    vector_repeat_merchant = num_gm * 16 // MASK
                    vector_repeat_remainder = num_gm * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    # order
                    with tik_instance.for_range(0, vector_repeat_merchant) as i:

                        tik_instance.vadds(self.mask,
                                           input_y_ub[i * dst_ub_gap],
                                           input_x_ub[i * src_ub_gap],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)
                        # fp32 need twice
                        if self.mask == 64:
                            tik_instance.vadds(self.mask,
                                               input_y_ub[i * dst_ub_gap + 8],
                                               input_x_ub[i * src_ub_gap + 8],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                    if vector_repeat_remainder != 0:
                        vadds_mask = int(vector_repeat_remainder / MASK * self.mask)
                        tik_instance.vadds(vadds_mask,
                                           input_y_ub[vector_repeat_merchant * dst_ub_gap],
                                           input_x_ub[vector_repeat_merchant * src_ub_gap],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)

                        if self.mask == 64:
                            tik_instance.vadds(vadds_mask,
                                               input_y_ub[vector_repeat_merchant * dst_ub_gap + 8],
                                               input_x_ub[vector_repeat_merchant * src_ub_gap + 8],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    if num_gm == 1:
                        utmerchant.set_as(use_time * self.output_shape[-1] - num_gm * 16)
                        use_time_merchant.set_as(utmerchant // self.output_shape[-1])
                        use_time_remainder.set_as(utmerchant % self.output_shape[-1])

                        with tik_instance.if_scope(utmerchant >= 0):
                            with tik_instance.if_scope(use_time_remainder != 0):
                                use_time_merchant.set_as(use_time_merchant + 1)

                                with tik_instance.for_range(0, use_time_merchant) as i:
                                    dst_usetime_gm = dst_gm + i * self.output_shape[-1]
                                    tik_instance.data_move(self.output_y_gm[dst_usetime_gm],
                                                           input_y_ub[i * num_gm * 16],
                                                           0,
                                                           1,
                                                           burstlen_ub2gm,
                                                           0,
                                                           0)

                                # begin index
                                # remainder line's data
                                src_ub = (use_time_merchant - 1) * num_gm * 16
                                src_ub_offset = self.output_shape[-1] - use_time_remainder

                                with tik_instance.for_range(0, src_ub_offset) as i:
                                    input_z_ub[i] = input_x_ub[src_ub + use_time_remainder + i]

                                with tik_instance.for_range(0, use_time - use_time_merchant) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[src_ub_offset + i *
                                                   self.output_shape[-1] + j] = \
                                            input_y_ub[(use_time_merchant + i) * num_gm * 16 + j]

                                dst_ub = dst_gm + use_time_merchant * \
                                         self.output_shape[-1] - src_ub_offset

                                tik_instance.data_move(self.output_y_gm[dst_ub],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                            with tik_instance.if_scope(use_time_remainder == 0):
                                with tik_instance.for_range(0, use_time_merchant) as i:
                                    dst_usetime_gm = dst_gm + i * self.output_shape[-1]
                                    tik_instance.data_move(self.output_y_gm[dst_usetime_gm],
                                                           input_y_ub[i * num_gm * 16],
                                                           0,
                                                           1,
                                                           burstlen_ub2gm,
                                                           0,
                                                           0)
                                src_ub = use_time_merchant * num_gm * 16
                                src_ub_offset = self.output_shape[-1] - use_time_remainder

                                with tik_instance.for_range(0, src_ub_offset) as i:
                                    input_z_ub[i] = input_x_ub[src_ub + use_time_remainder + i]

                                with tik_instance.for_range(0,
                                                            use_time - use_time_merchant - 1) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[src_ub_offset + i *
                                                   self.output_shape[-1] + j] = \
                                            input_y_ub[
                                                (use_time_merchant + i + 1) * num_gm * 16 + j]

                                dst_ub = dst_gm + use_time_merchant * self.output_shape[-1]

                                tik_instance.data_move(self.output_y_gm[dst_ub],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                        with tik_instance.else_scope():
                            one_block = DATA_MOVE_MIN_UNIT // self.num_bit
                            offset_z = one_block - use_time * self.output_shape[-1]

                            with tik_instance.if_scope(offset_z > 0):
                                with tik_instance.for_range(0, use_time) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[i * self.output_shape[-1] + j + offset_z] = \
                                            input_y_ub[i * num_gm * 16 + j]
                                # back one_block
                                tik_instance.data_move(input_y_ub[0],
                                                       self.output_y_gm[dst_gm - offset_z],
                                                       0,
                                                       1,
                                                       1,
                                                       0,
                                                       0)

                                with tik_instance.for_range(0, offset_z) as j:
                                    input_z_ub[j] = input_y_ub[j]
                                minimum_data_move = 32 // self.num_bit
                                last_psm = acture_memory % 16 / minimum_data_move
                                burstlen_gm2ub_sp = math.ceil(last_psm) * minimum_data_move
                                burstlen_gm2ub_sp = burstlen_gm2ub_sp * self.num_bit // \
                                                    DATA_MOVE_MIN_UNIT
                                if burstlen_gm2ub_sp > 0:
                                    tik_instance.data_move(self.output_y_gm[dst_gm - offset_z],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           burstlen_gm2ub_sp,
                                                           0,
                                                           0)
                            with tik_instance.else_scope():
                                with tik_instance.for_range(0, use_time) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[i * self.output_shape[-1] + j] = \
                                            input_y_ub[i * num_gm * 16 + j]

                                minimum_data_move = DATA_MOVE_MIN_UNIT // self.num_bit
                                last_psm = acture_memory % 16 / minimum_data_move
                                burstlen_gm2ub_sp = math.floor(last_psm) * minimum_data_move
                                burstlen_gm2ub_sp = burstlen_gm2ub_sp * self.num_bit // \
                                                    DATA_MOVE_MIN_UNIT

                                if burstlen_gm2ub_sp > 0:
                                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           burstlen_gm2ub_sp,
                                                           0,
                                                           0)

                                    offset_z = -1 * offset_z
                                    with tik_instance.for_range(0, minimum_data_move) as i:
                                        input_z_ub[i] = input_z_ub[offset_z + i]
                                    dst_gm += offset_z
                                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           1,
                                                           0,
                                                           0)

    def nz2nd_special_case0_int32(self, tik_instance, params):
        """
        padding
        [5,1,1,16,16]
        total_num = 5
        output = 5, 14, 14
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        tail = 0
        tail_block = 0
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num

        num_gm2ub = self.maximum_size_ub // (17 * 16)
        num_gm = tiling_shape[-3]
        use_time = tik_instance.Scalar(dtype="int64")
        use_time_merchant = tik_instance.Scalar(dtype="int64")
        use_time_remainder = tik_instance.Scalar(dtype="int64")
        utmerchant = tik_instance.Scalar(dtype="int64")

        input_x_ub, input_y_ub, \
        input_y_ub_vconv, \
        input_z_ub, tail, \
        tail_block, psm_vconv = self.set_ub_tensor_int32(tik_instance, num_gm,
                                                         num_gm2ub, tiling_shape,
                                                         tiling_shape_gm2ub,
                                                         tail, tail_block)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                if tail == 0:
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16

                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               num_gm, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # -------------------Vconv int32 to fp32---------------------
                    vector_repeat_merchant = num_gm * 16 // MASK
                    vector_repeat_remainder = num_gm * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                         vector_repeat_remainder,
                                         self.mask, input_y_ub, input_x_ub, dst_ub_gap, src_ub_gap,
                                         repeat_times, dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride)

                    # -------------------Vconv fp32 to int32---------------------
                    all_rep = psm_vconv // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    if num_gm == 1:
                        utmerchant.set_as(use_time * self.output_shape[-1] - num_gm * 16)
                        use_time_merchant.set_as(utmerchant // self.output_shape[-1])
                        use_time_remainder.set_as(utmerchant % self.output_shape[-1])
                        with tik_instance.if_scope(utmerchant >= 0):
                            with tik_instance.if_scope(use_time_remainder != 0):
                                use_time_merchant.set_as(use_time_merchant + 1)

                                with tik_instance.for_range(0, use_time_merchant) as i:
                                    dst_usetime_gm = dst_gm + i * self.output_shape[-1]
                                    tik_instance.data_move(self.output_y_gm[dst_usetime_gm],
                                                           input_y_ub_vconv[i * num_gm * 16],
                                                           0,
                                                           1,
                                                           burstlen_ub2gm,
                                                           0,
                                                           0)

                                # begin index
                                # remainder line's data
                                src_ub = (use_time_merchant - 1) * num_gm * 16
                                src_ub_offset = self.output_shape[-1] - use_time_remainder

                                with tik_instance.for_range(0, src_ub_offset) as i:
                                    input_z_ub[i] = input_x_ub[src_ub + use_time_remainder + i]

                                with tik_instance.for_range(0, use_time - use_time_merchant) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[src_ub_offset + i *
                                                   self.output_shape[-1] + j] = \
                                            input_y_ub_vconv[
                                                (use_time_merchant + i) * num_gm * 16 + j]

                                dst_ub = dst_gm + use_time_merchant * \
                                         self.output_shape[-1] - src_ub_offset

                                tik_instance.data_move(self.output_y_gm[dst_ub],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                            with tik_instance.if_scope(use_time_remainder == 0):
                                with tik_instance.for_range(0, use_time_merchant) as i:
                                    dst_usetime_gm = dst_gm + i * self.output_shape[-1]
                                    tik_instance.data_move(self.output_y_gm[dst_usetime_gm],
                                                           input_y_ub_vconv[i * num_gm * 16],
                                                           0,
                                                           1,
                                                           burstlen_ub2gm,
                                                           0,
                                                           0)
                                src_ub = use_time_merchant * num_gm * 16
                                src_ub_offset = self.output_shape[-1] - use_time_remainder

                                with tik_instance.for_range(0, src_ub_offset) as i:
                                    input_z_ub[i] = input_x_ub[src_ub + use_time_remainder + i]

                                with tik_instance.for_range(0,
                                                            use_time - use_time_merchant - 1) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[src_ub_offset + i *
                                                   self.output_shape[-1] + j] = \
                                            input_y_ub_vconv[
                                                (use_time_merchant + i + 1) * num_gm * 16 + j]

                                dst_ub = dst_gm + use_time_merchant * self.output_shape[-1]

                                tik_instance.data_move(self.output_y_gm[dst_ub],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                        with tik_instance.else_scope():
                            one_block = DATA_MOVE_MIN_UNIT // self.num_bit
                            offset_z = one_block - use_time * self.output_shape[-1]

                            with tik_instance.if_scope(offset_z > 0):
                                with tik_instance.for_range(0, use_time) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[i * self.output_shape[-1] + j + offset_z] = \
                                            input_y_ub_vconv[i * num_gm * 16 + j]
                                # back one_block
                                tik_instance.data_move(input_y_ub_vconv[0],
                                                       self.output_y_gm[dst_gm - offset_z],
                                                       0,
                                                       1,
                                                       1,
                                                       0,
                                                       0)

                                with tik_instance.for_range(0, offset_z) as j:
                                    input_z_ub[j] = input_y_ub_vconv[j]
                                minimum_data_move = 32 // self.num_bit
                                last_psm = acture_memory % 16 / minimum_data_move
                                burstlen_gm2ub_sp = math.ceil(last_psm) * minimum_data_move
                                burstlen_gm2ub_sp = burstlen_gm2ub_sp * self.num_bit // \
                                                    DATA_MOVE_MIN_UNIT
                                if burstlen_gm2ub_sp > 0:
                                    tik_instance.data_move(self.output_y_gm[dst_gm - offset_z],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           burstlen_gm2ub_sp,
                                                           0,
                                                           0)
                            with tik_instance.else_scope():
                                with tik_instance.for_range(0, use_time) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[i * self.output_shape[-1] + j] = \
                                            input_y_ub_vconv[i * num_gm * 16 + j]

                                minimum_data_move = DATA_MOVE_MIN_UNIT // self.num_bit
                                last_psm = acture_memory % 16 / minimum_data_move
                                burstlen_gm2ub_sp = math.floor(last_psm) * minimum_data_move
                                burstlen_gm2ub_sp = burstlen_gm2ub_sp * self.num_bit // \
                                                    DATA_MOVE_MIN_UNIT

                                if burstlen_gm2ub_sp > 0:
                                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           burstlen_gm2ub_sp,
                                                           0,
                                                           0)

                                    offset_z = -1 * offset_z
                                    with tik_instance.for_range(0, minimum_data_move) as i:
                                        input_z_ub[i] = input_z_ub[offset_z + i]
                                    dst_gm += offset_z
                                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           1,
                                                           0,
                                                           0)

    def nz2nd_special_case0_int32_mini(self, tik_instance, params):
        """
        padding
        [5,1,1,16,16]
        total_num = 5
        output = 5, 14, 14
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        tail = 0
        tail_block = 0
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num

        num_gm2ub = self.maximum_size_ub // (17 * 16)
        num_gm = tiling_shape[-3]
        use_time = tik_instance.Scalar(dtype="int64")
        use_time_merchant = tik_instance.Scalar(dtype="int64")
        use_time_remainder = tik_instance.Scalar(dtype="int64")
        utmerchant = tik_instance.Scalar(dtype="int64")

        input_x_ub, input_y_ub, input_z_ub, input_v_ub, \
        tail, tail_block = self.set_ub_tensor_int32_mini(tik_instance, num_gm, num_gm2ub,
                                                         tiling_shape, tiling_shape_gm2ub,
                                                         tail, tail_block)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                self.set_vector_dup(tik_instance, 16, input_v_ub, 0)
                if tail == 0:
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16

                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               num_gm, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # -------------------Vconv int32 to fp32---------------------
                    vector_repeat_merchant = num_gm * 16 // MASK
                    vector_repeat_remainder = num_gm * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_vadd_mini(tik_instance, vector_repeat_merchant,
                                           vector_repeat_remainder,
                                           self.mask, input_y_ub, input_x_ub, input_v_ub,
                                           dst_ub_gap, src_ub_gap,
                                           repeat_times, dst_blk_stride, src_blk_stride,
                                           dst_rep_stride, src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    if num_gm == 1:
                        utmerchant.set_as(use_time * self.output_shape[-1] - num_gm * 16)
                        use_time_merchant.set_as(utmerchant // self.output_shape[-1])
                        use_time_remainder.set_as(utmerchant % self.output_shape[-1])
                        with tik_instance.if_scope(utmerchant >= 0):
                            with tik_instance.if_scope(use_time_remainder != 0):
                                use_time_merchant.set_as(use_time_merchant + 1)

                                with tik_instance.for_range(0, use_time_merchant) as i:
                                    dst_usetime_gm = dst_gm + i * self.output_shape[-1]
                                    tik_instance.data_move(self.output_y_gm[dst_usetime_gm],
                                                           input_y_ub[i * num_gm * 16],
                                                           0,
                                                           1,
                                                           burstlen_ub2gm,
                                                           0,
                                                           0)

                                # begin index
                                # remainder line's data
                                src_ub = (use_time_merchant - 1) * num_gm * 16
                                src_ub_offset = self.output_shape[-1] - use_time_remainder

                                with tik_instance.for_range(0, src_ub_offset) as i:
                                    input_z_ub[i] = input_x_ub[src_ub + use_time_remainder + i]

                                with tik_instance.for_range(0, use_time - use_time_merchant) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[src_ub_offset + i *
                                                   self.output_shape[-1] + j] = \
                                            input_y_ub[(use_time_merchant + i) * num_gm * 16 + j]

                                dst_ub = dst_gm + use_time_merchant * \
                                         self.output_shape[-1] - src_ub_offset

                                tik_instance.data_move(self.output_y_gm[dst_ub],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                            with tik_instance.if_scope(use_time_remainder == 0):
                                with tik_instance.for_range(0, use_time_merchant) as i:
                                    dst_usetime_gm = dst_gm + i * self.output_shape[-1]
                                    tik_instance.data_move(self.output_y_gm[dst_usetime_gm],
                                                           input_y_ub[i * num_gm * 16],
                                                           0,
                                                           1,
                                                           burstlen_ub2gm,
                                                           0,
                                                           0)
                                src_ub = use_time_merchant * num_gm * 16
                                src_ub_offset = self.output_shape[-1] - use_time_remainder

                                with tik_instance.for_range(0, src_ub_offset) as i:
                                    input_z_ub[i] = input_x_ub[src_ub + use_time_remainder + i]

                                with tik_instance.for_range(0,
                                                            use_time - use_time_merchant - 1) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[src_ub_offset + i *
                                                   self.output_shape[-1] + j] = \
                                            input_y_ub[
                                                (use_time_merchant + i + 1) * num_gm * 16 + j]

                                dst_ub = dst_gm + use_time_merchant * self.output_shape[-1]

                                tik_instance.data_move(self.output_y_gm[dst_ub],
                                                       input_z_ub[0],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                        with tik_instance.else_scope():
                            one_block = DATA_MOVE_MIN_UNIT // self.num_bit
                            offset_z = one_block - use_time * self.output_shape[-1]

                            with tik_instance.if_scope(offset_z > 0):
                                with tik_instance.for_range(0, use_time) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[i * self.output_shape[-1] + j + offset_z] = \
                                            input_y_ub[i * num_gm * 16 + j]
                                # back one_block
                                tik_instance.data_move(input_y_ub[0],
                                                       self.output_y_gm[dst_gm - offset_z],
                                                       0,
                                                       1,
                                                       1,
                                                       0,
                                                       0)

                                with tik_instance.for_range(0, offset_z) as j:
                                    input_z_ub[j] = input_y_ub[j]
                                minimum_data_move = 32 // self.num_bit
                                last_psm = acture_memory % 16 / minimum_data_move
                                burstlen_gm2ub_sp = math.ceil(last_psm) * minimum_data_move
                                burstlen_gm2ub_sp = burstlen_gm2ub_sp * self.num_bit // \
                                                    DATA_MOVE_MIN_UNIT
                                if burstlen_gm2ub_sp > 0:
                                    tik_instance.data_move(self.output_y_gm[dst_gm - offset_z],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           burstlen_gm2ub_sp,
                                                           0,
                                                           0)
                            with tik_instance.else_scope():
                                with tik_instance.for_range(0, use_time) as i:
                                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                                        input_z_ub[i * self.output_shape[-1] + j] = \
                                            input_y_ub[i * num_gm * 16 + j]

                                minimum_data_move = DATA_MOVE_MIN_UNIT // self.num_bit
                                last_psm = acture_memory % 16 / minimum_data_move
                                burstlen_gm2ub_sp = math.floor(last_psm) * minimum_data_move
                                burstlen_gm2ub_sp = burstlen_gm2ub_sp * self.num_bit // \
                                                    DATA_MOVE_MIN_UNIT

                                if burstlen_gm2ub_sp > 0:
                                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           burstlen_gm2ub_sp,
                                                           0,
                                                           0)

                                    offset_z = -1 * offset_z
                                    with tik_instance.for_range(0, minimum_data_move) as i:
                                        input_z_ub[i] = input_z_ub[offset_z + i]
                                    dst_gm += offset_z
                                    tik_instance.data_move(self.output_y_gm[dst_gm],
                                                           input_z_ub[0],
                                                           0,
                                                           1,
                                                           1,
                                                           0,
                                                           0)

    def nz2nd_special_case1(self, tik_instance, params):
        """
        padding
        [X,1,1,16,16]
        deal 16 16
        output = X,2,2
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        core_number = 1
        num_gm = tiling_shape[-3]
        use_time = tik_instance.Scalar(dtype="uint64")

        input_y_ub = tik_instance.Tensor(self.dtype, tiling_shape,
                                         name="input_x_ub", scope=tik.scope_ubuf)

        input_z_ub = tik_instance.Tensor(self.dtype, [64, ],
                                         name="input_z_ub", scope=tik.scope_ubuf)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16

                tik_instance.data_move(input_y_ub[0],
                                       self.input_x_gm[src_x_index],
                                       0,
                                       self.input_shape[-4],
                                       16 * 16 * self.num_bit // 32,
                                       0,
                                       0)
                # out
                dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                burstlen_ub2gm = int(num_gm * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                use_time.set_as(dst_gm_temp / self.output_shape[-1])
                with tik_instance.if_scope(use_time >= 16):
                    use_time.set_as(16)

                with tik_instance.for_range(0, use_time) as i:
                    with tik_instance.for_range(0, self.output_shape[-1]) as j:
                        input_z_ub[i * self.output_shape[-1] + j] = \
                            input_y_ub[i * num_gm * 16 + j]

                tik_instance.data_move(self.output_y_gm[dst_gm],
                                       input_z_ub[0],
                                       0,
                                       1,
                                       burstlen_ub2gm,
                                       0,
                                       0)

    def nz2nd_special_case2(self, tik_instance, params):
        """
        padding
        [5,32,10,16,16]
        total_num = 50
        output = 5,158,510
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        tail = 0
        tail_block = 0
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num

        num_gm2ub = self.maximum_size_ub // (17 * 16)
        num_gm = tiling_shape[-3]
        use_time = tik_instance.Scalar(dtype="uint64")

        input_x_ub, input_y_ub, input_z_ub, \
        tail, tail_block = self.set_ub_tensor(tik_instance, num_gm, num_gm2ub,
                                              tiling_shape, tiling_shape_gm2ub,
                                              tail, tail_block)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                if tail == 0:
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16

                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    if srcstride_gm2ub <= MAX_STRIDE:

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               num_gm,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)
                    else:
                        src_x_index_gap = self.input_shape[-3] * 16 * 16
                        dst_x_index_gap = 16 * 17
                        dst_x_index = 0
                        src_x_gm_index = src_x_index
                        with tik_instance.for_range(0, num_gm) as i:
                            src_x_gm_index = src_x_gm_index + i * src_x_index_gap
                            dst_x_index = dst_x_index + i * dst_x_index_gap
                            tik_instance.data_move(input_x_ub[dst_x_index],
                                                   self.input_x_gm[src_x_gm_index],
                                                   0,
                                                   1,
                                                   burstlen_gm2ub,
                                                   0,
                                                   0)

                    # num_gm * 16
                    # vector maximum compute 128
                    vector_repeat_merchant = num_gm * 16 // MASK
                    vector_repeat_remainder = num_gm * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    # order
                    with tik_instance.for_range(0, vector_repeat_merchant) as i:

                        tik_instance.vadds(self.mask,
                                           input_y_ub[i * dst_ub_gap],
                                           input_x_ub[i * src_ub_gap],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)
                        # fp32 need twice
                        if self.mask == 64:
                            tik_instance.vadds(self.mask,
                                               input_y_ub[i * dst_ub_gap + 8],
                                               input_x_ub[i * src_ub_gap + 8],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                    if vector_repeat_remainder != 0:
                        vadds_mask = int(vector_repeat_remainder / MASK * self.mask)
                        tik_instance.vadds(vadds_mask,
                                           input_y_ub[vector_repeat_merchant * dst_ub_gap],
                                           input_x_ub[vector_repeat_merchant * src_ub_gap],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)

                        if self.mask == 64:
                            tik_instance.vadds(vadds_mask,
                                               input_y_ub[vector_repeat_merchant * dst_ub_gap + 8],
                                               input_x_ub[vector_repeat_merchant * src_ub_gap + 8],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:

                        tik_instance.data_move(self.output_y_gm[dst_gm + i * self.output_shape[-1]],
                                               input_y_ub[i * num_gm * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )
                    dst_ub = dst_gm + (use_time - 1) * self.output_shape[-1] + \
                             (self.input_shape[-4] - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)

                    burstlen_ub2gm_sp = int((num_gm - 1) * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    dst_gm_last = dst_gm + (use_time - 1) * self.output_shape[-1]
                    tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                           input_y_ub[(use_time - 1) * num_gm * 16],
                                           0,
                                           1,
                                           burstlen_ub2gm_sp,
                                           0,
                                           0)

                elif (tail != 0) and (tail_block != 0):

                    # deal tail_block firstly
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + \
                                  tail * num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = tail_block
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    if srcstride_gm2ub <= MAX_STRIDE:

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)
                    else:
                        src_x_index_gap = self.input_shape[-3] * 16 * 16
                        dst_x_index_gap = 16 * 17
                        dst_x_index = 0
                        with tik_instance.for_range(0, nburst_gm2ub) as i:
                            src_x_index = src_x_index + i * src_x_index_gap
                            dst_x_index = dst_x_index + i * dst_x_index_gap
                            tik_instance.data_move(input_x_ub[dst_x_index],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   1,
                                                   burstlen_gm2ub,
                                                   0,
                                                   0)

                    # tail_block * 16
                    # vector maximum compute 128
                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = tail_block * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    # order
                    with tik_instance.for_range(0, vector_repeat_merchant) as i:

                        tik_instance.vadds(self.mask,
                                           input_y_ub[i * dst_ub_gap],
                                           input_x_ub[i * src_ub_gap],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)
                        # fp32 need twice
                        if self.mask == 64:
                            tik_instance.vadds(self.mask,
                                               input_y_ub[i * dst_ub_gap + 8],
                                               input_x_ub[i * src_ub_gap + 8],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                    if vector_repeat_remainder != 0:
                        vadds_mask = int(vector_repeat_remainder / MASK * self.mask)
                        tik_instance.vadds(vadds_mask,
                                           input_y_ub[vector_repeat_merchant * dst_ub_gap],
                                           input_x_ub[vector_repeat_merchant * src_ub_gap],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)

                        if self.mask == 64:
                            tik_instance.vadds(vadds_mask,
                                               input_y_ub[vector_repeat_merchant * dst_ub_gap + 8],
                                               input_x_ub[vector_repeat_merchant * src_ub_gap + 8],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(tail_block * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:
                        dst_use_time_gm = dst_gm + tail * num_gm2ub * 16 + i * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                               input_y_ub[i * tail_block * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index_temp + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )

                    dst_ub = dst_gm + tail * num_gm2ub * 16 + (use_time - 1) * \
                             self.output_shape[-1] + (tail_block - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)
                    if (tail_block - 1) > 0:
                        burstlen_ub2gm_sp = int((tail_block - 1) * 16 *
                                                self.num_bit / DATA_MOVE_MIN_UNIT)
                        dst_gm_last = dst_gm + tail * num_gm2ub * 16 + \
                                      (use_time - 1) * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                               input_y_ub[(use_time - 1) * tail_block * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm_sp,
                                               0,
                                               0)

                    # deal tail data
                    with tik_instance.for_range(0, tail) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        if srcstride_gm2ub <= MAX_STRIDE:

                            tik_instance.data_move(input_x_ub[0],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub)
                        else:
                            src_x_index_gap = self.input_shape[-3] * 16 * 16
                            dst_x_index_gap = 16 * 17
                            dst_x_index = 0
                            with tik_instance.for_range(0, nburst_gm2ub) as i:
                                src_x_index = src_x_index + i * src_x_index_gap
                                dst_x_index = dst_x_index + i * dst_x_index_gap
                                tik_instance.data_move(input_x_ub[dst_x_index],
                                                       self.input_x_gm[src_x_index],
                                                       0,
                                                       1,
                                                       burstlen_gm2ub,
                                                       0,
                                                       0)

                        # tail_block * 16
                        # vector maximum compute 128
                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        # order
                        with tik_instance.for_range(0, vector_repeat_merchant) as i:

                            tik_instance.vadds(self.mask,
                                               input_y_ub[i * dst_ub_gap],
                                               input_x_ub[i * src_ub_gap],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)
                            # fp32 need twice
                            if self.mask == 64:
                                tik_instance.vadds(self.mask,
                                                   input_y_ub[i * dst_ub_gap + 8],
                                                   input_x_ub[i * src_ub_gap + 8],
                                                   0,
                                                   repeat_times,
                                                   dst_blk_stride,
                                                   src_blk_stride,
                                                   dst_rep_stride,
                                                   src_rep_stride)

                        if vector_repeat_remainder != 0:
                            vadds_mask = int(vector_repeat_remainder / MASK * self.mask)
                            tik_instance.vadds(vadds_mask,
                                               input_y_ub[vector_repeat_merchant * dst_ub_gap],
                                               input_x_ub[vector_repeat_merchant * src_ub_gap],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                            if self.mask == 64:
                                dst_y_ub = vector_repeat_merchant * dst_ub_gap + 8
                                src_x_ub = vector_repeat_merchant * src_ub_gap + 8
                                tik_instance.vadds(vadds_mask,
                                                   input_y_ub[dst_y_ub],
                                                   input_x_ub[src_x_ub],
                                                   0,
                                                   repeat_times,
                                                   dst_blk_stride,
                                                   src_blk_stride,
                                                   dst_rep_stride,
                                                   src_rep_stride)

                        # out
                        dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                        dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                        use_time.set_as(dst_gm_temp / self.output_shape[-1])

                        with tik_instance.if_scope(use_time >= 16):
                            use_time.set_as(16)

                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                        with tik_instance.for_range(0, use_time) as i:
                            dst_use_time_gm = dst_gm + i * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub[i * num_gm2ub * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

                else:
                    # no tail_block discrimination between final and (0,...final-1)
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + (tail - 1) * \
                                  num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = num_gm2ub
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    if srcstride_gm2ub <= MAX_STRIDE:

                        tik_instance.data_move(input_x_ub[0],
                                               self.input_x_gm[src_x_index],
                                               0,
                                               nburst_gm2ub,
                                               burstlen_gm2ub,
                                               srcstride_gm2ub,
                                               dststride_gm2ub)
                    else:
                        src_x_index_gap = self.input_shape[-3] * 16 * 16
                        dst_x_index_gap = 16 * 17
                        dst_x_index = 0
                        with tik_instance.for_range(0, nburst_gm2ub) as i:
                            src_x_index = src_x_index + i * src_x_index_gap
                            dst_x_index = dst_x_index + i * dst_x_index_gap
                            tik_instance.data_move(input_x_ub[dst_x_index],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   1,
                                                   burstlen_gm2ub,
                                                   0,
                                                   0)

                    # num_gm2ub * 16
                    # vector maxcompute 128
                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    # order
                    with tik_instance.for_range(0, vector_repeat_merchant) as i:

                        tik_instance.vadds(self.mask,
                                           input_y_ub[i * dst_ub_gap],
                                           input_x_ub[i * src_ub_gap],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)
                        # fp32 need twice
                        if self.mask == 64:
                            tik_instance.vadds(self.mask,
                                               input_y_ub[i * dst_ub_gap + 8],
                                               input_x_ub[i * src_ub_gap + 8],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                    if vector_repeat_remainder != 0:
                        vadds_mask = int(vector_repeat_remainder / MASK * self.mask)
                        tik_instance.vadds(vadds_mask,
                                           input_y_ub[vector_repeat_merchant * dst_ub_gap],
                                           input_x_ub[vector_repeat_merchant * src_ub_gap],
                                           0,
                                           repeat_times,
                                           dst_blk_stride,
                                           src_blk_stride,
                                           dst_rep_stride,
                                           src_rep_stride)

                        if self.mask == 64:
                            tik_instance.vadds(vadds_mask,
                                               input_y_ub[vector_repeat_merchant * dst_ub_gap + 8],
                                               input_x_ub[vector_repeat_merchant * src_ub_gap + 8],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:
                        dst_use_time_gm = dst_gm + (tail - 1) * \
                                          num_gm2ub * 16 + i * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                               input_y_ub[i * num_gm2ub * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index_temp + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )

                    dst_ub = dst_gm + (tail - 1) * num_gm2ub * 16 + (use_time - 1) * \
                             self.output_shape[-1] + (num_gm2ub - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)

                    burstlen_ub2gm_sp = int(
                        (num_gm2ub - 1) * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    dst_gm_last = dst_gm + (tail - 1) * num_gm2ub * 16 + \
                                  (use_time - 1) * self.output_shape[-1]
                    tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                           input_y_ub[(use_time - 1) * num_gm2ub * 16],
                                           0,
                                           1,
                                           burstlen_ub2gm_sp,
                                           0,
                                           0)

                    # deal tail-1 data
                    with tik_instance.for_range(0, tail - 1) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        if srcstride_gm2ub <= MAX_STRIDE:

                            tik_instance.data_move(input_x_ub[0],
                                                   self.input_x_gm[src_x_index],
                                                   0,
                                                   nburst_gm2ub,
                                                   burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub)
                        else:
                            src_x_index_gap = self.input_shape[-3] * 16 * 16
                            dst_x_index_gap = 16 * 17
                            dst_x_index = 0
                            with tik_instance.for_range(0, nburst_gm2ub) as i:
                                src_x_index = src_x_index + i * src_x_index_gap
                                dst_x_index = dst_x_index + i * dst_x_index_gap
                                tik_instance.data_move(input_x_ub[dst_x_index],
                                                       self.input_x_gm[src_x_index],
                                                       0,
                                                       1,
                                                       burstlen_gm2ub,
                                                       0,
                                                       0)

                        # num_gm2ub * 16
                        # vector maximum compute 128
                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        # order
                        with tik_instance.for_range(0, vector_repeat_merchant) as i:

                            tik_instance.vadds(self.mask,
                                               input_y_ub[i * dst_ub_gap],
                                               input_x_ub[i * src_ub_gap],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)
                            # fp32 need twice
                            if self.mask == 64:
                                tik_instance.vadds(self.mask,
                                                   input_y_ub[i * dst_ub_gap + 8],
                                                   input_x_ub[i * src_ub_gap + 8],
                                                   0,
                                                   repeat_times,
                                                   dst_blk_stride,
                                                   src_blk_stride,
                                                   dst_rep_stride,
                                                   src_rep_stride)

                        if vector_repeat_remainder != 0:
                            vadds_mask = int(vector_repeat_remainder / MASK * self.mask)
                            tik_instance.vadds(vadds_mask,
                                               input_y_ub[vector_repeat_merchant * dst_ub_gap],
                                               input_x_ub[vector_repeat_merchant * src_ub_gap],
                                               0,
                                               repeat_times,
                                               dst_blk_stride,
                                               src_blk_stride,
                                               dst_rep_stride,
                                               src_rep_stride)

                            if self.mask == 64:
                                dst_y_ub = vector_repeat_merchant * dst_ub_gap + 8
                                src_x_ub = vector_repeat_merchant * src_ub_gap + 8
                                tik_instance.vadds(vadds_mask,
                                                   input_y_ub[dst_y_ub],
                                                   input_x_ub[src_x_ub],
                                                   0,
                                                   repeat_times,
                                                   dst_blk_stride,
                                                   src_blk_stride,
                                                   dst_rep_stride,
                                                   src_rep_stride)

                        # out
                        dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                        dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                        use_time.set_as(dst_gm_temp / self.output_shape[-1])

                        with tik_instance.if_scope(use_time >= 16):
                            use_time.set_as(16)

                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                        with tik_instance.for_range(0, use_time) as i:
                            dst_use_time_gm = dst_gm + i * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub[i * num_gm2ub * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

    def nz2nd_special_case2_int32(self, tik_instance, params):
        """
        padding
        [5,32,10,16,16]
        total_num = 50
        output = 5,158,510
        support int32
        int32->fp32->sorted->int32
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        tail = 0
        tail_block = 0
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num

        num_gm2ub = self.maximum_size_ub // (17 * 16)
        num_gm = tiling_shape[-3]
        use_time = tik_instance.Scalar(dtype="uint64")

        input_x_ub, input_y_ub, input_y_ub_vconv, input_z_ub, \
        tail, tail_block, psm_vconv = self.set_ub_tensor_int32(tik_instance, num_gm, num_gm2ub,
                                                               tiling_shape, tiling_shape_gm2ub,
                                                               tail, tail_block)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                if tail == 0:
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16

                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               num_gm, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    vector_repeat_merchant = num_gm * 16 // MASK
                    vector_repeat_remainder = num_gm * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                         vector_repeat_remainder,
                                         self.mask, input_y_ub, input_x_ub, dst_ub_gap, src_ub_gap,
                                         repeat_times, dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride)

                    # all_rep must be int
                    all_rep = psm_vconv // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:

                        tik_instance.data_move(self.output_y_gm[dst_gm + i * self.output_shape[-1]],
                                               input_y_ub_vconv[i * num_gm * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    # ------------------reg_move---------------------- #
                    # input_z_ub is int32
                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )
                    dst_ub = dst_gm + (use_time - 1) * self.output_shape[-1] + \
                             (self.input_shape[-4] - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)

                    burstlen_ub2gm_sp = int((num_gm - 1) * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    dst_gm_last = dst_gm + (use_time - 1) * self.output_shape[-1]
                    tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                           input_y_ub_vconv[(use_time - 1) * num_gm * 16],
                                           0,
                                           1,
                                           burstlen_ub2gm_sp,
                                           0,
                                           0)

                elif (tail != 0) and (tail_block != 0):
                    # deal tail_block firstly
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + \
                                  tail * num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = tail_block
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # use vconv instead of vadds
                    # vector maximum compute 128
                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = tail_block * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                         vector_repeat_remainder,
                                         self.mask, input_y_ub, input_x_ub, dst_ub_gap, src_ub_gap,
                                         repeat_times, dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride)

                    # all_rep must be int
                    all_rep = tail_block * 16 * 16 // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(tail_block * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:
                        dst_use_time_gm = dst_gm + tail * num_gm2ub * 16 + i * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                               input_y_ub_vconv[i * tail_block * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index_temp + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )

                    dst_ub = dst_gm + tail * num_gm2ub * 16 + (use_time - 1) * \
                             self.output_shape[-1] + (tail_block - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)
                    if (tail_block - 1) > 0:
                        burstlen_ub2gm_sp = int((tail_block - 1) * 16 *
                                                self.num_bit / DATA_MOVE_MIN_UNIT)
                        dst_gm_last = dst_gm + tail * num_gm2ub * 16 + \
                                      (use_time - 1) * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                               input_y_ub_vconv[(use_time - 1) * tail_block * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm_sp,
                                               0,
                                               0)

                    # deal tail data
                    with tik_instance.for_range(0, tail) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                                   dststride_gm2ub, src_x_index, 0)

                        # use vconv instead of vadds
                        # vector maximum compute 128
                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                             vector_repeat_remainder,
                                             self.mask, input_y_ub, input_x_ub, dst_ub_gap,
                                             src_ub_gap,
                                             repeat_times, dst_blk_stride, src_blk_stride,
                                             dst_rep_stride, src_rep_stride)

                        # all_rep must be int
                        all_rep = psm_vconv // self.mask
                        vconv_repeat_merchant = all_rep // 255
                        vconv_repeat_remainder = all_rep % 255

                        src_ub_gap_vconv = 255 * self.mask
                        dst_ub_gap_vconv = 255 * self.mask
                        src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times_vconv = 255

                        self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                     self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                     src_ub_gap_vconv, repeat_times_vconv,
                                     dst_blk_stride_vconv, src_blk_stride_vconv,
                                     dst_rep_stride_vconv, src_rep_stride_vconv)

                        # out
                        dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                        dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                        use_time.set_as(dst_gm_temp / self.output_shape[-1])

                        with tik_instance.if_scope(use_time >= 16):
                            use_time.set_as(16)

                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                        with tik_instance.for_range(0, use_time) as i:
                            dst_use_time_gm = dst_gm + i * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub_vconv[i * num_gm2ub * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

                else:
                    # no tail_block discrimination between final and (0,...final-1)
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + (tail - 1) * \
                                  num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = num_gm2ub
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # use vconv instead of vadds
                    # vector maximum compute 128
                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                         vector_repeat_remainder,
                                         self.mask, input_y_ub, input_x_ub, dst_ub_gap, src_ub_gap,
                                         repeat_times, dst_blk_stride, src_blk_stride,
                                         dst_rep_stride, src_rep_stride)

                    # all_rep must be int
                    all_rep = psm_vconv // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:
                        dst_use_time_gm = dst_gm + (tail - 1) * \
                                          num_gm2ub * 16 + i * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                               input_y_ub_vconv[i * num_gm2ub * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index_temp + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )

                    dst_ub = dst_gm + (tail - 1) * num_gm2ub * 16 + (use_time - 1) * \
                             self.output_shape[-1] + (num_gm2ub - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)

                    burstlen_ub2gm_sp = int(
                        (num_gm2ub - 1) * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    dst_gm_last = dst_gm + (tail - 1) * num_gm2ub * 16 + \
                                  (use_time - 1) * self.output_shape[-1]
                    tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                           input_y_ub_vconv[(use_time - 1) * num_gm2ub * 16],
                                           0,
                                           1,
                                           burstlen_ub2gm_sp,
                                           0,
                                           0)

                    # deal tail-1 data
                    with tik_instance.for_range(0, tail - 1) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                                   dststride_gm2ub, src_x_index, 0)

                        # use vconv instead of vadds
                        # vector maximum compute 128
                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        self.reorder_s322f32(tik_instance, vector_repeat_merchant,
                                             vector_repeat_remainder,
                                             self.mask, input_y_ub, input_x_ub, dst_ub_gap,
                                             src_ub_gap,
                                             repeat_times, dst_blk_stride, src_blk_stride,
                                             dst_rep_stride, src_rep_stride)

                        # all_rep must be int
                        all_rep = psm_vconv // self.mask
                        vconv_repeat_merchant = all_rep // 255
                        vconv_repeat_remainder = all_rep % 255

                        src_ub_gap_vconv = 255 * self.mask
                        dst_ub_gap_vconv = 255 * self.mask
                        src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times_vconv = 255

                        self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                     self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                     src_ub_gap_vconv, repeat_times_vconv,
                                     dst_blk_stride_vconv, src_blk_stride_vconv,
                                     dst_rep_stride_vconv, src_rep_stride_vconv)

                        # out
                        dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                        dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                        use_time.set_as(dst_gm_temp / self.output_shape[-1])

                        with tik_instance.if_scope(use_time >= 16):
                            use_time.set_as(16)

                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                        with tik_instance.for_range(0, use_time) as i:
                            dst_use_time_gm = dst_gm + i * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub_vconv[i * num_gm2ub * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

    def nz2nd_special_case2_int32_mini(self, tik_instance, params):
        """
        padding
        [5,32,10,16,16]
        total_num = 50
        output = 5,158,510
        support int32
        int32->fp32->sorted->int32
        """
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        tail = 0
        tail_block = 0
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        tiling_shape_gm2ub = tiling_shape.copy()
        tiling_shape_gm2ub[-2] = 17
        core_number = core_num

        num_gm2ub = self.maximum_size_ub // (17 * 16)
        num_gm = tiling_shape[-3]
        use_time = tik_instance.Scalar(dtype="uint64")

        input_x_ub, input_y_ub, input_z_ub, input_v_ub, \
        tail, tail_block = self.set_ub_tensor_int32_mini(tik_instance, num_gm, num_gm2ub,
                                                         tiling_shape, tiling_shape_gm2ub,
                                                         tail, tail_block)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:
                self.set_vector_dup(tik_instance, 16, input_v_ub, 0)
                if tail == 0:
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16

                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               num_gm, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    vector_repeat_merchant = num_gm * 16 // MASK
                    vector_repeat_remainder = num_gm * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_vadd_mini(tik_instance, vector_repeat_merchant,
                                           vector_repeat_remainder,
                                           self.mask, input_y_ub, input_x_ub, input_v_ub,
                                           dst_ub_gap, src_ub_gap,
                                           repeat_times, dst_blk_stride, src_blk_stride,
                                           dst_rep_stride, src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:

                        tik_instance.data_move(self.output_y_gm[dst_gm + i * self.output_shape[-1]],
                                               input_y_ub[i * num_gm * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    # ------------------reg_move---------------------- #
                    # input_z_ub is int32
                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )
                    dst_ub = dst_gm + (use_time - 1) * self.output_shape[-1] + \
                             (self.input_shape[-4] - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)

                    burstlen_ub2gm_sp = int((num_gm - 1) * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    dst_gm_last = dst_gm + (use_time - 1) * self.output_shape[-1]
                    tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                           input_y_ub[(use_time - 1) * num_gm * 16],
                                           0,
                                           1,
                                           burstlen_ub2gm_sp,
                                           0,
                                           0)

                elif (tail != 0) and (tail_block != 0):
                    # deal tail_block firstly
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + \
                                  tail * num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = tail_block
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # use vconv instead of vadds
                    # vector maximum compute 128
                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = tail_block * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_vadd_mini(tik_instance, vector_repeat_merchant,
                                           vector_repeat_remainder,
                                           self.mask, input_y_ub, input_x_ub, input_v_ub,
                                           dst_ub_gap, src_ub_gap,
                                           repeat_times, dst_blk_stride, src_blk_stride,
                                           dst_rep_stride, src_rep_stride)
                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(tail_block * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:
                        dst_use_time_gm = dst_gm + tail * num_gm2ub * 16 + i * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                               input_y_ub[i * tail_block * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index_temp + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )

                    dst_ub = dst_gm + tail * num_gm2ub * 16 + (use_time - 1) * \
                             self.output_shape[-1] + (tail_block - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)
                    if (tail_block - 1) > 0:
                        burstlen_ub2gm_sp = int((tail_block - 1) * 16 *
                                                self.num_bit / DATA_MOVE_MIN_UNIT)
                        dst_gm_last = dst_gm + tail * num_gm2ub * 16 + \
                                      (use_time - 1) * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                               input_y_ub[(use_time - 1) * tail_block * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm_sp,
                                               0,
                                               0)

                    # deal tail data
                    with tik_instance.for_range(0, tail) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                                   dststride_gm2ub, src_x_index, 0)

                        # use vconv instead of vadds
                        # vector maximum compute 128
                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        self.reorder_vadd_mini(tik_instance, vector_repeat_merchant,
                                               vector_repeat_remainder,
                                               self.mask, input_y_ub, input_x_ub, input_v_ub,
                                               dst_ub_gap, src_ub_gap,
                                               repeat_times, dst_blk_stride, src_blk_stride,
                                               dst_rep_stride, src_rep_stride)

                        # out
                        dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                        dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                        use_time.set_as(dst_gm_temp / self.output_shape[-1])

                        with tik_instance.if_scope(use_time >= 16):
                            use_time.set_as(16)

                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                        with tik_instance.for_range(0, use_time) as i:
                            dst_use_time_gm = dst_gm + i * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub[i * num_gm2ub * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

                else:
                    # no tail_block discrimination between final and (0,...final-1)
                    merchant = (sum_core + num_core_loop) // self.input_shape[-3]
                    remainder = (sum_core + num_core_loop) % self.input_shape[-3]
                    src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + (tail - 1) * \
                                  num_gm2ub * 16 * 16 * self.input_shape[-3]

                    nburst_gm2ub = num_gm2ub
                    burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # use vconv instead of vadds
                    # vector maximum compute 128
                    vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                    src_ub_gap = MASK * 17
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = num_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = 16

                    self.reorder_vadd_mini(tik_instance, vector_repeat_merchant,
                                           vector_repeat_remainder,
                                           self.mask, input_y_ub, input_x_ub, input_v_ub,
                                           dst_ub_gap, src_ub_gap,
                                           repeat_times, dst_blk_stride, src_blk_stride,
                                           dst_rep_stride, src_rep_stride)

                    # out
                    dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                    dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                    burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    use_time.set_as(dst_gm_temp / self.output_shape[-1])
                    with tik_instance.if_scope(use_time >= 16):
                        use_time.set_as(16)

                    with tik_instance.for_range(0, use_time - 1) as i:
                        dst_use_time_gm = dst_gm + (tail - 1) * \
                                          num_gm2ub * 16 + i * self.output_shape[-1]
                        tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                               input_y_ub[i * num_gm2ub * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm,
                                               0,
                                               0)

                    small_offset_ub = self.input_shape[-1] * \
                                      self.input_shape[-4] - self.output_shape[-1]

                    src_ub = src_x_index_temp + (self.input_shape[-4] - 1) * \
                             self.input_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                    tik_instance.data_move(input_z_ub[0],
                                           self.input_x_gm[src_ub],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0, )

                    dst_ub = dst_gm + (tail - 1) * num_gm2ub * 16 + (use_time - 1) * \
                             self.output_shape[-1] + (num_gm2ub - 1) * 16 - small_offset_ub

                    tik_instance.data_move(self.output_y_gm[dst_ub],
                                           input_z_ub[0],
                                           0,
                                           1,
                                           self.num_bit // 2,
                                           0,
                                           0)

                    burstlen_ub2gm_sp = int(
                        (num_gm2ub - 1) * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    dst_gm_last = dst_gm + (tail - 1) * num_gm2ub * 16 + \
                                  (use_time - 1) * self.output_shape[-1]
                    tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                           input_y_ub[(use_time - 1) * num_gm2ub * 16],
                                           0,
                                           1,
                                           burstlen_ub2gm_sp,
                                           0,
                                           0)

                    # deal tail-1 data
                    with tik_instance.for_range(0, tail - 1) as num_ub_loop:
                        src_x_index = psm * self.input_shape[-3] * merchant + remainder * 16 * 16
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * \
                                      num_gm2ub * 16 * 16 * self.input_shape[-3]

                        nburst_gm2ub = num_gm2ub
                        burstlen_gm2ub = 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = (self.input_shape[-3] - 1) * 16 * 16 * self.num_bit // 32
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                                   dststride_gm2ub, src_x_index, 0)

                        # use vconv instead of vadds
                        # vector maximum compute 128
                        vector_repeat_merchant = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder = nburst_gm2ub * 16 % MASK

                        src_ub_gap = MASK * 17
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * 17 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = 16

                        self.reorder_vadd_mini(tik_instance, vector_repeat_merchant,
                                               vector_repeat_remainder,
                                               self.mask, input_y_ub, input_x_ub, input_v_ub,
                                               dst_ub_gap, src_ub_gap,
                                               repeat_times, dst_blk_stride, src_blk_stride,
                                               dst_rep_stride, src_rep_stride)

                        # out
                        dst_gm = acture_memory * merchant + remainder * self.output_shape[-1] * 16
                        dst_gm_temp = acture_memory * (merchant + 1) - dst_gm
                        use_time.set_as(dst_gm_temp / self.output_shape[-1])

                        with tik_instance.if_scope(use_time >= 16):
                            use_time.set_as(16)

                        dst_gm = dst_gm + num_ub_loop * num_gm2ub * 16
                        burstlen_ub2gm = int(num_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                        with tik_instance.for_range(0, use_time) as i:
                            dst_use_time_gm = dst_gm + i * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub[i * num_gm2ub * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)

    def nz2nd_special_case2_int32_pattern_zero(self, tik_instance, params):
        """
        padding
        [A,B,62500,1,16,16]
        A,B = 1,1
        total_num = 50
        output = [1,100W]
        [1250,1,16,16]
        support int32
        int32->fp32->sorted->int32
        """
        # if tiling_shape > maximum_size_ub, split D_axis ==> tail_d
        # in this case, C_axis * A * B < 32, so C_axis can not be splited
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        core_number = core_num
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        psm_map = self.calc_element(self.input_shape[-4:])
        c_value = tiling_shape[1]
        d_value = tiling_shape[0]
        for d_split in range(d_value):
            d_split = d_split + 1
            psm_ub_in = (c_value * 16 + 1) * (d_split * 16)
            psm_ub_out = d_split * 16 * c_value * 16
            if psm_ub_in > self.maximum_size_ub:
                d_split = d_split - 1
                psm_ub_in = (c_value * 16 + 1) * (d_split * 16)
                psm_ub_out = d_split * 16 * c_value * 16
                break

        # tail_d must be >= 1
        tail_d = d_value // d_split
        tail_block_d = d_value % d_split
        use_time = tik_instance.Scalar(dtype="uint64")

        input_x_ub = tik_instance.Tensor(self.dtype, [psm_ub_in, ], name="input_x_ub",
                                         scope=tik.scope_ubuf)

        input_y_ub = tik_instance.Tensor("float32", [psm_ub_out, ], name="input_y_ub",
                                         scope=tik.scope_ubuf)

        input_y_ub_vconv = tik_instance.Tensor(self.dtype, [psm_ub_out, ], name="input_y_ub_vconv",
                                               scope=tik.scope_ubuf)

        input_z_ub = tik_instance.Tensor(self.dtype, [16, ], name="input_z_ub",
                                         scope=tik.scope_ubuf)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:

                if tail_block_d != 0:
                    # deal tail_block firstly
                    # merchant: represent which map [D,C,16,16]
                    # remainder: pos in one map [d1,C,16,16]
                    merchant = (sum_core + num_core_loop) * tiling_shape[0] // self.input_shape[-4]
                    remainder = (sum_core + num_core_loop) * tiling_shape[0] % self.input_shape[-4]
                    src_x_index = psm_map * merchant + remainder * psm_map / self.input_shape[-4]
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + tail_d * psm_ub_out

                    nburst_gm2ub = tail_block_d
                    burstlen_gm2ub = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = 0
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # use vconv instead of vadds
                    # vector maximum compute 128
                    vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                    vector_repeat_merchant_c = tiling_shape[-3] * 16 // MAX_REPEAT
                    vector_repeat_remainder_c = tiling_shape[-3] * 16 % MAX_REPEAT

                    src_ub_gap = MASK * (tiling_shape[-3] * 16 + 1)
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * (tiling_shape[-3] * 16 + 1) * \
                                     self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = tail_block_d * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = MAX_REPEAT

                    self.reorder_s322f32_pattern_0(tik_instance, vector_repeat_merchant_d,
                                                   vector_repeat_remainder_d,
                                                   vector_repeat_merchant_c,
                                                   vector_repeat_remainder_c,
                                                   self.mask, input_y_ub, input_x_ub, dst_ub_gap,
                                                   src_ub_gap,
                                                   repeat_times, dst_blk_stride, src_blk_stride,
                                                   dst_rep_stride, src_rep_stride, nburst_gm2ub)

                    # all_rep must be int
                    all_rep = tail_block_d * 16 * tiling_shape[-3] * 16 // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # -----------------------Out-----------------------------------#
                    # 判断处理的tiling_shape是否为一张map的最后一个
                    # 判断处理的tail_block_d是否为tiling_shape的最后一个
                    # 条件融合：当tail_block_d存在时，判断其是否为map的结尾
                    # 是(flag == 0)：按照特殊方式输出，否(flag != 0)：按照常规方式输出
                    flag = (remainder + tiling_shape[0]) % self.input_shape[-4]
                    dst_gm = acture_memory * merchant + remainder * 16
                    use_time.set_as(self.output_shape[-2])
                    burstlen_ub2gm = int(tail_block_d * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    src_stride_ub2gm = 0
                    dst_stride_ub2gm = 0

                    with tik_instance.if_scope(flag == 0):
                        with tik_instance.for_range(0, use_time - 1) as i:
                            dst_use_time_gm = dst_gm + tail_d * d_split * 16 + i * \
                                              self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub_vconv[i * tail_block_d * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   src_stride_ub2gm,
                                                   dst_stride_ub2gm)

                        small_offset_ub = self.input_shape[-1] * \
                                          self.input_shape[-4] - self.output_shape[-1]

                        src_ub = src_x_index_temp + (tiling_shape[-4] - 1) * \
                                 tiling_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                        tik_instance.data_move(input_z_ub[0],
                                               self.input_x_gm[src_ub],
                                               0,
                                               1,
                                               self.num_bit // 2,
                                               0,
                                               0, )

                        dst_ub = dst_gm + tail_d * d_split * 16 + (use_time - 1) * \
                                 self.output_shape[-1] + (tail_block_d - 1) * 16 - small_offset_ub

                        tik_instance.data_move(self.output_y_gm[dst_ub],
                                               input_z_ub[0],
                                               0,
                                               1,
                                               self.num_bit // 2,
                                               0,
                                               0)
                        if (tail_block_d - 1) > 0:
                            burstlen_ub2gm_sp = int((tail_block_d - 1) * 16 *
                                                    self.num_bit / DATA_MOVE_MIN_UNIT)
                            dst_gm_last = dst_gm + tail_d * d_split * 16 + \
                                          (use_time - 1) * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                                   input_y_ub_vconv[
                                                       (use_time - 1) * tail_block_d * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm_sp,
                                                   0,
                                                   0)

                        # recover the begin of map
                        # just recover [1,C,16,16]
                        src_x_index_recover = psm_map * merchant
                        nburst_gm2ub_recover = 1
                        burstlen_gm2ub_recover = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub_recover = 0
                        dststride_gm2ub_recover = 0

                        self.data_move_gm_ub_int32(tik_instance, input_y_ub_vconv, self.input_x_gm,
                                                   nburst_gm2ub_recover, burstlen_gm2ub_recover,
                                                   srcstride_gm2ub_recover,
                                                   dststride_gm2ub_recover, src_x_index_recover, 0)

                        with tik_instance.if_scope(use_time > 1):
                            dst_gm_recover = acture_memory * merchant
                            burstlen_ub2gm_recover = int(16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                            # ***第0行没有被错误覆盖***
                            with tik_instance.for_range(1, use_time) as i:
                                dst_gm_recover = dst_gm_recover + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_gm_recover],
                                                       input_y_ub_vconv[i * 1 * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm_recover,
                                                       0,
                                                       0)

                    # ***消0场景(横轴消0)，无法保证dst_stride是整数***
                    with tik_instance.else_scope():

                        dst_stride_ub2gm = (self.output_shape[-1] - tail_block_d * 16) * \
                                           self.num_bit / DATA_MOVE_MIN_UNIT
                        if dst_stride_ub2gm == math.ceil(dst_stride_ub2gm):
                            dst_stride_ub2gm = int(dst_stride_ub2gm)

                            if dst_stride_ub2gm <= MAX_STRIDE:
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm + tail_d * d_split * 16],
                                    input_y_ub_vconv[0],
                                    0,
                                    use_time,
                                    burstlen_ub2gm,
                                    src_stride_ub2gm,
                                    dst_stride_ub2gm)
                            else:

                                src_y_ub_index = 0
                                src_y_ub_gap = tail_block_d * 16
                                dst_gm_gap = self.output_shape[-1]
                                with tik_instance.for_range(0, use_time) as i:
                                    src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                                    dst_gm = dst_gm + i * dst_gm_gap
                                    tik_instance.data_move(
                                        self.output_y_gm[dst_gm + tail_d * d_split * 16],
                                        input_y_ub_vconv[src_y_ub_index],
                                        0,
                                        1,
                                        burstlen_ub2gm,
                                        0,
                                        0)
                        else:

                            dst_use_time_gm = dst_gm + tail_d * d_split * 16
                            with tik_instance.for_range(0, use_time) as i:
                                dst_use_time_gm = dst_use_time_gm + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                       input_y_ub_vconv[i * tail_block_d * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                    # deal tail_d data
                    with tik_instance.for_range(0, tail_d) as num_ub_loop:

                        src_x_index = psm_map * merchant + remainder * psm_map / self.input_shape[
                            -4]
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * psm_ub_out

                        nburst_gm2ub = d_split
                        burstlen_gm2ub = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                                   dststride_gm2ub, src_x_index, 0)

                        # use vconv instead of vadds
                        # vector maximum compute 128
                        vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                        vector_repeat_merchant_c = tiling_shape[-3] * 16 // MAX_REPEAT
                        vector_repeat_remainder_c = tiling_shape[-3] * 16 % MAX_REPEAT

                        src_ub_gap = MASK * (tiling_shape[-3] * 16 + 1)
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * (tiling_shape[-3] * 16 + 1) * \
                                         self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = MAX_REPEAT

                        self.reorder_s322f32_pattern_0(tik_instance, vector_repeat_merchant_d,
                                                       vector_repeat_remainder_d,
                                                       vector_repeat_merchant_c,
                                                       vector_repeat_remainder_c,
                                                       self.mask, input_y_ub, input_x_ub,
                                                       dst_ub_gap, src_ub_gap,
                                                       repeat_times, dst_blk_stride, src_blk_stride,
                                                       dst_rep_stride, src_rep_stride, nburst_gm2ub)

                        # all_rep must be int
                        all_rep = nburst_gm2ub * 16 * tiling_shape[-3] * 16 // self.mask
                        vconv_repeat_merchant = all_rep // 255
                        vconv_repeat_remainder = all_rep % 255

                        src_ub_gap_vconv = 255 * self.mask
                        dst_ub_gap_vconv = 255 * self.mask
                        src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times_vconv = 255

                        self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                     self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                     src_ub_gap_vconv, repeat_times_vconv,
                                     dst_blk_stride_vconv, src_blk_stride_vconv,
                                     dst_rep_stride_vconv, src_rep_stride_vconv)

                        # tail_block_d存在时，消0发生在tail_block_d
                        dst_gm = acture_memory * merchant + remainder * 16
                        use_time.set_as(self.output_shape[-2])
                        burstlen_ub2gm = int(d_split * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride_ub2gm = 0
                        dst_stride_ub2gm = (self.output_shape[-1] - d_split * 16) * \
                                           self.num_bit / DATA_MOVE_MIN_UNIT

                        if dst_stride_ub2gm == math.ceil(dst_stride_ub2gm):
                            dst_stride_ub2gm = int(dst_stride_ub2gm)

                            if dst_stride_ub2gm <= MAX_STRIDE:
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm + num_ub_loop * d_split * 16],
                                    input_y_ub_vconv[0],
                                    0,
                                    use_time,
                                    burstlen_ub2gm,
                                    src_stride_ub2gm,
                                    dst_stride_ub2gm)
                            else:
                                src_y_ub_index = 0
                                src_y_ub_gap = d_split * 16
                                dst_gm_gap = self.output_shape[-1]
                                with tik_instance.for_range(0, use_time) as i:
                                    src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                                    dst_gm = dst_gm + i * dst_gm_gap
                                    tik_instance.data_move(
                                        self.output_y_gm[dst_gm + num_ub_loop * d_split * 16],
                                        input_y_ub_vconv[src_y_ub_index],
                                        0,
                                        1,
                                        burstlen_ub2gm,
                                        0,
                                        0)
                        else:
                            dst_gm = dst_gm + num_ub_loop * d_split * 16
                            with tik_instance.for_range(0, use_time) as i:
                                dst_gm = dst_gm + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_gm],
                                                       input_y_ub_vconv[i * d_split * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                else:
                    # tail_block_d == 0, discrimination between [0,...,tail_d-1) and tail_d-1
                    # deal tail_d-1 firstly

                    merchant = (sum_core + num_core_loop) * tiling_shape[0] // self.input_shape[-4]
                    remainder = (sum_core + num_core_loop) * tiling_shape[0] % self.input_shape[-4]
                    src_x_index = psm_map * merchant + remainder * psm_map / self.input_shape[-4]
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + (tail_d - 1) * psm_ub_out

                    nburst_gm2ub = d_split
                    burstlen_gm2ub = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = 0
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # use vconv instead of vadds
                    # vector maximum compute 128
                    vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                    vector_repeat_merchant_c = tiling_shape[-3] * 16 // MAX_REPEAT
                    vector_repeat_remainder_c = tiling_shape[-3] * 16 % MAX_REPEAT

                    src_ub_gap = MASK * (tiling_shape[-3] * 16 + 1)
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * (tiling_shape[-3] * 16 + 1) * \
                                     self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = MAX_REPEAT

                    self.reorder_s322f32_pattern_0(tik_instance, vector_repeat_merchant_d,
                                                   vector_repeat_remainder_d,
                                                   vector_repeat_merchant_c,
                                                   vector_repeat_remainder_c,
                                                   self.mask, input_y_ub, input_x_ub, dst_ub_gap,
                                                   src_ub_gap,
                                                   repeat_times, dst_blk_stride, src_blk_stride,
                                                   dst_rep_stride, src_rep_stride, nburst_gm2ub)

                    # all_rep must be int
                    all_rep = psm_ub_out // self.mask
                    vconv_repeat_merchant = all_rep // 255
                    vconv_repeat_remainder = all_rep % 255

                    src_ub_gap_vconv = 255 * self.mask
                    dst_ub_gap_vconv = 255 * self.mask
                    src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times_vconv = 255

                    self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                 self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                 src_ub_gap_vconv, repeat_times_vconv,
                                 dst_blk_stride_vconv, src_blk_stride_vconv,
                                 dst_rep_stride_vconv, src_rep_stride_vconv)

                    # -----------------------Out-----------------------------------#
                    # 判断处理的tiling_shape是否为一张map的最后一个
                    # 判断处理的tail_d(i)是否为tiling_shape的最后一个
                    # 条件融合：判断tail_d(i)是否为map的结尾
                    # 是(flag == 0)：按照特殊方式输出，否(flag != 0)：按照常规方式输出
                    flag = (remainder + tiling_shape[0]) % self.input_shape[-4]
                    dst_gm = acture_memory * merchant + remainder * 16
                    use_time.set_as(self.output_shape[-2])
                    burstlen_ub2gm = int(nburst_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    src_stride_ub2gm = 0
                    dst_stride_ub2gm = 0

                    with tik_instance.if_scope(flag == 0):

                        with tik_instance.for_range(0, use_time - 1) as i:
                            dst_use_time_gm = dst_gm + (tail_d - 1) * d_split * 16 + i * \
                                              self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub_vconv[i * d_split * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   src_stride_ub2gm,
                                                   dst_stride_ub2gm)

                        small_offset_ub = self.input_shape[-1] * \
                                          self.input_shape[-4] - self.output_shape[-1]

                        src_ub = src_x_index_temp + (tiling_shape[-4] - 1) * \
                                 tiling_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                        tik_instance.data_move(input_z_ub[0],
                                               self.input_x_gm[src_ub],
                                               0,
                                               1,
                                               self.num_bit // 2,
                                               0,
                                               0, )

                        dst_ub = dst_gm + (tail_d - 1) * d_split * 16 + (use_time - 1) * \
                                 self.output_shape[-1] + (d_split - 1) * 16 - small_offset_ub

                        tik_instance.data_move(self.output_y_gm[dst_ub],
                                               input_z_ub[0],
                                               0,
                                               1,
                                               self.num_bit // 2,
                                               0,
                                               0)

                        burstlen_ub2gm_sp = int(
                            (d_split - 1) * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                        dst_gm_last = dst_gm + (tail_d - 1) * d_split * 16 + \
                                      (use_time - 1) * self.output_shape[-1]

                        tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                               input_y_ub_vconv[(use_time - 1) * d_split * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm_sp,
                                               0,
                                               0)

                        # recover the begin of map
                        # just recover [1,C,16,16]
                        src_x_index_recover = psm_map * merchant
                        nburst_gm2ub_recover = 1
                        burstlen_gm2ub_recover = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub_recover = 0
                        dststride_gm2ub_recover = 0

                        self.data_move_gm_ub_int32(tik_instance, input_y_ub_vconv, self.input_x_gm,
                                                   nburst_gm2ub_recover, burstlen_gm2ub_recover,
                                                   srcstride_gm2ub_recover,
                                                   dststride_gm2ub_recover, src_x_index_recover, 0)

                        with tik_instance.if_scope(use_time > 1):
                            dst_gm_recover = acture_memory * merchant
                            burstlen_ub2gm_recover = int(16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                            # ***第0行没有被错误覆盖***
                            with tik_instance.for_range(1, use_time) as i:
                                dst_gm_recover = dst_gm_recover + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_gm_recover],
                                                       input_y_ub_vconv[i * 1 * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm_recover,
                                                       0,
                                                       0)
                    # ***消0场景(横轴消0)，无法保证dst_stride是整数***
                    with tik_instance.else_scope():

                        dst_use_time_gm = dst_gm + (tail_d - 1) * d_split * 16
                        with tik_instance.for_range(0, use_time) as i:
                            dst_use_time_gm = dst_use_time_gm + i * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub_vconv[i * d_split * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)
                    # deal tail_d-1 data
                    with tik_instance.for_range(0, tail_d - 1) as num_ub_loop:

                        src_x_index = psm_map * merchant + remainder * psm_map / self.input_shape[
                            -4]
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * psm_ub_out

                        nburst_gm2ub = d_split
                        burstlen_gm2ub = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                                   dststride_gm2ub, src_x_index, 0)

                        # use vconv instead of vadds
                        # vector maximum compute 128
                        vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                        vector_repeat_merchant_c = tiling_shape[-3] * 16 // MAX_REPEAT
                        vector_repeat_remainder_c = tiling_shape[-3] * 16 % MAX_REPEAT

                        src_ub_gap = MASK * (tiling_shape[-3] * 16 + 1)
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * (tiling_shape[-3] * 16 + 1) * \
                                         self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = MAX_REPEAT

                        self.reorder_s322f32_pattern_0(tik_instance, vector_repeat_merchant_d,
                                                       vector_repeat_remainder_d,
                                                       vector_repeat_merchant_c,
                                                       vector_repeat_remainder_c,
                                                       self.mask, input_y_ub, input_x_ub,
                                                       dst_ub_gap, src_ub_gap,
                                                       repeat_times, dst_blk_stride, src_blk_stride,
                                                       dst_rep_stride, src_rep_stride, nburst_gm2ub)

                        # all_rep must be int
                        all_rep = psm_ub_out // self.mask
                        vconv_repeat_merchant = all_rep // 255
                        vconv_repeat_remainder = all_rep % 255

                        src_ub_gap_vconv = 255 * self.mask
                        dst_ub_gap_vconv = 255 * self.mask
                        src_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride_vconv = 8 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride_vconv = 64 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times_vconv = 255

                        self.f322s32(tik_instance, vconv_repeat_merchant, vconv_repeat_remainder,
                                     self.mask, input_y_ub_vconv, input_y_ub, dst_ub_gap_vconv,
                                     src_ub_gap_vconv, repeat_times_vconv,
                                     dst_blk_stride_vconv, src_blk_stride_vconv,
                                     dst_rep_stride_vconv, src_rep_stride_vconv)

                        dst_gm = acture_memory * merchant + remainder * 16
                        use_time.set_as(self.output_shape[-2])
                        burstlen_ub2gm = int(d_split * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride_ub2gm = 0
                        dst_stride_ub2gm = (self.output_shape[-1] - d_split * 16) * \
                                           self.num_bit / DATA_MOVE_MIN_UNIT

                        if dst_stride_ub2gm == math.ceil(dst_stride_ub2gm):
                            dst_stride_ub2gm = int(dst_stride_ub2gm)
                            if dst_stride_ub2gm <= MAX_STRIDE:

                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm + num_ub_loop * d_split * 16],
                                    input_y_ub_vconv[0],
                                    0,
                                    use_time,
                                    burstlen_ub2gm,
                                    src_stride_ub2gm,
                                    dst_stride_ub2gm)
                            else:
                                src_y_ub_index = 0
                                src_y_ub_gap = d_split * 16
                                dst_gm_gap = self.output_shape[-1]
                                with tik_instance.for_range(0, use_time) as i:
                                    src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                                    dst_gm = dst_gm + i * dst_gm_gap
                                    tik_instance.data_move(
                                        self.output_y_gm[dst_gm + num_ub_loop * d_split * 16],
                                        input_y_ub_vconv[src_y_ub_index],
                                        0,
                                        1,
                                        burstlen_ub2gm,
                                        0,
                                        0)
                        else:
                            dst_gm = dst_gm + num_ub_loop * d_split * 16
                            with tik_instance.for_range(0, use_time) as i:
                                dst_gm = dst_gm + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_gm],
                                                       input_y_ub_vconv[i * d_split * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

    def nz2nd_special_case2_int32_pattern_zero_mini(self, tik_instance, params):
        """
        padding
        [A,B,62500,1,16,16]
        A,B = 1,1
        total_num = 50
        output = [1,100W]
        [1250,1,16,16]
        support int32
        int32->vadd->int32
        """
        # if tiling_shape > maximum_size_ub, split D_axis ==> tail_d
        # in this case, C_axis * A * B < 32, so C_axis can not be splited
        core_num = params.get("core_num")
        total_core_loop_num = params.get("total_core_loop_num")
        tiling_shape = params.get("tiling_shape")
        psm = params.get("psm")

        core_number = core_num
        acture_memory = self.output_shape[-1] * self.output_shape[-2]
        psm_map = self.calc_element(self.input_shape[-4:])
        c_value = tiling_shape[1]
        d_value = tiling_shape[0]
        for d_split in range(d_value):
            d_split = d_split + 1
            psm_ub_in = (c_value * 16 + 1) * (d_split * 16)
            psm_ub_out = d_split * 16 * c_value * 16
            if psm_ub_in > self.maximum_size_ub:
                d_split = d_split - 1
                psm_ub_in = (c_value * 16 + 1) * (d_split * 16)
                psm_ub_out = d_split * 16 * c_value * 16
                break

        # tail_d must be >= 1
        tail_d = d_value // d_split
        tail_block_d = d_value % d_split
        use_time = tik_instance.Scalar(dtype="uint64")

        input_x_ub = tik_instance.Tensor(self.dtype, [psm_ub_in, ], name="input_x_ub",
                                         scope=tik.scope_ubuf)

        input_y_ub = tik_instance.Tensor(self.dtype, [psm_ub_out, ], name="input_y_ub",
                                         scope=tik.scope_ubuf)

        input_z_ub = tik_instance.Tensor(self.dtype, [16, ], name="input_z_ub",
                                         scope=tik.scope_ubuf)

        input_v_ub = tik_instance.Tensor(self.dtype, [16, ], name="input_v_ub",
                                         scope=tik.scope_ubuf)

        with tik_instance.for_range(0, core_number, block_num=core_number) as num_core:

            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)

            with tik_instance.for_range(0, core_loop) as num_core_loop:

                self.set_vector_dup(tik_instance, 16, input_v_ub, 0)

                if tail_block_d != 0:
                    # deal tail_block firstly
                    # merchant: represent which map [D,C,16,16]
                    # remainder: pos in one map [d1,C,16,16]
                    merchant = (sum_core + num_core_loop) * tiling_shape[0] // self.input_shape[-4]
                    remainder = (sum_core + num_core_loop) * tiling_shape[0] % self.input_shape[-4]
                    src_x_index = psm_map * merchant + remainder * psm_map / self.input_shape[-4]
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + tail_d * psm_ub_out

                    nburst_gm2ub = tail_block_d
                    burstlen_gm2ub = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = 0
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # use vconv instead of vadds
                    # vector maximum compute 128
                    vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                    vector_repeat_merchant_c = tiling_shape[-3] * 16 // MAX_REPEAT
                    vector_repeat_remainder_c = tiling_shape[-3] * 16 % MAX_REPEAT

                    src_ub_gap = MASK * (tiling_shape[-3] * 16 + 1)
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * (tiling_shape[-3] * 16 + 1) * \
                                     self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = tail_block_d * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = MAX_REPEAT

                    self.reorder_vadd_pattern_0_mini(tik_instance, vector_repeat_merchant_d,
                                                     vector_repeat_remainder_d,
                                                     vector_repeat_merchant_c,
                                                     vector_repeat_remainder_c,
                                                     self.mask, input_y_ub, input_x_ub, input_v_ub,
                                                     dst_ub_gap, src_ub_gap,
                                                     repeat_times, dst_blk_stride, src_blk_stride,
                                                     dst_rep_stride, src_rep_stride, nburst_gm2ub)

                    # -----------------------Out-----------------------------------#
                    # 判断处理的tiling_shape是否为一张map的最后一个
                    # 判断处理的tail_block_d是否为tiling_shape的最后一个
                    # 条件融合：当tail_block_d存在时，判断其是否为map的结尾
                    # 是(flag == 0)：按照特殊方式输出，否(flag != 0)：按照常规方式输出
                    flag = (remainder + tiling_shape[0]) % self.input_shape[-4]
                    dst_gm = acture_memory * merchant + remainder * 16
                    use_time.set_as(self.output_shape[-2])
                    burstlen_ub2gm = int(tail_block_d * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    src_stride_ub2gm = 0
                    dst_stride_ub2gm = 0

                    with tik_instance.if_scope(flag == 0):
                        with tik_instance.for_range(0, use_time - 1) as i:
                            dst_use_time_gm = dst_gm + tail_d * d_split * 16 + i * \
                                              self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub[i * tail_block_d * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   src_stride_ub2gm,
                                                   dst_stride_ub2gm)

                        small_offset_ub = self.input_shape[-1] * \
                                          self.input_shape[-4] - self.output_shape[-1]

                        src_ub = src_x_index_temp + (tiling_shape[-4] - 1) * \
                                 tiling_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                        tik_instance.data_move(input_z_ub[0],
                                               self.input_x_gm[src_ub],
                                               0,
                                               1,
                                               self.num_bit // 2,
                                               0,
                                               0, )

                        dst_ub = dst_gm + tail_d * d_split * 16 + (use_time - 1) * \
                                 self.output_shape[-1] + (tail_block_d - 1) * 16 - small_offset_ub

                        tik_instance.data_move(self.output_y_gm[dst_ub],
                                               input_z_ub[0],
                                               0,
                                               1,
                                               self.num_bit // 2,
                                               0,
                                               0)
                        if (tail_block_d - 1) > 0:
                            burstlen_ub2gm_sp = int((tail_block_d - 1) * 16 *
                                                    self.num_bit / DATA_MOVE_MIN_UNIT)
                            dst_gm_last = dst_gm + tail_d * d_split * 16 + \
                                          (use_time - 1) * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                                   input_y_ub[(use_time - 1) * tail_block_d * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm_sp,
                                                   0,
                                                   0)

                        # recover the begin of map
                        # just recover [1,C,16,16]
                        src_x_index_recover = psm_map * merchant
                        nburst_gm2ub_recover = 1
                        burstlen_gm2ub_recover = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub_recover = 0
                        dststride_gm2ub_recover = 0

                        self.data_move_gm_ub_int32(tik_instance, input_y_ub, self.input_x_gm,
                                                   nburst_gm2ub_recover, burstlen_gm2ub_recover,
                                                   srcstride_gm2ub_recover,
                                                   dststride_gm2ub_recover, src_x_index_recover, 0)

                        with tik_instance.if_scope(use_time > 1):
                            dst_gm_recover = acture_memory * merchant
                            burstlen_ub2gm_recover = int(16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                            # ***第0行没有被错误覆盖***
                            with tik_instance.for_range(1, use_time) as i:
                                dst_gm_recover = dst_gm_recover + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_gm_recover],
                                                       input_y_ub[i * 1 * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm_recover,
                                                       0,
                                                       0)

                    # ***消0场景(横轴消0)，无法保证dst_stride是整数***
                    with tik_instance.else_scope():

                        dst_stride_ub2gm = (self.output_shape[-1] - tail_block_d * 16) * \
                                           self.num_bit / DATA_MOVE_MIN_UNIT
                        if dst_stride_ub2gm == math.ceil(dst_stride_ub2gm):
                            dst_stride_ub2gm = int(dst_stride_ub2gm)

                            if dst_stride_ub2gm <= MAX_STRIDE:
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm + tail_d * d_split * 16],
                                    input_y_ub[0],
                                    0,
                                    use_time,
                                    burstlen_ub2gm,
                                    src_stride_ub2gm,
                                    dst_stride_ub2gm)
                            else:

                                src_y_ub_index = 0
                                src_y_ub_gap = tail_block_d * 16
                                dst_gm_gap = self.output_shape[-1]
                                with tik_instance.for_range(0, use_time) as i:
                                    src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                                    dst_gm = dst_gm + i * dst_gm_gap
                                    tik_instance.data_move(
                                        self.output_y_gm[dst_gm + tail_d * d_split * 16],
                                        input_y_ub[src_y_ub_index],
                                        0,
                                        1,
                                        burstlen_ub2gm,
                                        0,
                                        0)
                        else:

                            dst_use_time_gm = dst_gm + tail_d * d_split * 16
                            with tik_instance.for_range(0, use_time) as i:
                                dst_use_time_gm = dst_use_time_gm + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                       input_y_ub[i * tail_block_d * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                    # deal tail_d data
                    with tik_instance.for_range(0, tail_d) as num_ub_loop:

                        src_x_index = psm_map * merchant + remainder * \
                                      psm_map / self.input_shape[-4]
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * psm_ub_out

                        nburst_gm2ub = d_split
                        burstlen_gm2ub = tiling_shape[-3] * 16 * \
                                         16 * self.num_bit // 32
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub,
                                                   self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub,
                                                   srcstride_gm2ub,
                                                   dststride_gm2ub,
                                                   src_x_index, 0)

                        # use vconv instead of vadds
                        # vector maximum compute 128
                        vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                        vector_repeat_merchant_c = tiling_shape[-3] * 16 \
                                                   // MAX_REPEAT
                        vector_repeat_remainder_c = tiling_shape[-3] * 16 \
                                                    % MAX_REPEAT

                        src_ub_gap = MASK * (tiling_shape[-3] * 16 + 1)
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * (tiling_shape[-3] * 16 + 1) * \
                                         self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // \
                                         DATA_MOVE_MIN_UNIT
                        repeat_times = MAX_REPEAT

                        self.reorder_vadd_pattern_0_mini(
                            tik_instance, vector_repeat_merchant_d,
                            vector_repeat_remainder_d,
                            vector_repeat_merchant_c,
                            vector_repeat_remainder_c,
                            self.mask, input_y_ub, input_x_ub,
                            input_v_ub, dst_ub_gap, src_ub_gap,
                            repeat_times, dst_blk_stride, src_blk_stride,
                            dst_rep_stride, src_rep_stride, nburst_gm2ub)

                        # tail_block_d存在时，消0发生在tail_block_d
                        dst_gm = acture_memory * merchant + remainder * 16
                        use_time.set_as(self.output_shape[-2])
                        burstlen_ub2gm = int(d_split * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride_ub2gm = 0
                        dst_stride_ub2gm = (self.output_shape[-1] - d_split * 16) * \
                                           self.num_bit / DATA_MOVE_MIN_UNIT

                        if dst_stride_ub2gm == math.ceil(dst_stride_ub2gm):
                            dst_stride_ub2gm = int(dst_stride_ub2gm)

                            if dst_stride_ub2gm <= MAX_STRIDE:
                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm + num_ub_loop * d_split * 16],
                                    input_y_ub[0],
                                    0,
                                    use_time,
                                    burstlen_ub2gm,
                                    src_stride_ub2gm,
                                    dst_stride_ub2gm)
                            else:
                                src_y_ub_index = 0
                                src_y_ub_gap = d_split * 16
                                dst_gm_gap = self.output_shape[-1]
                                with tik_instance.for_range(0, use_time) as i:
                                    src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                                    dst_gm = dst_gm + i * dst_gm_gap
                                    tik_instance.data_move(
                                        self.output_y_gm[dst_gm + num_ub_loop * d_split * 16],
                                        input_y_ub[src_y_ub_index],
                                        0,
                                        1,
                                        burstlen_ub2gm,
                                        0,
                                        0)
                        else:
                            dst_gm = dst_gm + num_ub_loop * d_split * 16
                            with tik_instance.for_range(0, use_time) as i:
                                dst_gm = dst_gm + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_gm],
                                                       input_y_ub[i * d_split * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

                else:
                    # tail_block_d == 0, discrimination between [0,...,tail_d-1) and tail_d-1
                    # deal tail_d-1 firstly

                    merchant = (sum_core + num_core_loop) * tiling_shape[0] // self.input_shape[-4]
                    remainder = (sum_core + num_core_loop) * tiling_shape[0] % self.input_shape[-4]
                    src_x_index = psm_map * merchant + remainder * psm_map / self.input_shape[-4]
                    src_x_index_temp = src_x_index
                    src_x_index = src_x_index_temp + (tail_d - 1) * psm_ub_out

                    nburst_gm2ub = d_split
                    burstlen_gm2ub = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                    srcstride_gm2ub = 0
                    dststride_gm2ub = 16 * self.num_bit // 32

                    self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                               nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                               dststride_gm2ub, src_x_index, 0)

                    # use vconv instead of vadds
                    # vector maximum compute 128
                    vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                    vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                    vector_repeat_merchant_c = tiling_shape[-3] * 16 // MAX_REPEAT
                    vector_repeat_remainder_c = tiling_shape[-3] * 16 % MAX_REPEAT

                    src_ub_gap = MASK * (tiling_shape[-3] * 16 + 1)
                    dst_ub_gap = MASK
                    src_blk_stride = 16 * (tiling_shape[-3] * 16 + 1) * \
                                     self.num_bit // DATA_MOVE_MIN_UNIT
                    src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                    repeat_times = MAX_REPEAT

                    self.reorder_vadd_pattern_0_mini(tik_instance, vector_repeat_merchant_d,
                                                     vector_repeat_remainder_d,
                                                     vector_repeat_merchant_c,
                                                     vector_repeat_remainder_c,
                                                     self.mask, input_y_ub, input_x_ub, input_v_ub,
                                                     dst_ub_gap, src_ub_gap,
                                                     repeat_times, dst_blk_stride, src_blk_stride,
                                                     dst_rep_stride, src_rep_stride, nburst_gm2ub)

                    # -----------------------Out-----------------------------------#
                    # 判断处理的tiling_shape是否为一张map的最后一个
                    # 判断处理的tail_d(i)是否为tiling_shape的最后一个
                    # 条件融合：判断tail_d(i)是否为map的结尾
                    # 是(flag == 0)：按照特殊方式输出，否(flag != 0)：按照常规方式输出
                    flag = (remainder + tiling_shape[0]) % self.input_shape[-4]
                    dst_gm = acture_memory * merchant + remainder * 16
                    use_time.set_as(self.output_shape[-2])
                    burstlen_ub2gm = int(nburst_gm2ub * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                    src_stride_ub2gm = 0
                    dst_stride_ub2gm = 0

                    with tik_instance.if_scope(flag == 0):

                        with tik_instance.for_range(0, use_time - 1) as i:
                            dst_use_time_gm = dst_gm + (tail_d - 1) * d_split * 16 + i * \
                                              self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub[i * d_split * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   src_stride_ub2gm,
                                                   dst_stride_ub2gm)

                        small_offset_ub = self.input_shape[-1] * \
                                          self.input_shape[-4] - self.output_shape[-1]

                        src_ub = src_x_index_temp + (tiling_shape[-4] - 1) * \
                                 tiling_shape[-3] * 16 * 16 + (use_time - 1) * 16 - small_offset_ub

                        tik_instance.data_move(input_z_ub[0],
                                               self.input_x_gm[src_ub],
                                               0,
                                               1,
                                               self.num_bit // 2,
                                               0,
                                               0, )

                        dst_ub = dst_gm + (tail_d - 1) * d_split * 16 + (use_time - 1) * \
                                 self.output_shape[-1] + (d_split - 1) * 16 - small_offset_ub

                        tik_instance.data_move(self.output_y_gm[dst_ub],
                                               input_z_ub[0],
                                               0,
                                               1,
                                               self.num_bit // 2,
                                               0,
                                               0)

                        burstlen_ub2gm_sp = int(
                            (d_split - 1) * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                        dst_gm_last = dst_gm + (tail_d - 1) * d_split * 16 + \
                                      (use_time - 1) * self.output_shape[-1]

                        tik_instance.data_move(self.output_y_gm[dst_gm_last],
                                               input_y_ub[(use_time - 1) * d_split * 16],
                                               0,
                                               1,
                                               burstlen_ub2gm_sp,
                                               0,
                                               0)

                        # recover the begin of map
                        # just recover [1,C,16,16]
                        src_x_index_recover = psm_map * merchant
                        nburst_gm2ub_recover = 1
                        burstlen_gm2ub_recover = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub_recover = 0
                        dststride_gm2ub_recover = 0

                        self.data_move_gm_ub_int32(tik_instance, input_y_ub, self.input_x_gm,
                                                   nburst_gm2ub_recover, burstlen_gm2ub_recover,
                                                   srcstride_gm2ub_recover,
                                                   dststride_gm2ub_recover, src_x_index_recover, 0)

                        with tik_instance.if_scope(use_time > 1):
                            dst_gm_recover = acture_memory * merchant
                            burstlen_ub2gm_recover = int(16 * self.num_bit / DATA_MOVE_MIN_UNIT)

                            # ***第0行没有被错误覆盖***
                            with tik_instance.for_range(1, use_time) as i:
                                dst_gm_recover = dst_gm_recover + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_gm_recover],
                                                       input_y_ub[i * 1 * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm_recover,
                                                       0,
                                                       0)
                    # ***消0场景(横轴消0)，无法保证dst_stride是整数***
                    with tik_instance.else_scope():

                        dst_use_time_gm = dst_gm + (tail_d - 1) * d_split * 16
                        with tik_instance.for_range(0, use_time) as i:
                            dst_use_time_gm = dst_use_time_gm + i * self.output_shape[-1]
                            tik_instance.data_move(self.output_y_gm[dst_use_time_gm],
                                                   input_y_ub[i * d_split * 16],
                                                   0,
                                                   1,
                                                   burstlen_ub2gm,
                                                   0,
                                                   0)
                    # deal tail_d-1 data
                    with tik_instance.for_range(0, tail_d - 1) as num_ub_loop:

                        src_x_index = psm_map * merchant + remainder * \
                                      psm_map / self.input_shape[-4]
                        src_x_index_temp = src_x_index
                        src_x_index = src_x_index_temp + num_ub_loop * psm_ub_out

                        nburst_gm2ub = d_split
                        burstlen_gm2ub = tiling_shape[-3] * 16 * 16 * self.num_bit // 32
                        srcstride_gm2ub = 0
                        dststride_gm2ub = 16 * self.num_bit // 32

                        self.data_move_gm_ub_int32(tik_instance, input_x_ub, self.input_x_gm,
                                                   nburst_gm2ub, burstlen_gm2ub, srcstride_gm2ub,
                                                   dststride_gm2ub, src_x_index, 0)

                        # use vconv instead of vadds
                        # vector maximum compute 128
                        vector_repeat_merchant_d = nburst_gm2ub * 16 // MASK
                        vector_repeat_remainder_d = nburst_gm2ub * 16 % MASK
                        vector_repeat_merchant_c = tiling_shape[-3] * 16 // MAX_REPEAT
                        vector_repeat_remainder_c = tiling_shape[-3] * 16 % MAX_REPEAT

                        src_ub_gap = MASK * (tiling_shape[-3] * 16 + 1)
                        dst_ub_gap = MASK
                        src_blk_stride = 16 * (tiling_shape[-3] * 16 + 1) * \
                                         self.num_bit // DATA_MOVE_MIN_UNIT
                        src_rep_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_blk_stride = 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        dst_rep_stride = nburst_gm2ub * 16 * self.num_bit // DATA_MOVE_MIN_UNIT
                        repeat_times = MAX_REPEAT

                        self.reorder_vadd_pattern_0_mini(tik_instance, vector_repeat_merchant_d,
                                                         vector_repeat_remainder_d,
                                                         vector_repeat_merchant_c,
                                                         vector_repeat_remainder_c,
                                                         self.mask, input_y_ub, input_x_ub,
                                                         input_v_ub, dst_ub_gap, src_ub_gap,
                                                         repeat_times, dst_blk_stride,
                                                         src_blk_stride,
                                                         dst_rep_stride, src_rep_stride,
                                                         nburst_gm2ub)

                        dst_gm = acture_memory * merchant + remainder * 16
                        use_time.set_as(self.output_shape[-2])
                        burstlen_ub2gm = int(d_split * 16 * self.num_bit / DATA_MOVE_MIN_UNIT)
                        src_stride_ub2gm = 0
                        dst_stride_ub2gm = (self.output_shape[-1] - d_split * 16) * \
                                           self.num_bit / DATA_MOVE_MIN_UNIT

                        if dst_stride_ub2gm == math.ceil(dst_stride_ub2gm):
                            dst_stride_ub2gm = int(dst_stride_ub2gm)
                            if dst_stride_ub2gm <= MAX_STRIDE:

                                tik_instance.data_move(
                                    self.output_y_gm[dst_gm + num_ub_loop * d_split * 16],
                                    input_y_ub[0],
                                    0,
                                    use_time,
                                    burstlen_ub2gm,
                                    src_stride_ub2gm,
                                    dst_stride_ub2gm)
                            else:
                                src_y_ub_index = 0
                                src_y_ub_gap = d_split * 16
                                dst_gm_gap = self.output_shape[-1]
                                with tik_instance.for_range(0, use_time) as i:
                                    src_y_ub_index = src_y_ub_index + i * src_y_ub_gap
                                    dst_gm = dst_gm + i * dst_gm_gap
                                    tik_instance.data_move(
                                        self.output_y_gm[dst_gm + num_ub_loop * d_split * 16],
                                        input_y_ub[src_y_ub_index],
                                        0,
                                        1,
                                        burstlen_ub2gm,
                                        0,
                                        0)
                        else:
                            dst_gm = dst_gm + num_ub_loop * d_split * 16
                            with tik_instance.for_range(0, use_time) as i:
                                dst_gm = dst_gm + i * self.output_shape[-1]
                                tik_instance.data_move(self.output_y_gm[dst_gm],
                                                       input_y_ub[i * d_split * 16],
                                                       0,
                                                       1,
                                                       burstlen_ub2gm,
                                                       0,
                                                       0)

    def nz_2_nd_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        if self.dtype == "int32":
            if not tbe_platform.cce_conf.api_check_support("tik.vconv", "s322f32"):
                self.maximum_size_ub = TOTAL_UB_MEMORY // self.num_bit
            else:
                self.maximum_size_ub = int(TOTAL_UB_MEMORY * 2 // self.num_bit) // 3
        condition_0 = self.input_sum_elements == self.output_sum_elements
        condition_1 = self.input_shape[-4] == 1
        condition_2 = self.output_shape[-1] * self.output_shape[-2] >= 16

        # not zero suppression
        if condition_0:
            params = self.split_shape_nzs(self.input_shape)
            self.pattern_case_zero(tik_instance, params, self.dtype)

        # zero suppression
        else:
            if condition_1 and condition_2:
                params = self.split_shape(self.input_shape)
                self.pattern_case_one(tik_instance, params, self.dtype)

            elif condition_1 and not condition_2:
                params = self.split_shape(self.input_shape)
                self.pattern_case_two(tik_instance, params)

            else:
                params = self.split_shape_zs(self.input_shape)
                self.pattern_case_three(tik_instance, params, self.dtype)

        return tik_instance


# pylint: disable=invalid-name,unused-argument
@util.check_input_type(dict, dict, str, str, str)
def nz_2_nd(src, dst, src_format, dst_format, kernel_name="Nz2ND"):
    """
    algorithm: Nz2Nd

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
        kernel name, default value is "Nz2Nd"

    Returns
    -------
    tik_instance : tik_instance
    """
    input_shape = src.get("shape")
    output_shape = dst.get("shape")
    dtype = src.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)
    check_list = ("float16", "float32", "int32")
    util.check_dtype_rule(dtype, check_list)
    src_format = src_format.upper()

    if src_format != "FRACTAL_NZ":
        raise RuntimeError("The src_format of Nz2Nd only support FRACTAL_NZ.")

    if dst_format.upper() not in {"ND", "NHWC", "NCHW"}:
        raise RuntimeError("The dst_format of Nz2Nd only support"
                           "ND, NHWC, NCHW.")

    result = Nz2NDCompute(input_shape, output_shape, dtype, kernel_name)

    return result.get_tik_instance()
